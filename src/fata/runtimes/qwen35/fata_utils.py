#!/usr/bin/env python
"""
FATA Cross-Model Utility Functions.
Dynamic token importance alignment, reconstruction, and answer scoring.
"""

import re
import torch
import torch.nn.functional as F


# ===========================================================================
# 1. Token Importance Mapping
# ===========================================================================

def get_merged_token_importance_qwen(raw_attention, image_grid_thw, merge_size=2):
    """
    Map Qwen3.5 raw vision attention to merged-token importance.

    Qwen has NO CLS token. Uses candidate model-agnostic importance:
    mean incoming attention across all queries per key token.

    Args:
        raw_attention: [batch*heads or flat, seq_len, seq_len] or [heads, seq, seq]
                       from the last vision attention block.
        image_grid_thw: tensor [[T, H, W]] defining grid dimensions in patches.
        merge_size: spatial merge factor (default 2 for Qwen3.5).

    Returns:
        merged_importance: [N_merged] importance scores, sum preserved.
    """
    # Handle various input shapes
    if raw_attention.dim() == 4:
        # [B, heads, seq, seq] -> mean over heads -> [B, seq, seq]
        attn = raw_attention.mean(dim=1)
    elif raw_attention.dim() == 3:
        attn = raw_attention
    elif raw_attention.dim() == 2:
        # [seq, seq] — single matrix
        attn = raw_attention.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected attention shape: {raw_attention.shape}")

    # Model-agnostic: mean incoming attention over all query tokens
    raw_score = attn.mean(dim=1)  # [B, seq] — mean attention each key token receives

    B = raw_score.shape[0]
    thw = image_grid_thw[0] if image_grid_thw.ndim == 2 else image_grid_thw
    T, H, W = int(thw[0].item()), int(thw[1].item()), int(thw[2].item())

    merged_scores = []
    for b in range(B):
        score = raw_score[b]  # [seq]
        seq_len = score.shape[0]
        expected_len = H * W * T

        if seq_len != expected_len:
            # Try truncating or padding
            if seq_len > expected_len:
                score = score[:expected_len]
            else:
                score = F.pad(score, (0, expected_len - seq_len))

        # Reshape to 2D spatial grid: [H, W] (T=1 for images)
        score_2d = score.reshape(H, W)

        # 2x2 spatial merge by summing
        # Pad if needed
        if H % merge_size != 0 or W % merge_size != 0:
            pad_h = (merge_size - H % merge_size) % merge_size
            pad_w = (merge_size - W % merge_size) % merge_size
            score_2d = F.pad(score_2d, (0, pad_w, 0, pad_h))

        merged_H = H // merge_size
        merged_W = W // merge_size

        # Reshape and sum over merge windows
        score_pooled = score_2d.reshape(merged_H, merge_size, merged_W, merge_size)
        score_pooled = score_pooled.sum(dim=(1, 3))  # [merged_H, merged_W]
        merged_score = score_pooled.flatten()

        merged_scores.append(merged_score)

    return torch.stack(merged_scores).squeeze(0)  # [N_merged] or [B, N_merged]


def get_merged_token_importance_internvl(cls_to_patch_attention, n_tiles):
    """
    Map InternVL CLS-to-patch attention to post-pixel-shuffle token importance.

    Args:
        cls_to_patch_attention: [tiles, heads, 1024] or [tiles, 1024]
                                CLS (token 0) attention to patch tokens 1..1024.
        n_tiles: number of tiles.

    Returns:
        merged_importance: [tiles * 256] importance scores.
    """
    if cls_to_patch_attention.dim() == 3:
        # [tiles, heads, 1024] -> cross-head mean
        attn = cls_to_patch_attention.mean(dim=1)  # [tiles, 1024]
    else:
        attn = cls_to_patch_attention  # [tiles, 1024]

    # Pixel shuffle: 1024 = 32x32 -> 256 = 16x16 (2x2 pooling)
    merged_scores = []
    for t in range(n_tiles):
        score_2d = attn[t].reshape(32, 32)  # 32x32 patch grid
        # 2x2 average pooling
        score_pooled = F.avg_pool2d(
            score_2d.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2
        ).squeeze()  # [16, 16]
        merged_scores.append(score_pooled.flatten())  # [256]

    return torch.cat(merged_scores, dim=0)  # [tiles * 256]


# ===========================================================================
# 2. Reconstruction
# ===========================================================================

def reconstruct_to_dynamic_length(original_embeddings, compressed_embeddings, target_length=None):
    """
    Reconstruct compressed embeddings back to original length using cosine NN.

    Args:
        original_embeddings: [N_full, D] — used for similarity reference.
        compressed_embeddings: [K, D] — selected tokens.
        target_length: output length (defaults to original_embeddings.shape[0]).

    Returns:
        reconstructed: [target_length, D]
        nn_indices: [target_length] — which compressed token each output maps to.
    """
    if target_length is None:
        target_length = original_embeddings.shape[0]

    N_full = original_embeddings.shape[0]
    K = compressed_embeddings.shape[0]
    D = compressed_embeddings.shape[1]

    # Truncate or pad original if needed
    if N_full > target_length:
        ref = original_embeddings[:target_length]
    elif N_full < target_length:
        ref = F.pad(original_embeddings, (0, 0, 0, target_length - N_full))
    else:
        ref = original_embeddings

    # Cosine similarity in FP32
    ref_norm = F.normalize(ref.float(), dim=-1)
    comp_norm = F.normalize(compressed_embeddings.float(), dim=-1)

    similarity = ref_norm @ comp_norm.T  # [target_length, K]
    nn_indices = similarity.argmax(dim=-1)  # [target_length]

    # Gather from compressed (preserve dtype)
    reconstructed = compressed_embeddings[nn_indices]  # [target_length, D]

    return reconstructed, nn_indices


# ===========================================================================
# 3. Answer Scoring (from original project)
# ===========================================================================

def normalize_answer(text):
    """Normalize text for VQA matching (from run_ultimate_benchmark.py)."""
    text = str(text).lower().replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'([^\w\s])', r' ', text)
    words = [w for w in text.split() if w not in ['a', 'an', 'the']]
    num_map = {'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
               'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
    return ' '.join([num_map.get(w, w) for w in words])


def compute_accuracy_open(pred_answer, gt_answers):
    """
    Compute VQA accuracy for open-ended questions.
    From run_ultimate_benchmark.py compute_accuracy().
    """
    pred_clean = normalize_answer(pred_answer)
    gts = [normalize_answer(gt) for gt in gt_answers]

    if not gts:
        return 0.0
    match_count = gts.count(pred_clean)
    if match_count == 0:
        for gt in gts:
            if gt and gt in pred_clean:
                match_count = 3
                break
    return min(1.0, float(match_count) / 3.0)


def compute_accuracy_mc(pred_answer, gt_data):
    """Compute accuracy for multiple-choice questions."""
    pred_ans = str(pred_answer).strip().lower()
    gt_letter = gt_data["answers"][0].lower()
    gt_text = gt_data.get("ground_truth_text", "").lower()
    match = re.search(r'(?i)(?:^|\s|\()(option\s+)?([a-f])(?:\)|\.|:|\s|$)', pred_ans)
    if match:
        extracted = match.group(2).lower()
    else:
        if gt_text and gt_text in pred_ans:
            return 1.0
        extracted = pred_ans[0] if len(pred_ans) > 0 else ""
    return 1.0 if extracted == gt_letter else 0.0
