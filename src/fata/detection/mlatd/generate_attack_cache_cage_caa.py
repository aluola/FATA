from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import re
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    CLIPImageProcessor,
    CLIPVisionModel,
    LlavaForConditionalGeneration,
)
from fata.attacks.linf_image import quantize_linf_image
from fata.runtimes.llava.attack_cache_io import (
    attack_cache_contract_path,
    attack_image_exists,
    attack_image_path,
    attack_namespace,
    baseline_attack_contract_extra,
    build_attack_cache_contract,
    save_attack_image,
)
from fata.utils.paths import (
    assert_output_separate,
    resolve_dataset_relative_path,
    resolve_owned_output_path,
)
from fata.utils.run_contract import ensure_cache_contract, ensure_run_contract, expected_unique_ids
from fata.runtimes.llava.image_span import expanded_image_and_trailing_text_span


ALL_DATASETS = ["TextVQA_Open", "VQAv2_Open", "ScienceQA_MC", "VQAv2_MC"]

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
)

# Canonical settings are consumed directly from the cache contract so the
# implementation and its resume identity cannot drift independently.
CAGE_EPS_255 = 2.0
CAGE_ALPHA_255 = 0.5
CAGE_STEPS = 100
CAGE_OBJECTIVE = baseline_attack_contract_extra("cage")["objective"]
CAGE_LAMBDA = CAGE_OBJECTIVE["lambda_cage"]
CAGE_K_MIN = CAGE_OBJECTIVE["k_min"]
CAGE_K_MAX = CAGE_OBJECTIVE["k_max"]

# Fair CAA setting used for the paper comparison.
CAA_EPS_255 = 2.0
CAA_ALPHA_255 = 1.0
CAA_STEPS = 100
CAA_OBJECTIVE = baseline_attack_contract_extra("caa")["objective"]
CAA_TARGET_LAYER = CAA_OBJECTIVE["target_layer"]
CAA_REGION_FRACTION = CAA_OBJECTIVE["least_important_region_fraction"]
CAA_W_BPR_INTER = CAA_OBJECTIVE["weights"]["bpr_inter"]
CAA_W_BPR_INTRA = CAA_OBJECTIVE["weights"]["bpr_intra"]
CAA_W_SE = CAA_OBJECTIVE["weights"]["semantic"]
CAA_W_QA = CAA_OBJECTIVE["weights"]["question_answer"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_seed(base_seed: int, sample_idx: int) -> int:
    return int(base_seed) * 1_000_003 + int(sample_idx)


def expand2square(
    pil_img: Image.Image,
    background_color=(122, 116, 104),
) -> Image.Image:
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        out = Image.new(pil_img.mode, (width, width), background_color)
        out.paste(pil_img, (0, (width - height) // 2))
        return out
    out = Image.new(pil_img.mode, (height, height), background_color)
    out.paste(pil_img, ((height - width) // 2, 0))
    return out


def build_prompt(gt_data: dict[str, Any]) -> str:
    if gt_data.get("type") == "multiple_choice":
        return (
            f"{SYSTEM_PROMPT}USER: <image>\n{gt_data['question']}\n"
            "Answer with the option's letter from the given choices directly. ASSISTANT:"
        )
    return (
        f"{SYSTEM_PROMPT}USER: <image>\n{gt_data['question']}\n"
        "Answer the question using a single word or phrase. ASSISTANT:"
    )


def load_dataset(project_root: Path, dataset: str) -> list[dict[str, Any]]:
    qa_file = (
        project_root
        / dataset
        / f"{dataset}_mapping.jsonl"
    )
    if not qa_file.exists():
        raise FileNotFoundError(qa_file)
    rows = []
    with qa_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict) or not row.get("image_filename"):
                    raise ValueError(f"invalid mapping row in {qa_file}")
                rows.append(row)
    if not rows:
        raise ValueError(f"mapping is empty: {qa_file}")
    expected_unique_ids(rows, "image_filename")
    return rows


def cache_path(
    cache_root: Path,
    attack: str,
    dataset: str,
    image_filename: str,
) -> Path:
    """
    Store adversarial-image caches in a canonical PNG format,
    independent of the source dataset image extension.
    """
    return attack_image_path(
        cache_root=cache_root,
        method="shared",
        dataset=dataset,
        attack=attack,
        image_filename=image_filename,
    )


MANIFEST_FIELDS = [
    "attack",
    "dataset",
    "sample_idx",
    "image_filename",
    "status",
    "seed",
    "eps_255",
    "alpha_255",
    "steps",
    "message",
]


def append_manifest(path: Path, row: dict[str, Any]) -> None:
    """Append a crash-diagnostic event; a successful run rewrites canonically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"manifest must not be a symbolic link: {path}")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "a", newline="", encoding="utf-8") as f:
        empty = os.fstat(f.fileno()).st_size == 0
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if empty:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
        f.flush()
        os.fsync(f.fileno())


def write_manifest_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one canonical, ordered status row per requested sample."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"manifest must not be a symbolic link: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(
                {key: row.get(key, "") for key in MANIFEST_FIELDS} for row in rows
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_complete_manifest(
    path: Path,
    *,
    attack: str,
    dataset: str,
    start: int,
    expected_image_ids: list[str],
    seed: int,
    eps_255: float,
    alpha_255: float,
    steps: int,
) -> None:
    """Fail closed unless the terminal manifest is exact and all-success."""

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_FIELDS:
            raise RuntimeError(
                f"attack manifest header mismatch: {reader.fieldnames!r}"
            )
        rows = list(reader)
    if len(rows) != len(expected_image_ids):
        raise RuntimeError(
            "attack manifest row-count mismatch: "
            f"found={len(rows)} expected={len(expected_image_ids)}"
        )
    for offset, (row, image_id) in enumerate(zip(rows, expected_image_ids)):
        index = start + offset
        expected_seed = sample_seed(seed, index)
        try:
            valid = (
                row["attack"] == attack
                and row["dataset"] == dataset
                and int(row["sample_idx"]) == index
                and row["image_filename"] == image_id
                and row["status"] == "success"
                and int(row["seed"]) == expected_seed
                and float(row["eps_255"]) == eps_255
                and float(row["alpha_255"]) == alpha_255
                and int(row["steps"]) == steps
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid attack manifest row for sample {index}: {exc}"
            ) from exc
        if not valid:
            raise RuntimeError(
                f"attack manifest contract mismatch for sample {index}: {row!r}"
            )


class CAGEGenerator:
    def __init__(self, clip_path: str, device: torch.device):
        self.device = device
        try:
            self.clip = CLIPVisionModel.from_pretrained(
                clip_path,
                output_attentions=True,
                attn_implementation="eager",
            )
        except TypeError:
            self.clip = CLIPVisionModel.from_pretrained(
                clip_path,
                output_attentions=True,
            )
        self.clip.to(device).eval()
        self.clip.requires_grad_(False)
        self.processor = CLIPImageProcessor.from_pretrained(clip_path)
        self.mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=device,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=device,
        ).view(1, 3, 1, 1)

    def clean_image_01(self, image: Image.Image) -> torch.Tensor:
        normalized = self.processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].to(self.device)
        return (normalized * self.std + self.mean).detach()

    def generate(
        self,
        image: Image.Image,
        seed: int,
        eps_255: float,
        alpha_255: float,
        steps: int,
    ) -> Image.Image:
        set_seed(seed)
        epsilon = eps_255 / 255.0
        alpha = alpha_255 / 255.0
        clean_image_01 = self.clean_image_01(image)

        with torch.no_grad():
            clean_outputs = self.clip(
                (clean_image_01 - self.mean) / self.std,
                output_attentions=True,
                return_dict=True,
            )
            clean_features = (
                clean_outputs.last_hidden_state.squeeze(0)[1:].detach()
            )

        delta = torch.empty_like(clean_image_01).uniform_(
            -epsilon,
            epsilon,
        )
        delta.requires_grad_(True)
        denominator = float(CAGE_K_MAX - CAGE_K_MIN + 1)

        for _ in range(steps):
            adv_img = torch.clamp(clean_image_01 + delta, 0.0, 1.0)
            outputs = self.clip(
                (adv_img - self.mean) / self.std,
                output_attentions=True,
                return_dict=True,
            )
            if outputs.attentions is None:
                raise RuntimeError("CAGE: CLIP attentions are None")

            s_adv = (
                outputs.attentions[-1][:, :, 0, 1:]
                .mean(dim=1)
                .squeeze(0)
            )
            z_adv = outputs.last_hidden_state.squeeze(0)[1:]

            d_i = 1.0 - F.cosine_similarity(
                z_adv,
                clean_features,
                dim=-1,
            )
            r_i = torch.argsort(
                torch.argsort(s_adv, descending=True)
            )
            pi_i = torch.zeros_like(r_i, dtype=torch.float32)
            pi_i[r_i < CAGE_K_MIN] = 1.0
            mid = (r_i >= CAGE_K_MIN) & (r_i <= CAGE_K_MAX)
            pi_i[mid] = (
                CAGE_K_MAX - r_i[mid].float()
            ) / denominator

            loss_efd = (
                torch.sum(pi_i * d_i)
                / (torch.sum(pi_i) + 1e-8)
            )
            p_d = F.softmax(d_i, dim=0).detach()
            p_s = F.softmax(s_adv, dim=0)
            loss_rda = torch.sum(
                p_d * torch.log(p_s + 1e-8)
            )
            loss = loss_efd + CAGE_LAMBDA * loss_rda

            grad = torch.autograd.grad(
                loss,
                delta,
                retain_graph=False,
                create_graph=False,
            )[0]
            delta = (delta + alpha * grad.sign()).detach()
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta.requires_grad_(True)

        final = torch.clamp(
            clean_image_01 + delta.detach(),
            0.0,
            1.0,
        )
        return quantize_linf_image(clean_image_01, final, epsilon)


class CAAGenerator:
    """
    CAA least-important-region implementation adapted from the project's
    run_caa_benchmark.py. This generator intentionally runs at full visual
    token count (K=576 / no compressor) so the cached adversarial image is
    independent of the downstream compression method.

    The default perturbation budget is the paper's fair setting eps=2/255.
    """

    def __init__(
        self,
        llava_path: str,
        device: torch.device,
        max_input_tokens: int,
    ):
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.processor = AutoProcessor.from_pretrained(
            llava_path,
            use_fast=False,
        )
        load_kwargs = dict(
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                llava_path,
                attn_implementation="eager",
                **load_kwargs,
            )
        except TypeError:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                llava_path,
                **load_kwargs,
            )
        if not torch.cuda.is_available():
            self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.device = next(self.model.parameters()).device

        if hasattr(self.model, "vision_tower"):
            self.vision_model = self.model.vision_tower.vision_model
        else:
            self.vision_model = self.model.model.vision_tower.vision_model

        llama_model = self.model.language_model.model
        target_layer = llama_model.layers[CAA_TARGET_LAYER]
        self.q_proj = target_layer.self_attn.q_proj
        self.k_proj = target_layer.self_attn.k_proj

        self.mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=self.device,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=self.device,
        ).view(1, 3, 1, 1)

    def get_pixel_mask_from_tokens(
        self,
        token_indices: torch.Tensor,
        image_size: int = 336,
        patch_size: int = 14,
    ) -> torch.Tensor:
        mask = torch.zeros(
            (1, 3, image_size, image_size),
            device=self.device,
        )
        grid_size = image_size // patch_size
        for idx_tensor in token_indices:
            idx = int(idx_tensor.item())
            row = idx // grid_size
            col = idx % grid_size
            mask[
                :,
                :,
                row * patch_size : (row + 1) * patch_size,
                col * patch_size : (col + 1) * patch_size,
            ] = 1.0
        return mask

    def _move_inputs(self, raw: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in raw.items():
            if v is None:
                continue
            if torch.is_tensor(v):
                if k == "pixel_values":
                    out[k] = v.to(
                        self.device,
                        dtype=torch.float16,
                    )
                else:
                    out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        seed: int,
        eps_255: float,
        alpha_255: float,
        steps: int,
    ) -> Image.Image | None:
        set_seed(seed)
        epsilon = eps_255 / 255.0
        alpha = alpha_255 / 255.0

        raw_inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = self._move_inputs(raw_inputs)
        input_ids = inputs["input_ids"]

        if (
            self.max_input_tokens > 0
            and input_ids.shape[1] > self.max_input_tokens
        ):
            return None

        tokenizer_img_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            "<image>"
        )
        config_img_token_id = getattr(
            self.model.config,
            "image_token_index",
            None,
        )

        clean_pixel_values = inputs["pixel_values"].to(
            torch.float16
        )
        clean_image_01 = (
            clean_pixel_values * self.std + self.mean
        ).detach()

        previous_k = getattr(
            self.vision_model,
            "current_K",
            None,
        )
        if hasattr(self.vision_model, "current_K"):
            self.vision_model.current_K = 576

        try:
            with torch.no_grad():
                clean_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=clean_pixel_values,
                    labels=None,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )

                hidden = clean_outputs.hidden_states[
                    CAA_TARGET_LAYER
                ][0]
                v_start, v_end, t_start = expanded_image_and_trailing_text_span(
                    input_ids,
                    output_sequence_length=int(hidden.shape[0]),
                    image_token_count=576,
                    config_token_id=config_img_token_id,
                    tokenizer_token_id=tokenizer_img_token_id,
                )

                clean_vis_hidden = hidden[v_start:v_end, :]
                clean_text_hidden = hidden[t_start:, :]
                if clean_text_hidden.numel() == 0:
                    raise RuntimeError(
                        "CAA text hidden span is empty"
                    )

                clean_k_vis = self.k_proj(
                    clean_vis_hidden
                )
                clean_q_text = self.q_proj(
                    clean_text_hidden
                )
                clean_logits = torch.matmul(
                    clean_q_text,
                    clean_k_vis.T,
                ).mean(dim=0)

                ranks = torch.argsort(
                    clean_logits,
                    descending=True,
                )
                region_n = int(576 * CAA_REGION_FRACTION)
                topk_most = ranks[:region_n]
                topk_least = ranks[-region_n:]
                pixel_least_mask = (
                    self.get_pixel_mask_from_tokens(
                        topk_least
                    )
                )

                ref_vis_hidden = (
                    clean_vis_hidden.detach()
                )
                ref_q_text = (
                    clean_q_text.mean(dim=0).detach()
                )

            delta = torch.empty_like(
                clean_image_01
            ).uniform_(-epsilon, epsilon)
            delta.requires_grad_(True)

            for _ in range(steps):
                adv_img = torch.clamp(
                    clean_image_01
                    + delta * pixel_least_mask,
                    0.0,
                    1.0,
                )
                norm_adv = (
                    (adv_img - self.mean) / self.std
                ).to(torch.float16)

                adv_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=inputs.get(
                        "attention_mask"
                    ),
                    pixel_values=norm_adv,
                    labels=None,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )

                adv_hidden = adv_outputs.hidden_states[
                    CAA_TARGET_LAYER
                ][0]
                if adv_hidden.shape[0] != hidden.shape[0]:
                    raise RuntimeError(
                        "CAA adversarial hidden sequence length changed unexpectedly: "
                        f"clean={hidden.shape[0]} adv={adv_hidden.shape[0]}"
                    )
                adv_vis_hidden = adv_hidden[
                    v_start:v_end,
                    :
                ]
                adv_k_vis = self.k_proj(
                    adv_vis_hidden
                )
                adv_logits = torch.matmul(
                    clean_q_text,
                    adv_k_vis.T,
                ).mean(dim=0)

                s_least = adv_logits[topk_least]
                s_most = adv_logits[topk_most]

                diff_inter = (
                    s_least.unsqueeze(1)
                    - s_most.unsqueeze(0)
                )
                loss_bpr_inter = -F.logsigmoid(
                    diff_inter
                ).mean()

                diff_intra = (
                    s_least.unsqueeze(1)
                    - s_least.unsqueeze(0)
                )
                mask_upper = torch.triu(
                    torch.ones_like(diff_intra),
                    diagonal=1,
                ).bool()
                loss_bpr_intra = -F.logsigmoid(
                    -diff_intra[mask_upper]
                ).mean()

                loss_se = -F.mse_loss(
                    adv_vis_hidden[topk_least],
                    ref_vis_hidden[topk_least],
                )

                adv_least_k = adv_k_vis[topk_least]
                alignment = torch.matmul(
                    adv_least_k,
                    ref_q_text.unsqueeze(-1),
                ).mean()
                loss_qa = -alignment

                loss_total = (
                    CAA_W_BPR_INTER
                    * loss_bpr_inter
                    + CAA_W_BPR_INTRA
                    * loss_bpr_intra
                    + CAA_W_SE
                    * loss_se
                    + CAA_W_QA
                    * loss_qa
                )

                grad = torch.autograd.grad(
                    loss_total,
                    delta,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                delta = (
                    delta - alpha * grad.sign()
                ).detach()
                delta = torch.clamp(
                    delta,
                    -epsilon,
                    epsilon,
                )
                delta.requires_grad_(True)

                del (
                    adv_outputs,
                    adv_hidden,
                    adv_vis_hidden,
                    adv_k_vis,
                    adv_logits,
                    s_least,
                    s_most,
                    diff_inter,
                    diff_intra,
                    loss_bpr_inter,
                    loss_bpr_intra,
                    loss_se,
                    adv_least_k,
                    alignment,
                    loss_qa,
                    loss_total,
                    grad,
                    adv_img,
                    norm_adv,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            final = torch.clamp(
                clean_image_01
                + delta.detach() * pixel_least_mask,
                0.0,
                1.0,
            )
            return quantize_linf_image(clean_image_01, final, epsilon)
        finally:
            if (
                previous_k is not None
                and hasattr(
                    self.vision_model,
                    "current_K",
                )
            ):
                self.vision_model.current_K = previous_k


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate method-independent CAGE/CAA adversarial-image cache "
            "for ML-ATD cross-attack ablation."
        )
    )
    parser.add_argument(
        "--attack",
        required=True,
        choices=["cage", "caa"],
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=ALL_DATASETS,
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-root",
        dest="project_root",
        type=Path,
        default=os.environ.get("FATA_DATA_ROOT"),
    )
    parser.add_argument(
        "--llava-path",
        dest="llava_path",
        default=os.environ.get("FATA_LLAVA_MODEL"),
    )
    parser.add_argument(
        "--clip-path",
        dest="clip_path",
        default=os.environ.get("FATA_CLIP_MODEL"),
    )
    parser.add_argument(
        "--cache-root",
        dest="cache_root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--manifest-dir",
        dest="manifest_dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=0,
        help=(
            "CAA only. Match the project's anti-OOM rule. "
            "0 (formal default) disables the historical length filter so the exact cohort is complete."
        ),
    )
    parser.add_argument(
        "--caa_eps_255",
        type=float,
        default=CAA_EPS_255,
    )
    parser.add_argument(
        "--caa_alpha_255",
        type=float,
        default=CAA_ALPHA_255,
    )
    parser.add_argument(
        "--caa_steps",
        type=int,
        default=CAA_STEPS,
    )
    parser.add_argument(
        "--cage_eps_255",
        type=float,
        default=CAGE_EPS_255,
    )
    parser.add_argument(
        "--cage_alpha_255",
        type=float,
        default=CAGE_ALPHA_255,
    )
    parser.add_argument(
        "--cage_steps",
        type=int,
        default=CAGE_STEPS,
    )
    args = parser.parse_args()
    if not all((args.project_root, args.llava_path, args.clip_path, args.cache_root, args.manifest_dir)):
        parser.error("provide data, model, CLIP, cache, and manifest paths")
    if args.start < 0 or args.limit <= 0 or args.seed < 0:
        parser.error("--start/--seed must be non-negative and --limit positive")
    if args.max_input_tokens < 0:
        parser.error("--max_input_tokens must be non-negative")
    floating_attack_values = (
        args.caa_eps_255,
        args.caa_alpha_255,
        args.cage_eps_255,
        args.cage_alpha_255,
    )
    if (
        not all(math.isfinite(value) and value > 0 for value in floating_attack_values)
        or args.caa_steps <= 0
        or args.cage_steps <= 0
    ):
        parser.error("attack epsilon, alpha, and steps must be positive")
    if (
        args.max_input_tokens != 0
        or args.caa_eps_255 != CAA_EPS_255
        or args.caa_alpha_255 != CAA_ALPHA_255
        or args.caa_steps != CAA_STEPS
        or args.cage_eps_255 != CAGE_EPS_255
        or args.cage_alpha_255 != CAGE_ALPHA_255
        or args.cage_steps != CAGE_STEPS
    ):
        parser.error(
            "formal CAGE/CAA cache generation is locked to max_input_tokens=0, "
            "CAGE eps/alpha/steps=2/0.5/100, and CAA=2/1/100"
        )
    protected_inputs = {
        "dataset_root": args.project_root,
        "LLaVA model": args.llava_path,
        "CLIP model": args.clip_path,
    }
    assert_output_separate(args.cache_root, protected_inputs)
    manifest_root = assert_output_separate(args.manifest_dir, protected_inputs)
    manifest_root = assert_output_separate(
        manifest_root, {"attack cache": args.cache_root}
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    qa = load_dataset(
        args.project_root,
        args.dataset,
    )
    if args.start + args.limit > len(qa):
        parser.error(
            f"requested exact slice [{args.start},{args.start + args.limit}) "
            f"exceeds mapping length {len(qa)}"
        )
    end = args.start + args.limit
    dataset_dir = (
        args.project_root
        / args.dataset
    )

    if args.attack == "cage":
        eps_255 = args.cage_eps_255
        alpha_255 = args.cage_alpha_255
        steps = args.cage_steps
    else:
        eps_255 = args.caa_eps_255
        alpha_255 = args.caa_alpha_255
        steps = args.caa_steps

    cache_attack = attack_namespace(
        args.attack,
        seed=args.seed,
        eps_255=eps_255,
        alpha_255=alpha_255,
        steps=steps,
    )
    manifest_root.mkdir(parents=True, exist_ok=True)
    if not manifest_root.is_dir():
        raise RuntimeError(f"manifest root is not a directory: {manifest_root}")
    manifest_name = (
        f"{cache_attack}_{args.dataset}_start{args.start}_limit{args.limit}.csv"
    )
    manifest = resolve_owned_output_path(manifest_root, manifest_name)
    manifest_meta = resolve_owned_output_path(
        manifest_root, str(Path(manifest_name).with_suffix(".meta.json"))
    )
    mapping_path = dataset_dir / f"{args.dataset}_mapping.jsonl"
    cache_contract = build_attack_cache_contract(
        definition=f"mlatd_{args.attack}_generator_v1",
        dataset=args.dataset,
        mapping_path=mapping_path,
        method="shared",
        model_path=args.llava_path,
        clip_path=args.clip_path,
        seed=args.seed,
        eps_255=eps_255,
        alpha_255=alpha_255,
        steps=steps,
        extra=baseline_attack_contract_extra(
            args.attack,
            max_input_tokens=args.max_input_tokens,
        ),
    )
    ensure_cache_contract(
        attack_cache_contract_path(
            cache_root=args.cache_root,
            method="shared",
            dataset=args.dataset,
            attack=cache_attack,
        ),
        cache_contract,
    )
    manifest_contract = {
        "schema_version": 1,
        "runtime": "mlatd_attack_cache_manifest_v1",
        "cache_contract": cache_contract,
        "cache_namespace": cache_attack,
        "start": args.start,
        "limit": args.limit,
        "expected_sample_indices": list(range(args.start, end)),
        "expected_image_ids": [
            str(qa[index]["image_filename"]) for index in range(args.start, end)
        ],
        "header": MANIFEST_FIELDS,
    }
    ensure_run_contract(
        manifest_meta,
        manifest_contract,
        result_path=manifest,
    )
    status_rows: dict[int, dict[str, Any]] = {}

    def record_status(row: dict[str, Any]) -> None:
        index = int(row["sample_idx"])
        status_rows[index] = row
        # Preserve the latest event if the process is interrupted.  Normal
        # completion atomically replaces this log with the canonical table.
        append_manifest(manifest, row)

    # Reconstruct the canonical success manifest directly from cryptographically
    # validated caches.  A no-op resume must not instantiate a GPU generator.
    if not args.overwrite:
        complete_cache = all(
            attack_image_exists(
                cache_root=args.cache_root,
                method="shared",
                dataset=args.dataset,
                attack=cache_attack,
                image_filename=str(qa[index]["image_filename"]),
            )
            for index in range(args.start, end)
        )
        if complete_cache:
            canonical_rows = [
                dict(
                    attack=args.attack,
                    dataset=args.dataset,
                    sample_idx=index,
                    image_filename=str(qa[index]["image_filename"]),
                    status="success",
                    seed=sample_seed(args.seed, index),
                    eps_255=eps_255,
                    alpha_255=alpha_255,
                    steps=steps,
                    message="reused_valid_cache",
                )
                for index in range(args.start, end)
            ]
            write_manifest_atomic(manifest, canonical_rows)
            validate_complete_manifest(
                manifest,
                attack=args.attack,
                dataset=args.dataset,
                start=args.start,
                expected_image_ids=[
                    str(qa[index]["image_filename"])
                    for index in range(args.start, end)
                ],
                seed=args.seed,
                eps_255=eps_255,
                alpha_255=alpha_255,
                steps=steps,
            )
            print(
                f"[SKIP] {args.attack} cache slice [{args.start},{end}) is "
                "exactly complete; model loading skipped"
            )
            return

    if args.attack == "cage":
        generator = CAGEGenerator(args.clip_path, device)
    else:
        generator = CAAGenerator(args.llava_path, device, args.max_input_tokens)

    for idx in tqdm(
        range(args.start, end),
        desc=f"cache {args.attack} {args.dataset}",
    ):
        gt = qa[idx]
        filename = gt["image_filename"]
        out_path = cache_path(
            args.cache_root,
            cache_attack,
            args.dataset,
            filename,
        )
        if attack_image_exists(
            cache_root=args.cache_root,
            method="shared",
            dataset=args.dataset,
            attack=cache_attack,
            image_filename=filename,
        ) and not args.overwrite:
            record_status(
                dict(
                    attack=args.attack,
                    dataset=args.dataset,
                    sample_idx=idx,
                    image_filename=filename,
                    status="success",
                    seed=sample_seed(args.seed, idx),
                    eps_255=eps_255,
                    alpha_255=alpha_255,
                    steps=steps,
                    message="reused_valid_cache",
                )
            )
            continue

        image_path = resolve_dataset_relative_path(dataset_dir, filename)
        if not image_path.exists():
            record_status(
                dict(
                    attack=args.attack,
                    dataset=args.dataset,
                    sample_idx=idx,
                    image_filename=filename,
                    status="missing_input",
                    seed=sample_seed(args.seed, idx),
                    eps_255=eps_255,
                    alpha_255=alpha_255,
                    steps=steps,
                    message=str(image_path),
                )
            )
            continue

        sseed = sample_seed(
            args.seed,
            idx,
        )
        set_seed(sseed)

        try:
            with Image.open(image_path) as raw:
                image = expand2square(
                    raw.convert("RGB")
                )

            if args.attack == "cage":
                adv = generator.generate(
                    image,
                    sseed,
                    eps_255,
                    alpha_255,
                    steps,
                )
            else:
                adv = generator.generate(
                    image,
                    build_prompt(gt),
                    sseed,
                    eps_255,
                    alpha_255,
                    steps,
                )

            if adv is None:
                record_status(
                    dict(
                        attack=args.attack,
                        dataset=args.dataset,
                        sample_idx=idx,
                        image_filename=filename,
                        status="skipped_long_prompt",
                        seed=sseed,
                        eps_255=eps_255,
                        alpha_255=alpha_255,
                        steps=steps,
                        message=(
                            f"input token length exceeds "
                            f"{args.max_input_tokens}"
                        ),
                    )
                )
                continue

            saved_path = save_attack_image(
                image=adv,
                cache_root=args.cache_root,
                method="shared",
                dataset=args.dataset,
                attack=cache_attack,
                image_filename=filename,
            )
            if saved_path != out_path:
                raise RuntimeError(f"cache path disagreement: {saved_path} != {out_path}")
            record_status(
                dict(
                    attack=args.attack,
                    dataset=args.dataset,
                    sample_idx=idx,
                    image_filename=filename,
                    status="success",
                    seed=sseed,
                    eps_255=eps_255,
                    alpha_255=alpha_255,
                    steps=steps,
                    message="",
                )
            )
        except RuntimeError as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                status = "oom"
            else:
                status = "runtime_error"
            record_status(
                dict(
                    attack=args.attack,
                    dataset=args.dataset,
                    sample_idx=idx,
                    image_filename=filename,
                    status=status,
                    seed=sseed,
                    eps_255=eps_255,
                    alpha_255=alpha_255,
                    steps=steps,
                    message=message[:1000],
                )
            )
        except Exception as exc:
            record_status(
                dict(
                    attack=args.attack,
                    dataset=args.dataset,
                    sample_idx=idx,
                    image_filename=filename,
                    status="error",
                    seed=sseed,
                    eps_255=eps_255,
                    alpha_255=alpha_255,
                    steps=steps,
                    message=str(exc)[:1000],
                )
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    missing = [
        str(qa[index]["image_filename"])
        for index in range(args.start, end)
        if not attack_image_exists(
            cache_root=args.cache_root,
            method="shared",
            dataset=args.dataset,
            attack=cache_attack,
            image_filename=qa[index]["image_filename"],
        )
    ]
    canonical_rows = []
    for index in range(args.start, end):
        filename = str(qa[index]["image_filename"])
        row = status_rows.get(index)
        if row is None:
            row = dict(
                attack=args.attack,
                dataset=args.dataset,
                sample_idx=index,
                image_filename=filename,
                status="missing_cache",
                seed=sample_seed(args.seed, index),
                eps_255=eps_255,
                alpha_255=alpha_255,
                steps=steps,
                message="no terminal status was recorded",
            )
        if filename not in missing:
            row = dict(row, status="success")
        canonical_rows.append(row)
    write_manifest_atomic(manifest, canonical_rows)
    if missing:
        raise RuntimeError(
            f"attack cache incomplete: missing_or_invalid={len(missing)} examples={missing[:5]}"
        )
    validate_complete_manifest(
        manifest,
        attack=args.attack,
        dataset=args.dataset,
        start=args.start,
        expected_image_ids=[
            str(qa[index]["image_filename"]) for index in range(args.start, end)
        ],
        seed=args.seed,
        eps_255=eps_255,
        alpha_255=alpha_255,
        steps=steps,
    )


if __name__ == "__main__":
    main()
