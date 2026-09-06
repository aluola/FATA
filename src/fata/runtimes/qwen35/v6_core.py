#!/usr/bin/env python
"""
Q-FATA V6 Compression-Specific — core module for Qwen3.5-9B.

Objective (V6): compressed-specific damage.
    D_full  = Acc_clean_full - Acc_adv_full         (constraint / penalty)
    D_13    = Acc_clean_1/3  - Acc_adv_1/3
    D_prac  = Acc_clean_practical - Acc_adv_practical
    Amp_13  = D_13   - D_full
    Amp_prac = D_prac - D_full                       (PRIMARY metric)

Threat model: task-aware white-box variant (uses GT answers + downstream Qwen3.5
gradients). Must be labeled as such; NOT the original gray-box FATA.

Differences vs V5 (v5_accuracy_first):
  1. Full preservation: P_full = w_full * CE_full_adv + w_distill * KL(full_adv || full_clean)
     (V5 had w_full=0). CE_full computed teacher-forced on the GT answer tokens with
     NO compression patch; distillation on the cached clean full-token answer logits.
  2. Gradient projection: g_task_proj removes the component of the compressed-task
     gradient that also damages the full-token prediction:
         g_full_dir = normalize(g_full_loss)          # ascent dir that hurts Full
         g_proj = g_task - max(0, <g_task,g_full_dir> / (||g_full_dir||^2+eps)) * g_full_dir
  3. Evaluation scorer: canonical strict VQA scorer (V6 audit decision), replacing
     the deprecated substring/1-len(refs) scorer.

Sign convention (unchanged from V5): gradient ASCENT on attack utility.
    delta = delta + alpha * sign(g)
    g = w_task*norm(g_task_CE_compressed) + w_rank*norm(g_rank) + w_comp*norm(g_comp)
        - w_full*norm(g_full_CE) - w_distill*norm(g_distill_KL)
  Maximize: compressed task CE, U_rank, U_comp.
  Minimize (penalty, subtracted): full task CE, full-vs-clean logit KL.
"""
import os, hashlib, re, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from fata.evaluation.scoring import score_vqa
from .fata_utils import reconstruct_to_dynamic_length
from .evaluation.output_parsing import extract_final_answer
from .compression import (
    VisionZIPDynamic, VisPrunerDynamic, FlowCutDynamic, PruMergeDynamic,
)
from .attacks.boundary_target_builder import BoundaryTargetBuilder
from .compression_surrogates import (
    VisionZIPSurrogate, VisPrunerSurrogate, FlowCutSurrogate, PruMergeSurrogate,
)
from .resume import DELTA_LINF_ATOL

METHODS_4 = ("VisionZIP", "VisPruner", "FlowCut", "PruMerge")
METHODS = METHODS_4
COMPRESSOR_CLS = {
    "VisionZIP": VisionZIPDynamic, "VisPruner": VisPrunerDynamic,
    "FlowCut": FlowCutDynamic, "PruMerge": PruMergeDynamic,
}
SURROGATE_CLS = {
    "VisionZIP": VisionZIPSurrogate, "VisPruner": VisPrunerSurrogate,
    "FlowCut": FlowCutSurrogate, "PruMerge": PruMergeSurrogate,
}

PIXEL_RANGE_MIN = -1.0
PIXEL_RANGE_MAX = 1.0

OPEN_SYSTEM = ("You are a visual question answering system. Answer using only the shortest possible answer. "
               "Do not explain, describe your reasoning, or repeat the question.")
USER_SUFFIX_OPEN = "\nAnswer with only the answer, preferably a single word or a short phrase."
MC_SYSTEM = ("You are a visual multiple-choice question answering system. Select the correct option. "
             "Output only its option letter, such as A, B, C, or D. Do not explain your reasoning.")
USER_SUFFIX_MC = "\nOutput only the correct option letter."


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class V6Config:
    name: str
    task_mode: str          # "ground_truth" | "proxy" | "none"
    w_task: float
    w_rank: float
    w_comp: float
    w_full: float           # full-token GT-CE preservation weight
    w_distill: float        # full-vs-clean logit KL preservation weight
    use_projection: bool = False
    epsilon: float = 2.0 / 255.0
    alpha: float = 0.5 / 255.0
    steps: int = 100
    is_mc: bool = False
    max_answer_tokens: int = 4
    rank_margin: float = 0.05
    comp_margin: float = 0.05
    boundary_window_ratio: float = 0.05
    eps_grad: float = 1e-8
    # per-step (method, budget) schedule; budget period 5
    budget_pattern: Tuple[str, ...] = ("Practical", "Practical", "1/3", "Practical", "1/3")
    # compressor universe used by the attack round-robin AND the surrogate term
    method_universe: Tuple[str, ...] = METHODS_4

    def practical_ratio(self) -> float:
        return 1.0 / 18.0 if self.is_mc else 1.0 / 9.0

    def ratio_for(self, budget: str) -> float:
        if budget == "1/3":
            return 1.0 / 3.0
        if budget == "Practical":
            return self.practical_ratio()
        return 1.0


# The release exposes exactly the frozen four-compressor paper configuration.
# Historical pilot/control variants are intentionally absent from the public
# runtime so an unreported configuration cannot be selected accidentally.
FORMAL_CONFIG_NAME = "v6_a_fullpres_light_4comp"
FORMAL_CONFIGS: Dict[str, V6Config] = {
    FORMAL_CONFIG_NAME: V6Config(
        FORMAL_CONFIG_NAME,
        task_mode="ground_truth",
        w_task=1.5,
        w_rank=0.25,
        w_comp=0.05,
        w_full=1.0,
        w_distill=1.0,
        use_projection=False,
        method_universe=METHODS_4,
    ),
}


# ===========================================================================
# Deterministic seed + initial delta (identical to V5 protocol)
# ===========================================================================
def deterministic_seed(dataset: str, image_id: str, sample_seed: int) -> int:
    h = hashlib.sha256(f"{dataset}|{image_id}|{sample_seed}".encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def make_initial_delta(pv: torch.Tensor, seed: int, epsilon: float) -> torch.Tensor:
    rng = np.random.RandomState(seed % (2 ** 32))
    delta_np = rng.uniform(-epsilon, epsilon, size=tuple(pv.shape)).astype(np.float32)
    delta = torch.from_numpy(delta_np).to(pv.device)
    delta.requires_grad_(True)
    return delta


def delta_sha256(delta: torch.Tensor) -> str:
    return hashlib.sha256(delta.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]


def get_schedule(steps: int, config: V6Config) -> List[Tuple[str, str]]:
    sched = []
    for step in range(steps):
        method = config.method_universe[step % len(config.method_universe)]
        budget = config.budget_pattern[step % len(config.budget_pattern)]
        sched.append((method, budget))
    return sched


# ===========================================================================
# Canonical STRICT VQA scorer (V6 audit decision; identical to the Aug-4
# forensic audit's authoritative_rescore.unified_score)
# ===========================================================================
def normalize_answer(text):
    text = str(text).lower().replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'([^\w\s])', r' ', text)
    words = [w for w in text.split() if w not in ['a', 'an', 'the']]
    num_map = {'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
               'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
    return ' '.join([num_map.get(w, w) for w in words])


def strict_score_open(pred_raw, refs) -> float:
    return score_vqa(extract_final_answer(pred_raw)[0], refs)


def strict_score_mc(pred_raw, gt_letter: str) -> float:
    pred = str(extract_final_answer(pred_raw)[0]).strip().lower()
    m = re.search(r'(?i)(?:^|\s|\()(option\s+)?([a-f])(?:\)|\.|:|\s|$)', pred)
    extracted = m.group(2).lower() if m else (pred[0] if len(pred) > 0 else '')
    return 1.0 if extracted == str(gt_letter).strip().lower() else 0.0


# ===========================================================================
# Answer token ids (teacher-forcing target)
# ===========================================================================
def answer_token_ids(processor, sample: dict, is_mc: bool, max_answer_tokens: int = 4) -> List[int]:
    if not sample.get("reference_answers"):
        return []
    if is_mc:
        letter = str(sample["reference_answers"][0]).strip().upper()
        ids = processor.tokenizer.encode(letter, add_special_tokens=False)
        if not ids:
            ids = processor.tokenizer.encode(" " + letter, add_special_tokens=False)
        return ids[:1]
    ans = str(sample["reference_answers"][0]).strip()
    ids = processor.tokenizer.encode(ans, add_special_tokens=False)
    return ids[:max_answer_tokens]


# ===========================================================================
# Model wrapper (identical to V5)
# ===========================================================================
class V6Model:
    def __init__(self, model_path, device_map="auto", enable_gradient_checkpointing: bool = True):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device_map,
            low_cpu_mem_usage=True, local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        if enable_gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass
        self._orig_visual_forward = self.model.model.visual.forward

    @property
    def device(self):
        return next(self.model.parameters()).device

    def build_prompt(self, sample: dict, is_mc: bool) -> tuple:
        if is_mc:
            opts = sample.get("options", [])
            opt_lines = [f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(opts) if o]
            opts_str = "\n".join(opt_lines)
            q = f"Question: {sample['question']}\n\n{opts_str}\n\n{USER_SUFFIX_MC}"
            return MC_SYSTEM, q
        return OPEN_SYSTEM, sample["question"] + USER_SUFFIX_OPEN

    def prepare_sample(self, sample: dict, is_mc: bool) -> dict:
        system, question = self.build_prompt(sample, is_mc)
        image = self._load_image(sample["image_path"])
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": [{"type": "image", "image": image},
                                              {"type": "text", "text": question}]}]
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", enable_thinking=False)
        return inputs

    @staticmethod
    def _load_image(path: str):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > 800:
            scale = 800 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)))
        return img

    def compress_pooler(self, emb: torch.Tensor, compressor_cls, ratio: float) -> torch.Tensor:
        N_full = emb.shape[0]
        comp = compressor_cls(ratio)
        compressed, _, _ = comp.compress(emb.float())
        recon, _ = reconstruct_to_dynamic_length(emb.float(), compressed, N_full)
        return recon

    def _patch_visual(self, compressor_cls, ratio: float):
        orig = self._orig_visual_forward

        def patched(self_module, pixel_values, grid_thw=None, **kw):
            o = orig(pixel_values, grid_thw=grid_thw, **kw)
            dtype = o.pooler_output.dtype
            emb = o.pooler_output.float()
            N_full = emb.shape[0]
            comp = compressor_cls(ratio)
            compressed, _, _ = comp.compress(emb)
            recon, _ = reconstruct_to_dynamic_length(emb, compressed, N_full)
            o.pooler_output = recon.to(dtype)
            return o

        self.model.model.visual.forward = patched.__get__(self.model.model.visual)

    def _restore_visual(self):
        self.model.model.visual.forward = self._orig_visual_forward

    def task_logits(self, adv_pv: torch.Tensor, inputs: dict, answer_ids: List[int],
                    compressor_cls=None, ratio: float = 1.0) -> torch.Tensor:
        """Teacher-forced logits for the answer tokens.
        compressor_cls=None or ratio>=1.0 -> NO compression patch (Full tokens)."""
        T = len(answer_ids)
        dev = inputs["input_ids"].device
        input_ids = inputs["input_ids"]
        ans_tensor = torch.tensor([answer_ids], device=dev, dtype=input_ids.dtype)
        full_ids = torch.cat([input_ids, ans_tensor], dim=1)
        attn = inputs["attention_mask"]
        full_attn = torch.cat([attn, torch.ones_like(ans_tensor)], dim=1)

        kwargs = dict(
            input_ids=full_ids,
            attention_mask=full_attn,
            pixel_values=adv_pv.to(dev).to(torch.bfloat16),
            image_grid_thw=inputs["image_grid_thw"].to(dev),
            logits_to_keep=T + 1,
        )
        mm = inputs.get("mm_token_type_ids")
        if mm is not None:
            full_mm = torch.cat([mm.to(dev), torch.zeros_like(ans_tensor)], dim=1)
            kwargs["mm_token_type_ids"] = full_mm

        patched = compressor_cls is not None and ratio < 1.0
        if patched:
            self._patch_visual(compressor_cls, ratio)
        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = self.model(**kwargs)
        finally:
            if patched:
                self._restore_visual()
        return out.logits.float()[:, :T, :]

    def generate_answer(self, pv: torch.Tensor, inputs: dict, compressor_cls, ratio: float,
                        max_tok: int, is_mc: bool = False) -> str:
        dev = inputs["input_ids"].device
        if ratio >= 1.0 or compressor_cls is None:
            k = dict(input_ids=inputs["input_ids"].to(dev),
                     pixel_values=pv.to(dev).to(torch.bfloat16),
                     max_new_tokens=max_tok, do_sample=False, use_cache=True)
            for key in ["image_grid_thw", "mm_token_type_ids", "attention_mask"]:
                if key in inputs and inputs[key] is not None:
                    k[key] = inputs[key].to(dev)
            with torch.no_grad():
                out = self.model.generate(**k)
        else:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                vo = self.model.model.visual(pv.to(dev).to(torch.bfloat16),
                                             grid_thw=inputs["image_grid_thw"].to(dev))
            emb = vo.pooler_output.float()
            N_full = emb.shape[0]
            comp = compressor_cls(ratio)
            compressed, _, _ = comp.compress(emb)
            recon, _ = reconstruct_to_dynamic_length(emb, compressed, N_full)

            def _p(self_module, pixel_values, grid_thw=None, **kw):
                o = self._orig_visual_forward(pixel_values, grid_thw=grid_thw, **kw)
                o.pooler_output = recon.to(device=o.pooler_output.device, dtype=o.pooler_output.dtype)
                return o

            self.model.model.visual.forward = _p.__get__(self.model.model.visual)
            try:
                k = dict(input_ids=inputs["input_ids"].to(dev),
                         pixel_values=pv.to(dev).to(torch.bfloat16),
                         max_new_tokens=max_tok, do_sample=False, use_cache=True)
                for key in ["image_grid_thw", "mm_token_type_ids", "attention_mask"]:
                    if key in inputs and inputs[key] is not None:
                        k[key] = inputs[key].to(dev)
                with torch.no_grad():
                    out = self.model.generate(**k)
            finally:
                self._restore_visual()
        return self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ===========================================================================
# Loss terms
# ===========================================================================
def _cos_global(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    af = a.reshape(-1).unsqueeze(0)
    bf = b.reshape(-1).unsqueeze(0)
    return F.cosine_similarity(af, bf, dim=-1, eps=1e-8).squeeze()


def task_ce(logits: torch.Tensor, answer_ids: List[int]) -> torch.Tensor:
    """Mean CE over answer token positions. MAXIMIZE to attack."""
    T = logits.shape[1]
    ce = 0.0
    for t in range(T):
        ce = ce + F.cross_entropy(logits[:, t, :], torch.tensor([answer_ids[t]], device=logits.device))
    return ce / T


def logit_kl(logits_adv: torch.Tensor, logits_clean: torch.Tensor) -> torch.Tensor:
    """KL(softmax(adv) || softmax(clean)) over answer positions, mean."""
    T = logits_adv.shape[1]
    kl = 0.0
    for t in range(T):
        lp_a = F.log_softmax(logits_adv[:, t, :], dim=-1)
        p_c = F.softmax(logits_clean[:, t, :].detach(), dim=-1)
        kl = kl + F.kl_div(lp_a, p_c, reduction="sum")
    return kl / T


def project_away(g_task: torch.Tensor, g_full: torch.Tensor, eps: float = 1e-8):
    """Project g_task away from the g_full direction iff they positively align.
    Returns (projected_gradient, triggered: bool, cosine: float, norm_before, norm_after)."""
    if g_full.numel() == 0 or g_task.numel() == 0:
        return g_task, False, 0.0, 0.0, 0.0
    if not (torch.isfinite(g_task).all() and torch.isfinite(g_full).all()):
        return g_task, False, float("nan"), float("nan"), float("nan")
    norm_before = float(g_task.norm())
    if norm_before == 0.0 or float(g_full.norm()) == 0.0:
        return g_task, False, 0.0, norm_before, norm_before
    cos = float(F.cosine_similarity(g_task.reshape(1, -1), g_full.reshape(1, -1)).item())
    if not math.isfinite(cos):
        return g_task, False, cos, norm_before, norm_before
    if cos <= 0.0:
        return g_task, False, cos, norm_before, norm_before
    # remove only the harmful (positively aligned) component
    coeff = torch.dot(g_task.reshape(-1), g_full.reshape(-1)) / (g_full.reshape(-1).dot(g_full.reshape(-1)) + eps)
    g_proj = g_task - max(coeff, torch.tensor(0.0, device=g_task.device)) * g_full
    if not torch.isfinite(g_proj).all():
        return g_task, False, cos, norm_before, norm_before
    return g_proj, True, cos, norm_before, float(g_proj.norm())


def _normalize(g: torch.Tensor, eps: float) -> torch.Tensor:
    return g / (g.norm() + eps)


def _safe_grad(loss, delta, retain_graph: bool) -> torch.Tensor:
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        return torch.zeros_like(delta)
    g = torch.autograd.grad(loss, delta, retain_graph=retain_graph, create_graph=False)[0]
    return g if g is not None else torch.zeros_like(delta)


def _final_diagnostic_status(
    *,
    delta_linf: float,
    epsilon: float,
    full_clean_gt_loss: float,
    full_adv_gt_loss: float,
    compressed_adv_gt_loss: float,
    full_logit_kl: float,
    mean_grad_cos_comp_full: float,
    projection_trigger_rate: float,
    diagnostic_exception: str = "",
) -> tuple[str, str]:
    """Classify the final attack result before any success rows are emitted."""

    if diagnostic_exception:
        return "diagnostic_error", diagnostic_exception
    diagnostics = {
        "delta_linf": delta_linf,
        "full_clean_GT_loss": full_clean_gt_loss,
        "full_adv_GT_loss": full_adv_gt_loss,
        "compressed_adv_GT_loss": compressed_adv_gt_loss,
        "full_logit_KL": full_logit_kl,
        "mean_grad_cos_comp_full": mean_grad_cos_comp_full,
        "projection_trigger_rate": projection_trigger_rate,
    }
    nonfinite = [name for name, value in diagnostics.items() if not math.isfinite(value)]
    if nonfinite:
        return "diagnostic_error", f"non-finite final diagnostics: {nonfinite}"
    if delta_linf < 0 or delta_linf > epsilon + DELTA_LINF_ATOL:
        return "delta_violation", "final delta_linf exceeds the threat-model bound"
    if min(full_clean_gt_loss, full_adv_gt_loss, compressed_adv_gt_loss) < 0:
        return "diagnostic_error", "negative final GT loss"
    if full_logit_kl < -1e-8:
        return "diagnostic_error", "materially negative final logit KL"
    if not -1.0 <= mean_grad_cos_comp_full <= 1.0:
        return "diagnostic_error", "final gradient cosine is outside [-1, 1]"
    if not 0.0 <= projection_trigger_rate <= 1.0:
        return "diagnostic_error", "final projection trigger rate is outside [0, 1]"
    return "success", ""


# ===========================================================================
# Attack
# ===========================================================================
def attack_v6(v6model: V6Model, sample: dict, inputs: dict, config: V6Config,
              seed: int, answer_ids: Optional[List[int]] = None,
              clean_repr_cache: Optional[Dict[Tuple[str, str], torch.Tensor]] = None,
              diag_interval: int = 10) -> Dict:
    model = v6model.model
    dev = inputs["input_ids"].device
    pv = inputs["pixel_values"].float().to(dev)
    grid_thw = inputs["image_grid_thw"].to(dev)
    is_mc = config.is_mc

    # ---- clean vision forward ----
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        clean_out = model.model.visual(pv.to(torch.bfloat16), grid_thw=grid_thw)
    clean_merged = clean_out.pooler_output.float().squeeze(0)
    clean_raw = getattr(clean_out, "last_hidden_state", None)
    if clean_raw is not None:
        clean_raw = clean_raw.float().squeeze(0)
    N_full = clean_merged.shape[0]
    clean_imp = clean_merged.norm(dim=-1).detach()

    # ---- structural targets / surrogates (restricted to the config's universe) ----
    btb = BoundaryTargetBuilder(margin=config.rank_margin, is_mc=is_mc)
    bt = btb.build(clean_imp, N_full)
    surrogates = {}
    if config.w_comp > 0:
        surrogates = {n: SURROGATE_CLS[n]() for n in config.method_universe if n in SURROGATE_CLS}

    clean_merged_det = clean_merged.detach()
    clean_raw_det = clean_raw.detach() if clean_raw is not None else None

    # ---- clean compressed reps (proxy mode) ----
    if config.task_mode == "proxy" and clean_repr_cache is None:
        clean_repr_cache = {}
        for (method, budget) in set(get_schedule(config.steps, config)):
            ratio = config.ratio_for(budget)
            H_clean = v6model.compress_pooler(clean_merged_det, COMPRESSOR_CLS[method], ratio)
            clean_repr_cache[(method, budget)] = H_clean.detach()

    # ---- clean FULL-token answer logits (distillation target) ----
    clean_full_logits = None
    full_clean_ce = float("nan")
    if answer_ids:
        with torch.no_grad():
            clean_full_logits = v6model.task_logits(pv, inputs, answer_ids,
                                                    compressor_cls=None, ratio=1.0).detach()
        full_clean_ce = float(task_ce(clean_full_logits, answer_ids).detach())

    # ---- deterministic delta ----
    delta = make_initial_delta(pv, seed, config.epsilon)
    init_sha = delta_sha256(delta)

    schedule = get_schedule(config.steps, config)
    diag = []
    proj_triggered_cnt = 0
    cos_comp_full_list = []

    for step in range(config.steps):
        method, budget = schedule[step]
        ratio = config.ratio_for(budget)
        delta.data.clamp_(-config.epsilon, config.epsilon)
        # adv_pv is a DETACHED-with-grad root: each loss term gets its own graph
        # rooted here and is backwarded + freed immediately (peak = 1 graph).
        # Gradients are converted to delta-space via the exact clamp-derivative mask.
        adv_pv = (pv + delta).clamp(PIXEL_RANGE_MIN, PIXEL_RANGE_MAX).detach().requires_grad_(True)
        mask = (((pv + delta) > PIXEL_RANGE_MIN) & ((pv + delta) < PIXEL_RANGE_MAX)).to(torch.float32)

        # ---- forward/backward interleaved per graph to bound peak memory ----
        # All graphs are rooted at the detached leaf adv_pv; the clamp-derivative
        # mask converts each gradient to delta-space exactly. Graphs are freed as
        # soon as their last consumer backward passes: A (vision) -> B (task LLM)
        # -> C (full-preservation LLM). Peak = ONE LLM graph at a time.

        # --- graph A: structural vision terms ---
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            adv_out = model.model.visual(adv_pv.to(torch.bfloat16), grid_thw=grid_thw)
        adv_merged = adv_out.pooler_output.float().squeeze(0)
        adv_raw = getattr(adv_out, "last_hidden_state", None)
        if adv_raw is not None:
            adv_raw = adv_raw.float().squeeze(0)
        adv_imp = adv_merged.norm(dim=-1)

        U_rank = -btb.compute_ranking_loss(adv_imp, bt)

        U_comp = torch.tensor(0.0, device=dev)
        if config.w_comp > 0 and surrogates:
            losses = [s.compute_boundary_loss(s.compute_scores(adv_merged, adv_imp),
                                              bt.all_core_indices, bt.all_decoy_indices,
                                              margin=config.comp_margin)
                      for s in surrogates.values()]
            U_comp = -sum(losses) / len(losses)

        g_rank_pv = _safe_grad(U_rank, adv_pv, retain_graph=True)
        U_task_proxy = None
        if config.task_mode == "proxy" and clean_repr_cache is not None:
            # proxy task loss also lives on graph A -> compute/backward within A's life
            g_comp_pv = _safe_grad(U_comp, adv_pv, retain_graph=True)
            H_adv = v6model.compress_pooler(adv_merged, COMPRESSOR_CLS[method], ratio)
            U_task_proxy = 1.0 - _cos_global(H_adv, clean_repr_cache[(method, budget)])
            g_task_pv = _safe_grad(U_task_proxy, adv_pv, retain_graph=False)  # frees A
        else:
            g_comp_pv = _safe_grad(U_comp, adv_pv, retain_graph=False)        # frees A
            g_task_pv = torch.zeros_like(adv_pv)

        # --- graph B: compressed task LLM term (MAXIMIZE) ---
        U_task = U_task_proxy if U_task_proxy is not None else torch.tensor(0.0, device=dev)
        if config.task_mode == "ground_truth" and answer_ids:
            logits = v6model.task_logits(adv_pv, inputs, answer_ids,
                                         COMPRESSOR_CLS[method], ratio)
            U_task = task_ce(logits, answer_ids)
            g_task_pv = _safe_grad(U_task, adv_pv, retain_graph=False)        # frees B

        # --- graph C: full-preservation terms (MINIMIZE -> subtracted) ---
        P_full_ce = torch.tensor(0.0, device=dev)
        P_distill = torch.tensor(0.0, device=dev)
        if answer_ids and (config.w_full > 0 or config.w_distill > 0):
            full_logits = v6model.task_logits(adv_pv, inputs, answer_ids,
                                              compressor_cls=None, ratio=1.0)
            if config.w_full > 0:
                P_full_ce = task_ce(full_logits, answer_ids)
            if config.w_distill > 0 and clean_full_logits is not None:
                P_distill = logit_kl(full_logits, clean_full_logits)
            g_full_ce_pv = _safe_grad(P_full_ce, adv_pv, retain_graph=True)
            g_distill_pv = _safe_grad(P_distill, adv_pv, retain_graph=False)  # frees C
        else:
            g_full_ce_pv = torch.zeros_like(adv_pv)
            g_distill_pv = torch.zeros_like(adv_pv)

        g_task = g_task_pv * mask
        g_rank = g_rank_pv * mask
        g_comp = g_comp_pv * mask
        g_full_ce = g_full_ce_pv * mask
        g_distill = g_distill_pv * mask

        # full-damage direction = ascent dir that hurts Full prediction
        g_full_damage = (config.w_full * g_full_ce + config.w_distill * g_distill)

        # ---- optional gradient projection (compressed attack dir vs full-damage dir) ----
        g_task_used = g_task
        proj_triggered = False
        if config.use_projection and float(g_full_damage.norm()) > 0 and float(g_task.norm()) > 0:
            g_task_used, proj_triggered, cos_cf, nb, na = project_away(g_task, g_full_damage)
            cos_comp_full_list.append(cos_cf)
            if proj_triggered:
                proj_triggered_cnt += 1
        else:
            cos_cf = float(F.cosine_similarity(g_task.reshape(1, -1), g_full_damage.reshape(1, -1)).item()) \
                if (float(g_task.norm()) > 0 and float(g_full_damage.norm()) > 0) else 0.0
            cos_comp_full_list.append(cos_cf)

        g = (config.w_task * _normalize(g_task_used, config.eps_grad)
             + config.w_rank * _normalize(g_rank, config.eps_grad)
             + config.w_comp * _normalize(g_comp, config.eps_grad)
             - config.w_full * _normalize(g_full_ce, config.eps_grad)
             - config.w_distill * _normalize(g_distill, config.eps_grad))

        if not torch.isfinite(g).all():
            return {"status": "bad_grad", "step": step, "N_full": N_full,
                    "initial_delta_sha256": init_sha}

        # ---- gradient ASCENT ----
        delta.data = delta.data + config.alpha * g.sign()
        delta.data.clamp_(-config.epsilon, config.epsilon)

        if step % diag_interval == 0 or step == config.steps - 1:
            diag.append({
                "step": step,
                "U_task": float(U_task.detach()) if isinstance(U_task, torch.Tensor) else 0.0,
                "U_rank": float(U_rank.detach()) if isinstance(U_rank, torch.Tensor) else 0.0,
                "U_comp": float(U_comp.detach()) if isinstance(U_comp, torch.Tensor) else 0.0,
                "P_full_ce": float(P_full_ce.detach()) if isinstance(P_full_ce, torch.Tensor) else 0.0,
                "P_distill": float(P_distill.detach()) if isinstance(P_distill, torch.Tensor) else 0.0,
                "g_task_l2": float(g_task.norm()),
                "g_full_l2": float(g_full_damage.norm()),
                "cos_comp_full": cos_cf,
                "proj_triggered": int(proj_triggered),
                "delta_linf": float(delta.data.abs().max()),
            })

    dlinf = float(delta.data.abs().max())
    adv_pv_final = (pv + delta.detach()).clamp(PIXEL_RANGE_MIN, PIXEL_RANGE_MAX)

    # ---- final diagnostics (differentiable logits, no generation) ----
    full_adv_ce = float("nan")
    comp_adv_ce = float("nan")
    full_logit_kl = float("nan")
    diagnostic_exception = ""
    if answer_ids:
        with torch.no_grad():
            try:
                full_adv_ce = float(task_ce(
                    v6model.task_logits(adv_pv_final, inputs, answer_ids,
                                        compressor_cls=None, ratio=1.0),
                    answer_ids).detach())
                if clean_full_logits is not None:
                    full_logit_kl = float(logit_kl(
                        v6model.task_logits(adv_pv_final, inputs, answer_ids,
                                            compressor_cls=None, ratio=1.0),
                        clean_full_logits).detach())
                # compressed CE at the 1/3 budget (representative) via first schedule method
                m0, b0 = schedule[-1]
                comp_adv_ce = float(task_ce(
                    v6model.task_logits(adv_pv_final, inputs, answer_ids,
                                        COMPRESSOR_CLS[m0], config.ratio_for(b0)),
                    answer_ids).detach())
            except Exception as error:
                diagnostic_exception = (
                    f"final diagnostics raised {type(error).__name__}: {error}"
                )[:300]

    mean_grad_cos = (
        float(np.mean(cos_comp_full_list))
        if cos_comp_full_list
        else float("nan")
    )
    projection_trigger_rate = proj_triggered_cnt / config.steps
    status, status_error = _final_diagnostic_status(
        delta_linf=dlinf,
        epsilon=config.epsilon,
        full_clean_gt_loss=full_clean_ce,
        full_adv_gt_loss=full_adv_ce,
        compressed_adv_gt_loss=comp_adv_ce,
        full_logit_kl=full_logit_kl,
        mean_grad_cos_comp_full=mean_grad_cos,
        projection_trigger_rate=projection_trigger_rate,
        diagnostic_exception=diagnostic_exception,
    )

    return {
        "status": status,
        "error": status_error,
        "delta": delta.detach(),
        "adv_pv": adv_pv_final,
        "diag": diag,
        "final_delta_linf": dlinf,
        "N_full": N_full,
        "seed": seed,
        "initial_delta_sha256": init_sha,
        "delta_sha256": delta_sha256(delta),
        "full_clean_GT_loss": full_clean_ce,
        "full_adv_GT_loss": full_adv_ce,
        "compressed_adv_GT_loss": comp_adv_ce,
        "full_logit_KL": full_logit_kl,
        "mean_grad_cos_comp_full": mean_grad_cos,
        "projection_trigger_rate": projection_trigger_rate,
    }
