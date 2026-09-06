from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPImageProcessor, CLIPVisionModel, LlavaForConditionalGeneration


from fata.attacks.linf_image import IMAGE_SERIALIZATION_CONTRACT, quantize_linf_image
from fata.runtimes.llava.compression_zoo import apply_compression_patch
from fata.runtimes.llava.image_span import expanded_image_and_trailing_text_span
from fata.utils.paths import (
    resolve_dataset_relative_path,
    resolve_owned_output_path,
    safe_filename_component,
)
from fata.utils.run_contract import (
    artifact_identity,
    dataset_image_identity,
    ensure_run_contract,
    expected_unique_ids,
    sha256_file,
)


ALL_METHODS = ["VisionZIP", "VisPruner", "PruMerge", "FlowCut"]
ALL_DATASETS = ["TextVQA_Open", "VQAv2_Open", "ScienceQA_MC", "VQAv2_MC"]
ALL_ATTACKS = [
    "clean_clip",
    "random_clip",
    "base",
    "fata",
    "cage",
    "caa",
]

PRACTICAL_K = {
    "TextVQA_Open": 64,
    "VQAv2_Open": 64,
    "ScienceQA_MC": 32,
    "VQAv2_MC": 32,
}
FORMAL_VISION_LAYERS = (6, 12, 18, 24)
FORMAL_LLM_LAYERS = (8, 16, 24, 32)

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
)


@dataclass(frozen=True)
class RunConfig:
    project_root: Path
    llava_path: str
    clip_path: str
    method: str
    seed: int
    eps_255: float
    alpha_255: float
    steps: int
    lam: float
    target_k: int
    token_budget_mode: str
    output_dir: Path
    attack_cache_root: Path | None = None

    @property
    def epsilon(self) -> float:
        return self.eps_255 / 255.0

    @property
    def alpha(self) -> float:
        return self.alpha_255 / 255.0


@dataclass
class SampleFeatures:
    arrays: dict[str, np.ndarray]
    image_span_valid: int
    image_token_count: int
    llm_sequence_length: int


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_sample_seed(base_seed: int, sample_idx: int) -> int:
    sample_seed = int(base_seed) * 1_000_003 + int(sample_idx)
    random.seed(sample_seed)
    np.random.seed(sample_seed % (2**32 - 1))
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)
    return sample_seed


def parse_csv_arg(value: str, all_values: Iterable[str]) -> list[str]:
    if value.lower() in {"all", "*"}:
        return list(all_values)
    return [x.strip() for x in value.split(",") if x.strip()]


def expand2square(pil_img: Image.Image, background_color=(122, 116, 104)) -> Image.Image:
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


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


def normalized_quarter_layers(num_layers: int) -> list[int]:
    """Return fixed 1-based quarter-depth layer indices, deduplicated."""
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    candidates = [
        max(1, math.ceil(num_layers * 0.25)),
        max(1, math.ceil(num_layers * 0.50)),
        max(1, math.ceil(num_layers * 0.75)),
        num_layers,
    ]
    return list(dict.fromkeys(candidates))


def safe_tensor_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
        return output.last_hidden_state
    raise TypeError(f"Unsupported hook output type: {type(output)!r}")


def pool_tokens(tensor: torch.Tensor, remove_cls: bool) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected [B, N, D], got {tuple(tensor.shape)}")
    tokens = tensor[:, 1:, :] if remove_cls and tensor.shape[1] > 1 else tensor
    pooled = tokens.float().mean(dim=1)
    return F.normalize(pooled, dim=-1)


def npz_row_count(path: Path) -> int:
    if not path.exists():
        return -1
    try:
        with np.load(path, allow_pickle=False) as arr:
            return int(len(arr["sample_idx"]))
    except Exception:
        return -1


def csv_row_count(path: Path) -> int:
    """
    Count logical CSV records rather than physical text lines.

    CSV fields such as ScienceQA questions may contain embedded newlines.
    Counting file lines would therefore overestimate the number of samples.
    """
    if not path.exists():
        return -1

    try:
        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)
            return sum(1 for _ in reader)
    except Exception:
        return -1


def feature_artifacts_complete(
    npz_path: Path,
    csv_path: Path,
    expected_indices: list[int],
    expected_image_ids: list[str],
    required_feature_names: list[str] | None = None,
) -> bool:
    """Validate exact row identity, uniqueness, and aligned per-sample arrays."""

    if not npz_path.is_file() or not csv_path.is_file() or not expected_indices:
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            required_metadata = {
                "sample_idx",
                "label",
                "image_span_valid",
                "image_token_count",
                "llm_sequence_length",
                "image_id",
            }
            if not required_metadata.issubset(archive.files):
                return False
            discovered_features = {
                name
                for name in archive.files
                if name.startswith(("vis_l", "proj_", "llm_last_l", "llm_image_l"))
            }
            if required_feature_names is not None:
                if not set(required_feature_names).issubset(discovered_features):
                    return False
            elif not (
                any(name.startswith("vis_l") for name in discovered_features)
                and {"proj_in", "proj_out"}.issubset(discovered_features)
                and any(name.startswith("llm_last_l") for name in discovered_features)
                and any(name.startswith("llm_image_l") for name in discovered_features)
            ):
                return False
            npz_indices = [int(value) for value in archive["sample_idx"].tolist()]
            npz_image_ids = [str(value) for value in archive["image_id"].tolist()]
            row_count = len(expected_indices)
            exempt = {"vision_layers", "llm_layers"}
            for name in archive.files:
                array = archive[name]
                if name not in exempt and (array.ndim == 0 or len(array) != row_count):
                    return False
                if name in discovered_features and (
                    array.ndim != 2 or not np.isfinite(array).all()
                ):
                    return False
            labels = np.asarray(archive["label"])
            spans = np.asarray(archive["image_span_valid"])
            token_counts = np.asarray(archive["image_token_count"])
            sequence_lengths = np.asarray(archive["llm_sequence_length"])
            if (
                labels.ndim != 1
                or not set(labels.tolist()).issubset({0, 1})
                or spans.ndim != 1
                or not set(spans.tolist()).issubset({0, 1})
                or np.any(token_counts < 0)
                or np.any((spans == 1) & (token_counts <= 0))
                or np.any(sequence_lengths <= 0)
            ):
                return False
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        csv_indices = [int(row["sample_idx"]) for row in rows]
        csv_image_ids = [str(row["Image_ID"]) for row in rows]
    except (KeyError, OSError, ValueError, TypeError):
        return False

    return (
        len(npz_indices) == len(set(npz_indices))
        and len(csv_indices) == len(set(csv_indices))
        and npz_indices == expected_indices
        and csv_indices == expected_indices
        and npz_image_ids == expected_image_ids
        and csv_image_ids == expected_image_ids
        and len(expected_image_ids) == len(set(expected_image_ids))
    )


def dataset_paths_for_config(cfg: RunConfig, dataset: str) -> tuple[Path, Path]:
    """Resolve one formal dataset without constructing a GPU model engine."""

    dataset_dir = Path(cfg.project_root) / dataset
    return dataset_dir, dataset_dir / f"{dataset}_mapping.jsonl"


def load_dataset_without_model(cfg: RunConfig, dataset: str) -> list[dict[str, Any]]:
    """Load and validate a mapping for CPU-only resume preflight."""

    _dataset_dir, qa_file = dataset_paths_for_config(cfg, dataset)
    if not qa_file.exists():
        raise FileNotFoundError(qa_file)
    rows: list[dict[str, Any]] = []
    with qa_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict) or not row.get("image_filename"):
                    raise ValueError(f"invalid mapping row in {qa_file}")
                rows.append(row)
    if not rows:
        raise ValueError(f"mapping is empty: {qa_file}")
    expected_unique_ids(rows, "image_filename")
    return rows


def token_budget_for_config(cfg: RunConfig, dataset: str) -> int:
    if cfg.token_budget_mode == "full":
        return 576
    if cfg.token_budget_mode == "practical":
        if dataset == "VQAv2_MC" and cfg.method == "FlowCut":
            return 64
        return PRACTICAL_K[dataset]
    raise ValueError(f"Unknown token_budget_mode: {cfg.token_budget_mode}")


def feature_chunk_paths(
    cfg: RunConfig,
    dataset: str,
    attack: str,
    token_budget: int,
    start: int,
    limit: int,
) -> tuple[Path, Path]:
    if token_budget <= 0 or start < 0 or limit <= 0 or cfg.seed < 0:
        raise ValueError("invalid ML-ATD feature chunk coordinates")
    method_component = safe_filename_component(cfg.method, label="method")
    dataset_component = safe_filename_component(dataset, label="dataset")
    attack_component = safe_filename_component(attack, label="attack")
    prefix = (
        f"mlat_feat_{method_component}_{dataset_component}_{attack_component}_k{token_budget}_"
        f"start{start}_limit{limit}_seed{cfg.seed}"
    )
    return (
        resolve_owned_output_path(cfg.output_dir, f"{prefix}.npz"),
        resolve_owned_output_path(cfg.output_dir, f"{prefix}.csv"),
    )


def build_feature_chunk_contract(
    cfg: RunConfig,
    *,
    dataset: str,
    mapping_path: Path,
    attack: str,
    token_budget: int,
    expected_indices: list[int],
    expected_image_ids: list[str],
    llava_identity: dict[str, Any],
    clip_identity: dict[str, Any],
    vision_layers: Iterable[int] = FORMAL_VISION_LAYERS,
    llm_layers: Iterable[int] = FORMAL_LLM_LAYERS,
    cache_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable ML-ATD feature contract without loading models."""

    return {
        "schema_version": 1,
        "image_serialization": IMAGE_SERIALIZATION_CONTRACT,
        "runtime": "mlatd_feature_extraction_v1",
        "feature_schema": "quarter_layers_projector_llm_last_image_v1",
        "dataset": dataset,
        "dataset_mapping_sha256": sha256_file(mapping_path),
        "dataset_images": dataset_image_identity(mapping_path, mapping_path.parent),
        "method": cfg.method,
        "attack": attack,
        "llava_model": llava_identity,
        "clip_model": clip_identity,
        "seed": cfg.seed,
        "eps_255": cfg.eps_255,
        "alpha_255": cfg.alpha_255,
        "steps": cfg.steps,
        "lam": cfg.lam,
        "target_k": cfg.target_k,
        "token_budget_mode": cfg.token_budget_mode,
        "token_budget": token_budget,
        "vision_layers": list(vision_layers),
        "llm_layers": list(llm_layers),
        "expected_sample_indices": expected_indices,
        "expected_image_ids": expected_image_ids,
        "source_cache_contract": cache_contract,
    }


def feature_chunk_complete_without_model(
    cfg: RunConfig,
    *,
    dataset: str,
    attack: str,
    qa_database: list[dict[str, Any]],
    start: int,
    limit: int,
    llava_identity: dict[str, Any],
    clip_identity: dict[str, Any],
    cache_contract: dict[str, Any] | None = None,
    contract_overrides: dict[str, Any] | None = None,
    eps_255: float | None = None,
    alpha_255: float | None = None,
    steps: int | None = None,
    expected_cache_root: Path | None = None,
    path_attack: str | None = None,
) -> bool:
    """Fail-closed CPU-only gate used before allocating LLaVA or CLIP.

    The gate compares the complete immutable contract to the current mapping,
    image tree and model bytes, then runs the same strict triplet checker used
    after writes.  Merely finding three plausibly named files is insufficient.
    """

    if start < 0 or limit <= 0 or start + limit > len(qa_database):
        raise ValueError(
            f"ML-ATD chunk exceeds exact mapping cohort: "
            f"start={start}, limit={limit}, rows={len(qa_database)}"
        )
    token_budget = token_budget_for_config(cfg, dataset)
    npz_path, csv_path = feature_chunk_paths(
        cfg, dataset, attack if path_attack is None else path_attack,
        token_budget, start, limit
    )
    meta_path = npz_path.with_suffix(".meta.json")
    if not (npz_path.is_file() and csv_path.is_file() and meta_path.is_file()):
        return False

    expected_indices = list(range(start, start + limit))
    expected_image_ids = [
        str(qa_database[index]["image_filename"]) for index in expected_indices
    ]
    expected_contract = build_feature_chunk_contract(
        cfg,
        dataset=dataset,
        mapping_path=dataset_paths_for_config(cfg, dataset)[1],
        attack=attack,
        token_budget=token_budget,
        expected_indices=expected_indices,
        expected_image_ids=expected_image_ids,
        llava_identity=llava_identity,
        clip_identity=clip_identity,
        cache_contract=cache_contract,
    )
    if contract_overrides:
        expected_contract.update(contract_overrides)
    try:
        stored_contract = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if stored_contract != expected_contract:
        return False

    required_features = [
        *(f"vis_l{layer:02d}" for layer in FORMAL_VISION_LAYERS),
        "proj_in",
        "proj_out",
        *(f"llm_last_l{layer:02d}" for layer in FORMAL_LLM_LAYERS),
        *(f"llm_image_l{layer:02d}" for layer in FORMAL_LLM_LAYERS),
    ]
    if not feature_artifacts_complete(
        npz_path,
        csv_path,
        expected_indices,
        expected_image_ids,
        required_features,
    ):
        return False

    from .check_mlat_full_grid import validate_triplet

    return not validate_triplet(
        npz_path,
        method=cfg.method,
        dataset=dataset,
        attack=attack,
        token_budget=token_budget,
        start=start,
        limit=limit,
        seed=cfg.seed,
        token_budget_mode=cfg.token_budget_mode,
        eps_255=cfg.eps_255 if eps_255 is None else eps_255,
        alpha_255=cfg.alpha_255 if alpha_255 is None else alpha_255,
        steps=cfg.steps if steps is None else steps,
        lam=cfg.lam,
        target_k=cfg.target_k,
        require_valid_spans=True,
        expected_cache_root=expected_cache_root,
    )

class MLATFeatureEngine:
    """
    HiddenDetect-inspired Multi-Level Activation Trajectory feature extractor.

    It preserves HiddenDetect's multi-layer activation-monitoring idea, while replacing
    the original refusal-oriented score with layer-wise Base-attack directions that are
    fitted later by the analysis script.
    """

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        if cfg.method not in ALL_METHODS:
            raise ValueError(f"Unknown method: {cfg.method}")

        set_global_seed(cfg.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llava_identity = artifact_identity(cfg.llava_path)
        self.clip_identity = artifact_identity(cfg.clip_path)

        print(f"[Info] Loading standalone CLIP from {cfg.clip_path}")
        try:
            self.clip_encoder = CLIPVisionModel.from_pretrained(
                cfg.clip_path,
                output_attentions=True,
                attn_implementation="eager",
            )
        except TypeError:
            self.clip_encoder = CLIPVisionModel.from_pretrained(
                cfg.clip_path,
                output_attentions=True,
            )
        self.clip_encoder.to(self.device).eval()
        self.clip_encoder.requires_grad_(False)
        self.clip_processor = CLIPImageProcessor.from_pretrained(cfg.clip_path)

        print(f"[Info] Loading LLaVA from {cfg.llava_path}")
        self.llava_processor = AutoProcessor.from_pretrained(cfg.llava_path, use_fast=False)
        llava_load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "cuda" if torch.cuda.is_available() else None,
            "low_cpu_mem_usage": True,
        }
        try:
            self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                cfg.llava_path,
                attn_implementation="eager",
                **llava_load_kwargs,
            )
        except TypeError:
            self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                cfg.llava_path,
                **llava_load_kwargs,
            )
        if not torch.cuda.is_available():
            self.llava_model.to(self.device)
        self.llava_model.eval()
        self.llava_model.requires_grad_(False)
        self.device = next(self.llava_model.parameters()).device

        self.vision_model = self._resolve_vision_model()
        self.projector = self._resolve_projector()
        apply_compression_patch(self.vision_model, cfg.method)

        vision_layers_module = self._resolve_vision_layers_module()
        self.num_vision_layers = len(vision_layers_module)
        self.vision_layers = normalized_quarter_layers(self.num_vision_layers)

        self.num_llm_layers = int(
            getattr(
                getattr(self.llava_model.config, "text_config", self.llava_model.config),
                "num_hidden_layers",
            )
        )
        self.llm_layers = normalized_quarter_layers(self.num_llm_layers)

        if tuple(self.vision_layers) != FORMAL_VISION_LAYERS:
            raise RuntimeError(
                "formal ML-ATD requires visual layers "
                f"{FORMAL_VISION_LAYERS}, found {tuple(self.vision_layers)}"
            )
        if tuple(self.llm_layers) != FORMAL_LLM_LAYERS:
            raise RuntimeError(
                "formal ML-ATD requires LLM layers "
                f"{FORMAL_LLM_LAYERS}, found {tuple(self.llm_layers)}"
            )

        print(f"[Info] Vision layers (1-based): {self.vision_layers}/{self.num_vision_layers}")
        print(f"[Info] LLM layers (1-based): {self.llm_layers}/{self.num_llm_layers}")

        self.mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=self.device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=self.device
        ).view(1, 3, 1, 1)

        self._vision_cache: dict[str, np.ndarray] = {}
        self._projector_cache: dict[str, Any] = {}
        self._hook_handles: list[Any] = []
        self._register_hooks(vision_layers_module)

    def close(self) -> None:
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles.clear()

    def _resolve_vision_model(self):
        candidates = []
        if hasattr(self.llava_model, "vision_tower"):
            candidates.append(self.llava_model.vision_tower)
        if hasattr(self.llava_model, "model") and hasattr(self.llava_model.model, "vision_tower"):
            candidates.append(self.llava_model.model.vision_tower)
        for tower in candidates:
            if hasattr(tower, "vision_model"):
                return tower.vision_model
            return tower
        raise AttributeError("Could not resolve LLaVA vision tower")

    def _resolve_projector(self):
        if hasattr(self.llava_model, "multi_modal_projector"):
            return self.llava_model.multi_modal_projector
        if hasattr(self.llava_model, "model") and hasattr(self.llava_model.model, "multi_modal_projector"):
            return self.llava_model.model.multi_modal_projector
        raise AttributeError("Could not resolve LLaVA multi_modal_projector")

    def _resolve_vision_layers_module(self):
        if hasattr(self.vision_model, "encoder") and hasattr(self.vision_model.encoder, "layers"):
            return self.vision_model.encoder.layers
        raise AttributeError("Could not resolve vision_model.encoder.layers")

    def _register_hooks(self, vision_layers_module) -> None:
        for layer_1b in self.vision_layers:
            module = vision_layers_module[layer_1b - 1]
            key = f"vis_l{layer_1b:02d}"

            def make_hook(cache_key: str):
                def hook(_module, _inputs, output):
                    tensor = safe_tensor_from_output(output)
                    pooled = pool_tokens(tensor, remove_cls=True)
                    self._vision_cache[cache_key] = (
                        pooled[0].detach().cpu().numpy().astype(np.float16)
                    )
                return hook

            self._hook_handles.append(module.register_forward_hook(make_hook(key)))

        def projector_pre_hook(_module, inputs):
            if not inputs:
                return
            tensor = safe_tensor_from_output(inputs[0])
            pooled = pool_tokens(tensor, remove_cls=False)
            self._projector_cache["proj_in"] = (
                pooled[0].detach().cpu().numpy().astype(np.float16)
            )
            self._projector_cache["token_count"] = int(tensor.shape[1])

        def projector_hook(_module, _inputs, output):
            tensor = safe_tensor_from_output(output)
            pooled = pool_tokens(tensor, remove_cls=False)
            self._projector_cache["proj_out"] = (
                pooled[0].detach().cpu().numpy().astype(np.float16)
            )
            self._projector_cache["token_count"] = int(tensor.shape[1])

        self._hook_handles.append(self.projector.register_forward_pre_hook(projector_pre_hook))
        self._hook_handles.append(self.projector.register_forward_hook(projector_hook))

    def token_budget_for_dataset(self, dataset: str) -> int:
        return token_budget_for_config(self.cfg, dataset)

    def dataset_paths(self, dataset: str) -> tuple[Path, Path]:
        return dataset_paths_for_config(self.cfg, dataset)

    def load_dataset(self, dataset: str) -> list[dict[str, Any]]:
        return load_dataset_without_model(self.cfg, dataset)

    def get_clean_image_01(self, image: Image.Image) -> torch.Tensor:
        inputs = self.clip_processor(images=image, return_tensors="pt")
        normalized = inputs["pixel_values"].to(self.device)
        return (normalized * self.std + self.mean).detach()

    def clean_clip_image(self, image: Image.Image) -> Image.Image:
        clean_image_01 = self.get_clean_image_01(image)
        return quantize_linf_image(clean_image_01, clean_image_01, 0.0)

    def random_clip_image(self, image: Image.Image, sample_seed: int) -> Image.Image:
        clean_image_01 = self.get_clean_image_01(image)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(sample_seed)
        noise = torch.empty_like(clean_image_01).uniform_(
            -self.cfg.epsilon,
            self.cfg.epsilon,
            generator=generator,
        )
        adversarial = torch.clamp(clean_image_01 + noise, 0, 1)
        return quantize_linf_image(clean_image_01, adversarial, self.cfg.epsilon)

    def prepare_attack_context(
        self, image: Image.Image, prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        previous_k = getattr(self.vision_model, "current_K", None)
        self.vision_model.current_K = 576
        raw_inputs = self.llava_processor(text=prompt, images=image, return_tensors="pt")
        pixel_values = raw_inputs["pixel_values"].to(self.device, dtype=torch.float16)

        with torch.no_grad():
            vision_outputs = self.vision_model(
                pixel_values,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            if vision_outputs.attentions is None:
                raise RuntimeError(
                    "Vision attentions are None. Ensure eager attention is enabled for LLaVA/CLIP."
                )
            clean_attn = vision_outputs.attentions[-1][:, :, 0, 1:].mean(dim=1).squeeze(0)
            clip_outputs = self.clip_encoder(
                pixel_values.float(),
                output_attentions=True,
                return_dict=True,
            )
            clean_features = clip_outputs.last_hidden_state.squeeze(0)[1:].detach()

        rank = torch.argsort(torch.argsort(clean_attn, descending=True))
        target_mask = (rank < self.cfg.target_k).float().detach()
        clean_image_01 = self.get_clean_image_01(image)

        if previous_k is not None:
            self.vision_model.current_K = previous_k
        return clean_image_01, clean_features, target_mask

    def generate_base_image(
        self, clean_image_01: torch.Tensor, target_mask: torch.Tensor
    ) -> Image.Image:
        delta = torch.empty_like(clean_image_01).uniform_(
            -self.cfg.epsilon, self.cfg.epsilon
        ).requires_grad_(True)

        for _ in range(self.cfg.steps):
            adv_img = torch.clamp(clean_image_01 + delta, 0, 1)
            outputs = self.clip_encoder(
                (adv_img - self.mean) / self.std,
                output_attentions=True,
                return_dict=True,
            )
            if outputs.attentions is None:
                raise RuntimeError("CLIP attentions are None during Base attack generation")
            score = outputs.attentions[-1][:, :, 0, 1:].mean(dim=1).squeeze(0)
            loss = torch.sum(score * target_mask)
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            delta = (delta - self.cfg.alpha * grad.sign()).detach()
            delta = torch.clamp(delta, -self.cfg.epsilon, self.cfg.epsilon)
            delta.requires_grad_(True)

        adversarial = torch.clamp(clean_image_01 + delta.detach(), 0, 1)
        return quantize_linf_image(clean_image_01, adversarial, self.cfg.epsilon)

    def generate_fata_image(
        self,
        clean_image_01: torch.Tensor,
        clean_features: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Image.Image:
        delta = torch.empty_like(clean_image_01).uniform_(
            -self.cfg.epsilon, self.cfg.epsilon
        ).requires_grad_(True)

        for _ in range(self.cfg.steps):
            adv_img = torch.clamp(clean_image_01 + delta, 0, 1)
            outputs = self.clip_encoder(
                (adv_img - self.mean) / self.std,
                output_attentions=True,
                return_dict=True,
            )
            if outputs.attentions is None:
                raise RuntimeError("CLIP attentions are None during FATA generation")
            score = outputs.attentions[-1][:, :, 0, 1:].mean(dim=1).squeeze(0)
            adv_features = outputs.last_hidden_state.squeeze(0)[1:]

            loss_attn = torch.sum(score * target_mask)
            similarity = F.cosine_similarity(
                adv_features,
                clean_features,
                dim=-1,
            )
            loss_sem = 1.0 - (
                torch.sum(similarity * target_mask) / (target_mask.sum() + 1e-8)
            )
            loss = loss_attn + self.cfg.lam * loss_sem
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            delta = (delta - self.cfg.alpha * grad.sign()).detach()
            delta = torch.clamp(delta, -self.cfg.epsilon, self.cfg.epsilon)
            delta.requires_grad_(True)

        adversarial = torch.clamp(clean_image_01 + delta.detach(), 0, 1)
        return quantize_linf_image(clean_image_01, adversarial, self.cfg.epsilon)

    def make_detection_image(
        self,
        image: Image.Image,
        gt_data: dict[str, Any],
        attack: str,
        sample_seed: int,
    ) -> Image.Image:
        if attack == "clean_clip":
            return self.clean_clip_image(image)
        if attack == "random_clip":
            return self.random_clip_image(image, sample_seed)
        if attack in {"base", "fata"}:
            prompt = build_prompt(gt_data)
            clean_image_01, clean_features, target_mask = self.prepare_attack_context(image, prompt)
            if attack == "base":
                return self.generate_base_image(clean_image_01, target_mask)
            return self.generate_fata_image(clean_image_01, clean_features, target_mask)
        raise ValueError(f"Unsupported attack type: {attack}")

    def _move_processor_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if value is None:
                continue
            if torch.is_tensor(value):
                if key == "pixel_values":
                    moved[key] = value.to(self.device, dtype=torch.float16)
                else:
                    moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _image_token_ids(self) -> tuple[int | None, int | None]:
        config_value = getattr(self.llava_model.config, "image_token_index", None)
        config_token_id = int(config_value) if config_value is not None else None
        tokenizer_token_id = None
        try:
            value = self.llava_processor.tokenizer.convert_tokens_to_ids("<image>")
            if value is not None and value >= 0:
                tokenizer_token_id = int(value)
        except Exception:
            pass
        return config_token_id, tokenizer_token_id

    def _infer_image_indices(
        self,
        input_ids: torch.Tensor | None,
        output_seq_len: int,
        image_token_count: int,
    ) -> torch.Tensor:
        if input_ids is None:
            raise RuntimeError("LLaVA processor returned no input_ids")
        config_token_id, tokenizer_token_id = self._image_token_ids()
        start, end, _ = expanded_image_and_trailing_text_span(
            input_ids,
            output_sequence_length=output_seq_len,
            image_token_count=image_token_count,
            config_token_id=config_token_id,
            tokenizer_token_id=tokenizer_token_id,
        )
        return torch.arange(start, end, device=self.device)

    @torch.inference_mode()
    def extract_multilevel_features(
        self,
        image: Image.Image,
        prompt: str,
        token_budget: int,
    ) -> SampleFeatures:
        self._vision_cache.clear()
        self._projector_cache.clear()
        self.vision_model.current_K = int(token_budget)

        raw_inputs = self.llava_processor(text=prompt, images=image, return_tensors="pt")
        inputs = self._move_processor_inputs(raw_inputs)
        outputs = self.llava_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("LLaVA did not return language-model hidden states")

        missing_visual = [
            f"vis_l{x:02d}" for x in self.vision_layers if f"vis_l{x:02d}" not in self._vision_cache
        ]
        if missing_visual:
            raise RuntimeError(f"Vision hooks did not fire: {missing_visual}")
        if "proj_in" not in self._projector_cache or "proj_out" not in self._projector_cache:
            raise RuntimeError("Projector hooks did not capture proj_in/proj_out")

        arrays = dict(self._vision_cache)
        arrays["proj_in"] = self._projector_cache["proj_in"]
        arrays["proj_out"] = self._projector_cache["proj_out"]

        image_token_count = int(self._projector_cache.get("token_count", 0))
        output_seq_len = int(hidden_states[-1].shape[1])
        image_indices = self._infer_image_indices(
            inputs.get("input_ids"), output_seq_len, image_token_count
        )
        image_span_valid = 1

        for layer_1b in self.llm_layers:
            hidden = hidden_states[layer_1b].float()
            last_feat = F.normalize(hidden[:, -1, :], dim=-1)[0]
            arrays[f"llm_last_l{layer_1b:02d}"] = (
                last_feat.detach().cpu().numpy().astype(np.float16)
            )

            image_feat = hidden.index_select(1, image_indices).mean(dim=1)
            image_feat = F.normalize(image_feat, dim=-1)[0]
            arrays[f"llm_image_l{layer_1b:02d}"] = (
                image_feat.detach().cpu().numpy().astype(np.float16)
            )

        return SampleFeatures(
            arrays=arrays,
            image_span_valid=image_span_valid,
            image_token_count=image_token_count,
            llm_sequence_length=output_seq_len,
        )

    def expected_paths(
        self,
        dataset: str,
        attack: str,
        token_budget: int,
        start: int,
        limit: int,
    ) -> tuple[Path, Path]:
        return feature_chunk_paths(
            self.cfg, dataset, attack, token_budget, start, limit
        )

    def chunk_contract(
        self,
        *,
        dataset: str,
        mapping_path: Path,
        attack: str,
        token_budget: int,
        expected_indices: list[int],
        expected_image_ids: list[str],
        cache_contract: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return build_feature_chunk_contract(
            self.cfg,
            dataset=dataset,
            mapping_path=mapping_path,
            attack=attack,
            token_budget=token_budget,
            expected_indices=expected_indices,
            expected_image_ids=expected_image_ids,
            llava_identity=self.llava_identity,
            clip_identity=self.clip_identity,
            vision_layers=self.vision_layers,
            llm_layers=self.llm_layers,
            cache_contract=cache_contract,
        )

    def output_complete(
        self,
        npz_path: Path,
        csv_path: Path,
        expected_indices: list[int],
        expected_image_ids: list[str],
        *,
        dataset: str,
        attack: str,
        token_budget: int,
        eps_255: float | None = None,
        alpha_255: float | None = None,
        steps: int | None = None,
    ) -> bool:
        if not feature_artifacts_complete(
            npz_path,
            csv_path,
            expected_indices,
            expected_image_ids,
            [
                *(f"vis_l{layer:02d}" for layer in self.vision_layers),
                "proj_in",
                "proj_out",
                *(f"llm_last_l{layer:02d}" for layer in self.llm_layers),
                *(f"llm_image_l{layer:02d}" for layer in self.llm_layers),
            ],
        ):
            return False
        if expected_indices != list(
            range(expected_indices[0], expected_indices[0] + len(expected_indices))
        ):
            return False
        # The fail-closed release checker is also the resume gate.  This keeps
        # a structurally plausible but stale/corrupt NPZ/CSV/meta triple from
        # being permanently skipped before the formal checker is run.
        from .check_mlat_full_grid import validate_triplet

        return not validate_triplet(
            npz_path,
            method=self.cfg.method,
            dataset=dataset,
            attack=attack,
            token_budget=token_budget,
            start=expected_indices[0],
            limit=len(expected_indices),
            seed=self.cfg.seed,
            token_budget_mode=self.cfg.token_budget_mode,
            eps_255=self.cfg.eps_255 if eps_255 is None else eps_255,
            alpha_255=self.cfg.alpha_255 if alpha_255 is None else alpha_255,
            steps=self.cfg.steps if steps is None else steps,
            lam=self.cfg.lam,
            target_k=self.cfg.target_k,
            require_valid_spans=True,
            expected_cache_root=(
                self.cfg.attack_cache_root if attack in {"cage", "caa"} else None
            ),
        )

    def process_chunk(
        self,
        dataset: str,
        qa_database: list[dict[str, Any]],
        attack: str,
        start: int,
        limit: int,
        overwrite_incomplete: bool,
    ) -> None:
        if attack in {"cage", "caa"}:
            raise ValueError(
                "CAGE/CAA require an explicit cache contract; run "
                "generate_attack_cache_cage_caa and extract_cached_attack_mlat"
            )
        if start < 0 or limit <= 0 or start + limit > len(qa_database):
            raise ValueError(
                f"ML-ATD chunk exceeds exact mapping cohort: "
                f"start={start}, limit={limit}, rows={len(qa_database)}"
            )
        dataset_dir, _qa_file = self.dataset_paths(dataset)
        end = start + limit
        expected_rows = limit
        token_budget = self.token_budget_for_dataset(dataset)
        npz_path, csv_path = self.expected_paths(
            dataset, attack, token_budget, start, expected_rows
        )

        expected_indices = list(range(start, end))
        expected_image_ids = [
            str(qa_database[index]["image_filename"]) for index in expected_indices
        ]
        meta_path = npz_path.with_suffix(".meta.json")
        contract = self.chunk_contract(
            dataset=dataset,
            mapping_path=_qa_file,
            attack=attack,
            token_budget=token_budget,
            expected_indices=expected_indices,
            expected_image_ids=expected_image_ids,
        )
        ensure_run_contract(
            meta_path,
            contract,
            result_path=[npz_path, csv_path],
        )

        if self.output_complete(
            npz_path,
            csv_path,
            expected_indices,
            expected_image_ids,
            dataset=dataset,
            attack=attack,
            token_budget=token_budget,
        ) and not overwrite_incomplete:
            print(f"[SKIP] complete {npz_path.name}")
            return
        if (npz_path.exists() or csv_path.exists()) and not overwrite_incomplete:
            raise FileExistsError(
                f"Incomplete output exists for {npz_path.name}. "
                "Use --overwrite_incomplete to replace it."
            )

        print(
            f"[START] method={self.cfg.method} dataset={dataset} attack={attack} "
            f"K={token_budget} range=[{start},{end})"
        )
        feature_lists: dict[str, list[np.ndarray]] = {}
        metadata_rows: list[dict[str, Any]] = []

        for sample_idx in tqdm(range(start, end), desc=npz_path.stem, leave=False):
            gt_data = qa_database[sample_idx]
            sample_seed = set_sample_seed(self.cfg.seed, sample_idx)
            img_filename = gt_data["image_filename"]
            img_path = resolve_dataset_relative_path(dataset_dir, img_filename)
            if not img_path.exists():
                raise FileNotFoundError(img_path)

            raw_image = Image.open(img_path).convert("RGB")
            image = expand2square(raw_image)
            detection_image = self.make_detection_image(
                image=image,
                gt_data=gt_data,
                attack=attack,
                sample_seed=sample_seed,
            )
            prompt = build_prompt(gt_data)
            sample_features = self.extract_multilevel_features(
                detection_image,
                prompt,
                token_budget,
            )

            for name, vector in sample_features.arrays.items():
                feature_lists.setdefault(name, []).append(vector)

            metadata_rows.append(
                {
                    "sample_idx": sample_idx,
                    "Image_ID": img_filename,
                    "Question": gt_data.get("question", ""),
                    "dataset": dataset,
                    "method": self.cfg.method,
                    "attack_for_detection": attack,
                    "label": 0 if attack in {"clean_clip", "random_clip"} else 1,
                    "seed": self.cfg.seed,
                    "sample_seed": sample_seed,
                    "token_budget": token_budget,
                    "token_budget_mode": self.cfg.token_budget_mode,
                    "eps_255": self.cfg.eps_255,
                    "alpha_255": self.cfg.alpha_255,
                    "steps": self.cfg.steps,
                    "lam": self.cfg.lam,
                    "target_k": self.cfg.target_k,
                    "vision_layers": ",".join(map(str, self.vision_layers)),
                    "llm_layers": ",".join(map(str, self.llm_layers)),
                    "image_span_valid": sample_features.image_span_valid,
                    "image_token_count": sample_features.image_token_count,
                    "llm_sequence_length": sample_features.llm_sequence_length,
                }
            )

            if torch.cuda.is_available() and (sample_idx - start + 1) % 20 == 0:
                torch.cuda.empty_cache()

        if len(metadata_rows) != expected_rows:
            raise RuntimeError(
                f"Expected {expected_rows} rows, extracted {len(metadata_rows)} "
                f"for {npz_path.name}"
            )

        output_dir = Path(self.cfg.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.resolve().is_dir():
            raise ValueError(f"ML-ATD output root is not a directory: {output_dir}")
        # Re-resolve after directory creation so a pre-existing child symlink
        # can never be accepted as a formal feature artifact.
        npz_path, csv_path = self.expected_paths(
            dataset, attack, token_budget, start, expected_rows
        )
        fd_npz, tmp_npz_name = tempfile.mkstemp(
            prefix=f".{npz_path.name}.", suffix=".tmp.npz", dir=npz_path.parent
        )
        os.close(fd_npz)
        tmp_npz = Path(tmp_npz_name)
        fd_csv, tmp_csv_name = tempfile.mkstemp(
            prefix=f".{csv_path.name}.", suffix=".tmp", dir=csv_path.parent
        )
        tmp_csv = Path(tmp_csv_name)

        npz_payload: dict[str, Any] = {
            name: np.stack(vectors, axis=0).astype(np.float16)
            for name, vectors in feature_lists.items()
        }
        npz_payload.update(
            {
                "sample_idx": np.asarray([r["sample_idx"] for r in metadata_rows], dtype=np.int64),
                "label": np.asarray([r["label"] for r in metadata_rows], dtype=np.int8),
                "image_span_valid": np.asarray(
                    [r["image_span_valid"] for r in metadata_rows], dtype=np.int8
                ),
                "image_token_count": np.asarray(
                    [r["image_token_count"] for r in metadata_rows], dtype=np.int32
                ),
                "llm_sequence_length": np.asarray(
                    [r["llm_sequence_length"] for r in metadata_rows], dtype=np.int32
                ),
                "vision_layers": np.asarray(self.vision_layers, dtype=np.int16),
                "llm_layers": np.asarray(self.llm_layers, dtype=np.int16),
                "image_id": np.asarray([r["Image_ID"] for r in metadata_rows]),
            }
        )
        try:
            np.savez_compressed(tmp_npz, **npz_payload)
            with tmp_npz.open("rb") as handle:
                os.fsync(handle.fileno())

            fieldnames = list(metadata_rows[0].keys())
            with os.fdopen(fd_csv, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metadata_rows)
                handle.flush()
                os.fsync(handle.fileno())
            fd_csv = -1

            # Revalidate immediately before replace. os.replace replaces a
            # destination symlink itself, but rejecting it is safer and makes
            # tampering visible instead of silently repairing the path.
            npz_path, csv_path = self.expected_paths(
                dataset, attack, token_budget, start, expected_rows
            )
            os.replace(tmp_npz, npz_path)
            os.replace(tmp_csv, csv_path)
        finally:
            if fd_csv >= 0:
                os.close(fd_csv)
            for temporary in (tmp_npz, tmp_csv):
                if temporary.exists():
                    temporary.unlink()

        if not self.output_complete(
            npz_path,
            csv_path,
            expected_indices,
            expected_image_ids,
            dataset=dataset,
            attack=attack,
            token_budget=token_budget,
        ):
            raise RuntimeError(f"Post-write completeness check failed: {npz_path}")
        print(f"[DONE] {npz_path.name} rows={expected_rows}")
