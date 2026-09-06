# fata_detection_defense/mahalanobis/run_mahalanobis_detection.py

from __future__ import annotations

import os
import csv
import json
import re
import argparse
import random
import pickle
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import AutoProcessor, LlavaForConditionalGeneration

from fata.attacks.linf_image import IMAGE_SERIALIZATION_CONTRACT, quantize_linf_image
from fata.runtimes.llava.compression_zoo import apply_compression_patch
from fata.runtimes.llava.attack_cache_io import (
    attack_cache_contract_path,
    attack_namespace,
    baseline_attack_definition,
    baseline_attack_contract_extra,
    build_attack_cache_contract,
    load_attack_image,
)
from fata.utils.paths import (
    assert_no_output_file_collision,
    assert_output_separate,
    create_owned_output_directory,
    resolve_attack_cache_root,
    resolve_dataset_relative_path,
    resolve_owned_output_path,
)
from fata.utils.run_contract import (
    artifact_identity,
    assert_exact_completion,
    dataset_image_identity,
    ensure_run_contract,
    expected_unique_ids,
    require_run_contract,
    sha256_file,
)


# ================= 1. 参数 =================

parser = argparse.ArgumentParser(description="Mahalanobis Detection for FATA")

parser.add_argument("--mode", type=str, required=True, choices=["fit", "eval"])

parser.add_argument(
    "--method",
    type=str,
    default="VisionZIP",
    choices=["VisionZIP", "VisPruner", "PruMerge", "FlowCut"],
)

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["TextVQA_Open", "ScienceQA_MC", "VQAv2_Open", "VQAv2_MC"],
)

parser.add_argument(
    "--attack_for_detection",
    type=str,
    default="clean",
    choices=["clean", "random", "base", "fata","cage","caa"],
    help="eval mode only",
)

parser.add_argument("--fit_start", type=int, default=0)
parser.add_argument("--fit_limit", type=int, default=500)

parser.add_argument("--eval_start", type=int, default=500)
parser.add_argument("--eval_limit", type=int, default=200)

parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--lam", type=float, default=1.0)
parser.add_argument("--eps_255", type=float, default=2.0)
parser.add_argument("--alpha_255", type=float, default=None)
parser.add_argument("--steps", type=int, default=100)

parser.add_argument("--pca_dim", type=int, default=128)

parser.add_argument(
    "--stats_path",
    type=str,
    default=None,
    help=(
        "Path to save/load runner-generated Mahalanobis clean statistics. "
        "It must remain under --output_dir; never use a third-party pickle."
    ),
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="fata_detection_defense/mahalanobis/results",
)

parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--model-path", default=os.environ.get("FATA_LLAVA_MODEL"))
parser.add_argument("--clip-path", default=os.environ.get("FATA_CLIP_MODEL"))
parser.add_argument("--dataset-root", default=os.environ.get("FATA_DATA_ROOT"))
parser.add_argument("--output-root", default=os.environ.get("FATA_OUTPUT_ROOT"))
parser.add_argument("--attack-cache-root", default=None)

args = parser.parse_args()
if args.alpha_255 is None:
    args.alpha_255 = 1.0 if args.attack_for_detection == "caa" else 0.5
if args.fit_start < 0 or args.eval_start < 0 or args.seed < 0:
    parser.error("fit/eval start offsets and seed must be non-negative")
if args.mode == "fit" and args.fit_limit <= 1:
    parser.error("--fit_limit must be greater than 1")
if args.mode == "eval" and args.eval_limit <= 0:
    parser.error("--eval_limit must be positive")
if (
    args.steps <= 0
    or not all(math.isfinite(value) for value in (args.eps_255, args.alpha_255, args.lam))
    or args.eps_255 <= 0
    or args.alpha_255 <= 0
    or args.lam < 0
):
    parser.error("attack values must be finite; epsilon/alpha positive and lambda non-negative")
if args.pca_dim <= 0:
    parser.error("--pca_dim must be positive")


# ================= 2. 全局配置 =================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not args.model_path or not args.clip_path or not args.dataset_root or not (args.output_dir or args.output_root):
    parser.error("provide model, CLIP, dataset, and output roots")
LLAVA_PATH = args.model_path
CLIP_PATH = args.clip_path
derived_output = args.output_dir == "fata_detection_defense/mahalanobis/results"
if derived_output:
    if not args.output_root:
        parser.error("pass --output-root or an explicit --output_dir")
protected_inputs = {
    "dataset_root": args.dataset_root,
    "LLaVA model": LLAVA_PATH,
    "CLIP model": CLIP_PATH,
}
if derived_output:
    output_root = assert_output_separate(args.output_root, protected_inputs)
    output_dir_root = resolve_owned_output_path(
        output_root, "detection/mahalanobis"
    )
else:
    output_root = None
    output_dir_root = assert_output_separate(args.output_dir, protected_inputs)
args.output_dir = str(output_dir_root)
args.attack_cache_root = str(
    resolve_attack_cache_root(
        explicit_cache_root=args.attack_cache_root,
        output_root=args.output_root,
        explicit_output_dir=output_dir_root,
    )
)
assert_output_separate(args.attack_cache_root, protected_inputs)
assert_output_separate(output_dir_root, {"attack cache": args.attack_cache_root})

DATASET_DIR = os.path.join(args.dataset_root, args.dataset)
QA_FILE = os.path.join(DATASET_DIR, f"{args.dataset}_mapping.jsonl")

EPSILON = args.eps_255 / 255.0
ALPHA = args.alpha_255 / 255.0
STEPS = args.steps

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
)

if output_root is not None:
    output_dir_root = create_owned_output_directory(
        output_root, "detection/mahalanobis"
    )
else:
    output_dir_root.mkdir(parents=True, exist_ok=True)
    if not output_dir_root.is_dir():
        raise RuntimeError(f"Mahalanobis output is not a directory: {output_dir_root}")

if args.stats_path is None:
    stats_relative = (
        "stats/"
        f"maha_stats_{args.dataset}_fit{args.fit_start}_{args.fit_limit}_pca{args.pca_dim}.pkl"
    )
else:
    stats_argument = Path(args.stats_path).expanduser()
    if not stats_argument.is_absolute():
        stats_argument = (Path.cwd() / stats_argument).absolute()
    try:
        stats_relative = stats_argument.relative_to(output_dir_root).as_posix()
    except ValueError:
        parser.error("--stats_path must remain inside the trusted --output_dir tree")
stats_parent = str(Path(stats_relative).parent)
if stats_parent != ".":
    create_owned_output_directory(output_dir_root, stats_parent)
args.stats_path = str(resolve_owned_output_path(output_dir_root, stats_relative))
assert_output_separate(args.stats_path, protected_inputs)
assert_output_separate(args.stats_path, {"attack cache": args.attack_cache_root})
Path(args.stats_path).parent.mkdir(parents=True, exist_ok=True)


# ================= 3. 随机种子 =================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_sample_seed(base_seed: int, sample_idx: int) -> int:
    sample_seed = int(base_seed) * 1000003 + int(sample_idx)
    random.seed(sample_seed)
    np.random.seed(sample_seed % (2**32 - 1))
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)
    return sample_seed


set_global_seed(args.seed)


# ================= 4. 工具函数 =================

def expand2square(pil_img, background_color=(122, 116, 104)):
    width, height = pil_img.size
    if width == height:
        return pil_img

    if width > height:
        res = Image.new(pil_img.mode, (width, width), background_color)
        res.paste(pil_img, (0, (width - height) // 2))
        return res

    res = Image.new(pil_img.mode, (height, height), background_color)
    res.paste(pil_img, ((height - width) // 2, 0))
    return res


def build_prompt(gt_data: dict) -> str:
    if gt_data["type"] == "multiple_choice":
        return (
            f"{SYSTEM_PROMPT}USER: <image>\n{gt_data['question']}\n"
            "Answer with the option's letter from the given choices directly. ASSISTANT:"
        )

    return (
        f"{SYSTEM_PROMPT}USER: <image>\n{gt_data['question']}\n"
        "Answer the question using a single word or phrase. ASSISTANT:"
    )


def fs_random_noise(image: Image.Image, eps_255: float = 2.0, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    clean = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    noise = rng.uniform(-eps_255 / 255.0, eps_255 / 255.0, size=clean.shape)
    adversarial = np.clip(clean + noise, 0.0, 1.0)
    return quantize_linf_image(
        np.transpose(clean, (2, 0, 1)),
        np.transpose(adversarial, (2, 0, 1)),
        eps_255 / 255.0,
    )


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


# ================= 5. 读取数据 =================

print(f"[Info] Loading dataset: {QA_FILE}")

qa_database = []
with open(QA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        qa_database.append(json.loads(line))

print(f"[Info] Dataset size: {len(qa_database)}")

SOURCE_CACHE_CONTRACT = None
CACHE_NAMESPACE = None
if args.mode == "eval" and args.attack_for_detection in {"cage", "caa"}:
    cache_attack = attack_namespace(
        args.attack_for_detection,
        seed=args.seed,
        eps_255=args.eps_255,
        alpha_255=args.alpha_255,
        steps=args.steps,
    )
    source_cache_contract = build_attack_cache_contract(
        definition=baseline_attack_definition(args.attack_for_detection),
        dataset=args.dataset,
        mapping_path=QA_FILE,
        method=args.method,
        model_path=LLAVA_PATH,
        clip_path=None if args.attack_for_detection == "caa" else CLIP_PATH,
        seed=args.seed,
        eps_255=args.eps_255,
        alpha_255=args.alpha_255,
        steps=args.steps,
        extra=baseline_attack_contract_extra(
            args.attack_for_detection,
            max_input_tokens=0,
        ),
    )
    require_run_contract(
        attack_cache_contract_path(
            cache_root=args.attack_cache_root,
            method=args.method,
            dataset=args.dataset,
            attack=cache_attack,
        ),
        source_cache_contract,
    )
    SOURCE_CACHE_CONTRACT = source_cache_contract
    CACHE_NAMESPACE = cache_attack


# ================= 6. 加载 CLIP =================

print("[Info] Loading CLIP encoder")

clip_encoder = CLIPVisionModel.from_pretrained(
    CLIP_PATH,
    output_attentions=True,
).to(DEVICE)

clip_encoder.eval()

clip_processor = CLIPImageProcessor.from_pretrained(CLIP_PATH)

mean = torch.tensor(
    [0.48145466, 0.4578275, 0.40821073],
    device=DEVICE,
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.26862954, 0.26130258, 0.27577711],
    device=DEVICE,
).view(1, 3, 1, 1)


@torch.no_grad()
def extract_clip_features(image: Image.Image) -> dict:
    """
    Extract CLIP feature views for Mahalanobis detection.

    cls: CLS token feature
    mean_patch: mean pooled patch-token feature
    """
    image = image.convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    outputs = clip_encoder(pixel_values)

    hidden = outputs.last_hidden_state.float()  # [1, 257, D]
    cls = hidden[:, 0, :]                       # [1, D]
    mean_patch = hidden[:, 1:, :].mean(dim=1)   # [1, D]

    cls = F.normalize(cls, dim=-1)
    mean_patch = F.normalize(mean_patch, dim=-1)

    return {
        "cls": cls.squeeze(0).detach().cpu().numpy(),
        "mean_patch": mean_patch.squeeze(0).detach().cpu().numpy(),
    }


# ================= 7. 仅 eval base/fata 时加载 LLaVA =================

llava_processor = None
llava_model = None
vision_model = None

def maybe_load_llava_for_attack():
    global llava_processor, llava_model, vision_model

    if llava_model is not None:
        return

    print(f"[Info] Loading LLaVA for attack generation. Method = {args.method}")

    llava_processor = AutoProcessor.from_pretrained(LLAVA_PATH, use_fast=False)

    llava_model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_PATH,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )

    vision_model = (
        llava_model.vision_tower.vision_model
        if hasattr(llava_model, "vision_tower")
        else llava_model.model.vision_tower.vision_model
    )

    vision_model.config.output_attentions = True
    vision_model.config.output_hidden_states = True

    apply_compression_patch(vision_model, args.method)


def prepare_clean_features_and_mask(image: Image.Image, prompt: str):
    maybe_load_llava_for_attack()

    vision_model.current_K = 576

    raw_inputs_clean = llava_processor(text=prompt, images=image, return_tensors="pt")
    inputs_clean = {
        key: val.to(DEVICE)
        for key, val in raw_inputs_clean.items()
        if val is not None
    }

    inputs_clean["pixel_values"] = inputs_clean["pixel_values"].to(torch.float16)

    with torch.no_grad():
        vision_outputs = vision_model(
            inputs_clean["pixel_values"],
            output_attentions=True,
            return_dict=True,
        )

        clean_attn = vision_outputs.attentions[-1][:, :, 0, 1:].mean(dim=1).squeeze(0)

        clean_features = clip_encoder(
            inputs_clean["pixel_values"]
        ).last_hidden_state.squeeze(0)[1:]

    target_mask = (
        torch.argsort(torch.argsort(clean_attn, descending=True)) < 64
    ).float().detach()

    return clean_features, target_mask


def get_clean_image_01(image: Image.Image):
    inputs_clip = clip_processor(images=image, return_tensors="pt")
    clean_image_01 = (inputs_clip["pixel_values"].to(DEVICE) * std + mean).detach()
    return clean_image_01


def generate_base_image(clean_image_01: torch.Tensor, target_mask: torch.Tensor):
    delta_base = (
        torch.zeros_like(clean_image_01)
        .uniform_(-EPSILON, EPSILON)
        .to(DEVICE)
        .requires_grad_(True)
    )

    for _ in range(STEPS):
        adv_img = torch.clamp(clean_image_01 + delta_base, 0, 1)
        adv_outputs = clip_encoder((adv_img - mean) / std)

        s_adv = adv_outputs.attentions[-1][:, :, 0, 1:].mean(dim=1).squeeze(0)

        loss_base = torch.sum(s_adv * target_mask)

        clip_encoder.zero_grad(set_to_none=True)
        loss_base.backward()

        delta_base.data = delta_base.data - ALPHA * delta_base.grad.detach().sign()
        delta_base.data = torch.clamp(delta_base.data, -EPSILON, EPSILON)
        delta_base.grad.zero_()

    final_base = torch.clamp(clean_image_01 + delta_base, 0, 1)
    linf = (final_base - clean_image_01).abs().max().item()
    if linf > EPSILON + 1e-6:
        raise RuntimeError(f"Base L_inf violation: {linf:.8f} > {EPSILON:.8f}")
    img_base = quantize_linf_image(clean_image_01, final_base, EPSILON)

    return img_base


def generate_fata_image(
    clean_image_01: torch.Tensor,
    clean_features: torch.Tensor,
    target_mask: torch.Tensor,
):
    delta_fata = (
        torch.zeros_like(clean_image_01)
        .uniform_(-EPSILON, EPSILON)
        .to(DEVICE)
        .requires_grad_(True)
    )

    for _ in range(STEPS):
        adv_img = torch.clamp(clean_image_01 + delta_fata, 0, 1)
        adv_outputs = clip_encoder((adv_img - mean) / std)

        s_adv = adv_outputs.attentions[-1][:, :, 0, 1:].mean(dim=1).squeeze(0)
        adv_features = adv_outputs.last_hidden_state.squeeze(0)[1:]

        loss_attn = torch.sum(s_adv * target_mask)

        sim = F.cosine_similarity(
            adv_features,
            clean_features.detach(),
            dim=-1,
        )

        loss_sem = 1.0 - (
            torch.sum(sim * target_mask) / (target_mask.sum() + 1e-8)
        )

        loss_fata = loss_attn + args.lam * loss_sem

        clip_encoder.zero_grad(set_to_none=True)
        loss_fata.backward()

        delta_fata.data = delta_fata.data - ALPHA * delta_fata.grad.detach().sign()
        delta_fata.data = torch.clamp(delta_fata.data, -EPSILON, EPSILON)
        delta_fata.grad.zero_()

    final_fata = torch.clamp(clean_image_01 + delta_fata, 0, 1)
    linf = (final_fata - clean_image_01).abs().max().item()
    if linf > EPSILON + 1e-6:
        raise RuntimeError(f"FATA L_inf violation: {linf:.8f} > {EPSILON:.8f}")
    img_fata = quantize_linf_image(clean_image_01, final_fata, EPSILON)

    return img_fata


# ================= 8. Mahalanobis 拟合与打分 =================

def fit_one_view(features: np.ndarray, pca_dim: int):
    """
    features: [N, D], already normalized.
    """
    n, d = features.shape
    use_dim = min(pca_dim, n - 1, d)

    pca = PCA(n_components=use_dim, random_state=args.seed)
    z = pca.fit_transform(features)

    cov = LedoitWolf().fit(z)

    delta = z - cov.location_
    dist = np.sum((delta @ cov.precision_) * delta, axis=1)

    return {
        "pca": pca,
        "cov": cov,
        "train_dist_mean": float(dist.mean()),
        "train_dist_std": float(dist.std() + 1e-8),
        "pca_dim": use_dim,
    }


def score_one_view(feature: np.ndarray, stat: dict):
    x = feature.reshape(1, -1)
    z = stat["pca"].transform(x)

    cov = stat["cov"]
    delta = z - cov.location_

    dist = float(np.sum((delta @ cov.precision_) * delta, axis=1)[0])
    zscore = (dist - stat["train_dist_mean"]) / stat["train_dist_std"]

    return dist, zscore


def stats_contract() -> dict:
    return {
        "schema_version": 1,
        "detector": "mahalanobis_clip_cls_mean_patch",
        "dataset": args.dataset,
        "dataset_mapping_sha256": sha256_file(QA_FILE),
        "dataset_images": dataset_image_identity(QA_FILE, DATASET_DIR),
        "clip_model": artifact_identity(CLIP_PATH),
        "fit_start": args.fit_start,
        "fit_limit": args.fit_limit,
        "pca_dim_arg": args.pca_dim,
        "seed": args.seed,
    }


def fit_mahalanobis_stats():
    if os.path.exists(args.stats_path) and not args.overwrite:
        raise FileExistsError(
            f"Stats file exists: {args.stats_path}. Use --overwrite explicitly."
        )
    start = args.fit_start
    end = args.fit_start + args.fit_limit
    if end > len(qa_database):
        raise RuntimeError(
            f"fit slice [{start},{end}) exceeds mapping length {len(qa_database)}"
        )
    expected_count = args.fit_limit

    print(f"[Fit] Using clean samples [{start}, {end})")
    print(f"[Fit] stats_path = {args.stats_path}")

    cls_list = []
    mean_patch_list = []

    for idx in tqdm(range(start, end), desc="Fitting clean CLIP features"):
        gt_data = qa_database[idx]
        img_filename = gt_data["image_filename"]
        img_path = str(resolve_dataset_relative_path(DATASET_DIR, img_filename))

        if not os.path.exists(img_path):
            continue

        try:
            raw_image = Image.open(img_path).convert("RGB")
            image = expand2square(raw_image)
        except Exception as e:
            print(f"[Warning] Failed to load {img_path}: {e}")
            continue

        feat = extract_clip_features(image)
        cls_list.append(feat["cls"])
        mean_patch_list.append(feat["mean_patch"])

    if len(cls_list) != expected_count:
        raise RuntimeError(
            f"Mahalanobis fit incomplete: expected={expected_count}, extracted={len(cls_list)}"
        )

    cls_arr = l2_normalize_np(np.stack(cls_list, axis=0))
    mean_patch_arr = l2_normalize_np(np.stack(mean_patch_list, axis=0))

    stats = {
        "contract": stats_contract(),
        "dataset": args.dataset,
        "fit_start": args.fit_start,
        "fit_limit": args.fit_limit,
        "pca_dim_arg": args.pca_dim,
        "n_fit": int(cls_arr.shape[0]),
        "views": {
            "cls": fit_one_view(cls_arr, args.pca_dim),
            "mean_patch": fit_one_view(mean_patch_arr, args.pca_dim),
        },
    }

    stats_output = resolve_owned_output_path(output_dir_root, stats_relative)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{stats_output.name}.tmp.", dir=stats_output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as f:
            pickle.dump(stats, f)
            f.flush()
            os.fsync(f.fileno())
        if stats_output.is_symlink():
            raise RuntimeError(f"Mahalanobis stats became a symlink: {stats_output}")
        os.replace(temporary, stats_output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"[Fit] Saved stats to {args.stats_path}")
    print(f"[Fit] n_fit = {stats['n_fit']}")


def load_stats():
    if Path(args.stats_path).is_symlink():
        raise RuntimeError("refusing to load Mahalanobis statistics through a symlink")
    with open(args.stats_path, "rb") as f:
        stats = pickle.load(f)
    if stats.get("contract") != stats_contract():
        raise RuntimeError(
            "Mahalanobis stats contract does not match dataset, mapping, CLIP, fit slice, PCA, or seed"
        )
    return stats


def eval_mahalanobis():
    if not os.path.exists(args.stats_path):
        raise FileNotFoundError(f"Stats file not found: {args.stats_path}")

    stats = load_stats()
    stats_sha256 = sha256_file(args.stats_path)
    stats_digest = stats_sha256[:12]

    start = args.eval_start
    end = args.eval_start + args.eval_limit
    if end > len(qa_database):
        raise RuntimeError(
            f"evaluation slice [{start},{end}) exceeds mapping length {len(qa_database)}"
        )
    selected_rows = qa_database[start:end]
    expected_ids = expected_unique_ids(selected_rows, "image_filename")

    out_name = (
        f"maha_raw_"
        f"{args.method}_{args.dataset}_"
        f"{args.attack_for_detection}_"
        f"eval{args.eval_start}_{args.eval_limit}_"
        f"seed{args.seed}_eps{args.eps_255:g}_a{args.alpha_255:g}_"
        f"s{args.steps}_lam{args.lam:g}_stats{stats_digest}.csv"
    )
    out_path = resolve_owned_output_path(output_dir_root, out_name)
    meta_path = resolve_owned_output_path(
        output_dir_root, str(Path(out_name).with_suffix(".meta.json"))
    )
    assert_no_output_file_collision(
        {"Mahalanobis result": out_path, "Mahalanobis metadata": meta_path},
        {"Mahalanobis fitted statistics": args.stats_path},
    )

    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_path}. Use --overwrite.")

    label = 0 if args.attack_for_detection in ["clean", "random"] else 1

    header = [
        "Image_ID",
        "Question",
        "dataset",
        "method",
        "attack_for_detection",
        "label",
        "seed",
        "sample_seed",
        "eval_start",
        "eval_limit",
        "eps_255",
        "alpha_255",
        "steps",
        "lam",
        "stats_sha256",
        "maha_cls",
        "maha_cls_z",
        "maha_mean_patch",
        "maha_mean_patch_z",
        "maha_max_z",
        "maha_avg_z",
    ]

    rows = []
    ensure_run_contract(
        meta_path,
        {
            "schema_version": 1,
            "image_serialization": IMAGE_SERIALIZATION_CONTRACT,
            "runtime": "mahalanobis_eval_v1",
            "dataset": args.dataset,
            "dataset_mapping_sha256": sha256_file(QA_FILE),
            "dataset_images": dataset_image_identity(QA_FILE, DATASET_DIR),
            "expected_image_ids": expected_ids,
            "method": args.method,
            "attack_for_detection": args.attack_for_detection,
            "model": artifact_identity(LLAVA_PATH),
            "clip_model": artifact_identity(CLIP_PATH),
            "seed": args.seed,
            "eval_start": args.eval_start,
            "eval_limit": args.eval_limit,
            "eps_255": args.eps_255,
            "alpha_255": args.alpha_255,
            "steps": args.steps,
            "lambda": args.lam,
            "source_cache_contract": SOURCE_CACHE_CONTRACT,
            "cache_namespace": CACHE_NAMESPACE,
            "stats_sha256": stats_sha256,
            "header": header,
        },
        result_path=out_path,
    )

    print(f"[Eval] Attack = {args.attack_for_detection}")
    print(f"[Eval] Samples [{start}, {end})")
    print(f"[Eval] Output = {out_path}")

    for sample_idx in tqdm(range(start, end), desc=f"Mahalanobis {args.attack_for_detection}"):
        gt_data = qa_database[sample_idx]
        sample_seed = set_sample_seed(args.seed, sample_idx)

        img_filename = gt_data["image_filename"]
        img_path = str(resolve_dataset_relative_path(DATASET_DIR, img_filename))

        if not os.path.exists(img_path):
            print(f"[Warning] Missing image: {img_path}")
            continue

        try:
            raw_image = Image.open(img_path).convert("RGB")
            image = expand2square(raw_image)
        except Exception as e:
            print(f"[Warning] Failed to load {img_path}: {e}")
            continue

        if args.attack_for_detection == "clean":
            test_img = image

        elif args.attack_for_detection == "random":
            test_img = fs_random_noise(
                image,
                eps_255=args.eps_255,
                seed=sample_seed,
            )

        elif args.attack_for_detection in {"cage", "caa"}:

            test_img = load_attack_image(
                cache_root=args.attack_cache_root,
                method=args.method,
                dataset=args.dataset,
                attack=attack_namespace(
                    args.attack_for_detection,
                    seed=args.seed,
                    eps_255=args.eps_255,
                    alpha_255=args.alpha_255,
                    steps=args.steps,
                ),
                image_filename=img_filename,
            )

        elif args.attack_for_detection in ["base", "fata"]:
            prompt = build_prompt(gt_data)

            clean_features, target_mask = prepare_clean_features_and_mask(image, prompt)
            clean_image_01 = get_clean_image_01(image)

            if args.attack_for_detection == "base":
                test_img = generate_base_image(
                    clean_image_01=clean_image_01,
                    target_mask=target_mask,
                )
            else:
                test_img = generate_fata_image(
                    clean_image_01=clean_image_01,
                    clean_features=clean_features,
                    target_mask=target_mask,
                )
        else:
            raise ValueError(args.attack_for_detection)

        feat = extract_clip_features(test_img)

        cls_x = l2_normalize_np(feat["cls"].reshape(1, -1))[0]
        mean_x = l2_normalize_np(feat["mean_patch"].reshape(1, -1))[0]

        maha_cls, maha_cls_z = score_one_view(cls_x, stats["views"]["cls"])
        maha_mean, maha_mean_z = score_one_view(mean_x, stats["views"]["mean_patch"])

        maha_max_z = max(maha_cls_z, maha_mean_z)
        maha_avg_z = 0.5 * (maha_cls_z + maha_mean_z)

        row = [
            img_filename,
            f'"{gt_data["question"]}"',
            args.dataset,
            args.method,
            args.attack_for_detection,
            label,
            args.seed,
            sample_seed,
            args.eval_start,
            args.eval_limit,
            args.eps_255,
            args.alpha_255,
            args.steps,
            args.lam,
            stats_sha256,
            f"{maha_cls:.6f}",
            f"{maha_cls_z:.6f}",
            f"{maha_mean:.6f}",
            f"{maha_mean_z:.6f}",
            f"{maha_max_z:.6f}",
            f"{maha_avg_z:.6f}",
        ]

        rows.append(row)

        if torch.cuda.is_available() and (sample_idx + 1) % 20 == 0:
            torch.cuda.empty_cache()

    expected_count = end - start
    if len(rows) != expected_count:
        raise RuntimeError(
            f"Mahalanobis eval incomplete: expected={expected_count}, scored={len(rows)}. "
            "No partial result was written."
        )
    assert_exact_completion(
        expected_ids,
        [str(row[0]) for row in rows],
        "Mahalanobis eval rows",
    )
    out_path = resolve_owned_output_path(output_dir_root, out_name)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{Path(out_path).name}.tmp.", dir=Path(out_path).parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        if Path(out_path).is_symlink():
            raise RuntimeError(f"Mahalanobis result became a symlink: {out_path}")
        os.replace(temporary, out_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"[Eval] Done. Saved to {out_path}")


# ================= 9. Main =================

if __name__ == "__main__":
    if args.mode == "fit":
        fit_mahalanobis_stats()
    elif args.mode == "eval":
        eval_mahalanobis()
    else:
        raise ValueError(args.mode)
