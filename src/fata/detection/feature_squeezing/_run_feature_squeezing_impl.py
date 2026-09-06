# fata_detection_defense/feature_squeezing/run_feature_squeezing_detection.py

from __future__ import annotations

import os
import io
import csv
import json
import re
import argparse
import random
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from tqdm import tqdm

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
    assert_output_separate,
    create_owned_output_directory,
    resolve_attack_cache_root,
    resolve_dataset_relative_path,
    resolve_owned_output_path,
)
from .feature_squeezing_utils import detection_label
from fata.utils.run_contract import (
    artifact_identity,
    assert_exact_completion,
    dataset_image_identity,
    ensure_run_contract,
    expected_unique_ids,
    require_run_contract,
    sha256_file,
)


# ================= 1. 参数解析 =================

parser = argparse.ArgumentParser(description="Feature Squeezing Detection for FATA")

parser.add_argument(
    "--method",
    type=str,
    required=True,
    choices=["VisionZIP", "VisPruner", "PruMerge", "FlowCut"],
)

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["TextVQA_Open", "ScienceQA_MC", "VQAv2_Open", "VQAv2_MC"],
)

parser.add_argument("--lam", type=float, default=1.0, help="FATA lambda")
parser.add_argument("--limit", type=int, default=100, help="测试样本数；0 表示全量")
parser.add_argument("--seed", type=int, default=0, help="随机种子")

parser.add_argument(
    "--attack_for_detection",
    type=str,
    required=True,
    choices=["clean", "random", "base", "fata","cage","caa"],
    help="要检测的图像类型：clean/random 是负类，base/fata 是正类",
)

parser.add_argument(
    "--detect_k",
    type=int,
    default=576,
    choices=[576, 192, 128, 64, 32, 16],
    help="Feature Squeezing 检测时使用的 token budget",
)

parser.add_argument("--eps_255", type=float, default=2.0, help="扰动预算 epsilon，以 /255 为单位")
parser.add_argument("--alpha_255", type=float, default=None, help="PGD 步长 alpha，以 /255 为单位；CAA 默认 1，其他攻击默认 0.5")
parser.add_argument("--steps", type=int, default=100, help="PGD steps")

parser.add_argument(
    "--fs_output_dir",
    type=str,
    default=None,
    help="Feature Squeezing 检测结果输出目录",
)

parser.add_argument(
    "--overwrite",
    action="store_true",
    help="若输出文件已存在，是否覆盖。默认不覆盖，防止误删结果。",
)
parser.add_argument("--model-path", default=os.environ.get("FATA_LLAVA_MODEL"))
parser.add_argument("--clip-path", default=os.environ.get("FATA_CLIP_MODEL"))
parser.add_argument("--dataset-root", default=os.environ.get("FATA_DATA_ROOT"))
parser.add_argument("--output-root", default=os.environ.get("FATA_OUTPUT_ROOT"))
parser.add_argument("--attack-cache-root", default=None)

args = parser.parse_args()
if args.alpha_255 is None:
    args.alpha_255 = 1.0 if args.attack_for_detection == "caa" else 0.5
if (
    args.limit < 0
    or args.seed < 0
    or args.steps <= 0
    or not all(math.isfinite(value) for value in (args.eps_255, args.alpha_255, args.lam))
    or args.eps_255 <= 0
    or args.alpha_255 <= 0
    or args.lam < 0
):
    parser.error("limit/seed/lambda and attack parameter values are invalid")


# ================= 2. 全局配置 =================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not args.model_path or not args.clip_path or not args.dataset_root or not (args.fs_output_dir or args.output_root):
    parser.error("provide model, CLIP, dataset, and output roots")
LLAVA_PATH = args.model_path
CLIP_PATH = args.clip_path
protected_inputs = {
    "dataset_root": args.dataset_root,
    "LLaVA model": LLAVA_PATH,
    "CLIP model": CLIP_PATH,
}
if args.fs_output_dir is None:
    output_root = assert_output_separate(args.output_root, protected_inputs)
    fs_output_root = resolve_owned_output_path(
        output_root, "detection/feature_squeezing"
    )
else:
    output_root = None
    fs_output_root = assert_output_separate(args.fs_output_dir, protected_inputs)
args.fs_output_dir = str(fs_output_root)
args.attack_cache_root = str(
    resolve_attack_cache_root(
        explicit_cache_root=args.attack_cache_root,
        output_root=args.output_root,
        explicit_output_dir=fs_output_root,
    )
)
assert_output_separate(args.attack_cache_root, protected_inputs)
assert_output_separate(fs_output_root, {"attack cache": args.attack_cache_root})

DATASET_DIR = os.path.join(args.dataset_root, args.dataset)
QA_FILE = os.path.join(DATASET_DIR, f"{args.dataset}_mapping.jsonl")

BUDGETS = [576, 192, 128, 64, 32, 16]

EPSILON = args.eps_255 / 255.0
ALPHA = args.alpha_255 / 255.0
STEPS = args.steps

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
)

if output_root is not None:
    fs_output_root = create_owned_output_directory(
        output_root, "detection/feature_squeezing"
    )
else:
    fs_output_root.mkdir(parents=True, exist_ok=True)
    if not fs_output_root.is_dir():
        raise RuntimeError(f"Feature Squeezing output is not a directory: {fs_output_root}")

RESULTS_NAME = (
    f"fs_detection_raw_"
    f"{args.method}_{args.dataset}_"
    f"K{args.detect_k}_"
    f"{args.attack_for_detection}_"
    f"eps{args.eps_255:g}_a{args.alpha_255:g}_s{args.steps}_lam{args.lam:g}_"
    f"seed{args.seed}_"
    f"limit{args.limit}.csv"
)
RESULTS_FILE = resolve_owned_output_path(fs_output_root, RESULTS_NAME)


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


# ================= 4. 基础工具函数 =================

def expand2square(pil_img, background_color=(122, 116, 104)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        res = Image.new(pil_img.mode, (width, width), background_color)
        res.paste(pil_img, (0, (width - height) // 2))
        return res
    else:
        res = Image.new(pil_img.mode, (height, height), background_color)
        res.paste(pil_img, ((height - width) // 2, 0))
        return res


def compute_accuracy(pred_ans, gt_data):
    pred_ans = str(pred_ans).strip().lower()
    q_type = gt_data.get("type", "open")

    if q_type == "multiple_choice":
        gt_letter = gt_data["answers"][0].lower()
        gt_text = gt_data.get("ground_truth_text", "").lower()

        match = re.search(
            r"(?i)(?:^|\s|\()(option\s+)?([a-f])(?:\)|\.|:|\s|$)",
            pred_ans,
        )

        if match:
            extracted = match.group(2).lower()
        else:
            if gt_text and gt_text in pred_ans:
                return 1.0
            extracted = pred_ans[0] if len(pred_ans) > 0 else ""

        return 1.0 if extracted == gt_letter else 0.0

    else:
        def official_vqa_process(text):
            text = str(text).lower().replace("\n", " ").replace("\r", " ")
            text = re.sub(r"([^\w\s])", r" ", text)
            words = [w for w in text.split() if w not in ["a", "an", "the"]]
            num_map = {
                "zero": "0",
                "one": "1",
                "two": "2",
                "three": "3",
                "four": "4",
                "five": "5",
            }
            return " ".join([num_map.get(w, w) for w in words])

        pred_clean = official_vqa_process(pred_ans)
        gts = [official_vqa_process(gt) for gt in gt_data["answers"]]

        if not gts:
            return 0.0

        match_count = gts.count(pred_clean)

        if match_count == 0:
            for gt in gts:
                if gt and gt in pred_clean:
                    match_count = 3
                    break

        return min(1.0, float(match_count) / 3.0)


def normalize_answer(ans: str) -> str:
    return str(ans).strip().lower()


def answer_changed(ans1: str, ans2: str) -> int:
    return int(normalize_answer(ans1) != normalize_answer(ans2))


# ================= 5. Feature Squeezing 变换 =================

def fs_reduce_bit_depth(image: Image.Image, bits: int = 5) -> Image.Image:
    image = image.convert("RGB")
    arr = np.asarray(image).astype(np.uint8)

    if bits == 8:
        return image

    levels = 2 ** bits
    arr = np.floor(arr.astype(np.float32) / 256.0 * levels)
    arr = np.clip(arr, 0, levels - 1)
    arr = np.round(arr * (255.0 / (levels - 1)))
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")


def fs_median_filter(image: Image.Image, size: int = 3) -> Image.Image:
    return image.convert("RGB").filter(ImageFilter.MedianFilter(size=size))


def fs_jpeg_compress(image: Image.Image, quality: int = 75) -> Image.Image:
    image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def fs_get_squeezers():
    return {
        "bit5": lambda img: fs_reduce_bit_depth(img, bits=5),
        "bit4": lambda img: fs_reduce_bit_depth(img, bits=4),
        "median3": lambda img: fs_median_filter(img, size=3),
        "jpeg75": lambda img: fs_jpeg_compress(img, quality=75),
        "jpeg50": lambda img: fs_jpeg_compress(img, quality=50),
    }


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


# ================= 6. 加载数据 =================

print(f"[Info] Loading dataset: {QA_FILE}")

qa_database = []
with open(QA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        qa_database.append(json.loads(line))

if args.limit > 0:
    if args.limit > len(qa_database):
        parser.error(
            f"--limit {args.limit} exceeds mapping length {len(qa_database)}"
        )
    qa_database = qa_database[: args.limit]
    print(f"[Info] Limit enabled: using first {len(qa_database)} samples")
else:
    print(f"[Info] Full dataset mode: {len(qa_database)} samples")

expected_ids = expected_unique_ids(qa_database, "image_filename")
if not expected_ids:
    parser.error("selected dataset slice is empty")

SOURCE_CACHE_CONTRACT = None
CACHE_NAMESPACE = None
if args.attack_for_detection in {"cage", "caa"}:
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


# ================= 7. 加载模型 =================

print(f"[Info] Loading LLaVA and CLIP. Method = {args.method}")

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


# ================= 8. 推理函数 =================

def evaluate_image_at_k(image: Image.Image, prompt: str, gt_data: dict, k: int):
    raw_inputs = llava_processor(text=prompt, images=image, return_tensors="pt")
    inputs = {key: val.to(DEVICE) for key, val in raw_inputs.items() if val is not None}

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    input_len = inputs["input_ids"].shape[1]

    vision_model.current_K = k

    with torch.no_grad():
        out = llava_model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            num_beams=1,
        )

    ans = llava_processor.decode(
        out[0][input_len:],
        skip_special_tokens=True,
    ).strip()

    score = float(compute_accuracy(ans, gt_data))

    return ans, score


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


# ================= 9. 攻击生成函数 =================

def prepare_clean_features_and_mask(image: Image.Image, prompt: str):
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


def generate_base_image(
    clean_image_01: torch.Tensor,
    target_mask: torch.Tensor,
):
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


# ================= 10. 初始化 CSV =================

if os.path.exists(RESULTS_FILE) and not args.overwrite:
    raise FileExistsError(
        f"Output file already exists: {RESULTS_FILE}\n"
        "Use --overwrite if you want to replace it."
    )

squeezers = fs_get_squeezers()

header = [
    "Image_ID",
    "Question",
    "dataset",
    "method",
    "detect_k",
    "attack_for_detection",
    "label",
    "seed",
    "sample_seed",
    "limit",
    "eps_255",
    "alpha_255",
    "steps",
    "lam",
    "orig_answer",
    "orig_score",
    "max_fs_score_diff",
    "max_answer_changed",
]

for sq_name in squeezers.keys():
    header += [
        f"{sq_name}_answer",
        f"{sq_name}_score",
        f"{sq_name}_score_diff",
        f"{sq_name}_answer_changed",
    ]

META_FILE = resolve_owned_output_path(
    fs_output_root, str(Path(RESULTS_NAME).with_suffix(".meta.json"))
)
ensure_run_contract(
    META_FILE,
    {
        "schema_version": 1,
        "image_serialization": IMAGE_SERIALIZATION_CONTRACT,
        "runtime": "feature_squeezing_detection_v1",
        "dataset": args.dataset,
        "dataset_mapping_sha256": sha256_file(QA_FILE),
        "dataset_images": dataset_image_identity(QA_FILE, DATASET_DIR),
        "expected_image_ids": expected_ids,
        "method": args.method,
        "detect_k": args.detect_k,
        "attack_for_detection": args.attack_for_detection,
        "model": artifact_identity(LLAVA_PATH),
        "clip_model": artifact_identity(CLIP_PATH),
        "seed": args.seed,
        "limit": args.limit,
        "selected_count": len(expected_ids),
        "eps_255": args.eps_255,
        "alpha_255": args.alpha_255,
        "steps": args.steps,
        "lambda": args.lam,
        "source_cache_contract": SOURCE_CACHE_CONTRACT,
        "cache_namespace": CACHE_NAMESPACE,
        "squeezers": list(squeezers),
        "header": header,
    },
    result_path=RESULTS_FILE,
)


# ================= 11. 主循环 =================

print(f"[Info] Output file: {RESULTS_FILE}")
print(f"[Info] attack_for_detection = {args.attack_for_detection}")
print(f"[Info] detect_k = {args.detect_k}")
print(f"[Info] eps = {args.eps_255}/255, alpha = {args.alpha_255}/255, steps = {args.steps}")

result_rows = []
for sample_idx, gt_data in enumerate(
    tqdm(qa_database, desc=f"FeatureSqueezing {args.method} {args.dataset}")
):
    sample_seed = set_sample_seed(args.seed, sample_idx)

    img_filename = gt_data["image_filename"]
    img_path = str(resolve_dataset_relative_path(DATASET_DIR, img_filename))

    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)

    try:
        raw_image = Image.open(img_path).convert("RGB")
        image = expand2square(raw_image)
    except Exception as e:
        raise RuntimeError(f"failed to decode image {img_path}: {e}") from e

    prompt = build_prompt(gt_data)
    label = detection_label(args.attack_for_detection)

    # ---------- 选择要检测的图像 ----------
    if args.attack_for_detection == "clean":
        test_img = image

    elif args.attack_for_detection == "random":
        test_img = fs_random_noise(
            image,
            eps_255=args.eps_255,
            seed=sample_seed,
        )

    elif args.attack_for_detection in ["base", "fata"]:
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

    else:
        raise ValueError(f"Unknown attack_for_detection: {args.attack_for_detection}")

    # ---------- 原始图像推理 ----------
    try:
        orig_answer, orig_score = evaluate_image_at_k(
            test_img,
            prompt,
            gt_data,
            args.detect_k,
        )
    except Exception as e:
        raise RuntimeError(
            f"original inference failed on {img_filename}: {e}"
        ) from e

    row = [
        img_filename,
        f'"{gt_data["question"]}"',
        args.dataset,
        args.method,
        args.detect_k,
        args.attack_for_detection,
        label,
        args.seed,
        sample_seed,
        args.limit,
        args.eps_255,
        args.alpha_255,
        args.steps,
        args.lam,
        orig_answer,
        f"{orig_score:.4f}",
    ]

    max_fs_score_diff = 0.0
    max_answer_changed = 0
    sq_values = []

    # ---------- Squeezing 后推理 ----------
    for sq_name, sq_fn in squeezers.items():
        try:
            sq_img = sq_fn(test_img)

            sq_answer, sq_score = evaluate_image_at_k(
                sq_img,
                prompt,
                gt_data,
                args.detect_k,
            )

            score_diff = abs(float(orig_score) - float(sq_score))
            ans_changed = answer_changed(orig_answer, sq_answer)

        except Exception as e:
            raise RuntimeError(
                f"squeezer {sq_name} failed on {img_filename}: {e}"
            ) from e

        if score_diff >= 0:
            max_fs_score_diff = max(max_fs_score_diff, score_diff)

        if ans_changed >= 0:
            max_answer_changed = max(max_answer_changed, ans_changed)

        sq_values += [
            sq_answer,
            f"{sq_score:.4f}",
            f"{score_diff:.4f}",
            ans_changed,
        ]

    row += [
        f"{max_fs_score_diff:.4f}",
        max_answer_changed,
    ]

    row += sq_values

    if len(row) != len(header):
        raise RuntimeError(
            f"Feature Squeezing row/header mismatch: {len(row)} != {len(header)}"
        )

    result_rows.append(row)

    # 稍微释放显存碎片
    if torch.cuda.is_available() and (sample_idx + 1) % 20 == 0:
        torch.cuda.empty_cache()

completed_ids = [str(row[0]) for row in result_rows]
assert_exact_completion(expected_ids, completed_ids, "Feature Squeezing rows")
RESULTS_FILE = resolve_owned_output_path(fs_output_root, RESULTS_NAME)
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{Path(RESULTS_FILE).name}.tmp.", dir=Path(RESULTS_FILE).parent
)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(result_rows)
        f.flush()
        os.fsync(f.fileno())
    if Path(RESULTS_FILE).is_symlink():
        raise RuntimeError(f"Feature Squeezing result became a symlink: {RESULTS_FILE}")
    os.replace(temporary, RESULTS_FILE)
finally:
    if temporary.exists():
        temporary.unlink()

print(f"[Done] Feature Squeezing detection finished.")
print(f"[Done] Saved to: {RESULTS_FILE}")
