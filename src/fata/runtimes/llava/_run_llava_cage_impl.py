import os
import torch
import torch.nn.functional as F
from PIL import Image
import csv
import json
import re
import argparse
import hashlib
import random
import numpy as np
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import AutoProcessor, LlavaForConditionalGeneration
from fata.attacks.linf_image import IMAGE_SERIALIZATION_CONTRACT, quantize_linf_image
from .compression_zoo import apply_compression_patch
from .result_schema import (
    answer_score_cells,
    read_completed_result_ids,
    result_header,
    result_schema,
)
from .attack_cache_io import (
    attack_cache_contract_path,
    attack_image_exists,
    attack_namespace,
    baseline_attack_definition,
    baseline_attack_contract_extra,
    build_attack_cache_contract,
    save_attack_image,
)
from .dataset_paths import add_dataset_root_argument, resolve_dataset_paths
from fata.utils.paths import (
    assert_output_separate,
    create_owned_output_directory,
    metadata_sidecar_path,
    resolve_dataset_relative_path,
    resolve_owned_output_path,
)
from fata.utils.run_contract import (
    artifact_identity,
    assert_exact_completion,
    dataset_image_identity,
    ensure_cache_contract,
    ensure_run_contract,
    expected_unique_ids,
    sha256_file,
)

# ================= 1. 参数解析 =================
parser = argparse.ArgumentParser(description="CAGE Attack Benchmark")
parser.add_argument("--method", type=str, required=True, choices=["VisionZIP", "VisPruner", "PruMerge", "FlowCut"])
parser.add_argument("--dataset", type=str, required=True, choices=["TextVQA_Open", "ScienceQA_MC", "VQAv2_Open", "VQAv2_MC"])
parser.add_argument("--limit", type=int, default=0, help="限制测试的样本数量；0 表示完整 mapping")
parser.add_argument("--seed", type=int, default=0, help="deterministic per-sample random-start seed")
parser.add_argument(
    "--attack-cache-root",
    type=str,
    default=None,
)

parser.add_argument(
    "--cache-only",
    action="store_true",
    help="Only generate and cache adversarial images; skip VQA evaluation.",
)
parser.add_argument("--model-path", default=os.environ.get("FATA_LLAVA_MODEL"))
parser.add_argument("--clip-path", default=os.environ.get("FATA_CLIP_MODEL"))
parser.add_argument("--output-root", default=os.environ.get("FATA_OUTPUT_ROOT"))
add_dataset_root_argument(parser)
args = parser.parse_args()
if args.limit < 0 or args.seed < 0:
    parser.error("--limit and --seed must be non-negative")

# ================= 2. 全局配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not args.model_path or not args.clip_path or not args.output_root:
    parser.error("pass --model-path, --clip-path and --output-root (or set their FATA_* environment variables)")
LLAVA_PATH = args.model_path
CLIP_PATH = args.clip_path

DATASET_DIR, QA_FILE = resolve_dataset_paths(args.dataset_root, args.dataset)
protected_inputs = {
    "dataset_root": args.dataset_root,
    "LLaVA model": LLAVA_PATH,
    "CLIP model": CLIP_PATH,
}
OUTPUT_ROOT = assert_output_separate(args.output_root, protected_inputs)
if args.attack_cache_root is None:
    args.attack_cache_root = resolve_owned_output_path(
        OUTPUT_ROOT, "adversarial_images"
    )
else:
    args.attack_cache_root = os.path.abspath(os.path.expanduser(args.attack_cache_root))
assert_output_separate(args.attack_cache_root, protected_inputs)

RESULTS_DIR = create_owned_output_directory(OUTPUT_ROOT, "llava/cage")
assert_output_separate(RESULTS_DIR, {"attack cache": args.attack_cache_root})
# 命名为 cage_pilot，避免与 FATA 的数据冲突
RESULTS_RELATIVE = (
    "llava/cage/"
    f"cage_eps2_a0.5_s100_seed{args.seed}_{args.method}_{args.dataset}.csv"
)
RESULTS_FILE = resolve_owned_output_path(OUTPUT_ROOT, RESULTS_RELATIVE)
META_RELATIVE = str(metadata_sidecar_path(RESULTS_RELATIVE))
META_FILE = str(resolve_owned_output_path(OUTPUT_ROOT, META_RELATIVE))

SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
BUDGETS = [576, 192, 128, 64, 32, 16]

# CAGE 官方超参数
EPSILON = 2 / 255.0  
ALPHA = 0.5 / 255.0     
STEPS = 100            
CAGE_CACHE_EXTRA = baseline_attack_contract_extra("cage", max_input_tokens=0)
CAGE_OBJECTIVE = CAGE_CACHE_EXTRA["objective"]
LAMBDA_CAGE = CAGE_OBJECTIVE["lambda_cage"]
K_MIN = CAGE_OBJECTIVE["k_min"]
K_MAX = CAGE_OBJECTIVE["k_max"]
SCORE_COLUMNS = ["Lang_Prior_K0"] + [
    f"{mode}_K{k}" for mode in ("Clean", "Base", "CAGE") for k in BUDGETS
]
HEADER = result_header(["Image_ID", "Question"], SCORE_COLUMNS)
RESULT_SCHEMA = result_schema(SCORE_COLUMNS)
CACHE_ATTACK = attack_namespace(
    "cage", seed=args.seed, eps_255=EPSILON * 255.0,
    alpha_255=ALPHA * 255.0, steps=STEPS,
)

# ================= 3. 核心工具函数 =================
def expand2square(pil_img, background_color=(122, 116, 104)):
    width, height = pil_img.size
    if width == height: return pil_img
    if width > height:
        res = Image.new(pil_img.mode, (width, width), background_color)
        res.paste(pil_img, (0, (width - height) // 2))
        return res
    else:
        res = Image.new(pil_img.mode, (height, height), background_color)
        res.paste(pil_img, ((height - width) // 2, 0))
        return res

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sample_generator(sample_id, stream):
    digest = hashlib.sha256(
        f"{args.seed}|{sample_id}|{stream}".encode("utf-8")
    ).digest()
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(int.from_bytes(digest[:8], "big") & 0x7FFFFFFF)
    return generator


set_global_seed(args.seed)

# ================= 4. 读取数据与恢复状态 =================
qa_database = []
with open(QA_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        qa_database.append(json.loads(line))

if args.limit > 0:
    if args.limit > len(qa_database):
        parser.error(
            f"--limit {args.limit} exceeds mapping length {len(qa_database)}"
        )
    qa_database = qa_database[:args.limit]
    print(f"⚡ 快速验证模式已开启：本次仅测试前 {args.limit} 个样本！")

expected_ids = expected_unique_ids(qa_database, "image_filename")
if not expected_ids:
    parser.error("selected dataset slice is empty")
EXPECTED_ROW_VALUES = {
    row["image_filename"]: {"Question": f'"{row.get("question", "")}"'}
    for row in qa_database
}
CACHE_CONTRACT = build_attack_cache_contract(
    definition=baseline_attack_definition("cage"),
    dataset=args.dataset,
    mapping_path=QA_FILE,
    method=args.method,
    model_path=LLAVA_PATH,
    clip_path=CLIP_PATH,
    seed=args.seed,
    eps_255=EPSILON * 255.0,
    alpha_255=ALPHA * 255.0,
    steps=STEPS,
    extra=CAGE_CACHE_EXTRA,
)
ensure_cache_contract(
    attack_cache_contract_path(
        cache_root=args.attack_cache_root,
        method=args.method,
        dataset=args.dataset,
        attack=CACHE_ATTACK,
    ),
    CACHE_CONTRACT,
)
ensure_run_contract(
    META_FILE,
    {
        "schema_version": 1,
        "image_serialization": IMAGE_SERIALIZATION_CONTRACT,
        "runtime": "llava_cage",
        "dataset": args.dataset,
        "dataset_mapping_sha256": sha256_file(QA_FILE),
        "dataset_images": dataset_image_identity(QA_FILE, DATASET_DIR),
        "method": args.method,
        "model": artifact_identity(LLAVA_PATH),
        "clip_model": artifact_identity(CLIP_PATH),
        "seed": args.seed,
        "epsilon_over_255": EPSILON * 255.0,
        "alpha_over_255": ALPHA * 255.0,
        "steps": STEPS,
        "lambda_cage": LAMBDA_CAGE,
        "k_min": K_MIN,
        "k_max": K_MAX,
        "budgets": BUDGETS,
        "header": HEADER,
        "result_schema": RESULT_SCHEMA,
        "cache_namespace": CACHE_ATTACK,
    },
    result_path=RESULTS_FILE,
)

processed_ids = read_completed_result_ids(
    RESULTS_FILE,
    prefix_columns=["Image_ID", "Question"],
    score_columns=SCORE_COLUMNS,
    ground_truth_rows=qa_database,
    expected_values=EXPECTED_ROW_VALUES,
)
missing_resume_caches = [
    image_id for image_id in expected_ids
    if not attack_image_exists(
        cache_root=args.attack_cache_root,
        method=args.method,
        dataset=args.dataset,
        attack=CACHE_ATTACK,
        image_filename=image_id,
    )
]
if args.cache_only and not missing_resume_caches:
    print(
        f"✅ 精确恢复：{len(expected_ids)} 个 CAGE 缓存已完整，"
        "跳过模型加载。"
    )
    raise SystemExit(0)
if not args.cache_only and processed_ids == set(expected_ids):
    if missing_resume_caches:
        raise RuntimeError(
            "completed LLaVA CAGE result has incomplete attack caches: "
            f"missing_or_invalid={len(missing_resume_caches)} "
            f"examples={missing_resume_caches[:5]}"
        )
    print(
        f"✅ 精确恢复：{len(processed_ids)} 个结果行与全部 CAGE 缓存已完整，"
        "跳过模型加载。"
    )
    raise SystemExit(0)
if not args.cache_only and not os.path.exists(RESULTS_FILE):
    with open(
        resolve_owned_output_path(OUTPUT_ROOT, RESULTS_RELATIVE),
        mode='w', newline='', encoding='utf-8'
    ) as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

# ================= 5. 加载模型 =================
print(f"🔄 正在加载模型与算法 [{args.method}] (CAGE 复现模式)...")
llava_processor = AutoProcessor.from_pretrained(LLAVA_PATH, use_fast=False)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    LLAVA_PATH, torch_dtype=torch.float16, device_map="cuda", attn_implementation="eager"
)
vision_model = llava_model.vision_tower.vision_model if hasattr(llava_model, 'vision_tower') else llava_model.model.vision_tower.vision_model
vision_model.config.output_attentions = True
vision_model.config.output_hidden_states = True 
apply_compression_patch(vision_model, args.method)

clip_encoder = CLIPVisionModel.from_pretrained(CLIP_PATH, output_attentions=True).to(DEVICE)
clip_encoder.eval()
clip_processor = CLIPImageProcessor.from_pretrained(CLIP_PATH)

mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(DEVICE)
std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(DEVICE)

def evaluate_image_across_k(image, prompt, score_prefix):
    raw_inputs = llava_processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in raw_inputs.items() if v is not None}
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        
    input_len = inputs['input_ids'].shape[1]
    
    answers = {}
    for k in BUDGETS:
        vision_model.current_K = k
        with torch.no_grad():
            out = llava_model.generate(**inputs, max_new_tokens=32, do_sample=False, num_beams=1)
        ans = llava_processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
        answers[f"{score_prefix}_K{k}"] = ans
    return answers

# ================= 6. 核心生成与攻击循环 =================
for gt_data in tqdm(qa_database, desc=f"Evaluating CAGE on {args.dataset}"):
    img_filename = gt_data["image_filename"]
    cache_complete = attack_image_exists(
        cache_root=args.attack_cache_root,
        method=args.method,
        dataset=args.dataset,
        attack=CACHE_ATTACK,
        image_filename=img_filename,
    )
    if cache_complete and (args.cache_only or img_filename in processed_ids):
        continue
    img_path = str(resolve_dataset_relative_path(DATASET_DIR, img_filename))
    if not os.path.exists(img_path): continue
    try: 
        raw_image = Image.open(img_path).convert("RGB")
        image = expand2square(raw_image)
    except: continue
    
    prompt = f"{SYSTEM_PROMPT}USER: <image>\n{gt_data['question']}\nAnswer with the option's letter from the given choices directly. ASSISTANT:" if gt_data["type"] == "multiple_choice" else f"{SYSTEM_PROMPT}USER: <image>\n{gt_data['question']}\nAnswer the question using a single word or phrase. ASSISTANT:"
    
    # 1. K=0 语言先验
    prompt_k0 = prompt.replace("<image>\n", "")
    raw_inputs_k0 = llava_processor(text=prompt_k0, return_tensors="pt")
    inputs_k0 = {k: v.to(DEVICE) for k, v in raw_inputs_k0.items() if v is not None}
    len_k0 = inputs_k0['input_ids'].shape[1]
    with torch.no_grad():
        out_k0 = llava_model.generate(**inputs_k0, max_new_tokens=32, do_sample=False, num_beams=1)
    ans_k0 = llava_processor.decode(out_k0[0][len_k0:], skip_special_tokens=True).strip()

    # 2. 获取干净特征 (CAGE 需要的基准特征)
    inputs_clip = clip_processor(images=image, return_tensors="pt")
    clean_image_01 = (inputs_clip['pixel_values'].to(DEVICE) * std + mean).detach()
    with torch.no_grad():
        clean_outputs = clip_encoder((clean_image_01 - mean) / std)
        clean_features = clean_outputs.last_hidden_state.squeeze(0)[1:]

    # === CAGE 的 Baseline (VEAttack): 全局特征破坏 ===
    delta_base = torch.empty_like(clean_image_01).uniform_(
        -EPSILON, EPSILON, generator=make_sample_generator(img_filename, "base")
    ).requires_grad_(True)
    for step in range(STEPS):
        adv_img = torch.clamp(clean_image_01 + delta_base, 0, 1)
        adv_outputs = clip_encoder((adv_img - mean) / std)
        z_adv = adv_outputs.last_hidden_state.squeeze(0)[1:]
        
        # 论文 Eq. 4: 直接最大化余弦距离，不带任何掩码
        loss_base = (1.0 - F.cosine_similarity(z_adv, clean_features.detach(), dim=-1)).mean()
        
        clip_encoder.zero_grad()
        loss_base.backward() 
        delta_base.data = delta_base.data + ALPHA * delta_base.grad.detach().sign()
        delta_base.data = torch.clamp(delta_base.data, -EPSILON, EPSILON)
        delta_base.grad.zero_()
    final_base = torch.clamp(clean_image_01 + delta_base, 0, 1).detach()
    base_linf = (final_base - clean_image_01).abs().max().item()
    if base_linf > EPSILON + 1e-6:
        raise RuntimeError(f"CAGE baseline L_inf violation: {base_linf:.8f} > {EPSILON:.8f}")
    img_base = quantize_linf_image(clean_image_01, final_base, EPSILON)

    # === CAGE Attack (算法严格复现) ===
    delta_cage = torch.empty_like(clean_image_01).uniform_(
        -EPSILON, EPSILON, generator=make_sample_generator(img_filename, "cage")
    ).requires_grad_(True)
    denominator = float(K_MAX - K_MIN + 1)
    
    for step in range(STEPS):
        adv_img = torch.clamp(clean_image_01 + delta_cage, 0, 1)
        adv_outputs = clip_encoder((adv_img - mean) / std)
        
        # 获取最新的 Adversarial 注意力分数 s 和 特征 z_adv
        s_adv = adv_outputs.attentions[-1][:, :, 0, 1:].mean(dim=1).squeeze(0) 
        z_adv = adv_outputs.last_hidden_state.squeeze(0)[1:]
        
        # 1. 距离度量 d_i (Eq. 7)
        d_i = 1.0 - F.cosine_similarity(z_adv, clean_features.detach(), dim=-1)
        
        # 2. 存活概率 \pi_i 计算 (Eq. 6)
        r_i = torch.argsort(torch.argsort(s_adv, descending=True)) # 获取排名 0~575
        pi_i = torch.zeros_like(r_i, dtype=torch.float32)
        
        pi_i[r_i < K_MIN] = 1.0
        mask_mid = (r_i >= K_MIN) & (r_i <= K_MAX)
        pi_i[mask_mid] = (K_MAX - r_i[mask_mid].float()) / denominator
        # >= K_MAX 的概率保持为 0
        
        # 3. L_EFD 计算 (Eq. 8)
        L_EFD = torch.sum(pi_i * d_i) / (torch.sum(pi_i) + 1e-8)
        
        # 4. L_RDA 计算 (Eq. 10, Eq. 11)
        p_d = F.softmax(d_i, dim=0).detach() # 作为对齐目标停止梯度
        p_s = F.softmax(s_adv, dim=0)
        L_RDA = torch.sum(p_d * torch.log(p_s + 1e-8)) 
        
        # 5. 总损失 (Eq. 12) 梯度上升
        loss_cage = L_EFD + LAMBDA_CAGE * L_RDA
        
        clip_encoder.zero_grad()
        loss_cage.backward() 
        delta_cage.data = delta_cage.data + ALPHA * delta_cage.grad.detach().sign()
        delta_cage.data = torch.clamp(delta_cage.data, -EPSILON, EPSILON)
        delta_cage.grad.zero_()
        
    final_cage = torch.clamp(clean_image_01 + delta_cage, 0, 1).detach()
    observed_linf = (final_cage - clean_image_01).abs().max().item()
    if observed_linf > EPSILON + 1e-6:
        raise RuntimeError(
            f"CAGE L_inf violation: {observed_linf:.8f} > {EPSILON:.8f}"
        )
    img_cage = quantize_linf_image(clean_image_01, final_cage, EPSILON)

    save_attack_image(
        image=img_cage,
        cache_root=args.attack_cache_root,
        method=args.method,
        dataset=args.dataset,
        attack=CACHE_ATTACK,
        image_filename=img_filename,
    )

    if args.cache_only:
        continue
    if img_filename in processed_ids:
        # The result row was already valid; this pass only repaired its cache.
        continue

    # 4. 评估所有 K 值并落盘
    raw_answers = {"Lang_Prior_K0": ans_k0}
    raw_answers.update(evaluate_image_across_k(image, prompt, "Clean"))
    raw_answers.update(evaluate_image_across_k(img_base, prompt, "Base"))
    raw_answers.update(evaluate_image_across_k(img_cage, prompt, "CAGE"))

    with open(
        resolve_owned_output_path(OUTPUT_ROOT, RESULTS_RELATIVE),
        mode='a', newline='', encoding='utf-8'
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [img_filename, f'"{gt_data["question"]}"']
            + answer_score_cells(SCORE_COLUMNS, raw_answers, gt_data)
        )

missing_caches = [
    image_id for image_id in expected_ids
    if not attack_image_exists(
        cache_root=args.attack_cache_root,
        method=args.method,
        dataset=args.dataset,
        attack=CACHE_ATTACK,
        image_filename=image_id,
    )
]
if missing_caches:
    raise RuntimeError(
        f"CAGE cache incomplete: missing_or_invalid={len(missing_caches)} "
        f"examples={missing_caches[:5]}"
    )
if not args.cache_only:
    final_ids = read_completed_result_ids(
        RESULTS_FILE,
        prefix_columns=["Image_ID", "Question"],
        score_columns=SCORE_COLUMNS,
        ground_truth_rows=qa_database,
        expected_values=EXPECTED_ROW_VALUES,
    )
    assert_exact_completion(expected_ids, final_ids, "LLaVA CAGE result rows")
