import os
import torch
import torch.nn.functional as F
from PIL import Image
import csv
import json
import re
import argparse
import hashlib
import math
import random
import numpy as np
from tqdm import tqdm
import gc
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
from .image_span import expanded_image_and_trailing_text_span
from fata.utils.paths import (
    assert_output_separate,
    create_owned_output_directory,
    metadata_sidecar_path,
    resolve_dataset_relative_path,
    resolve_owned_output_path,
    safe_filename_component,
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
parser = argparse.ArgumentParser(description="CAA Attack Benchmark (Anti-OOM Armor Version)")
parser.add_argument("--method", type=str, required=True, choices=["VisionZIP", "VisPruner", "PruMerge", "FlowCut"])
parser.add_argument("--dataset", type=str, required=True, choices=["TextVQA_Open", "ScienceQA_MC", "VQAv2_Open", "VQAv2_MC"])
parser.add_argument("--limit", type=int, default=0, help="限制测试的样本数量；0 表示全量")
parser.add_argument("--epsilon", type=float, default=2.0,
                    help="L_inf 扰动预算，单位为 /255；2 表示 2/255")
parser.add_argument("--alpha", type=float, default=1.0,
                    help="PGD 步长，单位为 /255；默认保持原 CAA 的 1/255")
parser.add_argument("--steps", type=int, default=100, help="PGD 优化步数")
parser.add_argument("--seed", type=int, default=0, help="基础随机种子")
parser.add_argument("--output-tag", type=str, default="eps2",
                    help="结果文件标签，用于与旧的 8/255 结果隔离")
parser.add_argument(
    "--attack-cache-root",
    type=str,
    default=None,
)

parser.add_argument(
    "--cache-only",
    action="store_true",
)
parser.add_argument("--model-path", default=os.environ.get("FATA_LLAVA_MODEL"))
parser.add_argument("--output-root", default=os.environ.get("FATA_OUTPUT_ROOT"))
add_dataset_root_argument(parser)
args = parser.parse_args()
if (
    args.limit < 0
    or args.seed < 0
    or args.steps <= 0
    or not math.isfinite(args.epsilon)
    or not math.isfinite(args.alpha)
    or args.epsilon <= 0
    or args.alpha <= 0
):
    parser.error("--limit/--seed must be non-negative and attack parameters finite/positive")
try:
    safe_tag = safe_filename_component(args.output_tag, label="output tag")
except ValueError as error:
    parser.error(str(error))

# ================= 2. 全局配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not args.model_path or not args.output_root:
    parser.error("pass --model-path and --output-root (or set FATA_LLAVA_MODEL and FATA_OUTPUT_ROOT)")
LLAVA_PATH = args.model_path

DATASET_DIR, QA_FILE = resolve_dataset_paths(args.dataset_root, args.dataset)
protected_inputs = {"dataset_root": args.dataset_root, "LLaVA model": LLAVA_PATH}
OUTPUT_ROOT = assert_output_separate(args.output_root, protected_inputs)
if args.attack_cache_root is None:
    args.attack_cache_root = resolve_owned_output_path(
        OUTPUT_ROOT, "adversarial_images"
    )
else:
    args.attack_cache_root = os.path.abspath(os.path.expanduser(args.attack_cache_root))
assert_output_separate(args.attack_cache_root, protected_inputs)

RESULTS_DIR = create_owned_output_directory(OUTPUT_ROOT, "llava/caa")
assert_output_separate(RESULTS_DIR, {"attack cache": args.attack_cache_root})
canonical_tag = f"eps{args.epsilon:g}"
tag_suffix = "" if safe_tag == canonical_tag else f"_tag-{safe_tag}"
RESULTS_RELATIVE = (
    "llava/caa/"
    f"caa_eps{args.epsilon:g}_a{args.alpha:g}_s{args.steps}_seed{args.seed}{tag_suffix}_{args.method}_{args.dataset}.csv"
)
RESULTS_FILE = resolve_owned_output_path(OUTPUT_ROOT, RESULTS_RELATIVE)
META_RELATIVE = str(metadata_sidecar_path(RESULTS_RELATIVE))
META_FILE = str(resolve_owned_output_path(OUTPUT_ROOT, META_RELATIVE))

SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
BUDGETS = [576, 192, 128, 64, 32, 16]

EPSILON = args.epsilon / 255.0
ALPHA = args.alpha / 255.0
STEPS = args.steps

CAA_CACHE_EXTRA = baseline_attack_contract_extra("caa", max_input_tokens=0)
CAA_OBJECTIVE = CAA_CACHE_EXTRA["objective"]
TARGET_LAYER = CAA_OBJECTIVE["target_layer"]
REGION_FRACTION = CAA_OBJECTIVE["least_important_region_fraction"]
W_BPR_INTER = CAA_OBJECTIVE["weights"]["bpr_inter"]
W_BPR_INTRA = CAA_OBJECTIVE["weights"]["bpr_intra"]
W_SE = CAA_OBJECTIVE["weights"]["semantic"]
W_QA = CAA_OBJECTIVE["weights"]["question_answer"]
SCORE_COLUMNS = [f"CAA_K{k}" for k in BUDGETS]
HEADER = result_header(["Image_ID", "Question"], SCORE_COLUMNS)
RESULT_SCHEMA = result_schema(SCORE_COLUMNS)
CACHE_ATTACK = attack_namespace(
    "caa", seed=args.seed, eps_255=args.epsilon,
    alpha_255=args.alpha, steps=args.steps,
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

def get_pixel_mask_from_tokens(token_indices, image_size=336, patch_size=14):
    mask = torch.zeros((1, 3, image_size, image_size), device=DEVICE)
    grid_size = image_size // patch_size
    for idx in token_indices:
        row = idx // grid_size
        col = idx % grid_size
        mask[:, :, row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = 1.0
    return mask

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_sample_generator(base_seed, sample_id, device):
    """每个样本使用稳定随机种子，保证断点续跑结果一致。"""
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode("utf-8")).digest()
    sample_seed = int.from_bytes(digest[:8], "little") % (2**31)
    generator = torch.Generator(device=device)
    generator.manual_seed(sample_seed)
    return generator

set_global_seed(args.seed)
print(
    f"CAA config | epsilon={args.epsilon}/255 | alpha={args.alpha}/255 | "
    f"steps={STEPS} | seed={args.seed} | output={RESULTS_FILE}"
)

# ================= 4. 读取数据与恢复状态 =================
qa_database = []
with open(QA_FILE, 'r', encoding='utf-8') as f:
    for line in f: qa_database.append(json.loads(line))

if args.limit > 0:
    if args.limit > len(qa_database):
        parser.error(
            f"--limit {args.limit} exceeds mapping length {len(qa_database)}"
        )
    qa_database = qa_database[:args.limit]

expected_ids = expected_unique_ids(qa_database, "image_filename")
if not expected_ids:
    parser.error("selected dataset slice is empty")
EXPECTED_ROW_VALUES = {
    row["image_filename"]: {"Question": f'"{row.get("question", "")}"'}
    for row in qa_database
}
CACHE_CONTRACT = build_attack_cache_contract(
    definition=baseline_attack_definition("caa"),
    dataset=args.dataset,
    mapping_path=QA_FILE,
    method=args.method,
    model_path=LLAVA_PATH,
    clip_path=None,
    seed=args.seed,
    eps_255=args.epsilon,
    alpha_255=args.alpha,
    steps=STEPS,
    extra=CAA_CACHE_EXTRA,
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
        "runtime": "llava_caa",
        "attack": "CAA",
        "region": "least-important-30-percent",
        "dataset": args.dataset,
        "dataset_mapping_sha256": sha256_file(QA_FILE),
        "dataset_images": dataset_image_identity(QA_FILE, DATASET_DIR),
        "method": args.method,
        "model": artifact_identity(LLAVA_PATH),
        "epsilon_over_255": args.epsilon,
        "alpha_over_255": args.alpha,
        "steps": STEPS,
        "seed": args.seed,
        "budgets": BUDGETS,
        "target_layer": TARGET_LAYER,
        "weights": {
            "W_BPR_INTER": W_BPR_INTER,
            "W_BPR_INTRA": W_BPR_INTRA,
            "W_SE": W_SE,
            "W_QA": W_QA,
        },
        "max_input_tokens": 0,
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
        f"✅ 精确恢复：{len(expected_ids)} 个 CAA 缓存已完整，"
        "跳过模型加载。"
    )
    raise SystemExit(0)
if not args.cache_only and processed_ids == set(expected_ids):
    if missing_resume_caches:
        raise RuntimeError(
            "completed LLaVA CAA result has incomplete attack caches: "
            f"missing_or_invalid={len(missing_resume_caches)} "
            f"examples={missing_resume_caches[:5]}"
        )
    print(
        f"✅ 精确恢复：{len(processed_ids)} 个结果行与全部 CAA 缓存已完整，"
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
print(f"🔄 加载模型与算法 [{args.method}] (CAA 防爆显存版)...")
llava_processor = AutoProcessor.from_pretrained(LLAVA_PATH, use_fast=False)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    LLAVA_PATH, torch_dtype=torch.float16, device_map="cuda", attn_implementation="eager"
)

for param in llava_model.parameters():
    param.requires_grad = False

vision_model = llava_model.vision_tower.vision_model if hasattr(llava_model, 'vision_tower') else llava_model.model.vision_tower.vision_model
apply_compression_patch(vision_model, args.method)

llama_model = llava_model.language_model.model
target_layer = llama_model.layers[TARGET_LAYER]
q_proj = target_layer.self_attn.q_proj
k_proj = target_layer.self_attn.k_proj

mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(DEVICE)
std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(DEVICE)

if os.path.exists(RESULTS_FILE):
    print(f"📦 断点续传：已存在 {len(processed_ids)} 个样本")

def evaluate_image_across_k(image, prompt):
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
        answers[f"CAA_K{k}"] = ans
    return answers

# ================= 6. 核心生成与攻击循环 =================
skipped_missing = 0
skipped_decode = 0
written_count = 0
max_observed_linf = 0.0

for gt_data in tqdm(qa_database, desc=f"CAA eps={args.epsilon}/255 on {args.dataset}"):
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
    if not os.path.exists(img_path):
        skipped_missing += 1
        continue
    try: 
        raw_image = Image.open(img_path).convert("RGB")
        image = expand2square(raw_image)
    except Exception as exc:
        skipped_decode += 1
        print(f"⚠️ 无法读取图像 {img_path}: {exc}")
        continue
    
    prompt = f"{SYSTEM_PROMPT}USER: <image>\n{gt_data['question']}\nAnswer with the option's letter from the given choices directly. ASSISTANT:" if gt_data["type"] == "multiple_choice" else f"{SYSTEM_PROMPT}USER: <image>\n{gt_data['question']}\nAnswer the question using a single word or phrase. ASSISTANT:"
    
    raw_inputs = llava_processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in raw_inputs.items() if v is not None}
    
    input_ids = inputs['input_ids']
    
    tokenizer_img_token_id = llava_processor.tokenizer.convert_tokens_to_ids("<image>")
    config_img_token_id = getattr(llava_model.config, "image_token_index", None)
    
    clean_pixel_values = inputs['pixel_values'].to(torch.float16)
    clean_image_01 = (clean_pixel_values * std + mean).detach()
    
    with torch.no_grad():
        vision_model.current_K = 576
        clean_outputs = llava_model(
            input_ids=input_ids,
            attention_mask=inputs['attention_mask'],
            pixel_values=clean_pixel_values,
            labels=None, # 【防爆装甲 2】：强制关闭底层 CrossEntropy 计算
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = clean_outputs.hidden_states[TARGET_LAYER][0] 
        v_start, v_end, t_start = expanded_image_and_trailing_text_span(
            input_ids,
            output_sequence_length=int(hidden_states.shape[0]),
            image_token_count=576,
            config_token_id=config_img_token_id,
            tokenizer_token_id=tokenizer_img_token_id,
        )
        clean_vis_hidden = hidden_states[v_start:v_end, :]
        clean_text_hidden = hidden_states[t_start:, :]
        
        clean_K_vis = k_proj(clean_vis_hidden)
        clean_Q_text = q_proj(clean_text_hidden)
        
        clean_logits = torch.matmul(clean_Q_text, clean_K_vis.T).mean(dim=0)
        
        clean_ranks = torch.argsort(clean_logits, descending=True)
        topk_most = clean_ranks[:int(576 * REGION_FRACTION)]
        topk_least = clean_ranks[-int(576 * REGION_FRACTION):]
        
        pixel_least_mask = get_pixel_mask_from_tokens(topk_least).to(DEVICE)
        
        ref_vis_hidden = clean_vis_hidden.detach()
        ref_Q_text = clean_Q_text.mean(dim=0).detach() 

    sample_generator = make_sample_generator(
        args.seed, img_filename, clean_image_01.device
    )
    delta_caa = torch.empty_like(clean_image_01).uniform_(
        -EPSILON, EPSILON, generator=sample_generator
    ).requires_grad_(True)
    
    for step in range(STEPS):
        adv_img = clean_image_01 + delta_caa * pixel_least_mask
        adv_img = torch.clamp(adv_img, 0, 1)
        norm_adv_img = ((adv_img - mean) / std).to(torch.float16)
        
        adv_outputs = llava_model(
            input_ids=input_ids,
            attention_mask=inputs['attention_mask'],
            pixel_values=norm_adv_img,
            labels=None, # 【防爆装甲 2】：彻底扼杀交叉熵分配
            output_hidden_states=True,
            return_dict=True
        )
        
        adv_hidden = adv_outputs.hidden_states[TARGET_LAYER][0]
        if adv_hidden.shape[0] != hidden_states.shape[0]:
            raise RuntimeError(
                "CAA adversarial hidden sequence length changed unexpectedly: "
                f"clean={hidden_states.shape[0]} adv={adv_hidden.shape[0]}"
            )
        adv_vis_hidden = adv_hidden[v_start:v_end, :]
        adv_K_vis = k_proj(adv_vis_hidden)
        adv_logits = torch.matmul(clean_Q_text, adv_K_vis.T).mean(dim=0)
        
        s_least = adv_logits[topk_least]
        s_most = adv_logits[topk_most]
        
        diff_inter = s_least.unsqueeze(1) - s_most.unsqueeze(0)
        loss_bpr_inter = -F.logsigmoid(diff_inter).mean()
        
        diff_intra = s_least.unsqueeze(1) - s_least.unsqueeze(0)
        mask_upper = torch.triu(torch.ones_like(diff_intra), diagonal=1).bool()
        loss_bpr_intra = -F.logsigmoid(-diff_intra[mask_upper]).mean()
        
        loss_se = - F.mse_loss(adv_vis_hidden[topk_least], ref_vis_hidden[topk_least])
        
        adv_least_K = adv_K_vis[topk_least]
        alignment = torch.matmul(adv_least_K, ref_Q_text.unsqueeze(-1)).mean()
        loss_qa = - alignment
        
        loss_total = (W_BPR_INTER * loss_bpr_inter) + (W_BPR_INTRA * loss_bpr_intra) + (W_SE * loss_se) + (W_QA * loss_qa)
        
        llava_model.zero_grad()
        loss_total.backward() 
        
        delta_caa.data = delta_caa.data - ALPHA * delta_caa.grad.detach().sign()
        delta_caa.data = torch.clamp(delta_caa.data, -EPSILON, EPSILON)
        delta_caa.grad.zero_()
        
        # 【防爆装甲 3】：极端深度的垃圾回收，连内存碎片都扬了
        del adv_outputs, adv_hidden, adv_vis_hidden, adv_K_vis, adv_logits
        del s_least, s_most, diff_inter, loss_bpr_inter, diff_intra, mask_upper, loss_bpr_intra
        del loss_se, adv_least_K, alignment, loss_qa, loss_total, adv_img, norm_adv_img
        torch.cuda.empty_cache()
        
    final_adv_tensor = torch.clamp(
        clean_image_01 + delta_caa * pixel_least_mask, 0, 1
    ).detach()
    observed_linf = (final_adv_tensor - clean_image_01).abs().max().item()
    max_observed_linf = max(max_observed_linf, observed_linf)
    if observed_linf > EPSILON + 1e-6:
        raise RuntimeError(
            f"L_inf 越界：{observed_linf:.8f} > {EPSILON:.8f}，样本={img_filename}"
        )

    img_caa = quantize_linf_image(clean_image_01, final_adv_tensor, EPSILON)
    save_attack_image(
        image=img_caa,
        cache_root=args.attack_cache_root,
        method=args.method,
        dataset=args.dataset,
        attack=CACHE_ATTACK,
        image_filename=img_filename,
    )

    if args.cache_only:
        written_count += 1

        gc.collect()
        torch.cuda.empty_cache()

        continue
    if img_filename in processed_ids:
        # The result row was already valid; this pass only repaired its cache.
        continue
    raw_answers = evaluate_image_across_k(img_caa, prompt)

    with open(
        resolve_owned_output_path(OUTPUT_ROOT, RESULTS_RELATIVE),
        mode='a', newline='', encoding='utf-8'
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [img_filename, f'"{gt_data["question"]}"']
            + answer_score_cells(SCORE_COLUMNS, raw_answers, gt_data)
        )

    written_count += 1
        
    # 每完成一个样本，强制大回收
    gc.collect()
    torch.cuda.empty_cache()

print("\n===== CAA 运行总结 =====")
print(f"结果文件: {RESULTS_FILE}")
print(f"元数据文件: {META_FILE}")
print(f"本次新写入: {written_count}")
print(f"已有并跳过: {len(processed_ids)}")
print(f"图像缺失: {skipped_missing}")
print(f"图像读取失败: {skipped_decode}")
print(f"观测到的最大 L_inf: {max_observed_linf * 255:.6f}/255")

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
        f"CAA cache incomplete: missing_or_invalid={len(missing_caches)} "
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
    assert_exact_completion(expected_ids, final_ids, "LLaVA CAA result rows")
