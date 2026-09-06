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
    ensure_run_contract,
    expected_unique_ids,
    sha256_file,
)

# ================= 1. 参数解析 =================
parser = argparse.ArgumentParser(description="Ultimate FATA Benchmark")
parser.add_argument("--ablation", type=str, default="full", choices=["full", "attn_only", "sem_only"], help="FATA消融实验模式")
parser.add_argument("--method", type=str, required=True, choices=["VisionZIP", "VisPruner", "PruMerge", "FlowCut"])
parser.add_argument("--dataset", type=str, required=True, choices=["TextVQA_Open", "ScienceQA_MC", "VQAv2_Open", "VQAv2_MC"])
parser.add_argument("--lam", type=float, default=1.0, help="黄金 Lambda 值")
parser.add_argument("--limit", type=int, default=0, help="限制测试的样本数量 (0表示全量1000个)")
parser.add_argument("--seed", type=int, default=0, help="deterministic per-sample random-start seed")
parser.add_argument("--model-path", default=os.environ.get("FATA_LLAVA_MODEL"))
parser.add_argument("--clip-path", default=os.environ.get("FATA_CLIP_MODEL"))
parser.add_argument("--output-root", default=os.environ.get("FATA_OUTPUT_ROOT"))
add_dataset_root_argument(parser)
args = parser.parse_args()
if args.limit < 0 or args.seed < 0 or not math.isfinite(args.lam) or args.lam < 0:
    parser.error("--limit/--seed must be non-negative and --lam finite/non-negative")

# 【Bug 修复】：将传入的参数提取为全局 LAMBDA 变量
LAMBDA = args.lam

# ================= 2. 全局配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not args.model_path or not args.clip_path or not args.output_root:
    parser.error("pass --model-path, --clip-path and --output-root")
LLAVA_PATH = args.model_path
CLIP_PATH = args.clip_path

# 自动定位到你分类好的黄金数据集
DATASET_DIR, QA_FILE = resolve_dataset_paths(args.dataset_root, args.dataset)
OUTPUT_ROOT = assert_output_separate(
    args.output_root,
    {
        "dataset_root": args.dataset_root,
        "LLaVA model": LLAVA_PATH,
        "CLIP model": CLIP_PATH,
    },
)

RESULTS_DIR = create_owned_output_directory(OUTPUT_ROOT, "llava/objective_ablation")
RESULTS_RELATIVE = (
    "llava/objective_ablation/"
    f"fata_ablation_{args.ablation}_lam{args.lam:g}_seed{args.seed}_{args.method}_{args.dataset}.csv"
)
RESULTS_FILE = resolve_owned_output_path(OUTPUT_ROOT, RESULTS_RELATIVE)
META_RELATIVE = str(metadata_sidecar_path(RESULTS_RELATIVE))
META_FILE = str(resolve_owned_output_path(OUTPUT_ROOT, META_RELATIVE))
SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
BUDGETS = [576, 192, 128, 64, 32, 16]
EPSILON = 2 / 255.0  
ALPHA = 0.5 / 255.0     
STEPS = 100            
SCORE_COLUMNS = ["Lang_Prior_K0"] + [
    f"{mode}_K{k}" for mode in ("Clean", "FATA") for k in BUDGETS
]
HEADER = result_header(["Image_ID", "Question"], SCORE_COLUMNS)
RESULT_SCHEMA = result_schema(SCORE_COLUMNS)

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

def sample_generator(sample_id: str) -> torch.Generator:
    digest = hashlib.sha256(
        f"{args.seed}|{args.ablation}|{sample_id}".encode("utf-8")
    ).digest()
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(int.from_bytes(digest[:8], "big") & 0x7FFFFFFF)
    return generator

def historical_compute_accuracy_unused(pred_ans, gt_data):
    pred_ans = str(pred_ans).strip().lower()
    q_type = gt_data.get("type", "open")
    
    if q_type == "multiple_choice":
        gt_letter = gt_data["answers"][0].lower()
        gt_text = gt_data.get("ground_truth_text", "").lower()
        match = re.search(r'(?i)(?:^|\s|\()(option\s+)?([a-f])(?:\)|\.|:|\s|$)', pred_ans)
        if match:
            extracted = match.group(2).lower()
        else:
            if gt_text and gt_text in pred_ans: return 1.0
            extracted = pred_ans[0] if len(pred_ans) > 0 else ""
        return 1.0 if extracted == gt_letter else 0.0
    else:
        def official_vqa_process(text):
            text = str(text).lower().replace('\n', ' ').replace('\r', ' ')
            text = re.sub(r'([^\w\s])', r' ', text)  
            words = [w for w in text.split() if w not in ['a', 'an', 'the']]
            num_map = {'zero':'0', 'one':'1', 'two':'2', 'three':'3', 'four':'4', 'five':'5'}
            return ' '.join([num_map.get(w, w) for w in words])

        pred_clean = official_vqa_process(pred_ans)
        gts = [official_vqa_process(gt) for gt in gt_data["answers"]]
        
        if not gts: return 0.0
        match_count = gts.count(pred_clean)
        if match_count == 0:
            for gt in gts:
                if gt and gt in pred_clean:
                    match_count = 3 
                    break
        return min(1.0, float(match_count) / 3.0)

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
ensure_run_contract(
    META_FILE,
    {
        "schema_version": 1,
        "image_serialization": IMAGE_SERIALIZATION_CONTRACT,
        "runtime": "llava_fata_objective_ablation",
        "ablation": args.ablation,
        "dataset": args.dataset,
        "dataset_mapping_sha256": sha256_file(QA_FILE),
        "dataset_images": dataset_image_identity(QA_FILE, DATASET_DIR),
        "method": args.method,
        "model": artifact_identity(LLAVA_PATH),
        "clip_model": artifact_identity(CLIP_PATH),
        "seed": args.seed,
        "lambda": args.lam,
        "epsilon_over_255": EPSILON * 255.0,
        "alpha_over_255": ALPHA * 255.0,
        "steps": STEPS,
        "budgets": BUDGETS,
        "header": HEADER,
        "result_schema": RESULT_SCHEMA,
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
if processed_ids == set(expected_ids):
    print(
        f"✅ 精确恢复：{len(processed_ids)} 个消融结果行已完整，"
        "跳过模型加载。"
    )
    raise SystemExit(0)
if not os.path.exists(RESULTS_FILE):
    with open(
        resolve_owned_output_path(OUTPUT_ROOT, RESULTS_RELATIVE),
        mode='w', newline='', encoding='utf-8'
    ) as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

# ================= 5. 加载模型 =================
print(f"🔄 正在加载模型与算法 [{args.method}] (消融模式: {args.ablation})...")
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

if os.path.exists(RESULTS_FILE):
    print(f"📦 [断点续传] 检测到已有结果文件，将自动跳过 {len(processed_ids)} 个已处理样本！")

def evaluate_image_across_k(image, prompt, score_prefix):
    # ... (这部分原来的代码保持不变) ...
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
for gt_data in tqdm(qa_database, desc=f"Evaluating {args.ablation} on {args.dataset}"):
    img_filename = gt_data["image_filename"]
    
    # 【核心续传逻辑】：如果这张图已经跑过了，直接光速跳过！
    if img_filename in processed_ids:
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

    # 2. 获取干净特征与掩码
    vision_model.current_K = 576 
    raw_inputs_clean = llava_processor(text=prompt, images=image, return_tensors="pt")
    inputs_clean = {k: v.to(DEVICE) for k, v in raw_inputs_clean.items() if v is not None}
    inputs_clean['pixel_values'] = inputs_clean['pixel_values'].to(torch.float16)
    
    with torch.no_grad():
        vision_outputs = vision_model(inputs_clean['pixel_values'], output_attentions=True, return_dict=True)
        clean_attn = vision_outputs.attentions[-1][:, :, 0, 1:].mean(dim=1).squeeze(0)
        clean_features = clip_encoder(inputs_clean['pixel_values']).last_hidden_state.squeeze(0)[1:]

    target_mask = (torch.argsort(torch.argsort(clean_attn, descending=True)) < 64).float().detach()

    # 3. 统一生成对抗扰动基底
    inputs_clip = clip_processor(images=image, return_tensors="pt")
    clean_image_01 = (inputs_clip['pixel_values'].to(DEVICE) * std + mean).detach()
    
    # 【已删除冗余的 Base Attack 计算逻辑】

    # === FATA Attack 消融版 ===
    delta_fata = torch.zeros_like(clean_image_01).uniform_(
        -EPSILON, EPSILON, generator=sample_generator(img_filename)
    ).to(DEVICE).requires_grad_(True)
    for step in range(STEPS):
        adv_img = torch.clamp(clean_image_01 + delta_fata, 0, 1)
        adv_outputs = clip_encoder((adv_img - mean) / std)
        s_adv = adv_outputs.attentions[-1][:, :, 0, 1:].mean(dim=1).squeeze(0) 
        adv_features = adv_outputs.last_hidden_state.squeeze(0)[1:]
        
        loss_attn = torch.sum(s_adv * target_mask)
        sim = F.cosine_similarity(adv_features, clean_features.detach(), dim=-1)
        loss_sem = 1.0 - (torch.sum(sim * target_mask) / (target_mask.sum() + 1e-8))
        
        # ================= 消融实验：动态 Loss 开关 =================
        if args.ablation == "attn_only":
            loss_fata = loss_attn 
        elif args.ablation == "sem_only":
            loss_fata = LAMBDA * loss_sem 
        else:
            loss_fata = loss_attn + LAMBDA * loss_sem
        # ============================================================
        
        clip_encoder.zero_grad()
        loss_fata.backward() 
        delta_fata.data = delta_fata.data - ALPHA * delta_fata.grad.detach().sign()
        delta_fata.data = torch.clamp(delta_fata.data, -EPSILON, EPSILON)
        delta_fata.grad.zero_()
        
    final_fata = torch.clamp(clean_image_01 + delta_fata, 0, 1).detach()
    observed_linf = (final_fata - clean_image_01).abs().max().item()
    if observed_linf > EPSILON + 1e-6:
        raise RuntimeError(
            f"objective-ablation L_inf violation: {observed_linf:.8f} > {EPSILON:.8f}"
        )
    img_fata = quantize_linf_image(clean_image_01, final_fata, EPSILON)

    # 4. 在所有 K 下执行评估并保存
    raw_answers = {"Lang_Prior_K0": ans_k0}
    raw_answers.update(evaluate_image_across_k(image, prompt, "Clean"))
    raw_answers.update(evaluate_image_across_k(img_fata, prompt, "FATA"))

    # 【修改】：写入 CSV 时移除 base_scores
    with open(
        resolve_owned_output_path(OUTPUT_ROOT, RESULTS_RELATIVE),
        mode='a', newline='', encoding='utf-8'
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [img_filename, f'"{gt_data["question"]}"']
            + answer_score_cells(SCORE_COLUMNS, raw_answers, gt_data)
        )

final_ids = read_completed_result_ids(
    RESULTS_FILE,
    prefix_columns=["Image_ID", "Question"],
    score_columns=SCORE_COLUMNS,
    ground_truth_rows=qa_database,
    expected_values=EXPECTED_ROW_VALUES,
)
assert_exact_completion(expected_ids, final_ids, "LLaVA objective-ablation result rows")
