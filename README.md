# FATA

FATA studies **compression-triggered adversarial failure** in vision-language
models: an image is optimized to preserve useful behavior with the full visual
representation while degrading behavior after visual-token compression.

This repository contains source code, configurations, tests, and small result
summaries. It does not contain model weights, dataset mappings, dataset images,
raw predictions, adversarial-image collections, activation features,
checkpoints, caches, or credentials.

## Threat Models and Scope

The model settings use distinct access assumptions and practical token
budgets:

| Model setting | Access and objective | Practical budget |
|---|---|---|
| LLaVA-1.5-7B | Compressor-agnostic gray-box attack using the vision encoder and its gradients, but not the deployment compressor, deployment budget, or downstream LLM gradient. Clean final-layer CLS-to-patch attention is averaged over heads, and a clean Top-64 target mask is fixed throughout PGD. `L_FATA = L_attn + lambda L_sem`, with epsilon `2/255`, alpha `0.5/255`, 100 steps, and lambda 1. | Per-dataset, per-compressor `K_prac`, selected from Clean results only. |
| Qwen3.5-9B | Task-aware white-box adaptation with downstream/task gradients, teacher forcing, compressed-task loss, rank and compressor disruption, Full ground-truth CE preservation, and clean-logit distillation. | Prespecified Open `1/9` and MC `1/18`. |
| InternVL3.5-8B-HF | Architecture-aware adaptation for dynamic tiling and dynamic visual-token counts. | Prespecified Open `1/9` and MC `1/18`; the evaluation trajectory is Full, `1/3`, `2/9`, `1/9`, `1/18`, and `1/36`. |

The four compressors in scope are VisionZIP, VisPruner, PruMerge, and FlowCut.

## Repository Layout

```text
.
├── configs/                 experiment and example path configurations
├── requirements/            separate environment specifications
├── src/fata/
│   ├── attacks/             PGD objectives and constraints
│   ├── compression/         token-budget and output-shape checks
│   ├── data/                dataset construction and record helpers
│   ├── detection/           detector implementations and metrics
│   ├── evaluation/          scoring, K_prac, and paper metrics
│   ├── runtimes/            model-specific runtimes
│   └── utils/               paths and checkpoint helpers
├── tests/                   CPU unit and integration tests
├── results/paper/           small reviewer-checkable summaries
├── docs/                    licensing and third-party review notes
└── MANIFEST.sha256          release-file integrity manifest
```

## Installation

Use Python 3.10. For the lightweight audit and test environment:

```bash
python -m venv .venv-audit
source .venv-audit/bin/activate
python -m pip install -e '.[test]'
```

The model families use different dependency stacks. Install only the stack
needed for the selected runtime:

```bash
python -m pip install -r requirements/llava.txt
python -m pip install -r requirements/dataset-builder.txt
python -m pip install -r requirements/qwen35.txt
python -m pip install -r requirements/internvl35.txt
python -m pip install -r requirements/detection.txt
```

Choose a compatible PyTorch/CUDA wheel for the target host. The requirement
files do not install model weights or dataset content.

## Models, Data, and Path Configuration

Copy `.env.example` to an untracked local file or export the variables in the
shell. Never commit `.env`, `paths.local.yaml`, or credentials.

```bash
export FATA_MODEL_ROOT=/path/to/local_models
export FATA_DATA_ROOT=/path/to/fata_datasets
export FATA_OUTPUT_ROOT=/path/to/outputs
export FATA_CACHE_ROOT=/path/to/huggingface_cache

export FATA_LLAVA_MODEL="$FATA_MODEL_ROOT/llava-1.5-7b-hf"
export FATA_CLIP_MODEL="$FATA_MODEL_ROOT/clip-vit-large"
export FATA_QWEN_MODEL="$FATA_MODEL_ROOT/Qwen3.5-9B"
export FATA_QWEN_MANIFEST_ROOT=/path/to/qwen_manifests

fata-validate-paths --create-output
```

The expected read-only layout is:

```text
FATA_DATA_ROOT/
├── TextVQA_Open/TextVQA_Open_mapping.jsonl
├── VQAv2_Open/VQAv2_Open_mapping.jsonl
├── VQAv2_MC/VQAv2_MC_mapping.jsonl
└── ScienceQA_MC/ScienceQA_MC_mapping.jsonl
```

Records contain `question`, a non-empty `answers` list, and
`image_filename`. Multiple-choice records use `type: multiple_choice` and an
option-letter ground truth. Mappings and images are intentionally excluded
from this repository.

## Evaluation Subset Construction

We construct four vision-dependent evaluation subsets, each containing 1,000
samples with unique image content: TextVQA-Open, VQAv2-Open, VQAv2-MC, and
ScienceQA-MC.

The sampling unit is a unique image paired with one question selected before
model inference. We first group QA records by a reliable source `image_id`;
canonical RGB SHA-256 hashes are additionally used to detect identical image
content. When multiple questions refer to the same image, the representative
question is selected deterministically according to the source order. We do
not try a second question from the same image based on whether the model
answers the first one correctly.

Each candidate is evaluated under two conditions:

1. **Black-image blind control:** the original image is replaced with a
   336 × 336 black RGB image while the question and prompt remain unchanged.
   The model must answer incorrectly.
2. **Full visual input:** the original image is evaluated by LLaVA-1.5-7B
   with all 576 visual tokens. The model must answer correctly.

Images are saved using deterministic JPEG settings and reloaded before the
final full-image check. Uniqueness is validated using the source `image_id`,
the canonical RGB SHA-256 hash, and the RGB SHA-256 hash obtained after
decoding the saved JPEG.

No adversarial attack or visual-token compressor is used during subset
construction. In particular, the selection process does not run or inspect
FATA, Base, CAA, CAGE, compressed-token accuracy, SR, ASR, or CBR.

The blind condition is a **black-image blind control**, not literal zero-token
inference: the black image is processed normally by the visual encoder and
therefore still produces visual tokens.

| Dataset | Source QA records | Source unique images | Eligible unique candidates | Unique images evaluated | Blind-control incorrect | Retained | Acceptance rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| TextVQA-Open | 5,000 | 3,166 | 3,166 | 2,128 | 1,978 | 1,000 | 46.99% |
| VQAv2-Open | 214,354 | 40,504 | 39,248 | 1,659 | 1,336 | 1,000 | 60.28% |
| VQAv2-MC | 214,354 | 40,504 | 39,235 | 3,361 | 1,259 | 1,000 | 29.75% |
| ScienceQA-MC | 21,208 | 7,803 | 7,792 | 7,792 | 3,151 | 1,000 | 12.83% |

`Source QA records` counts question-answer records rather than images. `Unique
images evaluated` counts the unique-image candidates actually passed through
the model. Candidate evaluation stops once the target of 1,000 retained
samples is reached, except for ScienceQA-MC, for which the complete eligible
pool is evaluated. Acceptance rates use the number of model-evaluated unique
images as the denominator.

### TextVQA-Open

We use the TextVQA validation split, which contains 5,000 QA records associated
with 3,166 unique images. Grouping by image removes the additional QA records
associated with already represented images before model inference.

The model evaluates 2,128 unique-image candidates. Of these, 1,978 are
answered incorrectly under the black-image blind control, and 1,000 are also
answered correctly with the full image. The 1,000th retained sample occurs at
source index 3,379, giving an acceptance rate of 1,000 / 2,128 = 46.99%.

### VQAv2-Open

We use the official VQAv2 validation questions and annotations. The split
contains 214,354 QA records and 40,504 unique COCO images. We remove 83,479
yes/no questions and collapse 91,627 additional records associated with image
IDs that already have a deterministically selected representative question,
leaving 39,248 eligible unique-image candidates.

The model evaluates 1,659 candidates. Among them, 1,336 are incorrect under
the black-image blind control, and 1,000 are correct under full-image
inference. The 1,000th retained sample occurs at source index 9,307, yielding
an acceptance rate of 60.28%.

### VQAv2-MC

VQAv2-MC is constructed from the same VQAv2 validation source after the yes/no
filter. The modal human answer is used as the correct option, and three
distinct distractors are generated deterministically. The random seed and the
`question_id` determine the option content, option order, and correct letter,
so the multiple-choice mapping is reproducible across runs.

After removing 83,479 yes/no records, collapsing 91,627 additional same-image
QA records, and excluding 13 invalid multiple-choice records, the eligible
pool contains 39,235 unique images. Image IDs already selected for VQAv2-Open
are excluded to prevent cross-task image overlap.

The procedure scans 4,359 candidates, excludes 998 because their image IDs
occur in VQAv2-Open, and evaluates the remaining 3,361 unique images. Of these,
1,259 are incorrect under the black-image blind control and 1,000 are correct
with the full image. The 1,000th retained sample occurs at source index 23,876.
The acceptance rate over model-evaluated candidates is 29.75%. The final
VQAv2-Open and VQAv2-MC subsets share no source image IDs.

### ScienceQA-MC

ScienceQA-MC uses the combined ScienceQA train, validation, and test splits.
The source contains 21,208 QA records and 7,803 unique images. We exclude
10,876 records without usable images, 15 records with invalid metadata or
answer choices, and 2,525 additional records associated with duplicate
source-image content across the three splits. This produces 7,792 eligible
unique-image candidates.

All 7,792 candidates are evaluated. Of these, 4,641 are already answered
correctly under the black-image blind control and are excluded, leaving 3,151
blind-control-incorrect candidates. Among those candidates, 1,004 are answered
correctly with the full image. Following the final exact-image uniqueness
validation, 1,000 samples are retained, corresponding to an acceptance rate
of 12.83%.

All four evaluation subsets therefore contain exactly 1,000 samples and 1,000
exact-unique images.

## Primary Entrypoints

Inspect every interface with `--help` before starting a model workload.

| Capability | Entrypoint |
|---|---|
| Dataset construction | `python -m fata.data.unique_builder.build_all` |
| Dataset validation | `python -m fata.data.unique_builder.validate_datasets` |
| Path validation | `fata-validate-paths` |
| Canonical VQA/MC scoring | `fata.evaluation.scoring` |
| Accuracy, SR, ASR, CBR, and transitions | `fata-metrics` |
| LLaVA Clean/Base/FATA | `python -m fata.runtimes.llava.run_llava_fata` |
| LLaVA CAA | `python -m fata.runtimes.llava.run_llava_caa` |
| LLaVA CAGE | `python -m fata.runtimes.llava.run_llava_cage` |
| LLaVA objective ablation | `python -m fata.runtimes.llava.run_llava_objective_ablation` |
| Qwen3.5 | `python -m fata.runtimes.qwen35.run_qwen35_v6` |
| InternVL3.5 status guard | `python -m fata.runtimes.internvl35.provenance_guard` |
| Feature Squeezing | `python -m fata.detection.feature_squeezing.run_feature_squeezing` |
| Mahalanobis-Max | `python -m fata.detection.mahalanobis.run_mahalanobis` |
| ML-ATD | `python -m fata.detection.mlatd.run_mlat_method_worker` |

### LLaVA Example

LLaVA evaluates budgets `576, 192, 128, 64, 32, 16`. The main entry emits
Clean, attention-only Base, and FATA outputs for one compressor/dataset pair:

```bash
python -m fata.runtimes.llava.run_llava_fata \
  --method VisionZIP \
  --dataset TextVQA_Open \
  --dataset-root "$FATA_DATA_ROOT" \
  --model-path "$FATA_LLAVA_MODEL" \
  --clip-path "$FATA_CLIP_MODEL" \
  --output-root "$FATA_OUTPUT_ROOT" \
  --lam 1 --seed 0 --limit 1
```

CAA, CAGE, and objective-ablation interfaces are available through the
entrypoints above. Keep all generated files beneath a dedicated
`FATA_OUTPUT_ROOT`; do not write into a dataset directory or `results/paper/`.

### Metrics and `K_prac`

The canonical open-ended scorer uses normalized exact VQA consensus;
multiple-choice scoring requires the option letter. Dataset-construction
filtering is separate from formal evaluation scoring.

`fata-metrics` consumes sample-aligned Clean-Full, Attack-Full,
Clean-practical, and Attack-practical columns:

```bash
fata-metrics /path/to/aligned_scores.csv \
  --output "$FATA_OUTPUT_ROOT/metrics/example.json"
```

For LLaVA,

```text
K_prac = min K in {16, 32, 64, 128, 192}
         such that mean(Clean@K) / mean(Clean@576) >= 0.80.
```

The selector uses Clean values only and excludes 576 as a candidate. Qwen and
InternVL use their prespecified task-specific fractions instead.

### Qwen3.5 Example

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m fata.runtimes.qwen35.run_qwen35_v6 \
  --dataset TextVQA_Open \
  --configs v6_a_fullpres_light_4comp \
  --limit 1 --start-index 0 --device-map balanced \
  --model-path "$FATA_QWEN_MODEL" \
  --data-root "$FATA_DATA_ROOT" \
  --manifest-dir "$FATA_QWEN_MANIFEST_ROOT" --manifest-suffix n1000 \
  --output-root "$FATA_OUTPUT_ROOT" --output-prefix v6_smoke
```

The manifest contains versioned, sample-aligned records and must resolve all
image paths beneath the configured data root.

### Detection Examples

Feature Squeezing:

```bash
python -m fata.detection.feature_squeezing.run_feature_squeezing \
  --method VisionZIP --dataset TextVQA_Open \
  --attack_for_detection fata --detect_k 64 --limit 1 --seed 0 \
  --model-path "$FATA_LLAVA_MODEL" --clip-path "$FATA_CLIP_MODEL" \
  --dataset-root "$FATA_DATA_ROOT" --output-root "$FATA_OUTPUT_ROOT"
```

Mahalanobis-Max uses separate `--mode fit` and `--mode eval` invocations.
ML-ATD feature extraction is exposed through
`fata.detection.mlatd.run_mlat_method_worker`. Use each module's `--help`
output for its complete argument contract.

Python pickle deserialization can execute code. Mahalanobis statistics must be
created and loaded only inside a trusted output directory; never load an
untrusted or downloaded pickle.

## Verification

Run lightweight checks from the repository root:

```bash
python -m compileall -q src tests scripts
python -m pytest -q
fata-kprac --help
fata-metrics --help
fata-validate-paths --help
python -m fata.data.unique_builder.build_all --help
python -m fata.runtimes.llava.run_llava_fata --help
```

These commands compile the package, exercise the CPU test suite, and inspect
CLI contracts without starting a full dataset build or model experiment.

## Results

`results/paper/` contains small CSV, JSON, and text summaries for reviewer
inspection. Treat them as immutable artifacts: do not edit scientific values
to satisfy packaging checks, and do not write smoke-test output into this
directory.

## GitHub Upload Boundary

Only this repository directory is within the GitHub payload. In particular,
exclude:

- model weights, processors, tokenizers, datasets, mappings, and images;
- raw predictions, adversarial images, activation features, and checkpoints;
- caches, virtual environments, W&B files, and temporary test output;
- `.env`, `paths.local.yaml`, credentials, host logs, and personal paths;
- sibling `audit/` and `local_only_artifacts/` directories.

`MANIFEST.sha256` covers every public file except itself.

## Licensing

This snapshot does not assign a project-wide open-source license. Publication
and redistribution remain blocked until first-party rights, upstream source
revisions, third-party licenses, notices, and dataset/model terms are reviewed
for the exact file manifest. See
[`docs/LICENSING_REVIEW_REQUIRED.md`](docs/LICENSING_REVIEW_REQUIRED.md) and
[`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md).

## Known Limitations

- A nominal token budget is an information-bottleneck count; model-specific
  adapters may expand the compressed representation to the sequence shape
  expected by the projector or language model. It should not be interpreted
  as a demonstrated end-to-end FLOP reduction.
- Model-family environments are separate and require operator-supplied model
  weights and dataset paths.
- The InternVL entry in this snapshot is a provenance guard rather than a
  formal attack launcher.
- Result summaries and incorporated implementations remain subject to the
  licensing review above.
- No final case-study export command is included.
