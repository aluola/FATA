# Third-party provenance and notice ledger

## Important: this is not a completed notice file

No authoritative upstream repository/commit and no license, notice, or copying
file was found in the audited source candidates. This document therefore
records unresolved provenance; it does **not** grant permission, identify a
copyright holder, or satisfy third-party attribution requirements.

The current tree must not be represented as having complete third-party
notices. Public redistribution is blocked until every included or derived
component is resolved and the required license texts and attributions are
added. See
[`LICENSING_REVIEW_REQUIRED.md`](LICENSING_REVIEW_REQUIRED.md) for the release
gate.

## Evidence standard

A local filename, method name, paper citation, package name, or source comment
is not sufficient provenance. In particular, comments such as “official,”
“ported,” “strict reproduction,” or “inspired by” are not evidence of:

- an upstream repository URL;
- an exact source revision;
- authorship or copyright ownership;
- source-code redistribution permission;
- model-weight or dataset redistribution permission;
- compliance with attribution, notice, copyleft, patent, or use restrictions.

For each component, release clearance requires an authoritative upstream
locator, pinned commit or release, applicable license at that revision, copied
license/notice text where required, a local similarity/adaptation review, and a
patch or written explanation of the local changes.

## Compression implementations requiring resolution

| Component | Release locations that require review | Evidence currently available | Missing before redistribution |
|---|---|---|---|
| VisionZIP | `src/fata/runtimes/llava/compression_zoo.py`; Qwen dynamic compressor/surrogate files; InternVL compressor adapter | local implementation names and comments; no verified upstream identity | upstream repository and revision, license, attribution, source-to-local diff, redistribution decision |
| VisPruner | same three model-family areas | local implementation name/comments only | same |
| PruMerge | same three model-family areas | local implementation name/comments only | same |
| FlowCut | same three model-family areas | local implementation name/comments only | same |
| DivPrune | excluded from the formal GitHub registries; present in historical/local-only candidate material and in the provenance of old InternVL/ML-ATD results | historical files/results demonstrate use; no verified upstream identity | resolve before distributing any local-only historical source; do not add to the formal public entry |
| FastV | not part of the formal release implementation | name appears only in historical/unreported audit scope | no public notice needed unless related code or material is later included |

The LLaVA `compression_zoo.py` comments label portions “official,” and Qwen
files say they were ported from that local file. Neither statement identifies
the upstream revision or terms. InternVL contains another model-specific
adaptation. These files require code-by-code comparison; do not assume a
single license applies to all three implementations of a named method.

## Attack and detector implementations requiring resolution

| Component | Release locations | Current provenance statement | Required resolution |
|---|---|---|---|
| CAA | LLaVA CAA runner/ablation and ML-ATD cache generator | project implementation with a located historical source chain and fixed local parameters; no verified upstream locator, revision, or license | identify exact upstream paper/code if any, compare objective and implementation, record changes, add license/attribution or remove/reimplement if permission is incompatible |
| CAGE | LLaVA CAGE runner/ablation and ML-ATD cache generator | project implementation with audited local constants; comments calling it official are not proof | same |
| HiddenDetect inspiration | `src/fata/detection/mlatd/` | local code calls ML-ATD “HiddenDetect-inspired”; no upstream URL, commit, snapshot, formula-to-code map, or license was found | identify the exact inspiration, determine whether code or only ideas were used, document similarities/changes, and satisfy citation/license requirements |
| Feature Squeezing | `src/fata/detection/feature_squeezing/` | historical project implementation; no upstream provenance recovered | identify cited method/source and determine whether local code is original, adapted, or copied |
| Mahalanobis-Max | `src/fata/detection/mahalanobis/` | historical project implementation using third-party ML libraries; no upstream provenance recovered | identify method/code lineage and satisfy any code/citation terms |

CAA and CAGE result existence proves that the methods were run historically;
it does not prove that the present source can be redistributed. Likewise,
describing ML-ATD as inspired by HiddenDetect is the maximum supportable claim
until the missing records are recovered.

## Model and processor dependencies

Model weights and processors are not included in this repository. The runtime
names the following operator-supplied families:

- LLaVA-1.5-7B-HF;
- CLIP ViT-Large used by LLaVA attacks and detectors;
- Qwen3.5-9B;
- InternVL3.5-8B-HF.

The audit did not recover an authoritative model-card snapshot, weight hash
ledger tied to a public revision, or redistribution license for these local
checkpoints. A user must obtain them from an authorized source and comply with
their model, base-model, tokenizer, processor, and acceptable-use terms. Do
not bundle weight, tokenizer, or processor files in the GitHub payload merely
because a local runtime expects them.

Before publishing commands that name a specific checkpoint, the owner should
record:

1. the authoritative model-card/repository locator;
2. exact revision and complete file hashes;
3. all upstream/base-model dependencies;
4. applicable license and use restrictions;
5. whether redistribution is permitted or users must fetch the files
   independently;
6. required citation and notice text.

## Dataset dependencies

Dataset files are also excluded from the GitHub payload. The unique builder
refers to these source families:

- TextVQA validation data through the `textvqa` dataset identifier;
- official VQAv2 validation questions and annotations;
- COCO validation images used by VQAv2;
- ScienceQA train, validation, and test data through the
  `derek-thomas/ScienceQA` dataset identifier.

These names and source endpoints in the loader are technical locators, not a
legal determination. The current package contains no verified consolidated
record of dataset licenses, image-level rights, terms of use, attribution
requirements, or redistribution permissions. The owner must review each
dataset and the interaction between dataset annotations and underlying image
rights. Generated mappings, hashes, thumbnails, cached archives, and retained
images may themselves be restricted even when code is distributable.

The `source_cache/` created by the builder is therefore local-only. Do not add
it, dataset mappings, or retained images to the GitHub repository without a
separate dataset-rights decision.

## Python and system dependencies

The requirement files name or imply the following direct runtime packages:

```text
NumPy, Pillow, PyYAML, tqdm, PyTorch, torchvision, Transformers,
Accelerate, safetensors, sentencepiece, SciPy, scikit-learn, pandas, timm
```

The unique builder additionally imports the Hugging Face `datasets` package.
It is pinned as `datasets==2.19.2` in `requirements/dataset-builder.txt`, based
on the captured environment, and must be included in the dependency/license
inventory. Fresh dependency resolution, a transitive SBOM, authoritative
license verification, and required license/notice collection have not been
completed. Python itself, CUDA libraries, GPU drivers, and all transitive
packages also have their own terms.

Package installation from an index does not permit copying package source or
binaries into this repository. Before release, generate a dependency bill of
materials from each supported environment, resolve each distribution to its
authoritative metadata, retain license texts where required, and review
binary/CUDA redistribution separately. This ledger does not assert licenses
for any named package.

## Results, figures, and generated artifacts

Facts and numeric measurements may have different legal treatment from source
code, but the audit makes no legal conclusion. Raw predictions, adversarial
images, activation features, cached source images, qualitative panels, and
embedded prompts can contain or derive from third-party model/dataset content.
Their inclusion in a local-only archive does not automatically authorize
public redistribution.

Before publishing any generated artifact, establish:

- the exact source dataset/model lineage;
- whether source images, annotations, prompts, or model outputs are embedded;
- whether the relevant model/data terms allow redistribution and modification;
- whether human-subject, privacy, trademark, or content restrictions apply;
- the attribution/citation required for the artifact and its sources.

Small aggregate CSV/JSON summaries in `results/paper/` still require owner and
legal review; their small size is a repository-hygiene property, not a license
clearance.

## First-party status is also unresolved

No project-level license was found for the audited first-party source. Until
the owner confirms authorship/contributor rights and selects a project license,
the absence of a license means downstream users should not assume permission
to copy, modify, or redistribute the code. A future first-party license cannot
override incompatible third-party terms.

## Required component record

For every component retained after review, add a row to a machine-readable
software bill of materials and a human-readable notice with at least:

```text
component name
component type (code/model/dataset/library/result)
authoritative upstream locator
pinned revision or release
copyright holder(s), copied from authoritative evidence
license identifier and full license-text path
NOTICE/attribution/citation requirements
files in this release that contain or adapt the component
local patch or similarity-review path
redistribution decision and reviewer/date
```

Do not fill unknown fields by guesswork. Use `UNRESOLVED` and keep the public
release gate closed.

## Current publication disposition

| Payload | Current disposition |
|---|---|
| This source snapshot | **Hold**: first-party license and third-party code lineage unresolved |
| Model weights/processors | **Exclude** |
| Datasets, mappings, images, source cache | **Exclude** pending separate rights review |
| Raw predictions, adversarial images, activations | **Local-only; public redistribution unresolved** |
| Small result summaries | **Technical GitHub candidate only; legal review still required** |
| Historical DivPrune source/results | **Not part of formal GitHub scope; review separately before any distribution** |

No `LICENSE`, `NOTICE`, or populated `third_party/` bundle should be synthesized
from memory. Add those files only after the evidence and approvals described
above are available.
