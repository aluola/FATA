# Licensing review required before publication

## Release gate: closed

This directory is a technical release candidate, not a legally cleared
open-source release. Do not upload it to a public repository, attach an
open-source license, or invite redistribution until the project owner and an
appropriate legal reviewer complete the checks in this document.

No project-wide license is assigned by this snapshot. Nothing here should be
read as permission to use, copy, modify, or redistribute code, model weights,
datasets, results, or generated media.

## Evidence recovered by the audit

The read-only source audit found:

- no `.git` repository metadata in the audited source/result scope;
- no recoverable remote, branch, commit, tag, dirty state, or submodule state;
- no `LICENSE*`, `COPYING*`, or `NOTICE*` file in the audited source
  candidates;
- no authoritative upstream locator, pinned revision, and license bundle for
  the local VisionZIP, VisPruner, PruMerge, FlowCut, DivPrune, CAA, CAGE, or
  HiddenDetect-related implementations;
- no contributor/assignment record establishing that every first-party file
  may be licensed by the current owner.

SHA-256 file identities and copy manifests establish which local bytes were
audited. They do not establish authorship, a source commit, or redistribution
rights. Modification times and names such as `final`, `frozen`, `SOURCE`, or
`backup` are not substitutes for repository provenance.

## Material requiring a rights decision

### First-party source

The owner must identify the author and contribution history for every file
proposed for publication, confirm that contributor agreements or employment
terms permit licensing, and decide which files are genuinely first-party. A
project license may be selected only after that review.

The following release areas contain first-party or mixed-origin candidates and
must be classified file by file:

```text
src/fata/attacks/
src/fata/compression/
src/fata/data/
src/fata/detection/
src/fata/evaluation/
src/fata/runtimes/
src/fata/utils/
tests/
configs/
```

Tests, configs, documentation, and generated wrappers are still copyrighted
works; creating them during packaging does not resolve the rights in code they
wrap or quote.

### Compressor implementations

Local compressor code exists in the LLaVA combined implementation, Qwen
dynamic compressors/surrogates, and InternVL adapters. Comments describe some
blocks as official or ported, but the audit found no evidence identifying the
exact upstream bytes and terms. Resolve VisionZIP, VisPruner, PruMerge, and
FlowCut independently, including whether model-family ports are derivative
works and whether required notices differ.

DivPrune is excluded from the formal public registry but remains in historical
and local-only provenance. Exclusion from the paper matrix does not authorize
redistribution of its old source or outputs.

### Attacks and detectors

CAA, CAGE, Feature Squeezing, Mahalanobis-Max, and the HiddenDetect-inspired
ML-ATD implementation require the same upstream/source-similarity review. The
current evidence supports only the descriptive phrase “HiddenDetect-inspired”;
it does not support a claim that ML-ATD is an authorized or version-pinned
reproduction of a particular upstream implementation.

### Models, processors, and tokenizers

The GitHub payload must not include model weights, processor/tokenizer assets,
or cached model files. For every documented model family, record its
authoritative distribution, exact revision/hash, license, acceptable-use terms,
base-model dependencies, and citation. If redistribution is not explicitly
permitted, require users to obtain it separately.

### Datasets and source caches

The GitHub payload must not include retained images, mappings, downloaded
archives, annotations, or the builder's source cache until a separate data
rights review covers TextVQA, VQAv2, COCO imagery, and ScienceQA. Review both
the dataset annotations and the underlying image/content rights. Derived
hashes, sample IDs, prompts, thumbnails, and qualitative case studies also need
a publication decision.

### Results and generated artifacts

Review raw predictions, adversarial images, activation features, checkpoints,
and small aggregate tables separately. Local-only placement reduces accidental
upload risk but does not grant publication rights. Confirm whether model and
dataset terms allow the intended generation, retention, modification, and
redistribution.

### Dependencies and binaries

Create a dependency bill of materials for every supported environment,
including Python, direct and transitive Python packages, CUDA/runtime
libraries, and any bundled binary. Review source and binary terms separately.
Do not copy installed packages into the repository as a substitute for a
dependency declaration.

The builder dependency is now explicitly pinned as `datasets==2.19.2` in
`requirements/dataset-builder.txt`, matching the captured environment. That
technical pin is not legal clearance: a fresh resolver install, transitive
software bill of materials, authoritative license metadata, and required
license/notice texts have not been verified.

## Required upstream review for each component

For each named compression, attack, detector, model, dataset, or incorporated
library, the reviewer must record:

1. authoritative upstream repository/model-card/dataset-page locator;
2. exact commit, tag, release, or content hashes;
3. license text applicable at that exact revision;
4. copyright and attribution text copied from authoritative evidence;
5. NOTICE, citation, patent, trademark, copyleft, and source-offer obligations;
6. files or outputs in this release that contain, adapt, or depend on it;
7. a source comparison and local patch for adapted code;
8. compatibility with the proposed first-party project license;
9. whether public source/result redistribution is permitted;
10. reviewer, decision date, and retained evidence location.

Unknown fields must remain `UNRESOLVED`; they may not be completed from model
memory, a paper title, or a comment in the local code.

## Project-license decision

Only after contributor rights and all incorporated third-party terms are
known should the owner choose a license for the first-party portions. The
decision must answer:

- which exact paths are covered by the project license;
- which paths remain under third-party licenses;
- whether any third-party terms are incompatible with the proposed license or
  distribution channel;
- whether the license covers code only or also documentation/configuration;
- how results, model outputs, and data-derived artifacts are treated;
- whether patent, trademark, or name-use language is required;
- which license and notice files must accompany source and binary forms.

Do not add a familiar permissive license simply because the intended venue is
GitHub or peer review. Do not label the whole tree with a single SPDX
identifier unless the file-level review supports it.

## Remediation options

For an unresolved or incompatible component, choose and document one of these
outcomes:

| Outcome | Required action |
|---|---|
| Retain | add verified upstream identity, license/notice/citation, local diff, and approval |
| Depend externally | remove copied implementation, pin a compatible upstream dependency, and document installation |
| Clean-room reimplement | preserve only unprotected interface/algorithm requirements as permitted, document separation and review, then validate scientifically |
| Exclude from public payload | remove it from the GitHub candidate while retaining an audit-only reference where legally permissible |
| Obtain permission | preserve written authorization and its scope/conditions with the release records |
| Abandon publication | keep the component and its dependent claims out of the release if no lawful resolution exists |

Removing a file may also remove the ability to reproduce a result. If so,
state that limitation explicitly; do not replace missing provenance with an
unverified implementation and retain the old number.

## Minimum publication checklist

The release gate may be opened only after all items are complete:

- [ ] owner/contributor rights for first-party files established;
- [ ] per-file origin classification completed;
- [ ] VisionZIP provenance/license/patch resolved;
- [ ] VisPruner provenance/license/patch resolved;
- [ ] PruMerge provenance/license/patch resolved;
- [ ] FlowCut provenance/license/patch resolved;
- [ ] CAA provenance/license/patch resolved;
- [ ] CAGE provenance/license/patch resolved;
- [ ] HiddenDetect/ML-ATD provenance and citation resolved;
- [ ] Feature Squeezing and Mahalanobis code lineage resolved;
- [ ] model/processor/tokenizer terms recorded; weights excluded or authorized;
- [ ] TextVQA, VQAv2, COCO, and ScienceQA terms recorded; data excluded or
      authorized;
- [ ] Python/system dependency bill of materials and required notices created;
- [ ] raw and aggregate result/artifact redistribution reviewed;
- [ ] compatible project license selected for first-party material only;
- [ ] full license texts, notices, attributions, and citations installed;
- [ ] `THIRD_PARTY_NOTICES.md` replaced or extended with verified facts;
- [ ] GitHub upload manifest regenerated after every inclusion/exclusion;
- [ ] credential, absolute-path, large-file, model, dataset, and cache scan
      passes;
- [ ] legal reviewer and project owner sign off on the exact manifest hash.

## Sign-off record to complete

```text
Release manifest SHA-256: UNRESOLVED
First-party license: UNRESOLVED
Third-party notice bundle: UNRESOLVED
Dataset redistribution decision: UNRESOLVED
Model redistribution decision: UNRESOLVED
Generated-artifact decision: UNRESOLVED
Project owner approval: UNRESOLVED
Legal reviewer approval: UNRESOLVED
Approval date: UNRESOLVED
```

Until every required field is resolved for the exact release manifest, the
correct public statement is: **licensing review required; redistribution not
cleared**. Technical limitations and scientific claim boundaries remain
separate from this legal review and are retained in the companion audit.
