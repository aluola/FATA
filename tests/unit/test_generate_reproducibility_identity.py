from __future__ import annotations

import copy
import json

import pytest
from PIL import Image

from fata.data.unique_builder import generate_reproducibility as repro
from fata.data.unique_builder.build_core import (
    CONSTRUCTION_PROTOCOL_VERSION,
    BuildConfig,
    build_dataset,
)
from fata.data.unique_builder.common import (
    CANONICAL_HASH_DESCRIPTION,
    DATASET_NAMES,
    JPEG_SETTINGS,
    canonical_rgb_sha256_path,
    dhash_path,
    sha256_file,
)
from fata.data.unique_builder.sources import Candidate, SourceBundle


def _model_manifest():
    files = [
        {"path": "config.json", "size_bytes": 10, "sha256": "a" * 64},
        {"path": "weights/model.bin", "size_bytes": 20, "sha256": "b" * 64},
    ]
    return {
        "path": "/models/fixture",
        "algorithm": "sha256-per-file-and-canonical-json-aggregate",
        "aggregate_sha256": repro._stable_json_digest(files),
        "files": files,
    }


def _protocol():
    return {
        "protocol_version": CONSTRUCTION_PROTOCOL_VERSION,
        "scorers": {
            "open": "legacy-official-vqa-process-plus-truth-substring-v1",
            "multiple_choice": (
                "legacy-letter-regex-then-ground-truth-substring-v1;"
                "VQAv2=A-D;ScienceQA=A-F"
            ),
        },
        "canonical_hash": CANONICAL_HASH_DESCRIPTION,
        "jpeg_settings": JPEG_SETTINGS,
        "runner": {
            "runner": "LlavaConstructionRunner",
            "model": _model_manifest(),
            "runtime": {
                "python": "3.10",
                "platform": "linux",
                "torch": "fixture",
                "transformers": "fixture",
                "datasets": "fixture",
                "numpy": "fixture",
                "pillow": "fixture",
                "libjpeg": "fixture",
                "cuda_runtime": "fixture",
                "cudnn": 1,
                "gpu": {
                    "name": "fixture GPU",
                    "total_memory_bytes": 1,
                    "compute_capability": [8, 0],
                },
            },
            "visual_patch_tokens": 576,
            "blind_control": {
                "name": "blind_black_image_control",
                "image_mode": "RGB",
                "image_size": [336, 336],
                "color": [0, 0, 0],
                "passes_through_visual_encoder": True,
            },
            "generation": {
                "max_new_tokens": 32,
                "do_sample": False,
                "num_beams": 1,
                "legacy_textvqa_temperature_zero": True,
            },
        },
    }


def _source_checkpoint(*, seed=20260904):
    protocol = _protocol()
    fingerprint = repro._stable_json_digest(protocol)
    source = {
        "construction_protocol": protocol,
        "construction_protocol_fingerprint": fingerprint,
    }
    checkpoint = {
        "construction_protocol_fingerprint": fingerprint,
        "seed": seed,
    }
    return source, checkpoint


def _formal_validation_gate_fixture():
    return {
        "hard_check_failures": [],
        "near_duplicate_audit": {
            "dhash_hamming_distance_threshold": 4,
            "unresolved_high_confidence_group_count": 0,
        },
        "overlap": {
            "pairwise": [
                {
                    "dataset_a": left,
                    "dataset_b": right,
                    "shared_canonical_rgb_sha256_count": 0,
                    "shared_source_canonical_rgb_sha256_count": 0,
                    "shared_source_image_id_count": 0,
                    "shared_canonical_rgb_sha256": [],
                    "shared_source_canonical_rgb_sha256": [],
                    "shared_source_image_ids": [],
                }
                for index, left in enumerate(DATASET_NAMES)
                for right in DATASET_NAMES[index + 1 :]
            ],
            "cross_dataset_exact_rgb_group_count": 0,
            "cross_dataset_exact_rgb_groups": [],
        },
    }


@pytest.mark.parametrize("threshold", (0, 64, True))
def test_formal_validation_gate_locks_near_duplicate_threshold(threshold):
    validation = _formal_validation_gate_fixture()
    validation["near_duplicate_audit"][
        "dhash_hamming_distance_threshold"
    ] = threshold

    with pytest.raises(RuntimeError, match="distance threshold 4"):
        repro._validate_formal_validation_gate(validation)


@pytest.mark.parametrize(
    "field",
    (
        "shared_canonical_rgb_sha256",
        "shared_source_canonical_rgb_sha256",
        "shared_source_image_ids",
    ),
)
def test_formal_validation_gate_rejects_forged_pairwise_details(field):
    validation = _formal_validation_gate_fixture()
    validation["overlap"]["pairwise"][0][field] = ["forged"]

    with pytest.raises(RuntimeError, match="nonzero exact overlap"):
        repro._validate_formal_validation_gate(validation)


def test_formal_validation_gate_requires_empty_hard_failures():
    validation = _formal_validation_gate_fixture()
    validation["hard_check_failures"] = ["forged passing report"]

    with pytest.raises(RuntimeError, match="hard_check_failures"):
        repro._validate_formal_validation_gate(validation)


def test_formal_exclusion_lineage_requires_exact_cumulative_sets():
    source_hash = "a" * 64
    saved_hash = "b" * 64
    open_id = "17"
    exclusions = {
        "source_image_ids": [open_id],
        "source_canonical_rgb_sha256": [source_hash],
        "saved_canonical_rgb_sha256": [saved_hash],
    }

    repro._validate_formal_exclusion_lineage(
        "VQAv2_MC",
        exclusions,
        prior_source_hashes={source_hash},
        prior_saved_hashes={saved_hash},
        vqav2_open_source_ids={open_id},
    )
    forged = copy.deepcopy(exclusions)
    forged["source_image_ids"] = []
    with pytest.raises(RuntimeError, match="cumulative exclusion lineage"):
        repro._validate_formal_exclusion_lineage(
            "VQAv2_MC",
            forged,
            prior_source_hashes={source_hash},
            prior_saved_hashes={saved_hash},
            vqav2_open_source_ids={open_id},
        )


@pytest.mark.parametrize("seed", (True, False, -1))
def test_protocol_identity_rejects_boolean_or_negative_seed(seed):
    source, checkpoint = _source_checkpoint(seed=seed)

    with pytest.raises(RuntimeError, match="invalid seed"):
        repro._validate_protocol_and_seed(
            source, checkpoint, dataset="TextVQA_Open"
        )


def test_protocol_identity_rejects_self_consistent_nonfrozen_seed():
    source, checkpoint = _source_checkpoint(seed=17)

    with pytest.raises(RuntimeError, match="frozen seed"):
        repro._validate_protocol_and_seed(
            source, checkpoint, dataset="TextVQA_Open"
        )


def test_protocol_identity_recomputes_body_fingerprint():
    source, checkpoint = _source_checkpoint()
    source["construction_protocol"]["runner"]["generation"]["do_sample"] = True

    with pytest.raises(RuntimeError, match="body does not match fingerprint"):
        repro._validate_protocol_and_seed(
            source, checkpoint, dataset="TextVQA_Open"
        )


def test_protocol_identity_rejects_self_consistent_nonformal_body():
    source, checkpoint = _source_checkpoint()
    source["construction_protocol"]["protocol_version"] = "attacker-protocol"
    forged = repro._stable_json_digest(source["construction_protocol"])
    source["construction_protocol_fingerprint"] = forged
    checkpoint["construction_protocol_fingerprint"] = forged

    with pytest.raises(RuntimeError, match="formal protocol version"):
        repro._validate_protocol_and_seed(
            source, checkpoint, dataset="TextVQA_Open"
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda model: model.update(aggregate_sha256="not-a-digest"),
        lambda model: model.update(aggregate_sha256="f" * 64),
        lambda model: model["files"][0].update(size_bytes=True),
        lambda model: model["files"][0].update(sha256="A" * 64),
        lambda model: model["files"].reverse(),
        lambda model: model["files"][0].update(path="../config.json"),
        lambda model: model.update(unexpected=True),
    ),
)
def test_model_manifest_requires_exact_structure_and_recomputed_aggregate(mutate):
    model = _model_manifest()
    mutate(model)

    with pytest.raises(RuntimeError):
        repro._validate_model_manifest(model, dataset="TextVQA_Open")


def _write_minimal_completed_tree(tmp_path):
    dataset_root = tmp_path / "datasets"
    reports_dir = tmp_path / "reports"
    manifests = dataset_root / "manifests"
    manifests.mkdir(parents=True)
    reports_dir.mkdir()
    source_cache = dataset_root / "source_cache"
    source_cache.mkdir()
    (source_cache / "coco_val2014").mkdir()
    (source_cache / "coco_val2014_identity").mkdir()
    (reports_dir / "final_validation.json").write_text(
        json.dumps(
            {
                "overall_pass": True,
                "hard_check_failures": [],
                "expected_count": 1000,
                "dataset_root": str(dataset_root.resolve()),
                "near_duplicate_audit": {
                    "dhash_hamming_distance_threshold": 4,
                    "unresolved_high_confidence_group_count": 0,
                },
                "overlap": {
                    "pairwise": [
                        {
                            "dataset_a": left,
                            "dataset_b": right,
                            "shared_canonical_rgb_sha256_count": 0,
                            "shared_source_canonical_rgb_sha256_count": 0,
                            "shared_source_image_id_count": 0,
                            "shared_canonical_rgb_sha256": [],
                            "shared_source_canonical_rgb_sha256": [],
                            "shared_source_image_ids": [],
                        }
                        for index, left in enumerate(DATASET_NAMES)
                        for right in DATASET_NAMES[index + 1 :]
                    ],
                    "cross_dataset_exact_rgb_group_count": 0,
                    "cross_dataset_exact_rgb_groups": [],
                },
                "datasets": {
                    dataset: {
                        "pass": True,
                        "counts": {
                            "mapping_rows": 1000,
                            "sample_sidecar_rows": 1000,
                            "actual_files_in_images_directory": 1000,
                            "existing_referenced_images": 1000,
                        },
                    }
                    for dataset in DATASET_NAMES
                },
            }
        ),
        encoding="utf-8",
    )
    (reports_dir / "final_validation.md").write_text("fixture\n", encoding="utf-8")
    (reports_dir / "image_hash_manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (reports_dir / "cross_dataset_overlap.csv").write_text("fixture\n", encoding="utf-8")
    protocol = _protocol()
    fingerprint = repro._stable_json_digest(protocol)
    for dataset in DATASET_NAMES:
        dataset_dir = dataset_root / dataset
        dataset_dir.mkdir()
        source = {
            "dataset": dataset,
            "construction_protocol": copy.deepcopy(protocol),
            "construction_protocol_fingerprint": fingerprint,
            "candidate_pool_fingerprint": "c" * 64,
            "source_metadata": {},
            "source_metadata_stable_fingerprint": repro._stable_json_digest({}),
            "build_exclusions": {
                "source_image_ids": [],
                "source_canonical_rgb_sha256": [],
                "saved_canonical_rgb_sha256": [],
            },
        }
        stats = {
            "dataset": dataset,
            "target": 1000,
            "seed": 20260904,
            "retained_count": 1000,
        }
        checkpoint = {
            "dataset": dataset,
            "target": 1000,
            "seed": 20260904,
            "candidate_pool_fingerprint": "c" * 64,
            "construction_protocol_fingerprint": fingerprint,
            "selection_stats": stats,
        }
        (manifests / f"{dataset}_source.json").write_text(
            json.dumps(source), encoding="utf-8"
        )
        (manifests / f"{dataset}_checkpoint.json").write_text(
            json.dumps(checkpoint), encoding="utf-8"
        )
        (manifests / f"{dataset}_selection_stats.json").write_text(
            json.dumps(stats), encoding="utf-8"
        )
        for suffix in ("candidate_pool", "screening", "samples"):
            (manifests / f"{dataset}_{suffix}.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
        (dataset_dir / f"{dataset}_mapping.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )
    return dataset_root, reports_dir


def test_generate_rejects_distinct_protocol_bodies_even_if_each_digest_is_valid(
    tmp_path, monkeypatch
):
    dataset_root, reports_dir = _write_minimal_completed_tree(tmp_path)
    dataset = DATASET_NAMES[1]
    source_path = dataset_root / "manifests" / f"{dataset}_source.json"
    checkpoint_path = dataset_root / "manifests" / f"{dataset}_checkpoint.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    source["construction_protocol"]["runner"]["runtime"]["python"] = "3.11-tampered"
    forged = repro._stable_json_digest(source["construction_protocol"])
    source["construction_protocol_fingerprint"] = forged
    checkpoint["construction_protocol_fingerprint"] = forged
    source_path.write_text(json.dumps(source), encoding="utf-8")
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    monkeypatch.setattr(repro, "_gpu_manifest", lambda: [])
    monkeypatch.setattr(
        repro,
        "_validate_image_hash_manifest",
        lambda *args, **kwargs: {"path": "fixture", "size_bytes": 3, "sha256": "d" * 64},
    )
    monkeypatch.setattr(
        repro, "_validate_completed_build_artifacts", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        repro, "_validate_vqa_source_archives", lambda *args, **kwargs: None
    )

    with pytest.raises(RuntimeError, match="non-identical construction protocol bodies"):
        repro.generate(dataset_root, reports_dir=reports_dir)


def _image_manifest_row(dataset_root, dataset="TextVQA_Open"):
    image_dir = dataset_root / dataset / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / f"{dataset}_0000.jpg"
    Image.new("RGB", (3, 2), (12, 34, 56)).save(image_path, format="JPEG")
    with Image.open(image_path) as image:
        width, height = image.size
    return image_path, {
        "dataset": dataset,
        "mapping_index": 0,
        "image_filename": f"images/{dataset}_0000.jpg",
        "width": width,
        "height": height,
        "file_size": image_path.stat().st_size,
        "saved_image_sha256": sha256_file(image_path),
        "canonical_rgb_sha256": canonical_rgb_sha256_path(image_path),
        "perceptual_hash": dhash_path(image_path),
        "source_dataset": "fixture",
        "source_split": "validation",
        "source_index": 0,
        "source_question_id": "q0",
        "source_image_id": "i0",
        "source_canonical_rgb_sha256": "e" * 64,
    }


def test_image_hash_manifest_rejects_jpeg_replaced_after_validation(tmp_path):
    dataset_root = tmp_path / "datasets"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    image_path, row = _image_manifest_row(dataset_root)
    (reports_dir / "image_hash_manifest.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    repro._validate_image_hash_manifest(
        dataset_root,
        reports_dir,
        expected_count=1,
        dataset_names=("TextVQA_Open",),
    )

    Image.new("RGB", (3, 2), (99, 88, 77)).save(image_path, format="JPEG")

    with pytest.raises(RuntimeError, match="validated image (size|bytes|RGB) changed"):
        repro._validate_image_hash_manifest(
            dataset_root,
            reports_dir,
            expected_count=1,
            dataset_names=("TextVQA_Open",),
        )


def test_image_hash_manifest_rejects_truncated_tail(tmp_path):
    dataset_root = tmp_path / "datasets"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _, row = _image_manifest_row(dataset_root)
    (reports_dir / "image_hash_manifest.jsonl").write_bytes(
        (json.dumps(row) + "\n" + '{"truncated"').encode("utf-8")
    )

    with pytest.raises(RuntimeError, match="image hash manifest"):
        repro._validate_image_hash_manifest(
            dataset_root,
            reports_dir,
            expected_count=1,
            dataset_names=("TextVQA_Open",),
        )


@pytest.mark.parametrize("kind", ("renamed_png", "grayscale_jpeg"))
def test_image_hash_manifest_requires_rgb_jpeg_encoding(tmp_path, kind):
    dataset_root = tmp_path / "datasets"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    image_path, row = _image_manifest_row(dataset_root)
    if kind == "renamed_png":
        Image.new("RGB", (3, 2), (1, 2, 3)).save(image_path, format="PNG")
    else:
        Image.new("L", (3, 2), 127).save(image_path, format="JPEG")
    row.update(
        file_size=image_path.stat().st_size,
        saved_image_sha256=sha256_file(image_path),
        canonical_rgb_sha256=canonical_rgb_sha256_path(image_path),
        perceptual_hash=dhash_path(image_path),
    )
    (reports_dir / "image_hash_manifest.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="not an RGB JPEG"):
        repro._validate_image_hash_manifest(
            dataset_root,
            reports_dir,
            expected_count=1,
            dataset_names=("TextVQA_Open",),
        )


def test_completed_artifact_validation_reconstructs_stats_from_screening(tmp_path):
    dataset = "TextVQA_Open"
    output = tmp_path / "datasets"
    reports = tmp_path / "reports"
    reports.mkdir()
    selection_stats = {
        "source_splits": ["validation"],
        "source_qa_records_total": 1,
        "source_unique_images_total": 1,
        "task_prefilter_unique_images": 1,
        "invalid_metadata_count": 0,
        "yes_no_removed_count": 0,
        "missing_or_corrupt_images": 0,
        "source_id_duplicate_count": 0,
        "exact_image_hash_duplicate_count": 0,
        "unique_images_evaluated": 0,
        "blind_wrong_count": 0,
        "full_correct_count": 0,
        "target_reached_source_index": None,
        "retained_count": 0,
        "acceptance_rate": 0.0,
        "source_hash_candidates_seen": 0,
        "scanned_candidates": 0,
    }
    candidate = Candidate(
        source_dataset="fixture/source",
        source_split="validation",
        source_index=0,
        source_split_index=0,
        source_question_id="q0",
        source_image_id="i0",
        question="fixture question",
        answers=("answer",),
        task_type="open",
        prompt="fixture prompt",
        mapping_payload={
            "dataset": dataset,
            "type": "open",
            "question": "fixture question",
            "answers": ["answer"],
        },
        image_loader=lambda: Image.new("RGB", (3, 2), (12, 34, 56)),
    )

    class Runner:
        def infer_blind(self, *_args):
            return "wrong"

        def infer(self, *_args):
            return "answer"

    bundle = SourceBundle(dataset, [candidate], selection_stats)
    result = build_dataset(
        bundle,
        Runner(),
        BuildConfig(output, target=1, seed=20260904),
    )
    sample = result.samples[0]
    image_path = output / dataset / sample["image_filename"]
    with Image.open(image_path) as image:
        width, height = image.size
    image_row = {
        "dataset": dataset,
        "mapping_index": 0,
        "image_filename": sample["image_filename"],
        "width": width,
        "height": height,
        "file_size": image_path.stat().st_size,
        "saved_image_sha256": sample["saved_image_sha256"],
        "canonical_rgb_sha256": sample["saved_canonical_rgb_sha256"],
        "perceptual_hash": sample["perceptual_hash"],
        "source_dataset": sample["source_dataset"],
        "source_split": sample["source_split"],
        "source_index": sample["source_index"],
        "source_question_id": sample["source_question_id"],
        "source_image_id": sample["source_image_id"],
        "source_canonical_rgb_sha256": sample[
            "source_canonical_rgb_sha256"
        ],
    }
    (reports / "image_hash_manifest.jsonl").write_text(
        json.dumps(image_row) + "\n", encoding="utf-8"
    )
    manifests = output / "manifests"
    source = json.loads(
        (manifests / f"{dataset}_source.json").read_text(encoding="utf-8")
    )
    checkpoint = json.loads(
        (manifests / f"{dataset}_checkpoint.json").read_text(encoding="utf-8")
    )
    stats = json.loads(
        (manifests / f"{dataset}_selection_stats.json").read_text(
            encoding="utf-8"
        )
    )
    stats["full_correct_count"] = 2
    checkpoint["selection_stats"] = copy.deepcopy(stats)

    with pytest.raises(RuntimeError, match="selection_stats reconstruction"):
        repro._validate_completed_build_artifacts(
            output,
            reports,
            dataset,
            source,
            checkpoint,
            stats,
            expected_count=1,
        )
