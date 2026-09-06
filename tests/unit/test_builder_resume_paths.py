from __future__ import annotations

import copy
import json
import sys
import zipfile
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image

from fata.data.unique_builder import build_all
from fata.data.unique_builder import generate_reproducibility as repro
from fata.data.unique_builder import sources as builder_sources
from fata.data.unique_builder.build_core import (
    CHECKPOINT_FORMAT_VERSION,
    BuildConfig,
    BuildExclusions,
    _initial_stats,
    _load_resume_state,
    _stable_json_digest,
    _stable_source_metadata,
    build_dataset,
    collect_completed_sibling_exclusions,
    load_completed_resume,
    load_stored_build_exclusions,
)
from fata.data.unique_builder.common import (
    DATASET_NAMES,
    JPEG_SETTINGS,
    canonical_rgb_sha256,
    canonical_rgb_sha256_path,
    dhash_path,
    extract_zip_member_atomic,
    model_directory_fingerprint,
    sha256_file,
)
from fata.data.unique_builder.sources import (
    VQA_ANNOTATION_MEMBER,
    VQA_QUESTION_MEMBER,
    Candidate,
    RecoverableSourceError,
    SourceBundle,
    prepare_vqav2_source,
)


POOL_FINGERPRINT = "pool-fingerprint"
EXCLUSIONS_FINGERPRINT = "exclusions-fingerprint"
PROTOCOL_FINGERPRINT = "protocol-fingerprint"


def test_textvqa_pool_materialization_transient_error_fails_closed(monkeypatch):
    class MetadataRows:
        def __getitem__(self, index):
            return {
                "image_id": f"image-{index}",
                "question_id": index,
                "question": "fixture question",
                "answers": ["fixture answer"],
            }

    class FailingTextVQA:
        column_names = ["image_id", "question_id", "question", "image", "answers"]

        def __init__(self):
            self.attempts = 0

        def __len__(self):
            return 5000

        def remove_columns(self, _columns):
            return MetadataRows()

        def __getitem__(self, key):
            if key == "image_id":
                return [f"image-{index}" for index in range(len(self))]
            self.attempts += 1
            raise OSError("fixture transient TextVQA decode")

    dataset = FailingTextVQA()
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *_args, **_kwargs: dataset),
    )

    with pytest.raises(RecoverableSourceError, match="candidate pool was not frozen"):
        builder_sources.load_textvqa()
    assert dataset.attempts == builder_sources.SOURCE_POOL_MATERIALIZATION_ATTEMPTS


def test_scienceqa_pool_materialization_transient_error_fails_closed(monkeypatch):
    required = ["image", "question", "choices", "answer", "hint"]

    class FailingSplit:
        column_names = required

        def __init__(self):
            self.attempts = 0

        def __len__(self):
            return 1

        def __getitem__(self, _index):
            self.attempts += 1
            raise OSError("fixture transient ScienceQA decode")

    class EmptySplit:
        column_names = required

        def __len__(self):
            return 0

    train = FailingSplit()
    dataset = {"train": train, "validation": EmptySplit(), "test": EmptySplit()}
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *_args, **_kwargs: dataset),
    )

    with pytest.raises(RecoverableSourceError, match="candidate pool was not frozen"):
        builder_sources.load_scienceqa()
    assert train.attempts == builder_sources.SOURCE_POOL_MATERIALIZATION_ATTEMPTS


@pytest.mark.parametrize("target", (True, False, 1.5, "1", 0, -1))
def test_build_config_requires_exact_positive_integer_target(tmp_path, target):
    with pytest.raises((TypeError, ValueError)):
        BuildConfig(tmp_path, target=target)


@pytest.mark.parametrize("seed", (True, False, 1.5, "1", -1))
def test_build_config_requires_exact_nonnegative_integer_seed(tmp_path, seed):
    with pytest.raises((TypeError, ValueError)):
        BuildConfig(tmp_path, seed=seed)


@pytest.mark.parametrize("resume", (0, 1, None, "true"))
def test_build_config_requires_exact_boolean_resume(tmp_path, resume):
    with pytest.raises(TypeError):
        BuildConfig(tmp_path, resume=resume)


def test_build_config_requires_valid_exclusions_instance(tmp_path):
    with pytest.raises(TypeError, match="BuildExclusions"):
        BuildConfig(tmp_path, exclusions=frozenset())


@pytest.mark.parametrize(
    "kwargs",
    (
        {"source_image_ids": "not-a-collection"},
        {"source_hashes": "a" * 64},
        {"saved_hashes": "b" * 64},
    ),
)
def test_build_exclusions_rejects_string_as_collection(kwargs):
    with pytest.raises(TypeError, match="collection"):
        BuildExclusions.from_iterables(**kwargs)


def test_build_exclusions_rejects_malformed_hash_collections():
    with pytest.raises(ValueError, match="SHA-256"):
        BuildExclusions.from_iterables(source_hashes=["not-a-digest"])


@pytest.mark.parametrize(
    "override",
    (
        {"target": True},
        {"target": 1.5},
        {"seed": True},
        {"seed": 1.5},
        {"resume": 1},
    ),
)
def test_run_all_api_rejects_nonexact_scalar_types(tmp_path, override):
    kwargs = {
        "output": tmp_path / "output",
        "target": 1,
        "model": tmp_path / "model",
        "hf_cache": None,
        "device": "cpu",
        "seed": 0,
        "resume": False,
        "runner": object(),
    }
    kwargs.update(override)

    with pytest.raises((TypeError, ValueError)):
        build_all.run_all(**kwargs)


def _selection_stats(dataset: str):
    stats = {
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
    if dataset == "ScienceQA_MC":
        stats["split_composition"] = {
            "train": 0,
            "validation": 0,
            "test": 0,
        }
    return stats


def _candidate(
    dataset: str,
    *,
    position: int,
    source_id: str,
    source_hash: str | None,
):
    question = f"fixture question {position}"
    answers = (f"answer-{position}",)
    mapping = {
        "dataset": dataset,
        "question": question,
        "answers": list(answers),
        "fixture_position": position,
    }
    return Candidate(
        source_dataset="fixture/source",
        source_split="validation",
        source_index=100 + position,
        source_split_index=position,
        source_question_id=f"question-{position}",
        source_image_id=source_id,
        question=question,
        answers=answers,
        task_type="open",
        prompt=f"fixture prompt {position}",
        mapping_payload=mapping,
        image_loader=lambda: Image.new("RGB", (3, 2), (12, 34, 56)),
        source_canonical_rgb_sha256=source_hash,
        options=("one", "two"),
        ground_truth_letter="A",
        ground_truth_text="one",
        legacy_temperature_zero=True,
        provenance={"fixture": True, "position": position},
    )


def _screening_base(position: int, candidate: Candidate):
    return {
        "candidate_position": position,
        "source_dataset": candidate.source_dataset,
        "source_split": candidate.source_split,
        "source_index": candidate.source_index,
        "source_split_index": candidate.source_split_index,
        "source_question_id": candidate.source_question_id,
        "source_image_id": candidate.source_image_id,
        "reason": None,
        "accepted": False,
    }


class _FixtureRunner:
    def infer_blind(self, prompt, legacy_temperature_zero):
        return "fixture-wrong"

    def infer(self, image, prompt, legacy_temperature_zero):
        return "answer-0"


class _ModelManifestFixtureRunner(_FixtureRunner):
    def __init__(self, model_path):
        self.model_path = model_path

    def construction_manifest(self):
        return {
            "runner": "ModelManifestFixtureRunner",
            "model": model_directory_fingerprint(self.model_path),
        }


def _run_fresh_builder(
    tmp_path,
    *,
    target: int = 1,
    candidates: list[Candidate] | None = None,
    source_metadata: dict | None = None,
    runner=None,
):
    if candidates is None:
        candidates = [
            _candidate(
                "TextVQA_Open",
                position=0,
                source_id="fresh-source-0",
                source_hash=None,
            )
        ]
    bundle = SourceBundle(
        dataset_name="TextVQA_Open",
        candidates=candidates,
        selection_stats=_selection_stats("TextVQA_Open"),
        source_metadata=source_metadata or {},
    )
    runner = _FixtureRunner() if runner is None else runner
    config = BuildConfig(output_root=tmp_path, target=target, seed=20260904)
    result = build_dataset(bundle, runner, config, fail_if_short=False)
    return bundle, runner, result


def test_recoverable_source_error_does_not_advance_durable_checkpoint(tmp_path):
    candidate = _candidate(
        "TextVQA_Open",
        position=0,
        source_id="retry-source-0",
        source_hash=None,
    )

    def unavailable():
        raise RecoverableSourceError("fixture transient failure")

    bundle = SourceBundle(
        dataset_name="TextVQA_Open",
        candidates=[replace(candidate, image_loader=unavailable)],
        selection_stats=_selection_stats("TextVQA_Open"),
    )
    config = BuildConfig(tmp_path, target=1, seed=20260904)

    with pytest.raises(RecoverableSourceError, match="resume will retry"):
        build_dataset(bundle, _FixtureRunner(), config)

    checkpoint_path = tmp_path / "manifests" / "TextVQA_Open_checkpoint.json"
    screening_path = tmp_path / "manifests" / "TextVQA_Open_screening.jsonl"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["next_candidate_position"] == 0
    assert checkpoint["selection_stats"]["scanned_candidates"] == 0
    assert screening_path.read_text(encoding="utf-8") == ""

    bundle.candidates[0] = candidate
    resumed = build_dataset(
        bundle,
        _FixtureRunner(),
        replace(config, resume=True),
    )
    assert resumed.target_reached is True


def test_shared_root_collects_only_validated_completed_sibling_identities(tmp_path):
    _, _, result = _run_fresh_builder(tmp_path)

    exclusions = collect_completed_sibling_exclusions(
        tmp_path,
        dataset_name="ScienceQA_MC",
        target=1,
        seed=20260904,
    )

    assert exclusions.source_canonical_rgb_sha256 == result.accepted_source_hashes
    assert exclusions.saved_canonical_rgb_sha256 == result.accepted_saved_hashes
    assert exclusions.source_image_ids == frozenset()


def test_resume_loads_original_immutable_exclusions_from_source_manifest(tmp_path):
    _run_fresh_builder(tmp_path)

    assert load_stored_build_exclusions(
        tmp_path, dataset_name="TextVQA_Open"
    ) == BuildExclusions()


def test_shared_root_rejects_incomplete_sibling_instead_of_ignoring_it(tmp_path):
    (tmp_path / "TextVQA_Open").mkdir()

    with pytest.raises(RuntimeError, match="incomplete TextVQA_Open build"):
        collect_completed_sibling_exclusions(
            tmp_path,
            dataset_name="ScienceQA_MC",
            target=1,
            seed=20260904,
        )


def test_shared_root_rejects_existing_cross_dataset_saved_duplicate(tmp_path):
    _run_fresh_builder(tmp_path)
    science_candidate = _candidate(
        "ScienceQA_MC",
        position=0,
        source_id="science-source-0",
        source_hash=None,
    )
    science_bundle = SourceBundle(
        dataset_name="ScienceQA_MC",
        candidates=[science_candidate],
        selection_stats=_selection_stats("ScienceQA_MC"),
    )
    build_dataset(
        science_bundle,
        _FixtureRunner(),
        BuildConfig(tmp_path, target=1, seed=20260904),
    )

    with pytest.raises(RuntimeError, match="cross-dataset .* RGB duplicate"):
        collect_completed_sibling_exclusions(
            tmp_path,
            dataset_name="VQAv2_Open",
            target=1,
            seed=20260904,
        )


@pytest.mark.parametrize("entrypoint", ("build", "complete_resume"))
def test_programmatic_dataset_name_cannot_escape_output_root(
    tmp_path, entrypoint
):
    output = tmp_path / "output"
    escaped = tmp_path / "escape"
    bundle = SourceBundle(
        dataset_name="../escape",
        candidates=[],
        selection_stats=_selection_stats("TextVQA_Open"),
    )
    config = BuildConfig(
        output_root=output,
        target=1,
        seed=20260904,
        resume=entrypoint == "complete_resume",
    )

    with pytest.raises(ValueError, match="formal datasets"):
        if entrypoint == "build":
            build_dataset(bundle, _FixtureRunner(), config, fail_if_short=False)
        else:
            load_completed_resume(
                bundle,
                config,
                current_runner=_FixtureRunner(),
            )

    assert not escaped.exists()
    assert not output.exists()


@pytest.mark.parametrize("owned_path", ("images", ".stage", "screening"))
def test_builder_rejects_symlinked_owned_paths_without_touching_target(
    tmp_path, owned_path
):
    output = tmp_path / "output"
    dataset_dir = output / "TextVQA_Open"
    manifests_dir = output / "manifests"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    if owned_path in {"images", ".stage"}:
        dataset_dir.mkdir(parents=True)
        (dataset_dir / owned_path).symlink_to(external, target_is_directory=True)
    else:
        manifests_dir.mkdir(parents=True)
        (manifests_dir / "TextVQA_Open_screening.jsonl").symlink_to(sentinel)
    bundle = SourceBundle(
        dataset_name="TextVQA_Open",
        candidates=[],
        selection_stats=_selection_stats("TextVQA_Open"),
    )

    with pytest.raises(ValueError, match="symlink"):
        build_dataset(
            bundle,
            _FixtureRunner(),
            BuildConfig(output, target=1, seed=20260904),
            fail_if_short=False,
        )

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert sorted(path.name for path in external.iterdir()) == ["sentinel.txt"]


def test_builder_exclusively_creates_predictable_stage_file(tmp_path):
    output = tmp_path / "output"
    stage_dir = output / "TextVQA_Open" / ".stage"
    stage_dir.mkdir(parents=True)
    sentinel = tmp_path / "external-sentinel.jpg"
    sentinel.write_bytes(b"do-not-overwrite")
    source_hash = canonical_rgb_sha256(
        Image.new("RGB", (3, 2), (12, 34, 56))
    )
    stage_path = stage_dir / f"candidate_000000_{source_hash[:16]}.jpg"
    stage_path.symlink_to(sentinel)
    bundle = SourceBundle(
        dataset_name="TextVQA_Open",
        candidates=[
            _candidate(
                "TextVQA_Open",
                position=0,
                source_id="stage-source-0",
                source_hash=None,
            )
        ],
        selection_stats=_selection_stats("TextVQA_Open"),
    )

    with pytest.raises(FileExistsError):
        build_dataset(
            bundle,
            _FixtureRunner(),
            BuildConfig(output, target=1, seed=20260904),
        )

    assert sentinel.read_bytes() == b"do-not-overwrite"
    assert stage_path.is_symlink()


def _resume_fixture(tmp_path, dataset: str, image_filename: str | None = None):
    dataset_dir = tmp_path / dataset
    images_dir = dataset_dir / "images"
    stage_dir = dataset_dir / ".stage"
    manifests_dir = tmp_path / "manifests"
    images_dir.mkdir(parents=True)
    stage_dir.mkdir()
    manifests_dir.mkdir()

    expected_filename = f"images/{dataset}_0000.jpg"
    expected_path = dataset_dir / expected_filename
    Image.new("RGB", (3, 2), (12, 34, 56)).save(expected_path, format="JPEG")
    saved_byte_hash = sha256_file(expected_path)
    saved_rgb_hash = canonical_rgb_sha256_path(expected_path)
    perceptual_hash = dhash_path(expected_path)
    source_id = f"{dataset}-source-0"
    candidate = _candidate(
        dataset,
        position=0,
        source_id=source_id,
        source_hash=saved_rgb_hash,
    )
    bundle = SourceBundle(
        dataset_name=dataset,
        candidates=[candidate],
        selection_stats=_selection_stats(dataset),
    )
    config = BuildConfig(output_root=tmp_path, target=1, seed=20260904, resume=True)
    mapping = copy.deepcopy(candidate.mapping_payload)
    mapping["image_filename"] = expected_filename
    sample = {
        "dataset": dataset,
        "retained_index": 0,
        "image_filename": expected_filename if image_filename is None else image_filename,
        "source_dataset": candidate.source_dataset,
        "source_split": candidate.source_split,
        "source_index": candidate.source_index,
        "source_split_index": candidate.source_split_index,
        "source_question_id": candidate.source_question_id,
        "source_image_id": source_id,
        "source_canonical_rgb_sha256": saved_rgb_hash,
        "canonical_rgb_sha256": saved_rgb_hash,
        "saved_image_sha256": saved_byte_hash,
        "saved_canonical_rgb_sha256": saved_rgb_hash,
        "perceptual_hash": perceptual_hash,
        "prompt": candidate.prompt,
        "blind_control": "blind_black_image_control",
        "blind_prediction": "fixture-wrong",
        "blind_correct": False,
        "full_prediction": candidate.answers[0],
        "full_correct": True,
        "blind_black_image_control_prediction": "fixture-wrong",
        "blind_black_image_control_correct": False,
        "full_visual_prediction": candidate.answers[0],
        "full_visual_correct": True,
        "mapping": mapping,
        "options": list(candidate.options),
        "ground_truth_letter": candidate.ground_truth_letter,
        "ground_truth_text": candidate.ground_truth_text,
        "legacy_temperature_zero": candidate.legacy_temperature_zero,
        "jpeg_settings": JPEG_SETTINGS,
        "provenance": candidate.provenance,
    }
    stats = _initial_stats(bundle, config)
    stats.update(
        scanned_candidates=1,
        source_hash_candidates_scanned=1,
        unique_images_evaluated=1,
        blind_wrong_count=1,
        full_correct_count=1,
        target_reached_source_index=candidate.source_index,
        retained_count=1,
        acceptance_rate=1.0,
        target_reached_candidate_position=0,
    )
    if dataset == "ScienceQA_MC":
        stats["split_composition"][candidate.source_split] = 1
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "dataset": dataset,
        "target": 1,
        "seed": 20260904,
        "candidate_pool_fingerprint": POOL_FINGERPRINT,
        "exclusions_fingerprint": EXCLUSIONS_FINGERPRINT,
        "construction_protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "next_candidate_position": 1,
        "selection_stats": stats,
        "samples": [sample],
        "seen_source_image_ids": [source_id],
        "seen_source_canonical_rgb_sha256": [saved_rgb_hash],
        "seen_saved_canonical_rgb_sha256": [saved_rgb_hash],
    }
    paths = {
        "dataset_dir": dataset_dir,
        "images_dir": images_dir,
        "stage_dir": stage_dir,
        "checkpoint": manifests_dir / f"{dataset}_checkpoint.json",
        "screening": manifests_dir / f"{dataset}_screening.jsonl",
        "mapping": dataset_dir / f"{dataset}_mapping.jsonl",
        "sidecar": manifests_dir / f"{dataset}_samples.jsonl",
        "stats": manifests_dir / f"{dataset}_stats.json",
    }
    paths["checkpoint"].write_text(json.dumps(checkpoint), encoding="utf-8")
    paths["screening"].write_text(
        json.dumps(
            {
                "candidate_position": 0,
                "source_dataset": candidate.source_dataset,
                "source_split": candidate.source_split,
                "source_index": candidate.source_index,
                "source_split_index": candidate.source_split_index,
                "source_question_id": candidate.source_question_id,
                "source_image_id": source_id,
                "source_canonical_rgb_sha256": saved_rgb_hash,
                "blind_prediction": "fixture-wrong",
                "blind_correct": False,
                "control": "blind_black_image_control",
                "saved_image_sha256": saved_byte_hash,
                "saved_canonical_rgb_sha256": saved_rgb_hash,
                "perceptual_hash": perceptual_hash,
                "full_prediction": candidate.answers[0],
                "full_correct": True,
                "reason": "accepted",
                "accepted": True,
                "retained_index": 0,
                "image_filename": expected_filename,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return bundle, config, paths, expected_filename


def _load(bundle, config, paths):
    return _load_resume_state(
        bundle,
        config,
        paths,
        POOL_FINGERPRINT,
        EXCLUSIONS_FINGERPRINT,
        PROTOCOL_FINGERPRINT,
    )


def _checkpoint(paths):
    return json.loads(paths["checkpoint"].read_text(encoding="utf-8"))


def _write_checkpoint(paths, checkpoint):
    paths["checkpoint"].write_text(json.dumps(checkpoint), encoding="utf-8")


@pytest.mark.parametrize("dataset", DATASET_NAMES)
def test_resume_accepts_only_the_builder_owned_dataset_filename(tmp_path, dataset):
    bundle, config, paths, expected_filename = _resume_fixture(tmp_path, dataset)

    next_position, _, samples, *_ = _load(bundle, config, paths)

    assert next_position == 1
    assert samples[0]["image_filename"] == expected_filename


@pytest.mark.parametrize(
    "image_filename",
    (
        "",
        "/tmp/TextVQA_Open_0000.jpg",
        "../images/TextVQA_Open_0000.jpg",
        "images/../TextVQA_Open_0000.jpg",
        r"images\TextVQA_Open_0000.jpg",
        "images/not-the-expected-name.jpg",
        "images/TextVQA_Open_0001.jpg",
        "images/VQAv2_Open_0000.jpg",
    ),
)
def test_resume_rejects_unsafe_or_wrong_checkpoint_image_filename(
    tmp_path, image_filename
):
    bundle, config, paths, _ = _resume_fixture(
        tmp_path, "TextVQA_Open", image_filename
    )
    paths["mapping"].write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"checkpoint sample.*image_filename"):
        _load(bundle, config, paths)

    assert paths["mapping"].read_text(encoding="utf-8") == "sentinel\n"


@pytest.mark.parametrize(
    "key",
    (
        "seen_source_image_ids",
        "seen_source_canonical_rgb_sha256",
        "seen_saved_canonical_rgb_sha256",
    ),
)
@pytest.mark.parametrize(
    "corruption",
    ("missing_key", "wrong_container", "non_string", "duplicate", "missing", "extra"),
)
def test_resume_rejects_malformed_or_nonexact_seen_sets(
    tmp_path, key, corruption
):
    bundle, config, paths, _ = _resume_fixture(tmp_path, "TextVQA_Open")
    checkpoint = _checkpoint(paths)
    original = list(checkpoint[key])
    if corruption == "missing_key":
        checkpoint.pop(key)
    elif corruption == "wrong_container":
        checkpoint[key] = {"not": "a list"}
    elif corruption == "non_string":
        checkpoint[key] = [123]
    elif corruption == "duplicate":
        checkpoint[key] = original + original
    elif corruption == "missing":
        checkpoint[key] = []
    else:
        extra = "f" * 64 if "sha256" in key else "extra-source-id"
        checkpoint[key] = sorted(original + [extra])
    _write_checkpoint(paths, checkpoint)

    with pytest.raises(RuntimeError, match=key):
        _load(bundle, config, paths)


@pytest.mark.parametrize(
    "key",
    (
        "seen_source_canonical_rgb_sha256",
        "seen_saved_canonical_rgb_sha256",
    ),
)
def test_resume_rejects_malformed_seen_digest(tmp_path, key):
    bundle, config, paths, _ = _resume_fixture(tmp_path, "TextVQA_Open")
    checkpoint = _checkpoint(paths)
    checkpoint[key] = ["not-a-sha256"]
    _write_checkpoint(paths, checkpoint)

    with pytest.raises(RuntimeError, match=r"lowercase SHA-256 digest"):
        _load(bundle, config, paths)


def test_resume_validates_seen_sets_before_removing_builder_orphans(tmp_path):
    bundle, config, paths, _ = _resume_fixture(tmp_path, "TextVQA_Open")
    orphan = paths["images_dir"] / "TextVQA_Open_9999.jpg"
    Image.new("RGB", (1, 1), (1, 2, 3)).save(orphan, format="JPEG")
    checkpoint = _checkpoint(paths)
    checkpoint["seen_source_image_ids"] = sorted(
        checkpoint["seen_source_image_ids"] + ["uncommitted-extra-id"]
    )
    _write_checkpoint(paths, checkpoint)

    with pytest.raises(RuntimeError, match="seen_source_image_ids"):
        _load(bundle, config, paths)

    assert orphan.is_file()


def test_resume_rebuilds_seen_sets_with_distinct_runtime_semantics(tmp_path):
    bundle, config, paths, _ = _resume_fixture(tmp_path, "TextVQA_Open")
    config = replace(config, target=2)
    checkpoint = _checkpoint(paths)
    first_screening = json.loads(
        paths["screening"].read_text(encoding="utf-8").strip()
    )
    first_source_id = first_screening["source_image_id"]
    first_source_hash = first_screening["source_canonical_rgb_sha256"]
    first_saved_hash = first_screening["saved_canonical_rgb_sha256"]
    second_source_hash = "1" * 64
    rejected_saved_hash = "2" * 64
    duplicate_candidate = _candidate(
        "TextVQA_Open",
        position=1,
        source_id=first_source_id,
        source_hash="4" * 64,
    )
    rejected_candidate = _candidate(
        "TextVQA_Open",
        position=2,
        source_id="second-source-id",
        source_hash=second_source_hash,
    )
    duplicate_row = _screening_base(1, duplicate_candidate)
    duplicate_row["reason"] = "duplicate_source_image_id"
    rejected_row = _screening_base(2, rejected_candidate)
    rejected_row.update(
        source_canonical_rgb_sha256=second_source_hash,
        blind_prediction="fixture-wrong-2",
        blind_correct=False,
        control="blind_black_image_control",
        saved_image_sha256="3" * 64,
        saved_canonical_rgb_sha256=rejected_saved_hash,
        perceptual_hash="0" * 16,
        full_prediction="fixture-full-wrong-2",
        full_correct=False,
        reason="full_visual_incorrect",
    )
    screening = [
        first_screening,
        duplicate_row,
        rejected_row,
    ]
    paths["screening"].write_text(
        "".join(json.dumps(row) + "\n" for row in screening), encoding="utf-8"
    )
    bundle.candidates = [bundle.candidates[0], duplicate_candidate, rejected_candidate]
    expected_stats = _initial_stats(bundle, config)
    expected_stats.update(
        scanned_candidates=3,
        source_hash_candidates_scanned=2,
        engine_source_id_duplicate_count=1,
        source_id_duplicate_count=1,
        unique_images_evaluated=2,
        blind_wrong_count=2,
        full_correct_count=1,
        full_incorrect_count=1,
        retained_count=1,
        acceptance_rate=0.5,
        candidate_pool_exhausted=True,
    )
    checkpoint["target"] = 2
    checkpoint["next_candidate_position"] = 3
    checkpoint["selection_stats"] = expected_stats
    checkpoint["seen_source_image_ids"] = sorted(
        [first_source_id, "second-source-id"]
    )
    checkpoint["seen_source_canonical_rgb_sha256"] = sorted(
        [first_source_hash, second_source_hash]
    )
    # Saved-image deduplication tracks accepted samples only. A staged image
    # rejected after encoding must not enter the durable seen-saved set.
    checkpoint["seen_saved_canonical_rgb_sha256"] = [first_saved_hash]
    _write_checkpoint(paths, checkpoint)

    _, _, _, source_ids, source_hashes, saved_hashes = _load(
        bundle, config, paths
    )

    assert source_ids == {first_source_id, "second-source-id"}
    assert source_hashes == {first_source_hash, second_source_hash}
    assert saved_hashes == {first_saved_hash}
    assert rejected_saved_hash not in saved_hashes


def test_resume_rejects_source_hash_for_pre_hash_screening_reason(tmp_path):
    bundle, config, paths, _ = _resume_fixture(tmp_path, "TextVQA_Open")
    checkpoint = _checkpoint(paths)
    row = _screening_base(0, bundle.candidates[0])
    row.update(
        reason="source_network_or_decode_error",
        error="fixture decode failure",
        source_canonical_rgb_sha256="a" * 64,
    )
    paths["screening"].write_text(json.dumps(row) + "\n", encoding="utf-8")
    checkpoint["samples"] = []
    checkpoint["selection_stats"]["retained_count"] = 0
    checkpoint["seen_saved_canonical_rgb_sha256"] = []
    _write_checkpoint(paths, checkpoint)

    with pytest.raises(RuntimeError, match="invalid schema"):
        _load(bundle, config, paths)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_canonical_rgb_sha256", "bad-source-digest"),
        ("saved_image_sha256", "bad-byte-digest"),
        ("saved_canonical_rgb_sha256", "bad-saved-digest"),
    ),
)
def test_resume_rejects_malformed_committed_screening_digest(
    tmp_path, field, value
):
    bundle, config, paths, _ = _resume_fixture(tmp_path, "TextVQA_Open")
    row = json.loads(paths["screening"].read_text(encoding="utf-8"))
    row[field] = value
    paths["screening"].write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"lowercase SHA-256 digest"):
        _load(bundle, config, paths)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("scanned_candidates", 2),
        ("source_hash_candidates_scanned", 0),
        ("blind_wrong_count", 0),
        ("full_correct_count", 0),
        ("acceptance_rate", 0.5),
        ("target_reached_source_index", 999),
        ("target_reached_candidate_position", None),
        ("candidate_pool_exhausted", True),
    ),
)
def test_resume_rejects_tampered_derived_selection_stats(tmp_path, field, value):
    bundle, config, paths, _ = _resume_fixture(tmp_path, "TextVQA_Open")
    checkpoint = _checkpoint(paths)
    checkpoint["selection_stats"][field] = value
    _write_checkpoint(paths, checkpoint)

    with pytest.raises(RuntimeError, match="selection_stats"):
        _load(bundle, config, paths)


@pytest.mark.parametrize(("field", "value"), (("target", True), ("seed", False)))
def test_resume_rejects_boolean_top_level_integer_fields(tmp_path, field, value):
    bundle, config, paths, _ = _resume_fixture(tmp_path, "TextVQA_Open")
    checkpoint = _checkpoint(paths)
    checkpoint[field] = value
    _write_checkpoint(paths, checkpoint)

    with pytest.raises(RuntimeError, match="incompatible resume checkpoint"):
        _load(bundle, config, paths)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_dataset", "tampered/source"),
        ("source_split", "test"),
        ("source_index", 999),
        ("source_split_index", 999),
        ("source_question_id", "tampered-question"),
        ("source_image_id", "tampered-image"),
        ("canonical_rgb_sha256", "a" * 64),
        ("prompt", "tampered prompt"),
        ("blind_control", "tampered control"),
        ("blind_prediction", "tampered blind prediction"),
        ("blind_correct", True),
        ("full_prediction", "tampered full prediction"),
        ("full_correct", False),
        ("blind_black_image_control_prediction", "tampered alias"),
        ("blind_black_image_control_correct", True),
        ("full_visual_prediction", "tampered alias"),
        ("full_visual_correct", False),
        ("mapping", {"image_filename": "images/TextVQA_Open_0000.jpg"}),
        ("options", ["tampered"]),
        ("ground_truth_letter", "B"),
        ("ground_truth_text", "tampered"),
        ("legacy_temperature_zero", False),
        ("jpeg_settings", {"format": "JPEG"}),
        ("provenance", {"tampered": True}),
    ),
)
def test_resume_reconstructs_accepted_sample_exactly_from_candidate(
    tmp_path, field, value
):
    bundle, config, paths, _ = _resume_fixture(tmp_path, "TextVQA_Open")
    checkpoint = _checkpoint(paths)
    checkpoint["samples"][0][field] = value
    _write_checkpoint(paths, checkpoint)

    with pytest.raises(RuntimeError, match="checkpoint sample"):
        _load(bundle, config, paths)


@pytest.mark.parametrize(
    ("reason", "accepted"),
    (("accepted", False), ("full_visual_incorrect", True)),
)
def test_resume_requires_accepted_iff_reason_is_accepted(
    tmp_path, reason, accepted
):
    bundle, config, paths, _ = _resume_fixture(tmp_path, "TextVQA_Open")
    row = json.loads(paths["screening"].read_text(encoding="utf-8"))
    row.update(reason=reason, accepted=accepted)
    paths["screening"].write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="accepted flag and reason disagree"):
        _load(bundle, config, paths)


def test_resume_recomputes_full_correct_despite_synchronized_state_tamper(tmp_path):
    bundle, config, paths, _ = _resume_fixture(tmp_path, "TextVQA_Open")
    checkpoint = _checkpoint(paths)
    row = json.loads(paths["screening"].read_text(encoding="utf-8"))
    # The prediction is actually correct, but all mutable decision fields are
    # synchronized to claim it was rejected.
    row.update(reason="full_visual_incorrect", accepted=False, full_correct=False)
    row.pop("retained_index")
    row.pop("image_filename")
    paths["screening"].write_text(json.dumps(row) + "\n", encoding="utf-8")
    checkpoint["samples"] = []
    checkpoint["seen_saved_canonical_rgb_sha256"] = []
    stats = _initial_stats(bundle, config)
    stats.update(
        scanned_candidates=1,
        source_hash_candidates_scanned=1,
        unique_images_evaluated=1,
        blind_wrong_count=1,
        full_incorrect_count=1,
        candidate_pool_exhausted=True,
    )
    checkpoint["selection_stats"] = stats
    _write_checkpoint(paths, checkpoint)

    with pytest.raises(RuntimeError, match="full_correct does not match"):
        _load(bundle, config, paths)


def test_resume_recomputes_blind_correct_despite_synchronized_state_tamper(tmp_path):
    bundle, config, paths, _ = _resume_fixture(tmp_path, "TextVQA_Open")
    checkpoint = _checkpoint(paths)
    row = json.loads(paths["screening"].read_text(encoding="utf-8"))
    # The prediction is wrong, but boolean, reason, stats, and sample state all
    # claim the blind control answered correctly.
    row.update(
        reason="blind_black_image_control_correct",
        accepted=False,
        blind_correct=True,
    )
    for key in (
        "saved_image_sha256",
        "saved_canonical_rgb_sha256",
        "perceptual_hash",
        "full_prediction",
        "full_correct",
        "retained_index",
        "image_filename",
    ):
        row.pop(key)
    paths["screening"].write_text(json.dumps(row) + "\n", encoding="utf-8")
    checkpoint["samples"] = []
    checkpoint["seen_saved_canonical_rgb_sha256"] = []
    stats = _initial_stats(bundle, config)
    stats.update(
        scanned_candidates=1,
        source_hash_candidates_scanned=1,
        unique_images_evaluated=1,
        blind_correct_count=1,
        candidate_pool_exhausted=True,
    )
    checkpoint["selection_stats"] = stats
    _write_checkpoint(paths, checkpoint)

    with pytest.raises(RuntimeError, match="blind_correct does not match"):
        _load(bundle, config, paths)


@pytest.mark.parametrize("case", ("early_saved", "prefull_full", "rejected_retained"))
def test_resume_enforces_reason_specific_screening_schema(tmp_path, case):
    bundle, config, paths, _ = _resume_fixture(tmp_path, "TextVQA_Open")
    accepted = json.loads(paths["screening"].read_text(encoding="utf-8"))
    if case == "early_saved":
        row = _screening_base(0, bundle.candidates[0])
        row.update(
            reason="source_network_or_decode_error",
            error="decode failed",
            saved_image_sha256="a" * 64,
        )
    elif case == "prefull_full":
        row = accepted
        row.update(reason="duplicate_saved_canonical_rgb_sha256", accepted=False)
        for key in ("full_correct", "retained_index", "image_filename"):
            row.pop(key)
        # full_prediction is deliberately retained even though this rejection
        # happens before Full inference.
    else:
        row = accepted
        row.update(reason="full_visual_incorrect", accepted=False, full_correct=False)
        # retained_index/image_filename are deliberately retained on a rejection.
    paths["screening"].write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid schema"):
        _load(bundle, config, paths)


def test_resume_rejects_more_samples_than_target_before_artifact_rewrite(tmp_path):
    bundle, config, paths, _ = _resume_fixture(tmp_path, "TextVQA_Open")
    checkpoint = _checkpoint(paths)
    checkpoint["samples"].append(copy.deepcopy(checkpoint["samples"][0]))
    checkpoint["selection_stats"]["retained_count"] = 2
    _write_checkpoint(paths, checkpoint)
    paths["mapping"].write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="more samples than its target"):
        _load(bundle, config, paths)

    assert paths["mapping"].read_text(encoding="utf-8") == "sentinel\n"


def test_resume_rejects_committed_rows_after_target_was_first_reached(tmp_path):
    bundle, config, paths, _ = _resume_fixture(tmp_path, "TextVQA_Open")
    checkpoint = _checkpoint(paths)
    duplicate = _candidate(
        "TextVQA_Open",
        position=1,
        source_id=bundle.candidates[0].source_image_id,
        source_hash="d" * 64,
    )
    bundle.candidates.append(duplicate)
    row = _screening_base(1, duplicate)
    row["reason"] = "duplicate_source_image_id"
    with paths["screening"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    checkpoint["next_candidate_position"] = 2
    checkpoint["selection_stats"]["scanned_candidates"] = 2
    checkpoint["selection_stats"]["source_id_duplicate_count"] = 1
    checkpoint["selection_stats"]["engine_source_id_duplicate_count"] = 1
    _write_checkpoint(paths, checkpoint)

    with pytest.raises(RuntimeError, match="continued scanning after the target"):
        _load(bundle, config, paths)


def test_resume_source_manifest_rejects_synchronized_protocol_body_tamper(tmp_path):
    bundle, runner, _ = _run_fresh_builder(tmp_path)
    source_path = tmp_path / "manifests" / "TextVQA_Open_source.json"
    checkpoint_path = tmp_path / "manifests" / "TextVQA_Open_checkpoint.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    source["construction_protocol"]["runner"]["tampered"] = True
    forged = _stable_json_digest(source["construction_protocol"])
    source["construction_protocol_fingerprint"] = forged
    checkpoint["construction_protocol_fingerprint"] = forged
    source_path.write_text(json.dumps(source), encoding="utf-8")
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(RuntimeError, match="construction_protocol"):
        build_dataset(
            bundle,
            runner,
            replace(BuildConfig(tmp_path, target=1), resume=True),
        )


def test_resume_candidate_pool_atomic_artifact_rejects_bad_tail(tmp_path):
    bundle, runner, _ = _run_fresh_builder(tmp_path)
    pool_path = tmp_path / "manifests" / "TextVQA_Open_candidate_pool.jsonl"
    pool_path.write_bytes(pool_path.read_bytes() + b'{"truncated"')

    with pytest.raises(RuntimeError, match="strict candidate-pool manifest"):
        build_dataset(
            bundle,
            runner,
            replace(BuildConfig(tmp_path, target=1), resume=True),
        )


@pytest.mark.parametrize(
    "relative",
    (
        "manifests/TextVQA_Open_source.json",
        "manifests/TextVQA_Open_candidate_pool.jsonl",
    ),
)
def test_resume_requires_precheckpoint_atomic_manifests(tmp_path, relative):
    bundle, runner, _ = _run_fresh_builder(tmp_path)
    (tmp_path / relative).unlink()

    with pytest.raises(RuntimeError, match="required atomic build artifacts are missing"):
        build_dataset(
            bundle,
            runner,
            replace(BuildConfig(tmp_path, target=1), resume=True),
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda metadata: metadata.update(dataset_fingerprint="forged"),
        lambda metadata: metadata["schema"].append("forged_column"),
        lambda metadata: metadata["cache_files"][0].update(sha256="f" * 64),
        lambda metadata: metadata.update(generic_pool_rule="forged rule"),
    ),
)
def test_resume_source_manifest_rejects_stable_metadata_tamper(
    tmp_path, mutate
):
    metadata = {
        "dataset_fingerprint": "hf-fingerprint",
        "schema": ["image", "question", "answers"],
        "cache_files": [
            {"path": "/cache/data.arrow", "size_bytes": 123, "sha256": "a" * 64}
        ],
        "archive": {
            "url": "https://example.invalid/data.zip",
            "size": 456,
            "sha256": "b" * 64,
            "cache_status": "downloaded",
            "attempt": 2,
            "cached_source": "/transient/cache.zip",
        },
        "generic_pool_rule": "frozen rule",
    }
    bundle, runner, _ = _run_fresh_builder(tmp_path, source_metadata=metadata)
    source_path = tmp_path / "manifests" / "TextVQA_Open_source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    mutate(source["source_metadata"])
    source["source_metadata_stable_fingerprint"] = _stable_json_digest(
        _stable_source_metadata(source["source_metadata"])
    )
    source_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(RuntimeError, match="stable source_metadata"):
        build_dataset(
            bundle,
            runner,
            replace(BuildConfig(tmp_path, target=1), resume=True),
        )


def test_resume_source_manifest_allows_only_explicit_transient_metadata_changes(
    tmp_path,
):
    initial_metadata = {
        "question_zip": {
            "path": "/cache/data.zip",
            "url": "https://example.invalid/data.zip",
            "size": 456,
            "sha256": "b" * 64,
            "cache_status": "downloaded",
            "attempt": 2,
            "cached_source": "/transient/cache.zip",
        }
    }
    bundle, runner, _ = _run_fresh_builder(
        tmp_path, source_metadata=initial_metadata
    )
    bundle.source_metadata["question_zip"].update(
        cache_status="existing", attempt=999, cached_source="/different/transient.zip"
    )

    resumed = build_dataset(
        bundle,
        runner,
        replace(BuildConfig(tmp_path, target=1), resume=True),
    )

    assert resumed.target_reached is True


@pytest.mark.parametrize(
    "metadata",
    (
        {
            "cache_status": "semantic-top-level-status",
            "attempt": 7,
            "cached_source": "semantic-top-level-source",
        },
        {
            "arbitrary_nested_record": {
                "cache_status": "semantic-nested-status",
                "attempt": 8,
                "cached_source": "semantic-nested-source",
            }
        },
    ),
)
def test_resume_does_not_ignore_transient_named_keys_outside_vqa_archives(
    tmp_path, metadata
):
    bundle, runner, _ = _run_fresh_builder(
        tmp_path, source_metadata=copy.deepcopy(metadata)
    )
    if "arbitrary_nested_record" in bundle.source_metadata:
        bundle.source_metadata["arbitrary_nested_record"]["attempt"] = 999
    else:
        bundle.source_metadata["attempt"] = 999

    with pytest.raises(RuntimeError, match="stable source_metadata"):
        build_dataset(
            bundle,
            runner,
            replace(BuildConfig(tmp_path, target=1), resume=True),
        )


def test_final_short_candidate_checkpoint_is_immediately_resume_safe(tmp_path):
    bundle, runner, result = _run_fresh_builder(tmp_path, target=2)
    checkpoint = json.loads(result.checkpoint_path.read_text(encoding="utf-8"))

    assert checkpoint["next_candidate_position"] == 1
    assert checkpoint["selection_stats"]["candidate_pool_exhausted"] is True

    resumed = build_dataset(
        bundle,
        runner,
        replace(BuildConfig(tmp_path, target=2), resume=True),
        fail_if_short=False,
    )
    assert resumed.stats["candidate_pool_exhausted"] is True


def test_empty_candidate_checkpoint_is_immediately_resume_safe(tmp_path):
    bundle, runner, result = _run_fresh_builder(
        tmp_path, target=1, candidates=[]
    )
    checkpoint = json.loads(result.checkpoint_path.read_text(encoding="utf-8"))

    assert checkpoint["next_candidate_position"] == 0
    assert checkpoint["selection_stats"]["candidate_pool_exhausted"] is True

    resumed = build_dataset(
        bundle,
        runner,
        replace(BuildConfig(tmp_path, target=1), resume=True),
        fail_if_short=False,
    )
    assert resumed.stats["candidate_pool_exhausted"] is True


def test_complete_resume_shared_preflight_never_calls_inference(tmp_path):
    bundle, runner, result = _run_fresh_builder(tmp_path)

    completed = load_completed_resume(
        bundle,
        BuildConfig(tmp_path, target=1, seed=20260904, resume=True),
        current_runner=runner,
    )

    assert completed is not None
    assert completed.target_reached is True
    assert completed.samples == result.samples


def test_complete_resume_binds_cpu_only_current_model_fingerprint(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("fixture-model", encoding="utf-8")
    bundle, _, _ = _run_fresh_builder(
        tmp_path / "output",
        runner=_ModelManifestFixtureRunner(model),
    )

    completed = load_completed_resume(
        bundle,
        BuildConfig(
            tmp_path / "output", target=1, seed=20260904, resume=True
        ),
        model_path=model,
    )

    assert completed is not None


@pytest.mark.parametrize("case", ("different", "missing"))
def test_complete_resume_rejects_wrong_model_without_loading_gpu(tmp_path, case):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("fixture-model", encoding="utf-8")
    bundle, _, _ = _run_fresh_builder(
        tmp_path / "output",
        runner=_ModelManifestFixtureRunner(model),
    )
    current = tmp_path / case
    if case == "different":
        current.mkdir()
        (current / "config.json").write_text("different-model", encoding="utf-8")

    with pytest.raises((FileNotFoundError, RuntimeError)):
        load_completed_resume(
            bundle,
            BuildConfig(
                tmp_path / "output", target=1, seed=20260904, resume=True
            ),
            model_path=current,
        )


def test_existing_extracted_vqa_json_must_match_current_zip_member(tmp_path):
    archive = tmp_path / "source.zip"
    member = "official/member.json"
    destination = tmp_path / "member.json"
    payload = b'{"official": true}\n'
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member, payload)

    identity = extract_zip_member_atomic(archive, member, destination)

    assert identity["member"] == member
    assert identity["sha256"] == sha256_file(destination)
    destination.write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs"):
        extract_zip_member_atomic(archive, member, destination)


def test_read_only_vqa_source_refuses_missing_cache_without_creating_it(tmp_path):
    source_cache = tmp_path / "missing-source-cache"

    with pytest.raises(FileNotFoundError, match="source_cache does not exist"):
        prepare_vqav2_source(source_cache, read_only=True)

    assert not source_cache.exists()


def test_read_only_vqa_source_verification_does_not_mutate_frozen_files(tmp_path):
    source_cache = tmp_path / "source-cache"
    source_cache.mkdir()
    pairs = (
        (
            source_cache / "v2_Questions_Val_mscoco.zip",
            VQA_QUESTION_MEMBER,
            b'{"questions": []}\n',
        ),
        (
            source_cache / "v2_Annotations_Val_mscoco.zip",
            VQA_ANNOTATION_MEMBER,
            b'{"annotations": []}\n',
        ),
    )
    tracked = [source_cache]
    for archive, member, payload in pairs:
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(member, payload)
        extracted = source_cache / member
        extracted.write_bytes(payload)
        tracked.extend((archive, extracted))
    before = {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in tracked
    }

    metadata = prepare_vqav2_source(source_cache, read_only=True)

    after = {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in tracked
    }
    assert before == after
    assert metadata["question_member"]["member"] == VQA_QUESTION_MEMBER
    assert metadata["annotation_member"]["member"] == VQA_ANNOTATION_MEMBER


def test_complete_read_only_resume_does_not_recreate_stage_or_public_state(tmp_path):
    bundle, runner, _ = _run_fresh_builder(tmp_path)
    stage_dir = tmp_path / "TextVQA_Open" / ".stage"
    stage_dir.rmdir()
    tracked = [path for path in tmp_path.rglob("*") if path.is_file()]
    before = {
        path: (path.stat().st_mtime_ns, path.stat().st_size, sha256_file(path))
        for path in tracked
    }

    completed = load_completed_resume(
        bundle,
        BuildConfig(tmp_path, target=1, seed=20260904, resume=True),
        repair_public_state=False,
        current_runner=runner,
    )

    after = {
        path: (path.stat().st_mtime_ns, path.stat().st_size, sha256_file(path))
        for path in tracked
    }
    assert completed is not None
    assert before == after
    assert not stage_dir.exists()


def test_build_all_rejects_symlinked_owned_source_cache(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    external = tmp_path / "external-cache"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    (output / "source_cache").symlink_to(external, target_is_directory=True)
    model = tmp_path / "model"
    model.mkdir()

    with pytest.raises(ValueError, match="source_cache must not be a symlink"):
        build_all.run_all(
            output=output,
            target=1,
            model=model,
            hf_cache=None,
            device="cpu",
            seed=20260904,
            resume=False,
            runner=_FixtureRunner(),
        )

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert sorted(path.name for path in external.iterdir()) == ["sentinel.txt"]


def test_reproducibility_rejects_symlinked_owned_source_cache(tmp_path):
    dataset_root = tmp_path / "datasets"
    reports = tmp_path / "reports"
    external = tmp_path / "external-cache"
    dataset_root.mkdir()
    reports.mkdir()
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    (dataset_root / "source_cache").symlink_to(
        external, target_is_directory=True
    )

    with pytest.raises(ValueError, match="source_cache must not be a symlink"):
        repro.generate(dataset_root, reports_dir=reports)

    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def _write_mini_vqa_archives(source_cache):
    payloads = {
        VQA_QUESTION_MEMBER: b'{"questions": []}\n',
        VQA_ANNOTATION_MEMBER: b'{"annotations": []}\n',
    }
    for archive_name, member in (
        ("v2_Questions_Val_mscoco.zip", VQA_QUESTION_MEMBER),
        ("v2_Annotations_Val_mscoco.zip", VQA_ANNOTATION_MEMBER),
    ):
        with zipfile.ZipFile(source_cache / archive_name, "w") as handle:
            handle.writestr(member, payloads[member])
    return payloads


def test_vqa_source_rejects_symlinked_zip_child(tmp_path):
    source_cache = tmp_path / "source-cache"
    source_cache.mkdir()
    external_zip = tmp_path / "external.zip"
    with zipfile.ZipFile(external_zip, "w") as handle:
        handle.writestr(VQA_QUESTION_MEMBER, b'{"questions": []}\n')
    (source_cache / "v2_Questions_Val_mscoco.zip").symlink_to(external_zip)

    with pytest.raises(ValueError, match="source-cache path contains a symlink"):
        prepare_vqav2_source(source_cache)

    assert external_zip.is_file()


def test_vqa_source_rejects_symlinked_extracted_json_child(tmp_path):
    source_cache = tmp_path / "source-cache"
    source_cache.mkdir()
    payloads = _write_mini_vqa_archives(source_cache)
    external_json = tmp_path / "external.json"
    external_json.write_bytes(payloads[VQA_QUESTION_MEMBER])
    (source_cache / VQA_QUESTION_MEMBER).symlink_to(external_json)

    with pytest.raises(ValueError, match="source-cache path contains a symlink"):
        prepare_vqav2_source(source_cache)

    assert external_json.read_bytes() == payloads[VQA_QUESTION_MEMBER]


def test_coco_loader_rejects_symlinked_cached_jpeg_child(tmp_path):
    source_cache = tmp_path / "source-cache"
    coco_dir = source_cache / "coco_val2014"
    coco_dir.mkdir(parents=True)
    external_jpeg = tmp_path / "external.jpg"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(external_jpeg, format="JPEG")
    cached = coco_dir / "COCO_val2014_000000000017.jpg"
    cached.symlink_to(external_jpeg)
    before = external_jpeg.read_bytes()

    with pytest.raises(ValueError, match="source-cache path contains a symlink"):
        builder_sources._coco_loader(source_cache, 17)()

    assert external_jpeg.read_bytes() == before


def _materialized_coco_fixture(tmp_path, *, image_id=17):
    source_cache = tmp_path / "source-cache"
    image_dir = source_cache / builder_sources.COCO_IMAGE_DIRECTORY
    image_dir.mkdir(parents=True)
    image_path = image_dir / f"COCO_val2014_{image_id:012d}.jpg"
    Image.new("RGB", (3, 2), (12, 34, 56)).save(image_path, format="JPEG")
    identities = builder_sources.validate_coco_identity_cache(
        source_cache, allow_create=True
    )
    manifest_path = (
        source_cache
        / builder_sources.COCO_IDENTITY_DIRECTORY
        / f"{image_path.name}.json"
    )
    return source_cache, image_path, manifest_path, identities[0]


def test_coco_identity_manifest_rejects_cross_resume_byte_replacement(tmp_path):
    source_cache, image_path, manifest_path, identity = _materialized_coco_fixture(
        tmp_path
    )
    assert manifest_path.is_file()
    assert identity["saved_image_sha256"] == sha256_file(image_path)
    Image.new("RGB", (3, 2), (99, 88, 77)).save(image_path, format="JPEG")

    with pytest.raises(RuntimeError, match="changed after first materialization"):
        builder_sources.validate_coco_identity_cache(
            source_cache, allow_create=False
        )


def test_coco_resume_rejects_existing_jpeg_without_identity_manifest(tmp_path):
    source_cache, _, manifest_path, _ = _materialized_coco_fixture(tmp_path)
    manifest_path.unlink()

    with pytest.raises(RuntimeError, match="lacks its immutable identity manifest"):
        builder_sources.validate_coco_identity_cache(
            source_cache, allow_create=False
        )


def test_coco_identity_first_transaction_recovers_publish_interruption(
    tmp_path, monkeypatch
):
    source_cache = tmp_path / "source-cache"
    source_cache.mkdir()
    image_id = 17
    image_path, manifest_path = builder_sources._coco_identity_paths(
        source_cache, image_id
    )

    def fake_download(destination, _url, **_kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            Image.new("RGB", (3, 2), (12, 34, 56)).save(
                destination, format="JPEG"
            )
        return {"path": str(destination)}

    monkeypatch.setattr(builder_sources, "copy_or_download", fake_download)
    real_replace = builder_sources.os.replace

    def interrupt_publication(source, destination):
        if destination == image_path:
            raise OSError("fixture stop between identity and JPEG publication")
        return real_replace(source, destination)

    monkeypatch.setattr(builder_sources.os, "replace", interrupt_publication)
    with pytest.raises(RecoverableSourceError, match="publication was interrupted"):
        builder_sources._coco_loader(source_cache, image_id)()

    assert manifest_path.is_file()
    assert not image_path.exists()
    assert builder_sources.validate_coco_identity_cache(
        source_cache,
        allow_create=False,
        allow_pending=True,
    ) == []

    monkeypatch.setattr(builder_sources.os, "replace", real_replace)
    recovered = builder_sources._coco_loader(source_cache, image_id)()
    assert recovered.mode == "RGB"
    assert image_path.is_file()
    assert builder_sources.validate_coco_identity_cache(
        source_cache, allow_create=False
    )[0]["image_id"] == image_id


@pytest.mark.parametrize("encoding", ("png", "grayscale_jpeg"))
def test_coco_first_materialization_requires_native_rgb_jpeg(tmp_path, encoding):
    source_cache = tmp_path / "source-cache"
    image_dir = source_cache / builder_sources.COCO_IMAGE_DIRECTORY
    image_dir.mkdir(parents=True)
    image_path = image_dir / "COCO_val2014_000000000017.jpg"
    if encoding == "png":
        Image.new("RGB", (3, 2), (1, 2, 3)).save(image_path, format="PNG")
    else:
        Image.new("L", (3, 2), 127).save(image_path, format="JPEG")

    with pytest.raises(RuntimeError, match="native RGB JPEG"):
        builder_sources.validate_coco_identity_cache(
            source_cache, allow_create=True
        )
