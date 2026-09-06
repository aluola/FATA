"""Deterministic source-pool preparation for the four FATA datasets.

This module deliberately does no model inference.  Its job is to validate the
upstream schema, apply task-specific metadata filters, and choose exactly one
question per reliable source image identity in native source order.  Exact RGB
deduplication is performed here when the images are embedded in the Hugging
Face dataset (TextVQA and ScienceQA), and is repeated unconditionally by the
build engine before inference.  COCO images are fetched lazily, so their exact
RGB deduplication happens in the engine as each representative is scanned.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from PIL import Image

from .common import (
    atomic_write_json,
    canonical_rgb_sha256,
    canonical_rgb_sha256_path,
    copy_or_download,
    extract_zip_member_atomic,
    generate_mc_options,
    is_yes_no_question,
    normalized_rgb,
    official_vqa_process,
    open_prompt,
    ordered_deduplicate,
    resolve_source_cache_child,
    resolve_source_cache_root,
    scienceqa_prompt,
    scienceqa_question,
    sha256_file,
    vqav2_mc_prompt,
    vqav2_mc_question,
)


TEXTVQA_DATASET = "textvqa"
TEXTVQA_CONFIG = "textvqa"
SCIENCEQA_DATASET = "derek-thomas/ScienceQA"
VQA_QUESTIONS_CACHE = (
    Path(os.environ["FATA_VQA_QUESTIONS_ARCHIVE"])
    if os.environ.get("FATA_VQA_QUESTIONS_ARCHIVE")
    else None
)
VQA_ANNOTATIONS_CACHE = (
    Path(os.environ["FATA_VQA_ANNOTATIONS_ARCHIVE"])
    if os.environ.get("FATA_VQA_ANNOTATIONS_ARCHIVE")
    else None
)
VQA_QUESTIONS_URL = (
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/"
    "v2_Questions_Val_mscoco.zip"
)
VQA_ANNOTATIONS_URL = (
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/"
    "v2_Annotations_Val_mscoco.zip"
)
VQA_QUESTION_MEMBER = "v2_OpenEnded_mscoco_val2014_questions.json"
VQA_ANNOTATION_MEMBER = "v2_mscoco_val2014_annotations.json"
COCO_VAL2014_URL = (
    "https://s3.amazonaws.com/images.cocodataset.org/val2014/{filename}"
)
COCO_IDENTITY_SCHEMA_VERSION = 1
COCO_IMAGE_DIRECTORY = "coco_val2014"
COCO_IDENTITY_DIRECTORY = "coco_val2014_identity"
COCO_TRANSACTION_DIRECTORY = "coco_val2014_transactions"
_COCO_FILENAME_RE = re.compile(r"COCO_val2014_([0-9]{12})\.jpg")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
SOURCE_POOL_MATERIALIZATION_ATTEMPTS = 3


class SourceSchemaError(RuntimeError):
    """An upstream dataset does not have the required, understood schema."""


class RecoverableSourceError(RuntimeError):
    """A bounded network or image-decode error for one source candidate."""


@dataclass(frozen=True)
class Candidate:
    """One preselected question for one visual source identity."""

    source_dataset: str
    source_split: str
    source_index: int
    source_split_index: int
    source_question_id: str | int
    source_image_id: str | int | None
    question: str
    answers: tuple[str, ...]
    task_type: str
    prompt: str
    mapping_payload: dict[str, Any]
    image_loader: Callable[[], Image.Image] = field(repr=False, compare=False)
    source_canonical_rgb_sha256: str | None = None
    options: tuple[str, ...] = ()
    ground_truth_letter: str | None = None
    ground_truth_text: str | None = None
    legacy_temperature_zero: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceBundle:
    """Prepared representatives plus source-level selection statistics."""

    dataset_name: str
    candidates: list[Candidate]
    selection_stats: dict[str, Any]
    source_metadata: dict[str, Any] = field(default_factory=dict)


def _require_columns(actual: Iterable[str], required: Sequence[str], label: str) -> None:
    actual_set = set(actual)
    missing = [name for name in required if name not in actual_set]
    if missing:
        raise SourceSchemaError(
            f"{label} schema is missing required fields {missing!r}; "
            f"actual fields are {sorted(actual_set)!r}"
        )


def native_first_by_image_id(
    records: Iterable[Any], image_id: Callable[[Any], Any]
) -> tuple[list[Any], int]:
    """Return the first record for each non-empty ID without hash-order effects."""

    representatives: list[Any] = []
    seen: set[str] = set()
    duplicates = 0
    for record in records:
        raw = image_id(record)
        key = str(raw) if raw is not None else ""
        if not key:
            continue
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        representatives.append(record)
    return representatives, duplicates


def native_first_by_canonical_hash(
    records: Iterable[Any], canonical_hash: Callable[[Any], str]
) -> tuple[list[Any], int]:
    """Return the native-first representative of each exact RGB image."""

    representatives: list[Any] = []
    seen: set[str] = set()
    duplicates = 0
    for record in records:
        digest = canonical_hash(record)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        representatives.append(record)
    return representatives, duplicates


def _base_stats(source_splits: Sequence[str], source_rows: int) -> dict[str, Any]:
    return {
        "source_splits": list(source_splits),
        "source_qa_records_total": int(source_rows),
        "source_unique_images_total": 0,
        "task_prefilter_unique_images": 0,
        "task_prefilter_unique_images_basis": None,
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


def _hf_cache_file_manifest(dataset: Any) -> list[dict[str, Any]]:
    """Record the exact Arrow/cache artifacts exposed by a HF Dataset."""

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in getattr(dataset, "cache_files", []):
        filename = entry.get("filename") if isinstance(entry, dict) else None
        if not isinstance(filename, str):
            continue
        path = Path(filename).resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.is_file():
            raise SourceSchemaError(f"Hugging Face cache file disappeared: {path}")
        output.append(
            {
                "path": key,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return output


def _open_mapping(dataset_name: str, question: str, answers: Sequence[str]) -> dict[str, Any]:
    return {
        "image_filename": "",  # filled only after final-file verification
        "dataset": dataset_name,
        "type": "open",
        "question": question,
        "answers": [str(answer) for answer in answers],
    }


def _mc_mapping(
    dataset_name: str,
    question: str,
    options: Sequence[str],
    answer_letter: str,
    answer_text: str,
) -> dict[str, Any]:
    return {
        "image_filename": "",  # filled only after final-file verification
        "dataset": dataset_name,
        "type": "multiple_choice",
        "question": question,
        "options": [str(option) for option in options],
        "answers": [answer_letter],
        "ground_truth_text": answer_text,
    }


def _hf_loader(dataset: Any, index: int) -> Callable[[], Image.Image]:
    def load() -> Image.Image:
        try:
            value = dataset[index]["image"]
            if value is None:
                raise RecoverableSourceError("source image is null")
            return normalized_rgb(value)
        except RecoverableSourceError:
            raise
        except OSError as error:
            raise RecoverableSourceError(f"image decode failed: {error}") from error

    return load


def _retry_pool_materialization(
    operation: Callable[[], Any], *, label: str
) -> Any:
    """Retry transient embedded-image reads, then fail the entire pool build.

    Silently dropping a temporarily unreadable item would change the frozen
    native-order candidate cohort while still allowing a nominally successful
    build. Pool preparation therefore retries a small fixed number of times
    and then fails closed; a fresh preparation can run once the source is
    healthy.
    """

    last_error: BaseException | None = None
    for _attempt in range(SOURCE_POOL_MATERIALIZATION_ATTEMPTS):
        try:
            return operation()
        except (OSError, RecoverableSourceError) as error:
            last_error = error
    raise RecoverableSourceError(
        f"{label} failed after {SOURCE_POOL_MATERIALIZATION_ATTEMPTS} attempts; "
        "candidate pool was not frozen, so rerun source preparation"
    ) from last_error


def load_textvqa(
    hf_cache_dir: Path | str | None = None,
) -> SourceBundle:
    """Load the original ``textvqa/textvqa`` validation split (exactly 5,000 rows)."""

    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": "validation", "trust_remote_code": True}
    if hf_cache_dir is not None:
        kwargs["cache_dir"] = str(hf_cache_dir)
    dataset = load_dataset(TEXTVQA_DATASET, TEXTVQA_CONFIG, **kwargs)
    required = ("image_id", "question_id", "question", "image", "answers")
    _require_columns(dataset.column_names, required, "TextVQA validation")
    if len(dataset) != 5000:
        raise SourceSchemaError(
            f"expected original TextVQA validation to contain 5000 rows, found {len(dataset)}"
        )

    stats = _base_stats(("validation",), len(dataset))
    metadata_dataset = dataset.remove_columns(["image"])
    # IDs can be inspected without decoding the image column.
    stats["source_unique_images_total"] = len(
        {str(value) for value in dataset["image_id"] if value is not None and str(value)}
    )

    eligible_indices: list[int] = []
    for index in range(len(dataset)):
        # The image column is removed once so metadata grouping never decodes a
        # duplicate image merely to read its ID.
        row = metadata_dataset[index]
        answers = row["answers"]
        if (
            row["image_id"] is None
            or not str(row["image_id"])
            or row["question_id"] is None
            or not isinstance(row["question"], str)
            or not row["question"].strip()
            or not isinstance(answers, list)
            or not answers
        ):
            stats["invalid_metadata_count"] += 1
            continue
        eligible_indices.append(index)

    id_representatives, duplicate_ids = native_first_by_image_id(
        eligible_indices, lambda index: metadata_dataset[index]["image_id"]
    )
    stats["source_id_duplicate_count"] = duplicate_ids
    stats["task_prefilter_source_id_representatives"] = len(id_representatives)

    candidates: list[Candidate] = []
    seen_hashes: set[str] = set()
    for index in id_representatives:
        def materialize_textvqa() -> tuple[Image.Image, str] | None:
            value = dataset[index]["image"]
            if value is None:
                return None
            image = normalized_rgb(value)
            return image, canonical_rgb_sha256(image)

        materialized = _retry_pool_materialization(
            materialize_textvqa,
            label=f"TextVQA validation row {index} image materialization",
        )
        if materialized is None:
            stats["missing_or_corrupt_images"] += 1
            continue
        _image, source_hash = materialized
        stats["source_hash_candidates_seen"] += 1
        if source_hash in seen_hashes:
            stats["exact_image_hash_duplicate_count"] += 1
            continue
        seen_hashes.add(source_hash)
        row = metadata_dataset[index]
        answers = tuple(str(value) for value in row["answers"])
        question = str(row["question"])
        candidates.append(
            Candidate(
                source_dataset="textvqa/textvqa",
                source_split="validation",
                source_index=index,
                source_split_index=index,
                source_question_id=row["question_id"],
                source_image_id=str(row["image_id"]),
                question=question,
                answers=answers,
                task_type="open",
                prompt=open_prompt(question),
                mapping_payload=_open_mapping("TextVQA_Open", question, answers),
                image_loader=_hf_loader(dataset, index),
                source_canonical_rgb_sha256=source_hash,
                legacy_temperature_zero=True,
                provenance={
                    "flickr_original_url": row.get("flickr_original_url"),
                    "flickr_300k_url": row.get("flickr_300k_url"),
                    "set_name": row.get("set_name"),
                },
            )
        )

    stats["source_exact_hash_unique_images"] = len(candidates)
    stats["task_prefilter_unique_images"] = len(candidates)
    stats["task_prefilter_unique_images_basis"] = (
        "metadata-valid native-first source_image_id representatives, then complete "
        "canonical RGB SHA-256 deduplication"
    )
    return SourceBundle(
        dataset_name="TextVQA_Open",
        candidates=candidates,
        selection_stats=stats,
        source_metadata={
            "dataset_id": TEXTVQA_DATASET,
            "dataset_url": "https://huggingface.co/datasets/textvqa",
            "config": TEXTVQA_CONFIG,
            "split": "validation",
            "schema": list(dataset.column_names),
            "native_rows": len(dataset),
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "cache_files": _hf_cache_file_manifest(dataset),
        },
    )


def prepare_vqav2_source(
    source_cache_dir: Path, *, read_only: bool = False
) -> dict[str, Any]:
    """Copy cached official ZIPs and safely extract the two required JSON members."""

    source_cache_dir = resolve_source_cache_root(
        source_cache_dir, require_existing=read_only
    )
    if read_only:
        if not source_cache_dir.is_dir():
            raise FileNotFoundError(
                f"read-only VQAv2 source cache does not exist: {source_cache_dir}"
            )
    else:
        source_cache_dir.mkdir(parents=True, exist_ok=True)
    question_zip = resolve_source_cache_child(
        source_cache_dir, "v2_Questions_Val_mscoco.zip"
    )
    annotation_zip = resolve_source_cache_child(
        source_cache_dir, "v2_Annotations_Val_mscoco.zip"
    )
    if read_only:
        missing = [
            str(path)
            for path in (question_zip, annotation_zip)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"read-only VQAv2 source cache is incomplete: {missing!r}"
            )
        question_info = {
            "path": str(question_zip),
            "url": VQA_QUESTIONS_URL,
            "size": question_zip.stat().st_size,
            "sha256": sha256_file(question_zip),
            "cache_status": "existing",
        }
        annotation_info = {
            "path": str(annotation_zip),
            "url": VQA_ANNOTATIONS_URL,
            "size": annotation_zip.stat().st_size,
            "sha256": sha256_file(annotation_zip),
            "cache_status": "existing",
        }
    else:
        question_info = copy_or_download(
            question_zip,
            VQA_QUESTIONS_URL,
            cached_source=VQA_QUESTIONS_CACHE,
            attempts=4,
        )
        annotation_info = copy_or_download(
            annotation_zip,
            VQA_ANNOTATIONS_URL,
            cached_source=VQA_ANNOTATIONS_CACHE,
            attempts=4,
        )
    questions_json = resolve_source_cache_child(
        source_cache_dir, VQA_QUESTION_MEMBER
    )
    annotations_json = resolve_source_cache_child(
        source_cache_dir, VQA_ANNOTATION_MEMBER
    )
    if read_only:
        missing = [
            str(path)
            for path in (questions_json, annotations_json)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"read-only VQAv2 extracted metadata is incomplete: {missing!r}"
            )
    question_member = extract_zip_member_atomic(
        question_zip, VQA_QUESTION_MEMBER, questions_json
    )
    annotation_member = extract_zip_member_atomic(
        annotation_zip, VQA_ANNOTATION_MEMBER, annotations_json
    )
    return {
        "question_zip": question_info,
        "annotation_zip": annotation_info,
        "questions_json": str(questions_json),
        "annotations_json": str(annotations_json),
        "question_member": question_member,
        "annotation_member": annotation_member,
    }


def _read_vqav2(
    source_cache_dir: Path, *, read_only: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metadata = prepare_vqav2_source(source_cache_dir, read_only=read_only)
    with Path(metadata["questions_json"]).open("r", encoding="utf-8") as handle:
        question_document = json.load(handle)
    with Path(metadata["annotations_json"]).open("r", encoding="utf-8") as handle:
        annotation_document = json.load(handle)
    _require_columns(question_document.keys(), ("questions",), "VQAv2 questions JSON")
    _require_columns(annotation_document.keys(), ("annotations",), "VQAv2 annotations JSON")
    questions = question_document["questions"]
    annotations = annotation_document["annotations"]
    if not isinstance(questions, list) or not isinstance(annotations, list):
        raise SourceSchemaError("VQAv2 questions/annotations must both be JSON arrays")
    if len(questions) != 214354 or len(annotations) != 214354:
        raise SourceSchemaError(
            "expected 214354 official VQAv2 validation questions and annotations; "
            f"found {len(questions)} and {len(annotations)}"
        )
    return questions, annotations, metadata


def _stable_modal_answer(answers: Sequence[str]) -> str:
    """Choose the mode, resolving ties by first occurrence in official order."""

    counts = Counter(str(answer) for answer in answers)
    if not counts:
        raise ValueError("cannot take mode of an empty answer list")
    maximum = max(counts.values())
    return next(str(answer) for answer in answers if counts[str(answer)] == maximum)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _coco_identity_paths(
    source_cache_root: Path, image_id: int
) -> tuple[Path, Path]:
    if type(image_id) is not int or image_id < 0:
        raise ValueError("COCO image_id must be a non-negative integer")
    filename = f"COCO_val2014_{image_id:012d}.jpg"
    image_path = resolve_source_cache_child(
        source_cache_root, f"{COCO_IMAGE_DIRECTORY}/{filename}"
    )
    manifest_path = resolve_source_cache_child(
        source_cache_root,
        f"{COCO_IDENTITY_DIRECTORY}/{filename}.json",
    )
    return image_path, manifest_path


def _current_coco_identity(
    image_path: Path,
    image_id: int,
    *,
    published_path: Path | None = None,
) -> dict[str, Any]:
    image_path = Path(image_path)
    published_path = image_path if published_path is None else Path(published_path)
    with Image.open(image_path) as opened:
        opened.load()
        encoded_format = opened.format
        encoded_mode = opened.mode
        width, height = opened.size
    if encoded_format != "JPEG" or encoded_mode != "RGB":
        raise RuntimeError(
            f"COCO cache image must be a native RGB JPEG: {image_path} "
            f"(format={encoded_format!r}, mode={encoded_mode!r})"
        )
    if width <= 0 or height <= 0:
        raise RuntimeError(f"COCO cache image has invalid dimensions: {image_path}")
    return {
        "schema_version": COCO_IDENTITY_SCHEMA_VERSION,
        "source": "official COCO val2014",
        "image_id": image_id,
        "filename": published_path.name,
        "relative_path": f"{COCO_IMAGE_DIRECTORY}/{published_path.name}",
        "file_size": image_path.stat().st_size,
        "saved_image_sha256": sha256_file(image_path),
        "canonical_rgb_sha256": canonical_rgb_sha256_path(image_path),
        "encoded_format": encoded_format,
        "encoded_mode": encoded_mode,
        "width": width,
        "height": height,
    }


def _read_coco_identity_manifest(
    manifest_path: Path, image_id: int
) -> dict[str, Any]:
    try:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"invalid COCO source identity manifest: {manifest_path}"
        ) from exc
    filename = f"COCO_val2014_{image_id:012d}.jpg"
    expected_fields = {
        "schema_version",
        "source",
        "image_id",
        "filename",
        "relative_path",
        "file_size",
        "saved_image_sha256",
        "canonical_rgb_sha256",
        "encoded_format",
        "encoded_mode",
        "width",
        "height",
    }
    if not isinstance(recorded, dict) or set(recorded) != expected_fields:
        raise RuntimeError(f"invalid COCO identity schema: {manifest_path}")
    if (
        recorded["schema_version"] != COCO_IDENTITY_SCHEMA_VERSION
        or recorded["source"] != "official COCO val2014"
        or recorded["image_id"] != image_id
        or recorded["filename"] != filename
        or recorded["relative_path"] != f"{COCO_IMAGE_DIRECTORY}/{filename}"
        or type(recorded["file_size"]) is not int
        or recorded["file_size"] <= 0
        or not isinstance(recorded["saved_image_sha256"], str)
        or _SHA256_RE.fullmatch(recorded["saved_image_sha256"]) is None
        or not isinstance(recorded["canonical_rgb_sha256"], str)
        or _SHA256_RE.fullmatch(recorded["canonical_rgb_sha256"]) is None
        or recorded["encoded_format"] != "JPEG"
        or recorded["encoded_mode"] != "RGB"
        or type(recorded["width"]) is not int
        or recorded["width"] <= 0
        or type(recorded["height"]) is not int
        or recorded["height"] <= 0
    ):
        raise RuntimeError(f"invalid COCO identity values: {manifest_path}")
    return recorded


def _validate_or_create_coco_identity(
    source_cache_root: Path,
    image_id: int,
    *,
    allow_create: bool,
) -> dict[str, Any]:
    image_path, manifest_path = _coco_identity_paths(source_cache_root, image_id)
    image_path = resolve_source_cache_child(
        source_cache_root,
        image_path.relative_to(source_cache_root).as_posix(),
        require_file=True,
    )
    current = _current_coco_identity(image_path, image_id)
    if manifest_path.exists():
        manifest_path = resolve_source_cache_child(
            source_cache_root,
            manifest_path.relative_to(source_cache_root).as_posix(),
            require_file=True,
        )
        recorded = _read_coco_identity_manifest(manifest_path, image_id)
        if _canonical_json(recorded) != _canonical_json(current):
            raise RuntimeError(
                f"COCO source image changed after first materialization: {image_path}"
            )
    else:
        if not allow_create:
            raise RuntimeError(
                f"COCO source image lacks its immutable identity manifest: {image_path}"
            )
        identity_dir = resolve_source_cache_child(
            source_cache_root, COCO_IDENTITY_DIRECTORY
        )
        identity_dir.mkdir(parents=False, exist_ok=True)
        manifest_path = resolve_source_cache_child(
            source_cache_root,
            f"{COCO_IDENTITY_DIRECTORY}/{image_path.name}.json",
        )
        atomic_write_json(manifest_path, current)
    return current


def validate_coco_identity_cache(
    source_cache_root: Path | str,
    *,
    allow_create: bool,
    allow_pending: bool = False,
) -> list[dict[str, Any]]:
    """Validate every materialized COCO JPEG and its immutable sidecar."""

    if type(allow_create) is not bool or type(allow_pending) is not bool:
        raise TypeError("allow_create and allow_pending must be booleans")
    root = resolve_source_cache_root(source_cache_root, require_existing=True)
    image_dir = resolve_source_cache_child(root, COCO_IMAGE_DIRECTORY)
    identity_dir = resolve_source_cache_child(root, COCO_IDENTITY_DIRECTORY)
    if allow_create:
        image_dir.mkdir(parents=False, exist_ok=True)
        identity_dir.mkdir(parents=False, exist_ok=True)
    elif not image_dir.is_dir() or not identity_dir.is_dir():
        raise RuntimeError(
            "resume requires the COCO image and identity-manifest directories"
        )

    image_ids: set[int] = set()
    if image_dir.is_dir():
        for path in sorted(image_dir.iterdir()):
            if path.is_symlink():
                raise ValueError(f"COCO source-cache image is a symlink: {path}")
            if not path.is_file():
                raise RuntimeError(f"unexpected COCO cache entry: {path}")
            match = _COCO_FILENAME_RE.fullmatch(path.name)
            if match is None:
                raise RuntimeError(f"unexpected COCO cache filename: {path.name}")
            image_ids.add(int(match.group(1)))

    manifest_ids: set[int] = set()
    if identity_dir.is_dir():
        for path in sorted(identity_dir.iterdir()):
            if path.is_symlink():
                raise ValueError(f"COCO identity manifest is a symlink: {path}")
            if not path.is_file() or not path.name.endswith(".jpg.json"):
                raise RuntimeError(f"unexpected COCO identity entry: {path}")
            match = _COCO_FILENAME_RE.fullmatch(path.name[:-5])
            if match is None:
                raise RuntimeError(f"unexpected COCO identity filename: {path.name}")
            manifest_ids.add(int(match.group(1)))
    orphan_manifests = manifest_ids.difference(image_ids)
    if orphan_manifests and not allow_pending:
        raise RuntimeError(
            "COCO identity manifests have no corresponding JPEG: "
            f"{sorted(orphan_manifests)[:10]!r}"
        )
    for image_id in sorted(orphan_manifests):
        _, manifest_path = _coco_identity_paths(root, image_id)
        manifest_path = resolve_source_cache_child(
            root,
            manifest_path.relative_to(root).as_posix(),
            require_file=True,
        )
        _read_coco_identity_manifest(manifest_path, image_id)

    return [
        _validate_or_create_coco_identity(
            root, image_id, allow_create=allow_create
        )
        for image_id in sorted(image_ids)
    ]


def _materialize_coco_transactionally(
    source_cache_root: Path, image_id: int, url: str
) -> Path:
    """Publish identity before JPEG so every interrupted state is recoverable."""

    destination, manifest_path = _coco_identity_paths(source_cache_root, image_id)
    image_dir = resolve_source_cache_child(source_cache_root, COCO_IMAGE_DIRECTORY)
    image_dir.mkdir(parents=True, exist_ok=True)
    resolve_source_cache_child(
        source_cache_root, COCO_IMAGE_DIRECTORY, require_directory=True
    )
    transaction_dir = resolve_source_cache_child(
        source_cache_root, COCO_TRANSACTION_DIRECTORY
    )
    transaction_dir.mkdir(parents=True, exist_ok=True)
    transaction_dir = resolve_source_cache_child(
        source_cache_root,
        COCO_TRANSACTION_DIRECTORY,
        require_directory=True,
    )
    staging = resolve_source_cache_child(
        source_cache_root,
        f"{COCO_TRANSACTION_DIRECTORY}/{destination.name}",
    )
    # Bytes staged before an identity commit are not authoritative.  A hard
    # stop in that window therefore causes an exact re-download, while a stage
    # paired with a committed identity may be reused only after hash equality.
    if staging.exists() and not manifest_path.exists():
        if staging.is_symlink() or not staging.is_file():
            raise RuntimeError(f"invalid COCO transaction stage: {staging}")
        staging.unlink()
    try:
        copy_or_download(staging, url, attempts=4)
    except (OSError, RuntimeError) as error:
        raise RecoverableSourceError(
            f"COCO image {image_id} could not be downloaded: {error}"
        ) from error
    try:
        current = _current_coco_identity(
            staging, image_id, published_path=destination
        )
    except (OSError, RuntimeError) as error:
        if staging.is_file() and not staging.is_symlink():
            staging.unlink()
        raise RecoverableSourceError(
            f"COCO image {image_id} could not be decoded: {error}"
        ) from error

    identity_dir = resolve_source_cache_child(
        source_cache_root, COCO_IDENTITY_DIRECTORY
    )
    identity_dir.mkdir(parents=False, exist_ok=True)
    manifest_path = resolve_source_cache_child(
        source_cache_root,
        f"{COCO_IDENTITY_DIRECTORY}/{destination.name}.json",
    )
    if manifest_path.exists():
        recorded = _read_coco_identity_manifest(manifest_path, image_id)
        if _canonical_json(recorded) != _canonical_json(current):
            raise RuntimeError(
                "COCO bytes fetched while recovering an interrupted transaction "
                f"do not match the committed identity: {destination}"
            )
    else:
        try:
            # Commit the immutable expected bytes first.  If the process stops
            # before os.replace, the manifest-only state is a recoverable
            # transaction journal and the next load re-fetches/verifies it.
            atomic_write_json(manifest_path, current)
        except OSError as error:
            raise RecoverableSourceError(
                f"COCO image {image_id} identity could not be committed: {error}"
            ) from error

    destination, manifest_path = _coco_identity_paths(source_cache_root, image_id)
    if destination.is_symlink() or destination.exists():
        raise RuntimeError(
            f"COCO transaction destination appeared concurrently: {destination}"
        )
    try:
        os.replace(staging, destination)
    except OSError as error:
        raise RecoverableSourceError(
            f"COCO image {image_id} publication was interrupted: {error}"
        ) from error
    _validate_or_create_coco_identity(
        source_cache_root, image_id, allow_create=False
    )
    return destination


def _coco_loader(source_cache_dir: Path, image_id: int) -> Callable[[], Image.Image]:
    filename = f"COCO_val2014_{image_id:012d}.jpg"
    source_cache_root = resolve_source_cache_root(source_cache_dir)
    url = COCO_VAL2014_URL.format(filename=filename)

    def load() -> Image.Image:
        destination, manifest_path = _coco_identity_paths(
            source_cache_root, image_id
        )
        if destination.is_file():
            if not manifest_path.is_file():
                raise RuntimeError(
                    "existing COCO source image lacks the identity manifest "
                    f"created during source preparation: {destination}"
                )
            _validate_or_create_coco_identity(
                source_cache_root,
                image_id,
                allow_create=False,
            )
        elif destination.exists():
            raise RuntimeError(f"COCO source image is not a regular file: {destination}")
        else:
            destination = _materialize_coco_transactionally(
                source_cache_root, image_id, url
            )
        with Image.open(destination) as image:
            return normalized_rgb(image)

    return load


def load_vqav2(
    dataset_name: str,
    source_cache_dir: Path | str,
    seed: int,
    *,
    read_only_source: bool = False,
    resume_source: bool = False,
) -> SourceBundle:
    """Prepare VQAv2 Open or MC representatives from official validation JSON."""

    if dataset_name not in {"VQAv2_Open", "VQAv2_MC"}:
        raise ValueError(f"unsupported VQAv2 dataset name: {dataset_name!r}")
    if type(read_only_source) is not bool or type(resume_source) is not bool:
        raise TypeError("read_only_source and resume_source must be booleans")
    source_cache_dir = resolve_source_cache_root(source_cache_dir)
    questions, annotations, source_metadata = _read_vqav2(
        source_cache_dir, read_only=read_only_source
    )
    validate_coco_identity_cache(
        source_cache_dir,
        allow_create=not (read_only_source or resume_source),
        allow_pending=not read_only_source,
    )
    source_metadata["coco_image_identity"] = {
        "schema_version": COCO_IDENTITY_SCHEMA_VERSION,
        "image_directory": str((source_cache_dir / COCO_IMAGE_DIRECTORY).resolve()),
        "manifest_directory": str(
            (source_cache_dir / COCO_IDENTITY_DIRECTORY).resolve()
        ),
        "identity_fields": [
            "file_size",
            "saved_image_sha256",
            "canonical_rgb_sha256",
            "encoded_format",
            "encoded_mode",
            "width",
            "height",
        ],
        "policy": "first-materialization immutable; exact resume verification",
    }

    annotation_by_qid: dict[int, dict[str, Any]] = {}
    for position, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise SourceSchemaError(f"annotation {position} is not an object")
        _require_columns(
            annotation.keys(), ("question_id", "image_id", "answers"), f"annotation {position}"
        )
        qid = annotation["question_id"]
        if not isinstance(qid, int):
            raise SourceSchemaError(f"annotation {position} question_id is not an integer")
        if qid in annotation_by_qid:
            raise SourceSchemaError(f"duplicate VQAv2 annotation question_id {qid}")
        annotation_by_qid[qid] = annotation

    generic_pool: list[str] = []
    if dataset_name == "VQAv2_MC":
        first_answers: list[str] = []
        # Preserve the legacy pool boundary exactly: the old comprehension
        # filtered literal yes/no answers first and only then sliced the first
        # 20,000 remaining answers.  Ordered deduplication replaces only the
        # nondeterministic list(set(...)); it must not move that boundary.
        for position, annotation in enumerate(annotations):
            values = annotation.get("answers")
            if not isinstance(values, list) or not values:
                raise SourceSchemaError(
                    f"annotation {position} has no answers for the legacy generic pool"
                )
            first = values[0]
            if not isinstance(first, dict) or "answer" not in first:
                raise SourceSchemaError(
                    f"annotation {position} has an invalid first answer object"
                )
            first_answer = str(first["answer"])
            if first_answer in {"yes", "no"}:
                continue
            first_answers.append(first_answer)
            if len(first_answers) == 20000:
                break
        if len(first_answers) != 20000:
            raise SourceSchemaError(
                "VQAv2 validation did not provide 20,000 non-yes/no first answers "
                "required by the frozen legacy generic-pool boundary"
            )
        generic_pool = ordered_deduplicate(first_answers)

    stats = _base_stats(("validation",), len(questions))
    stats["source_unique_images_total"] = len(
        {
            int(question["image_id"])
            for question in questions
            if isinstance(question, dict) and isinstance(question.get("image_id"), int)
        }
    )
    eligible: list[dict[str, Any]] = []
    for source_index, question_row in enumerate(questions):
        if not isinstance(question_row, dict):
            raise SourceSchemaError(f"question {source_index} is not an object")
        _require_columns(
            question_row.keys(), ("question_id", "image_id", "question"), f"question {source_index}"
        )
        qid = question_row["question_id"]
        image_id = question_row["image_id"]
        question = question_row["question"]
        if not isinstance(qid, int) or not isinstance(image_id, int) or not isinstance(question, str):
            raise SourceSchemaError(f"question {source_index} has invalid official field types")
        annotation = annotation_by_qid.get(qid)
        if annotation is None:
            stats["invalid_metadata_count"] += 1
            continue
        if annotation["image_id"] != image_id:
            raise SourceSchemaError(
                f"question/annotation image_id mismatch for question_id {qid}: "
                f"{image_id!r} != {annotation['image_id']!r}"
            )
        raw_answers = annotation.get("answers")
        if not isinstance(raw_answers, list) or not raw_answers:
            stats["invalid_metadata_count"] += 1
            continue
        if is_yes_no_question(question):
            stats["yes_no_removed_count"] += 1
            continue
        answer_texts: list[str] = []
        invalid_answer = False
        for value in raw_answers:
            if not isinstance(value, dict) or "answer" not in value:
                invalid_answer = True
                break
            answer_texts.append(str(value["answer"]))
        if invalid_answer or not any(value.strip() for value in answer_texts):
            stats["invalid_metadata_count"] += 1
            continue
        eligible.append(
            {
                "source_index": source_index,
                "question": question_row,
                "annotation": annotation,
                "answers": answer_texts,
            }
        )

    representatives, duplicates = native_first_by_image_id(
        eligible, lambda value: value["question"]["image_id"]
    )
    stats["source_id_duplicate_count"] = duplicates
    stats["task_prefilter_unique_images"] = len(representatives)
    stats["task_prefilter_source_id_representatives"] = len(representatives)
    stats["task_prefilter_unique_images_basis"] = (
        "metadata/task-filtered native-first reliable source_image_id representatives; "
        "COCO canonical RGB hashes are fetched lazily and enforced for every scanned "
        "representative before model inference, so this is not a complete full-pool "
        "exact-RGB count"
    )
    stats["task_prefilter_exact_rgb_unique_images"] = None
    candidates: list[Candidate] = []
    for value in representatives:
        source_index = int(value["source_index"])
        question_row = value["question"]
        annotation = value["annotation"]
        answers = tuple(str(answer) for answer in value["answers"])
        question = str(question_row["question"])
        qid = int(question_row["question_id"])
        image_id = int(question_row["image_id"])
        loader = _coco_loader(source_cache_dir, image_id)
        provenance = {
            "annotation_answer_type": annotation.get("answer_type"),
            "annotation_question_type": annotation.get("question_type"),
            "coco_url": COCO_VAL2014_URL.format(
                filename=f"COCO_val2014_{image_id:012d}.jpg"
            ),
        }
        if dataset_name == "VQAv2_Open":
            candidates.append(
                Candidate(
                    source_dataset="official VQAv2",
                    source_split="validation",
                    source_index=source_index,
                    source_split_index=source_index,
                    source_question_id=qid,
                    source_image_id=image_id,
                    question=question,
                    answers=answers,
                    task_type="open",
                    prompt=open_prompt(question),
                    mapping_payload=_open_mapping(dataset_name, question, answers),
                    image_loader=loader,
                    provenance=provenance,
                )
            )
            continue

        # Counter and tie-breaking retain the old raw-string semantics.  The
        # shared option generator internally lowercases the option copy, while
        # provenance/mapping keep this original modal answer text.
        ground_truth = _stable_modal_answer(answers)
        try:
            options, answer_letter, option_seed = generate_mc_options(
                ground_truth, generic_pool, seed, qid
            )
        except ValueError:
            stats["invalid_metadata_count"] += 1
            continue
        formatted_question = vqav2_mc_question(question, options)
        provenance = {**provenance, "option_rng_seed": option_seed}
        candidates.append(
            Candidate(
                source_dataset="official VQAv2",
                source_split="validation",
                source_index=source_index,
                source_split_index=source_index,
                source_question_id=qid,
                source_image_id=image_id,
                question=formatted_question,
                answers=(answer_letter,),
                task_type="multiple_choice",
                prompt=vqav2_mc_prompt(formatted_question),
                mapping_payload=_mc_mapping(
                    dataset_name, formatted_question, options, answer_letter, ground_truth
                ),
                image_loader=loader,
                options=tuple(options),
                ground_truth_letter=answer_letter,
                ground_truth_text=ground_truth,
                provenance=provenance,
            )
        )

    source_metadata.update(
        {
            "dataset_id": "official VQAv2 validation",
            "source_splits": ["validation"],
            "questions_count": len(questions),
            "annotations_count": len(annotations),
            "generic_pool_rule": (
                "stable ordered-dedup of the first 20000 non-literal-yes/no first "
                "answers, preserving the legacy filter-then-slice boundary"
                if dataset_name == "VQAv2_MC"
                else None
            ),
            "generic_pool_pre_dedup_count": (
                len(first_answers) if dataset_name == "VQAv2_MC" else None
            ),
            "generic_pool_size": len(generic_pool),
        }
    )
    # MC option validity is part of the task-specific prefilter (13 official
    # representatives normalize to an unusable ground truth in this release).
    stats["task_prefilter_unique_images"] = len(candidates)
    return SourceBundle(dataset_name, candidates, stats, source_metadata)


def load_scienceqa(hf_cache_dir: Path | str | None = None) -> SourceBundle:
    """Prepare ScienceQA in fixed train, validation, test native order."""

    from datasets import load_dataset

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if hf_cache_dir is not None:
        kwargs["cache_dir"] = str(hf_cache_dir)
    dataset_dict = load_dataset(SCIENCEQA_DATASET, **kwargs)
    split_order = ("train", "validation", "test")
    if tuple(name for name in split_order if name in dataset_dict) != split_order:
        raise SourceSchemaError(
            f"ScienceQA must provide train+validation+test; found {list(dataset_dict.keys())!r}"
        )
    required = ("image", "question", "choices", "answer", "hint")
    for split in split_order:
        _require_columns(dataset_dict[split].column_names, required, f"ScienceQA {split}")

    total_rows = sum(len(dataset_dict[split]) for split in split_order)
    stats = _base_stats(split_order, total_rows)
    stats["split_source_qa_records"] = {
        split: len(dataset_dict[split]) for split in split_order
    }
    stats["split_composition"] = {split: 0 for split in split_order}

    candidates: list[Candidate] = []
    seen_hashes: set[str] = set()
    all_non_null_hashes: set[str] = set()
    global_index = 0
    metadata_eligible_with_images = 0
    for split in split_order:
        split_dataset = dataset_dict[split]
        for split_index in range(len(split_dataset)):
            current_global_index = global_index
            global_index += 1
            def materialize_scienceqa() -> tuple[dict[str, Any], str | None]:
                row = split_dataset[split_index]
                image = row["image"]
                if image is None:
                    return row, None
                normalized = normalized_rgb(image)
                return row, canonical_rgb_sha256(normalized)

            row, source_hash = _retry_pool_materialization(
                materialize_scienceqa,
                label=f"ScienceQA {split} row {split_index} image materialization",
            )
            choices = row["choices"]
            answer_index = row["answer"]
            question = row["question"]
            hint = row["hint"]
            if source_hash is None:
                stats["missing_or_corrupt_images"] += 1
                continue
            all_non_null_hashes.add(source_hash)
            stats["source_hash_candidates_seen"] += 1
            if (
                not isinstance(question, str)
                or not question.strip()
                or not isinstance(choices, list)
                or not 2 <= len(choices) <= 6
                or not all(isinstance(choice, str) and choice.strip() for choice in choices)
                or not isinstance(answer_index, int)
                or not 0 <= answer_index < len(choices)
                or not isinstance(hint, str)
            ):
                stats["invalid_metadata_count"] += 1
                continue
            normalized_choices = [official_vqa_process(choice) for choice in choices]
            if (
                any(not choice for choice in normalized_choices)
                or len(set(normalized_choices)) != len(normalized_choices)
            ):
                # The frozen construction scorer removes punctuation/articles and
                # maps small number words.  Choices that collide under that exact
                # normalization do not have a unique construction-time answer.
                stats["invalid_metadata_count"] += 1
                continue
            metadata_eligible_with_images += 1
            if source_hash in seen_hashes:
                stats["exact_image_hash_duplicate_count"] += 1
                continue
            seen_hashes.add(source_hash)
            choice_texts = tuple(str(choice) for choice in choices)
            formatted_question = scienceqa_question(question, choice_texts, hint)
            letter = chr(ord("A") + answer_index)
            answer_text = choice_texts[answer_index]
            candidates.append(
                Candidate(
                    source_dataset="derek-thomas/ScienceQA",
                    source_split=split,
                    source_index=current_global_index,
                    source_split_index=split_index,
                    source_question_id=f"{split}:{split_index}",
                    source_image_id=None,
                    question=formatted_question,
                    answers=(letter,),
                    task_type="multiple_choice",
                    prompt=scienceqa_prompt(formatted_question),
                    mapping_payload=_mc_mapping(
                        "ScienceQA_MC",
                        formatted_question,
                        choice_texts,
                        letter,
                        answer_text,
                    ),
                    image_loader=_hf_loader(split_dataset, split_index),
                    source_canonical_rgb_sha256=source_hash,
                    options=choice_texts,
                    ground_truth_letter=letter,
                    ground_truth_text=answer_text,
                    provenance={
                        "hint": hint,
                        "task": row.get("task"),
                        "grade": row.get("grade"),
                        "subject": row.get("subject"),
                        "topic": row.get("topic"),
                        "category": row.get("category"),
                        "skill": row.get("skill"),
                    },
                )
            )

    stats["metadata_prefilter_image_records"] = metadata_eligible_with_images
    stats["task_prefilter_unique_images"] = len(candidates)
    stats["task_prefilter_unique_images_basis"] = (
        "metadata-valid records with construction-normalized unique options, then "
        "complete cross-split canonical RGB SHA-256 deduplication"
    )
    stats["source_unique_images_total"] = len(all_non_null_hashes)
    stats["source_exact_hash_unique_images"] = len(candidates)
    return SourceBundle(
        "ScienceQA_MC",
        candidates,
        stats,
        {
            "dataset_id": SCIENCEQA_DATASET,
            "dataset_url": "https://huggingface.co/datasets/derek-thomas/ScienceQA",
            "source_splits": list(split_order),
            "schemas": {
                split: list(dataset_dict[split].column_names) for split in split_order
            },
            "native_rows": {
                split: len(dataset_dict[split]) for split in split_order
            },
            "dataset_fingerprints": {
                split: getattr(dataset_dict[split], "_fingerprint", None)
                for split in split_order
            },
            "cache_files": {
                split: _hf_cache_file_manifest(dataset_dict[split])
                for split in split_order
            },
        },
    )


__all__ = [
    "Candidate",
    "SourceBundle",
    "SourceSchemaError",
    "RecoverableSourceError",
    "load_textvqa",
    "load_vqav2",
    "load_scienceqa",
    "native_first_by_image_id",
    "native_first_by_canonical_hash",
    "prepare_vqav2_source",
]
