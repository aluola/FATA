#!/usr/bin/env python3
"""CPU-only downstream dataset-loader smoke test.

This checks the legacy LLaVA JSONL/path contract and the authoritative
InternVL ``load_mapping`` implementation.  It deliberately imports no model,
attack, or compressor code.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from PIL import Image, ImageOps
from fata.utils.paths import assert_output_separate, resolve_owned_output_path


DATASETS: tuple[str, ...] = (
    "TextVQA_Open",
    "VQAv2_Open",
    "VQAv2_MC",
    "ScienceQA_MC",
)
EXPECTED_TYPES = {
    "TextVQA_Open": "open",
    "VQAv2_Open": "open",
    "VQAv2_MC": "multiple_choice",
    "ScienceQA_MC": "multiple_choice",
}
COMMON_FIELDS = ("image_filename", "dataset", "type", "question", "answers")
MC_FIELDS = ("options", "ground_truth_text")
INTERNVL_DATASETS_MODULE = "fata.runtimes.internvl35.evaluation.datasets"
JSON_REPORT = "downstream_loader_smoke_test.json"
MARKDOWN_REPORT = "downstream_loader_smoke_test.md"
MAX_REPORTED_ERRORS = 50


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate all four FATA mappings with the legacy LLaVA contract "
            "and InternVL evaluation.datasets.load_mapping (CPU/file I/O only)."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root containing <dataset>/<dataset>_mapping.jsonl.",
    )
    parser.add_argument(
        "--expected-count",
        type=_positive_int,
        default=1000,
        help="Required mapping rows per dataset (default: 1000).",
    )
    parser.add_argument(
        "--sample-limit",
        type=_positive_int,
        default=2,
        help="Prefix rows per dataset to schema-check and image-decode (default: 2).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=(
            Path(os.environ["FATA_OUTPUT_ROOT"]) / "loader_smoke"
            if os.environ.get("FATA_OUTPUT_ROOT")
            else None
        ),
        help="Writable report directory (or set FATA_OUTPUT_ROOT); never writes into the dataset root.",
    )
    return parser


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path = resolve_owned_output_path(path.parent, path.name)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        fd = -1
        path = resolve_owned_output_path(path.parent, path.name)
        os.replace(temporary, path)
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary.exists():
            temporary.unlink()


def _load_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
    return rows


def _safe_image_path(dataset_dir: Path, image_filename: Any) -> Path:
    if not isinstance(image_filename, str) or not image_filename.strip():
        raise ValueError("image_filename must be a non-empty string")
    if "\\" in image_filename:
        raise ValueError("image_filename must use POSIX separators")
    relative = PurePosixPath(image_filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("image_filename must be a safe relative path")
    image_path = (dataset_dir / Path(*relative.parts)).resolve()
    try:
        image_path.relative_to(dataset_dir)
    except ValueError as exc:
        raise ValueError("image_filename escapes the dataset directory") from exc
    return image_path


def _decode_image(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    # This mirrors the effective downstream operation and forces full decode,
    # rather than merely accepting a readable image header.
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        rgb.load()


def _record_error(result: dict[str, Any], message: str) -> None:
    result["error_count"] += 1
    if len(result["errors"]) < MAX_REPORTED_ERRORS:
        result["errors"].append(message)


def _validate_legacy_contract(
    dataset_root: Path,
    dataset: str,
    expected_count: int,
    sample_limit: int,
) -> tuple[dict[str, Any], list[Any]]:
    dataset_dir = (dataset_root / dataset).resolve()
    mapping_path = dataset_dir / f"{dataset}_mapping.jsonl"
    result: dict[str, Any] = {
        "status": "fail",
        "required_fields": list(COMMON_FIELDS + (MC_FIELDS if dataset.endswith("_MC") else ())),
        "rows_checked": 0,
        "images_opened": 0,
        "error_count": 0,
        "errors": [],
    }

    if not mapping_path.is_file():
        _record_error(result, f"mapping file not found: {mapping_path}")
        return result, []

    try:
        rows = _load_jsonl(mapping_path)
    except Exception as exc:  # keep the report available on malformed input
        _record_error(result, f"mapping read failed: {type(exc).__name__}: {exc}")
        return result, []

    result["row_count"] = len(rows)
    if len(rows) != expected_count:
        _record_error(
            result,
            f"mapping has {len(rows)} rows; expected {expected_count}",
        )

    expected_type = EXPECTED_TYPES[dataset]
    required = COMMON_FIELDS + (MC_FIELDS if expected_type == "multiple_choice" else ())
    checked_rows = rows[:sample_limit]
    result["sample_limit"] = sample_limit
    for index, raw in enumerate(checked_rows):
        result["rows_checked"] += 1
        location = f"row {index + 1}"
        if not isinstance(raw, dict):
            _record_error(result, f"{location}: mapping record must be an object")
            continue

        missing = [field for field in required if field not in raw]
        if missing:
            _record_error(result, f"{location}: missing fields: {', '.join(missing)}")

        if raw.get("dataset") != dataset:
            _record_error(
                result,
                f"{location}: dataset must be {dataset!r}, got {raw.get('dataset')!r}",
            )
        if raw.get("type") != expected_type:
            _record_error(
                result,
                f"{location}: type must be {expected_type!r}, got {raw.get('type')!r}",
            )
        if not isinstance(raw.get("question"), str) or not raw.get("question", "").strip():
            _record_error(result, f"{location}: question must be a non-empty string")
        answers = raw.get("answers")
        if (
            not isinstance(answers, list)
            or not answers
            or not all(isinstance(answer, str) for answer in answers)
        ):
            _record_error(result, f"{location}: answers must be a non-empty string list")

        if expected_type == "multiple_choice":
            options = raw.get("options")
            if (
                not isinstance(options, list)
                or not options
                or not all(isinstance(option, str) for option in options)
            ):
                _record_error(result, f"{location}: options must be a non-empty string list")
            ground_truth = raw.get("ground_truth_text")
            if not isinstance(ground_truth, str) or not ground_truth.strip():
                _record_error(
                    result,
                    f"{location}: ground_truth_text must be a non-empty string",
                )
            if isinstance(options, list) and all(
                isinstance(option, str) for option in options
            ):
                expected_option_count = 4 if dataset == "VQAv2_MC" else None
                if expected_option_count is not None and len(options) != expected_option_count:
                    _record_error(
                        result,
                        f"{location}: VQAv2_MC must have exactly 4 options",
                    )
                if dataset == "ScienceQA_MC" and not 2 <= len(options) <= 6:
                    _record_error(
                        result,
                        f"{location}: ScienceQA_MC must have 2 through 6 options",
                    )
                if isinstance(answers, list) and len(answers) == 1:
                    letter = answers[0].strip().upper() if isinstance(answers[0], str) else ""
                    option_index = ord(letter) - ord("A") if len(letter) == 1 else -1
                    if option_index < 0 or option_index >= len(options):
                        _record_error(
                            result,
                            f"{location}: MC answer must be one in-range option letter",
                        )
                    elif isinstance(ground_truth, str) and ground_truth != options[option_index]:
                        _record_error(
                            result,
                            f"{location}: ground_truth_text does not match the answer option",
                        )
                elif isinstance(answers, list):
                    _record_error(result, f"{location}: MC answers must contain one letter")

        try:
            image_path = _safe_image_path(dataset_dir, raw.get("image_filename"))
            _decode_image(image_path)
            result["images_opened"] += 1
        except Exception as exc:
            _record_error(
                result,
                f"{location}: dataset_dir/image_filename cannot be opened: "
                f"{type(exc).__name__}: {exc}",
            )

    if result["error_count"] == 0:
        result["status"] = "pass"
    return result, rows


def _internvl_loader(module_name: str) -> Callable[..., list[Any]]:
    module = importlib.import_module(module_name)
    loader = getattr(module, "load_mapping", None)
    if not callable(loader):
        raise AttributeError(f"{module_name} does not expose callable load_mapping")
    return loader


def _validate_internvl_contract(
    loader: Callable[..., list[Any]] | None,
    loader_error: str | None,
    dataset_root: Path,
    dataset: str,
    expected_count: int,
    legacy_rows: list[Any],
    sample_limit: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "fail",
        "expected_count_argument": expected_count,
        "error_count": 0,
        "errors": [],
    }
    if loader is None:
        _record_error(result, loader_error or "InternVL loader import failed")
        return result

    try:
        samples = loader(dataset_root, dataset, expected_count=expected_count)
        result["loaded_count"] = len(samples)
        if len(samples) != expected_count:
            _record_error(
                result,
                f"load_mapping returned {len(samples)} samples; expected {expected_count}",
            )

        # Confirm that the loader preserves authoritative JSONL order and
        # resolves exactly the path used by the LLaVA contract.
        if len(legacy_rows) == len(samples):
            dataset_dir = (dataset_root / dataset).resolve()
            for index, (raw, sample) in enumerate(
                zip(legacy_rows[:sample_limit], samples[:sample_limit])
            ):
                if not isinstance(raw, dict):
                    continue
                filename = raw.get("image_filename")
                if sample.image_filename != filename:
                    _record_error(
                        result,
                        f"row {index + 1}: InternVL changed image_filename order/value",
                    )
                    continue
                try:
                    expected_path = _safe_image_path(dataset_dir, filename)
                except ValueError:
                    continue
                if sample.image_path != expected_path:
                    _record_error(
                        result,
                        f"row {index + 1}: InternVL image_path mismatch: "
                        f"{sample.image_path} != {expected_path}",
                    )
                expected_values = {
                    "dataset": dataset,
                    "sample_index": index,
                    "image_filename": str(filename),
                    "task_type": str(raw.get("type")),
                    "question": str(raw.get("question")),
                    "answers": tuple(str(value) for value in raw.get("answers", ())),
                    "options": tuple(str(value) for value in raw.get("options", ())),
                    "ground_truth_text": (
                        None
                        if raw.get("ground_truth_text") is None
                        else str(raw.get("ground_truth_text"))
                    ),
                    "image_id": Path(str(filename)).stem,
                }
                for attribute, expected in expected_values.items():
                    actual = getattr(sample, attribute, object())
                    if actual != expected:
                        _record_error(
                            result,
                            f"row {index + 1}: InternVL {attribute} mismatch: "
                            f"{actual!r} != {expected!r}",
                        )
        elif legacy_rows:
            _record_error(
                result,
                "InternVL sample count differs from independently parsed mapping rows",
            )
    except Exception as exc:
        _record_error(
            result,
            f"load_mapping failed: {type(exc).__name__}: {exc}",
        )

    if result["error_count"] == 0:
        result["status"] = "pass"
    return result


def _markdown(report: dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# Downstream loader smoke test",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Dataset root: `{report['dataset_root']}`",
        f"- Expected rows per dataset: `{report['expected_count']}`",
        f"- InternVL loader: `{report['internvl_loader']}`",
        f"- Generated (UTC): `{report['generated_at_utc']}`",
        "- Scope: JSONL/schema/path/image-decode checks only; no model, attack, or compressor was run.",
        "",
        "| Dataset | Rows | LLaVA contract | Images opened | InternVL load_mapping |",
        "|---|---:|---|---:|---|",
    ]
    for item in report["datasets"]:
        legacy = item["legacy_llava_contract"]
        internvl = item["internvl_load_mapping"]
        lines.append(
            "| {dataset} | {rows} | {legacy} | {images} | {internvl} |".format(
                dataset=cell(item["dataset"]),
                rows=cell(legacy.get("row_count", "n/a")),
                legacy=cell(legacy["status"]),
                images=cell(legacy["images_opened"]),
                internvl=cell(internvl["status"]),
            )
        )

    failures = [item for item in report["datasets"] if item["status"] != "pass"]
    if failures:
        lines.extend(["", "## Failures", ""])
        for item in failures:
            lines.append(f"### {item['dataset']}")
            lines.append("")
            errors = (
                item["legacy_llava_contract"]["errors"]
                + item["internvl_load_mapping"]["errors"]
            )
            for error in errors:
                lines.append(f"- {error}")
            omitted = (
                item["legacy_llava_contract"]["error_count"]
                + item["internvl_load_mapping"]["error_count"]
                - len(errors)
            )
            if omitted > 0:
                lines.append(f"- … {omitted} additional errors omitted from this report")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run(
    dataset_root: Path,
    expected_count: int,
    sample_limit: int,
    report_dir: Path,
) -> tuple[dict[str, Any], Path, Path]:
    root = dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    report_dir = assert_output_separate(
        report_dir, {"dataset root": root}
    )

    loader: Callable[..., list[Any]] | None = None
    loader_error: str | None = None
    try:
        loader = _internvl_loader(INTERNVL_DATASETS_MODULE)
    except Exception as exc:
        loader_error = f"{type(exc).__name__}: {exc}"

    dataset_results: list[dict[str, Any]] = []
    for dataset in DATASETS:
        legacy, rows = _validate_legacy_contract(
            root, dataset, expected_count, sample_limit
        )
        internvl = _validate_internvl_contract(
            loader,
            loader_error,
            root,
            dataset,
            expected_count,
            rows,
            sample_limit,
        )
        status = (
            "pass"
            if legacy["status"] == "pass" and internvl["status"] == "pass"
            else "fail"
        )
        dataset_results.append(
            {
                "dataset": dataset,
                "status": status,
                "dataset_dir": str((root / dataset).resolve()),
                "mapping_path": str(
                    (root / dataset / f"{dataset}_mapping.jsonl").resolve()
                ),
                "legacy_llava_contract": legacy,
                "internvl_load_mapping": internvl,
            }
        )

    overall = "pass" if all(item["status"] == "pass" for item in dataset_results) else "fail"
    report: dict[str, Any] = {
        "schema_version": 1,
        "check": "downstream_loader_smoke_test",
        "status": overall,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "expected_count": expected_count,
        "sample_limit": sample_limit,
        "datasets_checked": list(DATASETS),
        "internvl_loader": str(INTERNVL_DATASETS_MODULE),
        "internvl_loader_import_status": "pass" if loader is not None else "fail",
        "internvl_loader_import_error": loader_error,
        "model_or_attack_executed": False,
        "datasets": dataset_results,
    }

    json_path = resolve_owned_output_path(report_dir, JSON_REPORT)
    markdown_path = resolve_owned_output_path(report_dir, MARKDOWN_REPORT)
    _atomic_write_text(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(markdown_path, _markdown(report))
    return report, json_path, markdown_path


def main() -> int:
    args = _parser().parse_args()
    if args.report_dir is None:
        _parser().error("provide --report-dir or set FATA_OUTPUT_ROOT")
    try:
        report, json_path, markdown_path = run(
            args.dataset_root,
            args.expected_count,
            args.sample_limit,
            args.report_dir,
        )
    except Exception as exc:
        print(f"loader smoke setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"status={report['status']}")
    print(f"json_report={json_path}")
    print(f"markdown_report={markdown_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
