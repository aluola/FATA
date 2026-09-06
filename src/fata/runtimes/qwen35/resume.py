"""Fail-closed CSV resume accounting for the Qwen V6 grid."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from fata.evaluation.scoring import score_vqa
from .evaluation.output_parsing import extract_final_answer


GroupKey = tuple[str, str]
CellKey = tuple[str, str]

SUCCESS_REQUIRED_FIELDS = frozenset(
    {
        "dataset",
        "config",
        "sample_index",
        "image_id",
        "compressor",
        "retention_label",
        "retention_ratio",
        "N_full",
        "K_actual",
        "clean_full_raw",
        "fata_full_raw",
        "clean_full_correct",
        "fata_full_correct",
        "clean_compressed_raw",
        "fata_compressed_raw",
        "clean_compressed_correct",
        "fata_compressed_correct",
        "delta_linf",
        "seed",
        "initial_delta_sha256",
        "delta_sha256",
        "epsilon",
        "alpha",
        "steps",
        "task_mode",
        "w_task",
        "w_rank",
        "w_comp",
        "w_full",
        "w_distill",
        "use_projection",
        "full_clean_GT_loss",
        "full_adv_GT_loss",
        "compressed_adv_GT_loss",
        "full_logit_KL",
        "mean_grad_cos_comp_full",
        "projection_trigger_rate",
        "attack_seconds",
        "status",
    }
)
RAW_ANSWER_FIELDS = frozenset(
    {
        "clean_full_raw",
        "fata_full_raw",
        "clean_compressed_raw",
        "fata_compressed_raw",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{16}$")
# ``attack_v6`` measures a float32 perturbation after pixel-range clipping.
# Share its numerical allowance with resume validation so the producer cannot
# emit a success row that the consumer rejects solely due to float32 rounding.
DELTA_LINF_ATOL = 1e-5
GROUP_SHARED_FIELDS = (
    "dataset",
    "config",
    "sample_index",
    "image_id",
    "N_full",
    "clean_full_raw",
    "fata_full_raw",
    "clean_full_correct",
    "fata_full_correct",
    "delta_linf",
    "seed",
    "initial_delta_sha256",
    "delta_sha256",
    "epsilon",
    "alpha",
    "steps",
    "task_mode",
    "w_task",
    "w_rank",
    "w_comp",
    "w_full",
    "w_distill",
    "use_projection",
    "full_clean_GT_loss",
    "full_adv_GT_loss",
    "compressed_adv_GT_loss",
    "full_logit_KL",
    "mean_grad_cos_comp_full",
    "projection_trigger_rate",
    "status",
)


def expected_cells(
    methods: Iterable[str], budgets: Iterable[str]
) -> frozenset[CellKey]:
    """Return the exact compressor-by-budget cell universe."""

    return frozenset((method, budget) for method in methods for budget in budgets)


def read_successful_cells(
    csv_path: str | Path,
    *,
    methods: Iterable[str],
    budgets: Iterable[str],
    expected_header: Sequence[str] | None = None,
    expected_dataset: str | None = None,
    allowed_configs: Iterable[str] | None = None,
    allowed_image_ids: Iterable[str] | None = None,
    sample_index_by_id: dict[str, int] | None = None,
    expected_seed_by_id: dict[str, int] | None = None,
    config_contracts: dict[str, dict[str, Any]] | None = None,
    scoring_targets_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[GroupKey, set[CellKey]]:
    """Read successful cells, rejecting duplicates and out-of-contract values.

    Non-success rows are retry diagnostics and do not make a group complete.
    A duplicate success cell is ambiguous evidence, so resume stops instead of
    silently blessing a corrupted or concatenated CSV.
    """

    path = Path(csv_path)
    if not path.exists() or path.stat().st_size == 0:
        return {}

    expected = expected_cells(methods, budgets)
    allowed_config_set = None if allowed_configs is None else set(allowed_configs)
    allowed_image_id_set = (
        None if allowed_image_ids is None else set(allowed_image_ids)
    )
    successes: defaultdict[GroupKey, set[CellKey]] = defaultdict(set)
    group_fingerprints: dict[GroupKey, tuple[Any, ...]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if expected_header is not None and reader.fieldnames != list(expected_header):
            raise ValueError(f"resume CSV {path} has an unexpected header")
        required = {
            "image_id",
            "config",
            "compressor",
            "retention_label",
            "status",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"resume CSV {path} is missing columns: {sorted(missing)}"
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"resume CSV {path}:{line_number} has a malformed row width"
                )
            image_id = row.get("image_id", "")
            config_name = row.get("config", "")
            if allowed_image_id_set is not None and image_id not in allowed_image_id_set:
                raise ValueError(
                    f"resume CSV {path}:{line_number} image_id {image_id!r} "
                    "is outside the immutable selected cohort"
                )
            if expected_dataset is not None and row.get("dataset") != expected_dataset:
                raise ValueError(
                    f"resume CSV {path}:{line_number} has a different dataset"
                )
            if allowed_config_set is not None and config_name not in allowed_config_set:
                raise ValueError(
                    f"resume CSV {path}:{line_number} has an unknown config {config_name!r}"
                )
            if sample_index_by_id is not None:
                if image_id not in sample_index_by_id:
                    raise ValueError(
                        f"resume CSV {path}:{line_number} has an unknown image_id {image_id!r}"
                    )
                try:
                    row_index = int(row.get("sample_index", ""))
                except ValueError as error:
                    raise ValueError(
                        f"resume CSV {path}:{line_number} has an invalid sample_index"
                    ) from error
                if row_index != int(sample_index_by_id[image_id]):
                    raise ValueError(
                        f"resume CSV {path}:{line_number} sample_index disagrees with manifest"
                    )
            if row.get("status") != "success":
                continue
            group = (image_id, config_name)
            cell = (row.get("compressor", ""), row.get("retention_label", ""))
            if not all(group):
                raise ValueError(
                    f"resume CSV {path}:{line_number} has an empty group key"
                )
            if cell not in expected:
                raise ValueError(
                    f"resume CSV {path}:{line_number} has out-of-contract "
                    f"success cell {cell!r}"
                )
            contract = None
            if config_contracts is not None:
                if config_name not in config_contracts:
                    raise ValueError(
                        f"resume CSV {path}:{line_number} has no contract for "
                        f"config {config_name!r}"
                    )
                contract = config_contracts[config_name]
                _validate_success_config(path, line_number, row, contract)
            fingerprint = _validate_success_row(
                path,
                line_number,
                row,
                contract=contract,
            )
            if scoring_targets_by_id is not None:
                target = scoring_targets_by_id.get(image_id)
                if target is None:
                    raise ValueError(
                        f"resume CSV {path}:{line_number} has no scoring target "
                        f"for image_id {image_id!r}"
                    )
                _validate_success_scores(
                    path,
                    line_number,
                    row,
                    target=target,
                )
            if expected_seed_by_id is not None:
                if image_id not in expected_seed_by_id:
                    raise ValueError(
                        f"resume CSV {path}:{line_number} has no expected seed "
                        f"for image_id {image_id!r}"
                    )
                if _integer(path, line_number, row, "seed") != int(
                    expected_seed_by_id[image_id]
                ):
                    raise ValueError(
                        f"resume CSV {path}:{line_number} seed disagrees with "
                        "the deterministic manifest seed"
                    )
            if group in group_fingerprints and group_fingerprints[group] != fingerprint:
                raise ValueError(
                    f"resume CSV {path}:{line_number} has inconsistent "
                    "sample/seed/hash/attack identity within a success group"
                )
            group_fingerprints.setdefault(group, fingerprint)
            if cell in successes[group]:
                raise ValueError(
                    f"resume CSV {path}:{line_number} duplicates successful "
                    f"cell {cell!r} for group {group!r}"
                )
            successes[group].add(cell)
    return dict(successes)


def _fail(path: Path, line_number: int, message: str) -> None:
    raise ValueError(f"resume CSV {path}:{line_number} {message}")


def _finite_float(
    path: Path,
    line_number: int,
    row: dict[str | None, str | list[str] | None],
    field: str,
) -> float:
    try:
        value = float(str(row.get(field, "")).strip())
    except (TypeError, ValueError):
        _fail(path, line_number, f"has invalid {field}")
    if not math.isfinite(value):
        _fail(path, line_number, f"has non-finite {field}")
    return value


def _integer(
    path: Path,
    line_number: int,
    row: dict[str | None, str | list[str] | None],
    field: str,
) -> int:
    raw = str(row.get(field, "")).strip()
    if not re.fullmatch(r"[+-]?\d+", raw):
        _fail(path, line_number, f"has invalid integer {field}")
    return int(raw)


def _boolean(
    path: Path,
    line_number: int,
    row: dict[str | None, str | list[str] | None],
    field: str,
) -> bool:
    raw = str(row.get(field, "")).strip().lower()
    if raw in {"1", "true"}:
        return True
    if raw in {"0", "false"}:
        return False
    _fail(path, line_number, f"has invalid boolean {field}")


def _correctness_score(
    path: Path,
    line_number: int,
    row: dict[str | None, str | list[str] | None],
    field: str,
    *,
    is_mc: bool,
) -> float:
    """Validate the score range emitted by the canonical VQA scorers."""

    value = _finite_float(path, line_number, row, field)
    if not 0.0 <= value <= 1.0:
        _fail(path, line_number, f"has {field} outside [0, 1]")
    if is_mc and value not in {0.0, 1.0}:
        _fail(path, line_number, f"has non-binary MC score {field}")
    return value


def _validate_success_scores(
    path: Path,
    line_number: int,
    row: dict[str | None, str | list[str] | None],
    *,
    target: dict[str, Any],
) -> None:
    """Recompute every stored correctness value from raw output + manifest GT."""

    is_mc = target.get("is_mc")
    references = target.get("reference_answers")
    if not isinstance(is_mc, bool) or not isinstance(references, list) or not references:
        _fail(path, line_number, "has an invalid manifest scoring target")
    normalized_references = [str(value) for value in references]
    gt_letter = normalized_references[0].strip().upper() if is_mc else ""
    for raw_field, score_field in (
        ("clean_full_raw", "clean_full_correct"),
        ("fata_full_raw", "fata_full_correct"),
        ("clean_compressed_raw", "clean_compressed_correct"),
        ("fata_compressed_raw", "fata_compressed_correct"),
    ):
        extracted = extract_final_answer(str(row.get(raw_field, "")))[0]
        if is_mc:
            pred = str(extracted).strip().lower()
            match = re.search(
                r"(?i)(?:^|\s|\()(option\s+)?([a-f])(?:\)|\.|:|\s|$)", pred
            )
            answer = match.group(2).lower() if match else (pred[0] if pred else "")
            expected = 1.0 if answer == gt_letter.strip().lower() else 0.0
        else:
            expected = score_vqa(extracted, normalized_references)
        recorded = _correctness_score(
            path, line_number, row, score_field, is_mc=is_mc
        )
        if not math.isclose(recorded, expected, rel_tol=0.0, abs_tol=1e-12):
            _fail(
                path,
                line_number,
                f"has {score_field} inconsistent with {raw_field} and manifest target",
            )


def _validate_success_row(
    path: Path,
    line_number: int,
    row: dict[str | None, str | list[str] | None],
    *,
    contract: dict[str, Any] | None,
) -> tuple[Any, ...]:
    """Validate every field that turns a row into resumable success evidence."""

    missing = sorted(field for field in SUCCESS_REQUIRED_FIELDS if field not in row)
    if missing:
        _fail(path, line_number, f"is missing success columns: {missing}")

    # Empty generated text is a valid immediate-EOS model response.  The four
    # raw-answer columns must exist, but unlike every other success field they
    # are deliberately allowed to contain the empty string.
    empty = sorted(
        field
        for field in SUCCESS_REQUIRED_FIELDS.difference(RAW_ANSWER_FIELDS)
        if not str(row.get(field, "")).strip()
    )
    if empty:
        _fail(path, line_number, f"has empty success fields: {empty}")
    if str(row.get("error", "")).strip():
        _fail(path, line_number, "has a non-empty error field on a success row")

    sample_index = _integer(path, line_number, row, "sample_index")
    n_full = _integer(path, line_number, row, "N_full")
    k_actual = _integer(path, line_number, row, "K_actual")
    seed = _integer(path, line_number, row, "seed")
    steps = _integer(path, line_number, row, "steps")
    if sample_index < 0:
        _fail(path, line_number, "has negative sample_index")
    if n_full <= 0:
        _fail(path, line_number, "has non-positive N_full")
    if not 1 <= k_actual <= n_full:
        _fail(path, line_number, "has K_actual outside [1, N_full]")
    if not 0 <= seed <= 0x7FFFFFFF:
        _fail(path, line_number, "has seed outside deterministic-seed range")
    if steps <= 0:
        _fail(path, line_number, "has non-positive steps")

    epsilon = _finite_float(path, line_number, row, "epsilon")
    alpha = _finite_float(path, line_number, row, "alpha")
    delta_linf = _finite_float(path, line_number, row, "delta_linf")
    retention = _finite_float(path, line_number, row, "retention_ratio")
    attack_seconds = _finite_float(path, line_number, row, "attack_seconds")
    if epsilon <= 0:
        _fail(path, line_number, "has non-positive epsilon")
    if not 0 < alpha <= epsilon:
        _fail(path, line_number, "has alpha outside (0, epsilon]")
    if delta_linf < 0 or delta_linf > epsilon + DELTA_LINF_ATOL:
        _fail(path, line_number, "has delta_linf outside [0, epsilon]")
    if attack_seconds <= 0:
        _fail(path, line_number, "has non-positive attack_seconds")

    for field in ("w_task", "w_rank", "w_comp", "w_full", "w_distill"):
        if _finite_float(path, line_number, row, field) < 0:
            _fail(path, line_number, f"has negative {field}")
    for field in (
        "full_clean_GT_loss",
        "full_adv_GT_loss",
        "compressed_adv_GT_loss",
    ):
        if _finite_float(path, line_number, row, field) < 0:
            _fail(path, line_number, f"has negative {field}")
    if _finite_float(path, line_number, row, "full_logit_KL") < -1e-8:
        _fail(path, line_number, "has materially negative full_logit_KL")
    grad_cos = _finite_float(path, line_number, row, "mean_grad_cos_comp_full")
    if not -1.0 <= grad_cos <= 1.0:
        _fail(path, line_number, "has mean_grad_cos_comp_full outside [-1, 1]")
    trigger_rate = _finite_float(path, line_number, row, "projection_trigger_rate")
    if not 0.0 <= trigger_rate <= 1.0:
        _fail(path, line_number, "has projection_trigger_rate outside [0, 1]")

    if contract is not None and "is_mc" in contract:
        if not isinstance(contract["is_mc"], bool):
            _fail(path, line_number, "has non-boolean is_mc in config contract")
        is_mc = contract["is_mc"]
    else:
        is_mc = str(row.get("dataset", "")).endswith("_MC")

    correctness: dict[str, float] = {}
    for field in (
        "clean_full_correct",
        "fata_full_correct",
        "clean_compressed_correct",
        "fata_compressed_correct",
    ):
        correctness[field] = _correctness_score(
            path, line_number, row, field, is_mc=is_mc
        )
    _boolean(path, line_number, row, "use_projection")
    if str(row.get("task_mode", "")) not in {"ground_truth", "proxy", "none"}:
        _fail(path, line_number, "has unknown task_mode")

    initial_hash = str(row.get("initial_delta_sha256", "")).strip()
    final_hash = str(row.get("delta_sha256", "")).strip()
    if not SHA256_RE.fullmatch(initial_hash):
        _fail(path, line_number, "has invalid initial_delta_sha256")
    if not SHA256_RE.fullmatch(final_hash):
        _fail(path, line_number, "has invalid delta_sha256")

    budget = str(row.get("retention_label", ""))
    if budget == "Full":
        expected_retention = 1.0
        for compressed, full in (
            ("clean_compressed_raw", "clean_full_raw"),
            ("fata_compressed_raw", "fata_full_raw"),
        ):
            if row.get(compressed) != row.get(full):
                _fail(
                    path,
                    line_number,
                    f"has Full {compressed} different from {full}",
                )
        for compressed, full in (
            ("clean_compressed_correct", "clean_full_correct"),
            ("fata_compressed_correct", "fata_full_correct"),
        ):
            if correctness[compressed] != correctness[full]:
                _fail(
                    path,
                    line_number,
                    f"has Full {compressed} different from {full}",
                )
    elif budget == "1/3":
        expected_retention = 1.0 / 3.0
    elif budget == "Practical":
        expected_retention = 1.0 / 18.0 if is_mc else 1.0 / 9.0
    else:
        _fail(path, line_number, f"has unknown retention_label {budget!r}")
    if not math.isclose(retention, expected_retention, rel_tol=1e-9, abs_tol=1e-12):
        _fail(path, line_number, "retention_ratio disagrees with retention_label")
    expected_k = max(1, round(n_full * expected_retention))
    if k_actual != expected_k:
        _fail(path, line_number, "K_actual disagrees with N_full/retention_label")

    # These values are produced once per attack and copied into all twelve
    # compressor/budget cells. Exact textual equality is intentional: a CSV
    # assembled from different attacks or partially rerun configurations must
    # not be accepted as one complete group. Wall-clock attack_seconds is the
    # sole shared-looking field excluded because legitimate retries can differ.
    return tuple(
        str(row.get(field, "")).strip() for field in GROUP_SHARED_FIELDS
    )


def _validate_success_config(
    path: Path,
    line_number: int,
    row: dict[str | None, str | list[str] | None],
    contract: dict[str, Any],
) -> None:
    """Require success-row hyperparameters to agree with the frozen config."""

    columns = {
        "epsilon": "epsilon",
        "alpha": "alpha",
        "steps": "steps",
        "task_mode": "task_mode",
        "w_task": "w_task",
        "w_rank": "w_rank",
        "w_comp": "w_comp",
        "w_full": "w_full",
        "w_distill": "w_distill",
        "use_projection": "use_projection",
    }
    for column, key in columns.items():
        actual = row.get(column)
        expected = contract[key]
        try:
            if isinstance(expected, bool):
                normalized = str(actual).strip().lower()
                matches = normalized in ({"1", "true"} if expected else {"0", "false"})
            elif isinstance(expected, (int, float)):
                matches = math.isclose(
                    float(str(actual)), float(expected), rel_tol=1e-9, abs_tol=1e-12
                )
            else:
                matches = str(actual) == str(expected)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(
                f"resume CSV {path}:{line_number} column {column} does not match config"
            )
    if contract.get("use_projection") is False and _finite_float(
        path, line_number, row, "projection_trigger_rate"
    ) != 0.0:
        raise ValueError(
            f"resume CSV {path}:{line_number} projection_trigger_rate must be zero "
            "when use_projection is false"
        )


def completed_groups(
    successes: dict[GroupKey, set[CellKey]],
    *,
    methods: Iterable[str],
    budgets: Iterable[str],
) -> set[GroupKey]:
    """Return only groups with the entire expected cell universe present."""

    expected = expected_cells(methods, budgets)
    return {group for group, cells in successes.items() if cells == expected}
