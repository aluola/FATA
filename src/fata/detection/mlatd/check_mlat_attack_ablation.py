"""Fail-closed completeness check for the release CAGE/CAA ML-ATD grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fata.attacks.linf_image import IMAGE_SERIALIZATION_CONTRACT
from fata.runtimes.llava.attack_cache_io import (
    attack_cache_contract_path,
    attack_cache_directory_path,
    attack_image_path,
    attack_namespace,
    baseline_attack_contract_extra,
)
from fata.utils.run_contract import sha256_file
from fata.utils.paths import resolve_owned_output_path
from .check_mlat_full_grid import (
    _artifact_identity_valid,
    _dataset_identity_valid,
    _is_sha256,
    validate_triplet,
)


DATASETS = (
    "TextVQA_Open",
    "VQAv2_Open",
    "ScienceQA_MC",
    "VQAv2_MC",
)
METHODS = (
    "VisionZIP",
    "VisPruner",
    "PruMerge",
    "FlowCut",
)
ATTACKS = ("cage", "caa")
PRACTICAL_K = {
    "TextVQA_Open": 64,
    "VQAv2_Open": 64,
    "ScienceQA_MC": 32,
    "VQAv2_MC": 32,
}

Issue = tuple[str, ...]


def practical_k(method: str, dataset: str) -> int:
    if method == "FlowCut" and dataset == "VQAv2_MC":
        return 64
    return PRACTICAL_K[dataset]


def _selection(value: str, allowed: Sequence[str]) -> list[str]:
    if value.strip().lower() in {"all", "*"}:
        return list(allowed)
    selected = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [item for item in selected if item not in allowed]
    if not selected or unknown:
        raise ValueError(
            f"invalid selection {value!r}; allowed={','.join(allowed)}"
        )
    return selected


def _attack_parameters(args: argparse.Namespace, attack: str) -> tuple[float, float, int]:
    return (
        float(getattr(args, f"{attack}_eps_255")),
        float(getattr(args, f"{attack}_alpha_255")),
        int(getattr(args, f"{attack}_steps")),
    )


def _contract_issues(
    path: Path,
    *,
    dataset: str,
    seed: int,
    eps_255: float,
    alpha_255: float,
    steps: int,
    max_input_tokens: int,
) -> list[Issue]:
    if not path.is_file():
        return [(str(path), "missing_cache_contract")]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [(str(path), "invalid_cache_contract", str(exc))]
    if not isinstance(payload, dict):
        return [(str(path), "invalid_cache_contract", "expected JSON object")]

    attack = path.parents[2].name.split("_eps", 1)[0]
    expected: dict[str, Any] = {
        "schema_version": 1,
        "image_serialization": IMAGE_SERIALIZATION_CONTRACT,
        "definition": f"mlatd_{attack}_generator_v1",
        "dataset": dataset,
        "method": "shared",
        "seed": seed,
        "eps_255": eps_255,
        "alpha_255": alpha_255,
        "steps": steps,
    }
    issues: list[Issue] = []
    for key, wanted in expected.items():
        actual = payload.get(key)
        if isinstance(wanted, float):
            try:
                matches = math.isfinite(float(actual)) and float(actual) == wanted
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == wanted
        if not matches:
            issues.append(
                (str(path), "cache_contract_mismatch", key, repr(actual), repr(wanted))
            )
    extra = payload.get("extra")
    wanted_extra = baseline_attack_contract_extra(
        attack,
        max_input_tokens=max_input_tokens,
    )
    if extra != wanted_extra:
        issues.append(
            (
                str(path),
                "cache_contract_mismatch",
                "extra",
                repr(extra),
                repr(wanted_extra),
            )
        )
    if not _is_sha256(payload.get("dataset_mapping_sha256")):
        issues.append((str(path), "invalid_cache_identity", "dataset_mapping_sha256"))
    if not _dataset_identity_valid(payload.get("dataset_images")):
        issues.append((str(path), "invalid_cache_identity", "dataset_images"))
    for key in ("model", "clip_model"):
        if not _artifact_identity_valid(payload.get(key)):
            issues.append((str(path), "invalid_cache_identity", key))
    return issues


def check_cache_directory(
    cache_root: Path,
    *,
    namespace: str,
    dataset: str,
    expected_count: int,
    seed: int,
    eps_255: float,
    alpha_255: float,
    steps: int,
    max_input_tokens: int,
) -> tuple[int, list[Issue], list[str]]:
    """Validate the recursive canonical shared-PNG cache directory."""

    try:
        directory = attack_cache_directory_path(
            cache_root=cache_root,
            attack=namespace,
            dataset=dataset,
            method="shared",
        )
    except ValueError as exc:
        return 0, [(str(cache_root), "unsafe_cache_directory", str(exc))], []
    if not directory.is_dir():
        return 0, [(str(directory), "missing_cache_directory")], []

    issues = _contract_issues(
        attack_cache_contract_path(
            cache_root=cache_root,
            attack=namespace,
            dataset=dataset,
            method="shared",
        ),
        dataset=dataset,
        seed=seed,
        eps_255=eps_255,
        alpha_255=alpha_255,
        steps=steps,
        max_input_tokens=max_input_tokens,
    )
    png_files = sorted(
        path for path in directory.rglob("*.png") if path.is_file()
    )
    cache_image_ids = [
        path.relative_to(directory).as_posix()[: -len(".png")]
        for path in png_files
    ]
    other_files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path != directory / "CACHE_CONTRACT.json"
        and path.suffix != ".png"
        and not path.name.endswith(".png.sha256")
    )
    for path in other_files:
        issues.append((str(path), "unexpected_cache_file"))

    for path in png_files:
        if path.is_symlink():
            issues.append((str(path), "cache_image_is_symlink"))
            continue
        if path.stat().st_size <= 0:
            issues.append((str(path), "empty_cache_image"))
            continue
        try:
            image_id = path.relative_to(directory).as_posix()[: -len(".png")]
            expected_path = attack_image_path(
                cache_root=cache_root,
                attack=namespace,
                dataset=dataset,
                method="shared",
                image_filename=image_id,
            )
            if path.resolve() != expected_path:
                issues.append((str(path), "cache_image_path_mismatch"))
                continue
            digest_path = path.with_name(path.name + ".sha256")
            if digest_path.is_symlink():
                issues.append((str(digest_path), "cache_digest_is_symlink"))
                continue
            expected_digest = digest_path.read_text(encoding="ascii").strip()
            if len(expected_digest) != 64 or sha256_file(path) != expected_digest:
                issues.append((str(path), "cache_image_digest_mismatch"))
            with Image.open(path) as image:
                image_format = image.format
                image_mode = image.mode
                image_size = image.size
                image.verify()
            if image_format != "PNG":
                issues.append((str(path), "cache_image_is_not_png", repr(image_format)))
            if image_mode != "RGB":
                issues.append((str(path), "cache_image_is_not_rgb", repr(image_mode)))
            if len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
                issues.append((str(path), "cache_image_has_invalid_size", repr(image_size)))
        except (OSError, ValueError) as exc:
            issues.append((str(path), "invalid_cache_image", str(exc)))

    for digest_path in directory.rglob("*.png.sha256"):
        image_path = digest_path.with_name(digest_path.name[: -len(".sha256")])
        if not image_path.is_file():
            issues.append((str(digest_path), "orphan_cache_digest"))

    if len(png_files) != expected_count:
        issues.append(
            (
                str(directory),
                "cache_count_mismatch",
                f"found={len(png_files)}",
                f"expected={expected_count}",
            )
        )
    return len(png_files), issues, cache_image_ids


def _feature_stem(
    *,
    method: str,
    dataset: str,
    namespace: str,
    token_budget: int,
    start: int,
    limit: int,
    seed: int,
) -> str:
    return (
        f"mlat_feat_{method}_{dataset}_{namespace}_k{token_budget}_"
        f"start{start}_limit{limit}_seed{seed}"
    )


def check_feature_pair(
    npz_path: Path,
    *,
    method: str,
    dataset: str,
    attack: str,
    namespace: str,
    token_budget: int,
    token_budget_mode: str,
    target_k: int,
    expected_start: int,
    expected_rows: int,
    seed: int,
    eps_255: float,
    alpha_255: float,
    steps: int,
    source_cache_contract: dict[str, Any],
    cache_root: Path,
) -> list[Issue]:
    raw_issues = validate_triplet(
        npz_path,
        method=method,
        dataset=dataset,
        attack=attack,
        token_budget=token_budget,
        start=expected_start,
        limit=expected_rows,
        seed=seed,
        eps_255=eps_255,
        alpha_255=alpha_255,
        steps=steps,
        token_budget_mode=token_budget_mode,
        target_k=target_k,
        expected_cache_root=cache_root,
    )
    issues: list[Issue] = [
        (str(npz_path), "feature_triplet_error", message) for message in raw_issues
    ]
    if raw_issues:
        return issues
    meta_path = npz_path.with_suffix(".meta.json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("cache_namespace") != namespace:
        issues.append((str(meta_path), "cache_namespace_mismatch"))
    if metadata.get("source_cache_contract") != source_cache_contract:
        issues.append((str(meta_path), "source_cache_contract_mismatch"))
    cross_fields = {
        "image_serialization": "image_serialization",
        "dataset_mapping_sha256": "dataset_mapping_sha256",
        "dataset_images": "dataset_images",
        "llava_model": "model",
        "clip_model": "clip_model",
        "seed": "seed",
        "eps_255": "eps_255",
        "alpha_255": "alpha_255",
        "steps": "steps",
    }
    for feature_key, source_key in cross_fields.items():
        if metadata.get(feature_key) != source_cache_contract.get(source_key):
            issues.append((str(meta_path), "source_feature_contract_mismatch", feature_key))
    return issues


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root", "--cache_root", dest="cache_root", type=Path, required=True
    )
    parser.add_argument(
        "--result-dir", "--result_dir", dest="result_dir", type=Path, required=True
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--attacks", default="cage,caa")
    parser.add_argument("--total_limit", type=int, default=1000)
    parser.add_argument("--chunk_size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cage_eps_255", type=float, default=2.0)
    parser.add_argument("--cage_alpha_255", type=float, default=0.5)
    parser.add_argument("--cage_steps", type=int, default=100)
    parser.add_argument("--caa_eps_255", type=float, default=2.0)
    parser.add_argument("--caa_alpha_255", type=float, default=1.0)
    parser.add_argument("--caa_steps", type=int, default=100)
    parser.add_argument("--max_input_tokens", type=int, default=0)
    parser.add_argument(
        "--token_budget_mode",
        choices=("practical",),
        default="practical",
        help="Formal release mode; uncompressed/full was not part of the paper grid.",
    )
    parser.add_argument("--target_k", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.total_limit <= 0 or args.chunk_size <= 0 or args.target_k <= 0 or args.seed < 0:
        parser.error("limits/K must be positive and --seed non-negative")
    if args.max_input_tokens < 0:
        parser.error("--max_input_tokens must be non-negative")
    for attack in ATTACKS:
        eps_255, alpha_255, steps = _attack_parameters(args, attack)
        if (
            not math.isfinite(eps_255)
            or not math.isfinite(alpha_255)
            or eps_255 <= 0
            or alpha_255 <= 0
            or steps <= 0
        ):
            parser.error("attack epsilon, alpha, and steps must be positive")
    if (
        args.max_input_tokens != 0
        or args.cage_eps_255 != 2.0
        or args.cage_alpha_255 != 0.5
        or args.cage_steps != 100
        or args.caa_eps_255 != 2.0
        or args.caa_alpha_255 != 1.0
        or args.caa_steps != 100
        or args.target_k != 64
    ):
        parser.error(
            "formal attack ablation is locked to max_input_tokens=0, "
            "target_k=64, CAGE=2/0.5/100, and CAA=2/1/100"
        )
    try:
        datasets = _selection(args.datasets, DATASETS)
        methods = _selection(args.methods, METHODS)
        attacks = _selection(args.attacks, ATTACKS)
    except ValueError as exc:
        parser.error(str(exc))

    namespaces: dict[str, str] = {}
    cache_contracts: dict[tuple[str, str], dict[str, Any]] = {}
    cache_image_ids: dict[tuple[str, str], list[str]] = {}
    result_root = args.result_dir.expanduser().resolve()
    if not result_root.is_dir():
        parser.error(f"--result-dir is not a directory: {result_root}")
    issues: list[Issue] = []
    print("=== Attack cache coverage ===")
    for attack in attacks:
        eps_255, alpha_255, steps = _attack_parameters(args, attack)
        namespace = attack_namespace(
            attack,
            seed=args.seed,
            eps_255=eps_255,
            alpha_255=alpha_255,
            steps=steps,
        )
        namespaces[attack] = namespace
        for dataset in datasets:
            count, cache_issues, cached_ids = check_cache_directory(
                args.cache_root,
                namespace=namespace,
                dataset=dataset,
                expected_count=args.total_limit,
                seed=args.seed,
                eps_255=eps_255,
                alpha_255=alpha_255,
                steps=steps,
                max_input_tokens=args.max_input_tokens,
            )
            print(f"{namespace} {dataset}: png={count} expected={args.total_limit}")
            issues.extend(cache_issues)
            cache_image_ids[(attack, dataset)] = cached_ids
            contract_path = (
                args.cache_root / namespace / dataset / "shared" / "CACHE_CONTRACT.json"
            )
            if contract_path.is_file():
                try:
                    loaded_contract = json.loads(contract_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_contract, dict):
                        cache_contracts[(attack, dataset)] = loaded_contract
                except (OSError, json.JSONDecodeError):
                    pass

    expected_feature_files = 0
    global_models: dict[str, Any] | None = None
    dataset_identities: dict[str, dict[str, Any]] = {}
    chunk_image_ids: dict[tuple[str, int, int], list[str]] = {}
    feature_ids_by_cache: dict[tuple[str, str], list[str]] = {}
    for method in methods:
        for dataset in datasets:
            token_budget = 576 if args.token_budget_mode == "full" else args.target_k
            if args.token_budget_mode == "practical":
                token_budget = practical_k(method, dataset)
            for attack in attacks:
                eps_255, alpha_255, steps = _attack_parameters(args, attack)
                namespace = namespaces[attack]
                for start in range(0, args.total_limit, args.chunk_size):
                    limit = min(args.chunk_size, args.total_limit - start)
                    stem = _feature_stem(
                        method=method,
                        dataset=dataset,
                        namespace=namespace,
                        token_budget=token_budget,
                        start=start,
                        limit=limit,
                        seed=args.seed,
                    )
                    expected_feature_files += 1
                    source_contract = cache_contracts.get((attack, dataset))
                    if source_contract is None:
                        issues.append((str(args.result_dir / f"{stem}.npz"), "unavailable_source_cache_contract"))
                        continue
                    try:
                        npz_path = resolve_owned_output_path(
                            result_root, f"{stem}.npz"
                        )
                    except ValueError as exc:
                        issues.append((stem, "unsafe_result_path", str(exc)))
                        continue
                    pair_issues = check_feature_pair(
                            npz_path,
                            method=method,
                            dataset=dataset,
                            attack=attack,
                            namespace=namespace,
                            token_budget=token_budget,
                            token_budget_mode=args.token_budget_mode,
                            target_k=args.target_k,
                            expected_start=start,
                            expected_rows=limit,
                            seed=args.seed,
                            eps_255=eps_255,
                            alpha_255=alpha_255,
                            steps=steps,
                            source_cache_contract=source_contract,
                            cache_root=args.cache_root,
                        )
                    issues.extend(pair_issues)
                    if pair_issues:
                        continue
                    meta = json.loads(
                        npz_path.with_suffix(".meta.json").read_text(encoding="utf-8")
                    )
                    models = {"llava_model": meta["llava_model"], "clip_model": meta["clip_model"]}
                    if global_models is not None and models != global_models:
                        issues.append((str(npz_path), "cross_grid_model_identity_mismatch"))
                    if global_models is None:
                        global_models = models
                    identity = {
                        "dataset_mapping_sha256": meta["dataset_mapping_sha256"],
                        "dataset_images": meta["dataset_images"],
                    }
                    if dataset in dataset_identities and identity != dataset_identities[dataset]:
                        issues.append((str(npz_path), "cross_grid_dataset_identity_mismatch"))
                    dataset_identities.setdefault(dataset, identity)
                    ids = [str(value) for value in meta["expected_image_ids"]]
                    chunk_key = (dataset, start, limit)
                    if chunk_key in chunk_image_ids and ids != chunk_image_ids[chunk_key]:
                        issues.append((str(npz_path), "cross_grid_image_id_mismatch"))
                    chunk_image_ids.setdefault(chunk_key, ids)
                    feature_ids_by_cache.setdefault((attack, dataset), []).extend(ids)

    for key, cached_ids in cache_image_ids.items():
        feature_ids = feature_ids_by_cache.get(key, [])
        if (
            len(feature_ids) != args.total_limit * len(methods)
            or set(feature_ids) != set(cached_ids)
            or len(cached_ids) != len(set(cached_ids))
        ):
            issues.append((str(args.cache_root), "cache_feature_image_id_mismatch", *key))

    print(f"expected feature NPZ/CSV/meta triples: {expected_feature_files}")
    print(f"issues: {len(issues)}")
    for issue in issues[:100]:
        print(issue)
    if len(issues) > 100:
        print(f"... {len(issues) - 100} additional issues omitted")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
