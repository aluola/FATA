"""Explicit replacement for the missing historical attack-cache helper."""

from __future__ import annotations

import os
import math
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any

from fata.attacks.linf_image import IMAGE_SERIALIZATION_CONTRACT
from fata.utils.run_contract import (
    artifact_identity,
    artifact_identity_valid,
    dataset_identity_valid,
    dataset_image_identity,
    is_sha256_digest,
    sha256_file,
)


def baseline_attack_definition(attack: str) -> str:
    """Return the immutable implementation identifier for a cached baseline."""

    normalized = str(attack).strip().lower()
    if normalized == "caa":
        return "llava_caa_layer1_region30_w10_10_2_5_v1"
    if normalized == "cage":
        return "llava_cage_efd_rda_lam0.005_k16_192_v1"
    raise ValueError(f"no cached baseline definition for {attack!r}")


def baseline_attack_contract_extra(
    attack: str,
    *,
    max_input_tokens: int = 0,
) -> dict:
    """Return the pixel-affecting CAA/CAGE objective as structured metadata."""

    if (
        isinstance(max_input_tokens, bool)
        or not isinstance(max_input_tokens, int)
        or max_input_tokens < 0
    ):
        raise ValueError("max_input_tokens must be a non-negative integer")
    normalized = str(attack).strip().lower()
    if normalized == "caa":
        objective = {
            "target_layer": 1,
            "least_important_region_fraction": 0.30,
            "weights": {
                "bpr_inter": 10.0,
                "bpr_intra": 10.0,
                "semantic": 2.0,
                "question_answer": 5.0,
            },
        }
    elif normalized == "cage":
        objective = {
            "lambda_cage": 0.005,
            "k_min": 16,
            "k_max": 192,
        }
    else:
        raise ValueError(f"no structured baseline objective contract for {attack!r}")
    return {
        "max_input_tokens": max_input_tokens,
        "objective": objective,
    }


def attack_namespace(
    attack: str,
    *,
    seed: int,
    eps_255: float,
    alpha_255: float,
    steps: int,
) -> str:
    """Encode cache-changing attack parameters in the cache namespace."""

    if (
        steps <= 0
        or not math.isfinite(float(eps_255))
        or not math.isfinite(float(alpha_255))
        or eps_255 <= 0
        or alpha_255 <= 0
    ):
        raise ValueError("epsilon, alpha, and steps must be finite and positive")
    return _safe(
        f"{attack}_eps{eps_255:g}_a{alpha_255:g}_s{steps}_seed{seed}"
    )


def validate_embedded_baseline_cache_contract(
    source: Any,
    *,
    cache_namespace: Any,
    attack: str,
    dataset: str,
    method: str,
    seed: int,
    eps_255: float,
    alpha_255: float,
    steps: int,
    result_dataset_mapping_sha256: str,
    result_dataset_images: dict[str, Any],
    result_model: dict[str, Any],
    result_clip_model: dict[str, Any],
    max_input_tokens: int = 0,
) -> None:
    """Validate a self-contained CAA/CAGE cache lineage embedded in a result.

    Consumers call this after loading only JSON metadata.  The exact cache
    objective, dataset/model bytes and namespace therefore remain auditable
    even when a result is copied away from its original attack-cache tree.
    """

    normalized = str(attack).strip().lower()
    if normalized not in {"caa", "cage"}:
        raise ValueError(f"unsupported cached baseline attack: {attack!r}")
    expected_keys = {
        "schema_version",
        "image_serialization",
        "definition",
        "dataset",
        "dataset_mapping_sha256",
        "dataset_images",
        "method",
        "model",
        "clip_model",
        "seed",
        "eps_255",
        "alpha_255",
        "steps",
        "extra",
    }
    if not isinstance(source, dict) or set(source) != expected_keys:
        raise ValueError("cached baseline source contract has an unexpected schema")
    if not is_sha256_digest(result_dataset_mapping_sha256):
        raise ValueError("result dataset mapping identity is invalid")
    if not dataset_identity_valid(result_dataset_images):
        raise ValueError("result dataset image identity is invalid")
    if not artifact_identity_valid(result_model) or not artifact_identity_valid(
        result_clip_model
    ):
        raise ValueError("result model identity is invalid")
    source_clip = None if normalized == "caa" else result_clip_model
    expected_values = {
        "schema_version": 1,
        "image_serialization": IMAGE_SERIALIZATION_CONTRACT,
        "definition": baseline_attack_definition(normalized),
        "dataset": dataset,
        "dataset_mapping_sha256": result_dataset_mapping_sha256,
        "dataset_images": result_dataset_images,
        "method": method,
        "model": result_model,
        "clip_model": source_clip,
        "seed": int(seed),
        "eps_255": float(eps_255),
        "alpha_255": float(alpha_255),
        "steps": int(steps),
        "extra": baseline_attack_contract_extra(
            normalized, max_input_tokens=max_input_tokens
        ),
    }
    for key, expected in expected_values.items():
        if source.get(key) != expected:
            raise ValueError(f"cached baseline source contract mismatch: {key}")
    if not dataset_identity_valid(source["dataset_images"]):
        raise ValueError("cached baseline source dataset identity is invalid")
    if not artifact_identity_valid(source["model"]):
        raise ValueError("cached baseline source model identity is invalid")
    if source["clip_model"] is not None and not artifact_identity_valid(
        source["clip_model"]
    ):
        raise ValueError("cached baseline source CLIP identity is invalid")
    expected_namespace = attack_namespace(
        normalized,
        seed=int(seed),
        eps_255=float(eps_255),
        alpha_255=float(alpha_255),
        steps=int(steps),
    )
    if cache_namespace != expected_namespace:
        raise ValueError("cached baseline namespace mismatch")


def _safe(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not rendered:
        raise ValueError("empty cache path component")
    return rendered


def _safe_relative_image_path(image_filename: str) -> PurePosixPath:
    """Validate a dataset-relative POSIX image path without flattening it."""

    raw = str(image_filename)
    relative = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or relative.is_absolute()
        or relative.as_posix() != raw
        or any(part in {"", ".", ".."} or _safe(part) != part for part in relative.parts)
    ):
        raise ValueError(f"unsafe relative image filename: {image_filename!r}")
    return relative


def _rooted_cache_path(cache_root: str | Path, *parts: str) -> Path:
    """Resolve a cache path below one immutable root without following children.

    Resolving the method directory and then treating that resolved directory as
    the containment anchor is unsafe: an attacker-controlled symlink at the
    attack/dataset/method level can move the anchor outside ``cache_root``.
    Anchor first, reject every existing child symlink, and only then resolve the
    complete candidate path.
    """

    root = Path(cache_root).expanduser().resolve()
    candidate = root.joinpath(*parts)
    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"cache path contains a symbolic link: {current}")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f"cache path escapes root: {candidate}")
    return resolved


def _cache_companion_path(
    cache_root: str | Path, primary: Path, suffix: str
) -> Path:
    root = Path(cache_root).expanduser().resolve()
    try:
        relative = primary.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"cache path escapes root: {primary}") from exc
    return _rooted_cache_path(
        cache_root, *relative.parts[:-1], relative.name + suffix
    )


def attack_image_path(*, cache_root, method, dataset, attack, image_filename) -> Path:
    """Return the canonical lossless cache path for one mapped image.

    Dataset-relative subdirectories are preserved, and ``.png`` is appended
    to the full source filename (for example ``images/a.jpg.png``). Appending
    instead of replacing the suffix avoids collisions between source files
    that differ only by extension.
    """

    if cache_root is None:
        raise ValueError("cache_root is required")
    relative = _safe_relative_image_path(str(image_filename))
    return _rooted_cache_path(
        cache_root,
        _safe(attack),
        _safe(dataset),
        _safe(method),
        *relative.parts[:-1],
        relative.name + ".png",
    )


def attack_cache_contract_path(*, cache_root, method, dataset, attack) -> Path:
    if cache_root is None:
        raise ValueError("cache_root is required")
    return _rooted_cache_path(
        cache_root,
        _safe(attack),
        _safe(dataset),
        _safe(method),
        "CACHE_CONTRACT.json",
    )


def attack_cache_directory_path(*, cache_root, method, dataset, attack) -> Path:
    """Return a symlink-safe cache namespace directory below ``cache_root``."""

    if cache_root is None:
        raise ValueError("cache_root is required")
    return _rooted_cache_path(
        cache_root,
        _safe(attack),
        _safe(dataset),
        _safe(method),
    )


def build_attack_cache_contract(
    *,
    definition: str,
    dataset: str,
    mapping_path: str | Path,
    method: str,
    model_path: str | Path,
    clip_path: str | Path | None,
    seed: int,
    eps_255: float,
    alpha_255: float,
    steps: int,
    extra: dict | None = None,
) -> dict:
    """Build the producer/consumer contract for a cache namespace."""

    contract = {
        "schema_version": 1,
        "image_serialization": IMAGE_SERIALIZATION_CONTRACT,
        "definition": definition,
        "dataset": dataset,
        "dataset_mapping_sha256": sha256_file(mapping_path),
        "dataset_images": dataset_image_identity(mapping_path),
        "method": method,
        "model": artifact_identity(model_path),
        "clip_model": None if clip_path is None else artifact_identity(clip_path),
        "seed": int(seed),
        "eps_255": float(eps_255),
        "alpha_255": float(alpha_255),
        "steps": int(steps),
    }
    if extra is not None:
        contract["extra"] = extra
    return contract


def save_attack_image(*, image, cache_root, method, dataset, attack, image_filename) -> Path:
    output = attack_image_path(
        cache_root=cache_root, method=method, dataset=dataset,
        attack=attack, image_filename=image_filename,
    )
    destination = output.parent
    destination.mkdir(parents=True, exist_ok=True)
    # Revalidate after creating parents so a pre-existing or concurrently
    # substituted child symlink is not trusted for the write.
    output = attack_image_path(
        cache_root=cache_root, method=method, dataset=dataset,
        attack=attack, image_filename=image_filename,
    )
    digest_path = _cache_companion_path(cache_root, output, ".sha256")
    image_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp.", dir=destination
    )
    temporary = Path(temporary_name)
    digest_fd = -1
    digest_temporary: Path | None = None
    try:
        with os.fdopen(image_fd, "wb") as handle:
            image.save(handle, format="PNG")
            handle.flush()
            os.fsync(handle.fileno())
        digest_fd, digest_name = tempfile.mkstemp(
            prefix=f".{digest_path.name}.tmp.", dir=destination
        )
        digest_temporary = Path(digest_name)
        with os.fdopen(digest_fd, "w", encoding="ascii") as handle:
            digest_fd = -1
            handle.write(sha256_file(temporary) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Refuse a final-path symlink that appeared after the initial check.
        attack_image_path(
            cache_root=cache_root, method=method, dataset=dataset,
            attack=attack, image_filename=image_filename,
        )
        _cache_companion_path(cache_root, output, ".sha256")
        temporary.replace(output)
        digest_temporary.replace(digest_path)
    finally:
        if digest_fd >= 0:
            os.close(digest_fd)
        if temporary.exists():
            temporary.unlink()
        if digest_temporary is not None and digest_temporary.exists():
            digest_temporary.unlink()
    return output


def load_attack_image(*, cache_root, method, dataset, attack, image_filename):
    from PIL import Image

    path = attack_image_path(
        cache_root=cache_root, method=method, dataset=dataset,
        attack=attack, image_filename=image_filename,
    )
    if not attack_image_exists(
        cache_root=cache_root,
        method=method,
        dataset=dataset,
        attack=attack,
        image_filename=image_filename,
    ):
        raise RuntimeError(f"cached attack image is missing, corrupt, or unhashed: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def attack_image_exists(*, cache_root, method, dataset, attack, image_filename) -> bool:
    if cache_root is None:
        return False
    path = attack_image_path(
        cache_root=cache_root, method=method, dataset=dataset,
        attack=attack, image_filename=image_filename,
    )
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    digest_path = _cache_companion_path(cache_root, path, ".sha256")
    if not digest_path.is_file():
        return False
    try:
        expected_digest = digest_path.read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            return False
        if sha256_file(path) != expected_digest:
            return False
        from PIL import Image

        with Image.open(path) as image:
            image_format = image.format
            image_mode = image.mode
            image_size = image.size
            image.verify()
        if image_format != "PNG" or image_mode != "RGB":
            return False
        if len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
            return False
        return True
    except (OSError, ValueError):
        return False
