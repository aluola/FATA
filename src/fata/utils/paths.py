"""Centralized, explicit path configuration with no server-specific defaults."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from pathlib import Path, PurePosixPath
from typing import Mapping


def safe_filename_component(value: str, *, label: str = "value") -> str:
    """Validate a user-controlled value used as one filename component."""

    rendered = str(value)
    if (
        not rendered
        or rendered in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", rendered) is None
    ):
        raise ValueError(f"unsafe {label}: {value!r}")
    return rendered


def metadata_sidecar_path(result_path: str | Path) -> Path:
    """Return a sibling ``.meta.json`` path without rewriting parent names."""

    path = Path(result_path)
    if path.suffix != ".csv":
        raise ValueError(f"result path must end in .csv: {path}")
    return path.with_suffix(".meta.json")


def resolve_owned_output_path(
    output_root: str | Path,
    relative_path: str | PurePosixPath,
) -> Path:
    """Resolve one program-owned path without trusting child symlinks.

    The configured root itself is resolved first (so an explicitly selected
    mounted/symlinked root remains usable).  Every existing component beneath
    that anchor must then be a real directory/file rather than a symlink.
    """

    root = Path(output_root).expanduser().resolve()
    raw = str(relative_path)
    relative = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or relative.is_absolute()
        or relative.as_posix() != raw
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"unsafe output-relative path: {relative_path!r}")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"output path contains a symbolic link: {current}")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f"output-relative path escapes root: {relative_path!r}")
    return resolved


def create_owned_output_directory(
    output_root: str | Path,
    relative_path: str | PurePosixPath,
) -> Path:
    """Create and revalidate a real directory below a configured output root."""

    destination = resolve_owned_output_path(output_root, relative_path)
    destination.mkdir(parents=True, exist_ok=True)
    destination = resolve_owned_output_path(output_root, relative_path)
    if not destination.is_dir():
        raise ValueError(f"output path is not a directory: {destination}")
    return destination


def assert_no_output_file_collision(
    outputs: Mapping[str, str | Path],
    protected_inputs: Mapping[str, str | Path],
) -> None:
    """Reject exact file collisions before an atomic writer can replace input."""

    resolved_outputs = {
        label: Path(path).expanduser().resolve() for label, path in outputs.items()
    }
    if len(set(resolved_outputs.values())) != len(resolved_outputs):
        raise ValueError(f"duplicate output file paths: {resolved_outputs}")
    for output_label, output in resolved_outputs.items():
        for input_label, raw_input in protected_inputs.items():
            protected = Path(raw_input).expanduser().resolve()
            if output == protected:
                raise ValueError(
                    f"output file {output_label} would overwrite protected "
                    f"{input_label}: {protected}"
                )


def resolve_attack_cache_root(
    *,
    explicit_cache_root: str | Path | None,
    output_root: str | Path | None,
    explicit_output_dir: str | Path | None,
) -> Path:
    """Resolve an attack cache without ever constructing ``Path(None)``."""

    if explicit_cache_root:
        return Path(explicit_cache_root).expanduser().resolve()
    if output_root:
        return resolve_owned_output_path(output_root, "adversarial_images")
    if explicit_output_dir:
        output_dir = Path(explicit_output_dir).expanduser().resolve()
        return resolve_owned_output_path(output_dir.parent, "adversarial_images")
    raise ValueError("provide an attack cache root, output root, or output directory")


def resolve_dataset_relative_path(
    dataset_dir: str | Path,
    relative_path: str,
) -> Path:
    """Resolve a canonical POSIX mapping path inside one dataset directory."""

    root = Path(dataset_dir).expanduser().resolve()
    raw = str(relative_path)
    relative = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or relative.is_absolute()
        or relative.as_posix() != raw
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"unsafe dataset-relative path: {relative_path!r}")
    resolved = root.joinpath(*relative.parts).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"dataset-relative path escapes root: {relative_path!r}")
    return resolved


def assert_output_separate(
    output: str | Path,
    protected: Mapping[str, str | Path],
) -> Path:
    """Reject output roots that overlap any immutable input/cache tree."""

    resolved_output = Path(output).expanduser().resolve()
    for label, raw_path in protected.items():
        resolved_input = Path(raw_path).expanduser().resolve()
        if (
            resolved_output == resolved_input
            or resolved_output.is_relative_to(resolved_input)
            or resolved_input.is_relative_to(resolved_output)
        ):
            raise ValueError(
                f"output path {resolved_output} overlaps protected {label}: {resolved_input}"
            )
    return resolved_output


@dataclass(frozen=True)
class PathConfig:
    model_root: Path
    data_root: Path
    output_root: Path
    cache_root: Path

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> "PathConfig":
        source = os.environ if env is None else env
        names = ("FATA_MODEL_ROOT", "FATA_DATA_ROOT", "FATA_OUTPUT_ROOT", "FATA_CACHE_ROOT")
        missing = [name for name in names if not source.get(name)]
        if missing:
            raise ValueError("missing required path variables: " + ", ".join(missing))
        return cls(*(Path(source[name]).expanduser().resolve() for name in names))

    def validate(self, *, require_inputs: bool = True, create_output: bool = False) -> None:
        assert_output_separate(
            self.output_root,
            {
                "model_root": self.model_root,
                "data_root": self.data_root,
                "cache_root": self.cache_root,
            },
        )
        if require_inputs:
            for label, path in (("model_root", self.model_root), ("data_root", self.data_root)):
                if not path.is_dir():
                    raise FileNotFoundError(f"{label} is not a directory: {path}")
        if create_output:
            self.output_root.mkdir(parents=True, exist_ok=True)
        elif not self.output_root.is_dir():
            raise FileNotFoundError(f"output_root is not a directory: {self.output_root}")
        if not self.cache_root.is_dir():
            raise FileNotFoundError(f"cache_root is not a directory: {self.cache_root}")

    def as_environment(self) -> dict[str, str]:
        return {
            "FATA_MODEL_ROOT": str(self.model_root),
            "FATA_DATA_ROOT": str(self.data_root),
            "FATA_OUTPUT_ROOT": str(self.output_root),
            "FATA_CACHE_ROOT": str(self.cache_root),
        }
