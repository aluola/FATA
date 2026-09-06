from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import random
import re
import shutil
import struct
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, ImageOps


DATASET_NAMES = ("TextVQA_Open", "VQAv2_Open", "VQAv2_MC", "ScienceQA_MC")
SEED = 20260904
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
)
OPEN_SUFFIX = "\nAnswer the question using a single word or phrase. ASSISTANT:"
MC_SUFFIX = "Answer with the option's letter from the given choices directly. ASSISTANT:"
CANONICAL_HASH_DESCRIPTION = (
    "SHA256(b'RGB\\0' + width_uint64_be + height_uint64_be + "
    "EXIF-transposed RGB pixel bytes)"
)
JPEG_SETTINGS = {
    "format": "JPEG",
    "quality": 75,
    "subsampling": 2,
    "optimize": False,
    "progressive": False,
}


def configure_reproducibility(seed: int = SEED) -> None:
    """Configure all RNGs used by the builder.

    PYTHONHASHSEED is recorded/set for child processes, while correctness never
    relies on hash-table iteration order in the current process.
    """

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink():
            raise RuntimeError(f"refusing to replace symbolic link: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, payload: str) -> None:
    atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, payload)


def append_jsonl_durable(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        os.write(descriptor, line)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    """Read complete JSONL records, tolerating only a truncated final line."""

    if not path.exists():
        return []
    raw = path.read_bytes()
    records: list[dict[str, Any]] = []
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            is_last = index == len(lines) - 1
            if is_last and not line.endswith((b"\n", b"\r")):
                break
            raise
    return records


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def model_directory_fingerprint(model_path: Path | str) -> dict[str, Any]:
    """Hash every regular model artifact so resume cannot silently mix weights."""

    root = Path(model_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise RuntimeError(f"model directory contains no regular files: {root}")
    aggregate_payload = json.dumps(
        files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "path": str(root),
        "algorithm": "sha256-per-file-and-canonical-json-aggregate",
        "aggregate_sha256": hashlib.sha256(aggregate_payload).hexdigest(),
        "files": files,
    }


def canonical_rgb_sha256(image: Image.Image) -> str:
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    normalized.load()
    width, height = normalized.size
    digest = hashlib.sha256()
    digest.update(b"RGB\0")
    digest.update(struct.pack(">QQ", width, height))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def canonical_rgb_sha256_path(path: Path) -> str:
    with Image.open(path) as image:
        return canonical_rgb_sha256(image)


def dhash(image: Image.Image) -> str:
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    gray = normalized.resize((9, 8), Image.Resampling.LANCZOS).convert("L")
    array = np.asarray(gray, dtype=np.int16)
    bits = array[:, 1:] > array[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def dhash_path(path: Path) -> str:
    with Image.open(path) as image:
        return dhash(image)


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def normalized_rgb(image: Image.Image) -> Image.Image:
    result = ImageOps.exif_transpose(image).convert("RGB")
    result.load()
    return result


def expand2square(
    image: Image.Image, background_color: tuple[int, int, int] = (122, 116, 104)
) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    if width > height:
        result = Image.new(image.mode, (width, width), background_color)
        result.paste(image, (0, (width - height) // 2))
        return result
    result = Image.new(image.mode, (height, height), background_color)
    result.paste(image, ((height - width) // 2, 0))
    return result


def save_deterministic_jpeg(
    image: Image.Image, path: Path, *, exclusive: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = normalized_rgb(image)
    if exclusive:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            image.save(handle, **JPEG_SETTINGS)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        image.save(path, **JPEG_SETTINGS)
        with path.open("rb") as handle:
            os.fsync(handle.fileno())


def official_vqa_process(text: Any) -> str:
    """The legacy construction normalizer, intentionally preserved verbatim."""

    text = str(text).lower().replace("\n", " ").replace("\r", " ")
    text = re.sub(r"([^\w\s])", r" ", text)
    words = [word for word in text.split() if word not in ["a", "an", "the"]]
    number_map = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
    }
    return " ".join(number_map.get(word, word) for word in words)


def is_correct_open(prediction: Any, ground_truth_answers: Sequence[Any]) -> bool:
    """The legacy construction scorer, including its substring behavior."""

    prediction_normalized = official_vqa_process(prediction)
    truths = [official_vqa_process(answer) for answer in ground_truth_answers]
    if prediction_normalized in truths:
        return True
    for truth in truths:
        if truth and truth in prediction_normalized:
            return True
    return False


def is_correct_mc(
    prediction: Any, ground_truth_letter: str, ground_truth_text: str, max_letter: str
) -> bool:
    """The legacy MC construction scorer, generalized only by its old A-D/A-F limit."""

    prediction_normalized = str(prediction).strip().lower()
    letter = ground_truth_letter.lower()
    match = re.search(
        rf"(?i)(?:^|\s|\()(option\s+)?([a-{max_letter.lower()}])(?:\)|\.|:|\s|$)",
        prediction_normalized,
    )
    if match:
        extracted = match.group(2).lower()
    else:
        if ground_truth_text.lower() in prediction_normalized:
            return True
        extracted = prediction_normalized[0] if prediction_normalized else ""
    return extracted == letter


YES_NO_PREFIXES = (
    "is ",
    "are ",
    "does ",
    "do ",
    "can ",
    "could ",
    "was ",
    "were ",
    "has ",
    "have ",
)


def is_yes_no_question(question: Any) -> bool:
    return str(question).strip().lower().startswith(YES_NO_PREFIXES)


def ordered_deduplicate(values: Iterable[str], key=official_vqa_process) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = key(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(value)
    return output


COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "orange",
    "brown",
    "pink",
    "purple",
    "gray",
    "silver",
    "gold",
]
NUMBERS = [str(index) for index in range(21)] + [
    "none",
    "many",
    "few",
    "one",
    "two",
    "three",
    "four",
    "five",
]


def derived_question_seed(global_seed: int, question_id: Any) -> int:
    payload = f"vqav2-mc-options-v1\0{global_seed}\0{question_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def generate_mc_options(
    ground_truth_answer: str,
    generic_pool: Sequence[str],
    global_seed: int,
    question_id: Any,
) -> tuple[list[str], str, int]:
    """Generate stable VQAv2-MC options without set/hash-order dependence."""

    ground_truth = str(ground_truth_answer).lower()
    if not official_vqa_process(ground_truth):
        raise ValueError("empty normalized ground-truth answer")
    seed = derived_question_seed(global_seed, question_id)
    rng = random.Random(seed)
    if ground_truth in COLORS:
        preferred_pool: Sequence[str] = COLORS
    elif ground_truth in NUMBERS or ground_truth.isdigit():
        preferred_pool = NUMBERS
    else:
        preferred_pool = generic_pool
    # The source adapter constructs ``generic_pool`` once with stable ordered
    # deduplication.  Re-deduplicating its thousands of values for every one of
    # ~39k VQAv2 questions would make source preparation quadratic.
    fallback_pool = generic_pool
    options = [ground_truth]
    normalized = {official_vqa_process(ground_truth)}

    def try_add(pool: Sequence[str], max_attempts: int) -> None:
        attempts = 0
        while len(options) < 4 and attempts < max_attempts:
            candidate = str(rng.choice(pool))
            candidate_normalized = official_vqa_process(candidate)
            if candidate_normalized and candidate_normalized not in normalized:
                options.append(candidate)
                normalized.add(candidate_normalized)
            attempts += 1

    if preferred_pool:
        try_add(preferred_pool, max(50, len(preferred_pool) * 4))
    if len(options) < 4:
        try_add(fallback_pool, max(200, len(fallback_pool) * 2))
    if len(options) != 4:
        raise ValueError("unable to create four normalized-unique MC options")
    rng.shuffle(options)
    correct_index = next(
        index
        for index, value in enumerate(options)
        if official_vqa_process(value) == official_vqa_process(ground_truth)
    )
    correct_letter = chr(ord("A") + correct_index)
    return options, correct_letter, seed


def open_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}USER: <image>\n{question}{OPEN_SUFFIX}"


def vqav2_mc_question(question: str, options: Sequence[str]) -> str:
    return (
        f"{question}\nOptions:\nA. {options[0]}\nB. {options[1]}\n"
        f"C. {options[2]}\nD. {options[3]}"
    )


def vqav2_mc_prompt(question_with_options: str) -> str:
    return f"{SYSTEM_PROMPT}USER: <image>\n{question_with_options}\n{MC_SUFFIX}"


def scienceqa_question(question: str, choices: Sequence[str], hint: str) -> str:
    letters = ["A", "B", "C", "D", "E", "F"]
    context = f"Context: {hint}\n" if hint else ""
    options = "Options:\n" + "".join(
        f"{letters[index]}. {choice}\n" for index, choice in enumerate(choices)
    )
    return f"{context}{question}\n{options}"


def scienceqa_prompt(question_with_options: str) -> str:
    # question_with_options intentionally ends in a newline, preserving the old prompt.
    return f"{SYSTEM_PROMPT}USER: <image>\n{question_with_options}{MC_SUFFIX}"


def resolve_source_cache_root(
    source_cache_root: Path | str, *, require_existing: bool = False
) -> Path:
    """Resolve a real source-cache directory without following a root symlink."""

    raw = Path(source_cache_root).expanduser()
    if raw.is_symlink():
        raise ValueError(f"source_cache must not be a symlink: {raw}")
    resolved = raw.resolve()
    if raw.exists() and not raw.is_dir():
        raise ValueError(f"source_cache is not a directory: {raw}")
    if require_existing and not resolved.is_dir():
        raise FileNotFoundError(f"source_cache does not exist: {resolved}")
    return resolved


def resolve_builder_source_cache(
    output_root: Path | str, *, require_existing: bool = False
) -> Path:
    """Anchor the owned source cache to the resolved builder output root."""

    root = Path(output_root).expanduser().resolve()
    lexical = root / "source_cache"
    if lexical.is_symlink():
        raise ValueError(f"builder source_cache must not be a symlink: {lexical}")
    resolved = resolve_source_cache_root(
        lexical, require_existing=require_existing
    )
    if not resolved.is_relative_to(root):
        raise ValueError("resolved builder source_cache escapes output_root")
    return resolved


def resolve_source_cache_child(
    source_cache_root: Path | str,
    relative_path: Path | str,
    *,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    """Resolve one cache child, rejecting every existing symlink component."""

    if require_file and require_directory:
        raise ValueError("source-cache child cannot require both file and directory")
    root = resolve_source_cache_root(source_cache_root)
    raw_relative = str(relative_path)
    relative = PurePosixPath(raw_relative)
    if (
        not raw_relative
        or "\\" in raw_relative
        or relative.is_absolute()
        or relative.as_posix() != raw_relative
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"unsafe source-cache relative path: {relative_path!r}")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"source-cache path contains a symlink: {current}")
    resolved = root.joinpath(*relative.parts).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("resolved source-cache child escapes source_cache")
    if require_file and not resolved.is_file():
        raise FileNotFoundError(f"source-cache file does not exist: {resolved}")
    if require_directory and not resolved.is_dir():
        raise FileNotFoundError(f"source-cache directory does not exist: {resolved}")
    return resolved


def copy_or_download(
    destination: Path,
    url: str,
    cached_source: Path | None = None,
    attempts: int = 4,
    timeout: tuple[float, float] = (15.0, 120.0),
) -> dict[str, Any]:
    """Populate a cache file with verified HTTPS, retrying bounded failures."""

    if not url.lower().startswith("https://"):
        raise ValueError(f"refusing non-HTTPS source URL: {url}")
    if destination.is_symlink() or destination.parent.is_symlink():
        raise ValueError(f"cache destination must not contain a symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.parent.is_symlink():
        raise ValueError(f"cache destination must not contain a symlink: {destination}")
    if destination.is_file():
        return {
            "path": str(destination),
            "url": url,
            "size": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "cache_status": "existing",
        }
    if cached_source is not None and cached_source.is_file():
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.tmp-", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as target, cached_source.open(
                "rb"
            ) as source:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
            if destination.is_symlink():
                raise ValueError(
                    f"cache destination became a symlink: {destination}"
                )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return {
            "path": str(destination),
            "url": url,
            "size": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "cache_status": "copied_from_hf_cache",
            "cached_source": str(cached_source),
        }

    import requests

    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        temporary: Path | None = None
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.part-", dir=destination.parent
                )
                temporary = Path(temporary_name)
                with os.fdopen(descriptor, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            if destination.is_symlink():
                raise ValueError(
                    f"cache destination became a symlink: {destination}"
                )
            assert temporary is not None
            os.replace(temporary, destination)
            return {
                "path": str(destination),
                "url": url,
                "size": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "cache_status": "downloaded",
                "attempt": attempt,
            }
        except (requests.RequestException, OSError) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    raise RuntimeError(f"failed after {attempts} attempts: {url}") from last_error


def extract_zip_member_atomic(
    zip_path: Path, member: str, destination: Path
) -> dict[str, Any]:
    import zipfile

    if zip_path.is_symlink() or destination.is_symlink() or destination.parent.is_symlink():
        raise ValueError(
            "ZIP source and extracted destination must not contain symlinks"
        )
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failure for member {bad!r} in {zip_path}")
        names = archive.namelist()
        if member not in names:
            raise RuntimeError(f"missing {member!r} in {zip_path}; found {names!r}")
        with archive.open(member) as source:
            member_bytes = source.read()
    if destination.is_file():
        if destination.read_bytes() != member_bytes:
            raise RuntimeError(
                f"existing extracted member differs from {member!r} in {zip_path}: "
                f"{destination}"
            )
    else:
        atomic_write_bytes(destination, member_bytes)
    return {
        "member": member,
        "path": str(destination.resolve()),
        "size_bytes": len(member_bytes),
        "sha256": hashlib.sha256(member_bytes).hexdigest(),
    }


def write_csv_atomic(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


class LlavaConstructionRunner:
    """Uncompressed 576-patch LLaVA construction inference only."""

    control_name = "blind_black_image_control"

    def __init__(self, model_path: Path, device: str = "cuda:0") -> None:
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.torch = torch
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=False)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map={"": str(self.device)},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        vision = self.model.config.vision_config
        patch_tokens = (int(vision.image_size) // int(vision.patch_size)) ** 2
        if patch_tokens != 576:
            raise RuntimeError(f"expected 576 visual patch tokens, found {patch_tokens}")
        self.visual_patch_tokens = patch_tokens
        self.black_control = Image.new("RGB", (336, 336), (0, 0, 0))
        self._model_artifact_fingerprint: dict[str, Any] | None = None

    def construction_manifest(self) -> dict[str, Any]:
        """Return the immutable model/inference identity committed to checkpoints."""

        import datasets
        import transformers
        from PIL import __version__ as pillow_version
        from PIL import features as pillow_features

        if self._model_artifact_fingerprint is None:
            self._model_artifact_fingerprint = model_directory_fingerprint(
                self.model_path
            )
        device_properties = self.torch.cuda.get_device_properties(self.device)
        return {
            "runner": "LlavaConstructionRunner",
            "model": self._model_artifact_fingerprint,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": self.torch.__version__,
                "transformers": transformers.__version__,
                "datasets": datasets.__version__,
                "numpy": np.__version__,
                "pillow": pillow_version,
                "libjpeg": pillow_features.version("jpg"),
                "cuda_runtime": self.torch.version.cuda,
                "cudnn": self.torch.backends.cudnn.version(),
                "gpu": {
                    "name": device_properties.name,
                    "total_memory_bytes": device_properties.total_memory,
                    "compute_capability": [
                        device_properties.major,
                        device_properties.minor,
                    ],
                },
            },
            "visual_patch_tokens": self.visual_patch_tokens,
            "blind_control": {
                "name": self.control_name,
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
        }

    def infer(
        self, image: Image.Image, prompt: str, legacy_temperature_zero: bool = False
    ) -> str:
        torch = self.torch
        image = expand2square(normalized_rgb(image))
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        if "pixel_values" not in inputs:
            raise RuntimeError("processor did not return pixel_values")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        generation = {
            "max_new_tokens": 32,
            "do_sample": False,
            "num_beams": 1,
        }
        if legacy_temperature_zero:
            generation["temperature"] = 0.0
        with torch.no_grad():
            output = self.model.generate(**inputs, **generation)
        return self.processor.decode(
            output[0][input_length:], skip_special_tokens=True
        ).strip()

    def infer_blind(self, prompt: str, legacy_temperature_zero: bool = False) -> str:
        return self.infer(self.black_control, prompt, legacy_temperature_zero)
