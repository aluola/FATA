from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from fata.runtimes.llava.result_schema import RAW_ANSWER_SUFFIX


REPO_ROOT = Path(__file__).resolve().parents[2]


CASES = {
    "fata": {
        "module": "fata.runtimes.llava.run_llava_fata",
        "extra": ["--clip-path", "{clip}", "--attack-cache-root", "{cache}"],
        "relative": (
            "llava/main/"
            "ultimate_benchmark_lam1_seed0_VisionZIP_ScienceQA_MC.csv"
        ),
        "scores": ["Lang_Prior_K0"]
        + [
            f"{mode}_K{k}"
            for mode in ("Clean", "Base", "FATA")
            for k in (576, 192, 128, 64, 32, 16)
        ],
    },
    "caa": {
        "module": "fata.runtimes.llava.run_llava_caa",
        "extra": ["--attack-cache-root", "{cache}"],
        "relative": (
            "llava/caa/"
            "caa_eps2_a1_s100_seed0_VisionZIP_ScienceQA_MC.csv"
        ),
        "scores": [f"CAA_K{k}" for k in (576, 192, 128, 64, 32, 16)],
    },
    "cage": {
        "module": "fata.runtimes.llava.run_llava_cage",
        "extra": ["--clip-path", "{clip}", "--attack-cache-root", "{cache}"],
        "relative": (
            "llava/cage/"
            "cage_eps2_a0.5_s100_seed0_VisionZIP_ScienceQA_MC.csv"
        ),
        "scores": ["Lang_Prior_K0"]
        + [
            f"{mode}_K{k}"
            for mode in ("Clean", "Base", "CAGE")
            for k in (576, 192, 128, 64, 32, 16)
        ],
    },
    "objective": {
        "module": "fata.runtimes.llava.run_llava_objective_ablation",
        "extra": ["--clip-path", "{clip}"],
        "relative": (
            "llava/objective_ablation/"
            "fata_ablation_full_lam1_seed0_VisionZIP_ScienceQA_MC.csv"
        ),
        "scores": ["Lang_Prior_K0"]
        + [
            f"{mode}_K{k}"
            for mode in ("Clean", "FATA")
            for k in (576, 192, 128, 64, 32, 16)
        ],
    },
}


def _prepare(tmp_path: Path):
    data_root = tmp_path / "data"
    dataset = data_root / "ScienceQA_MC"
    images = dataset / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (2, 2), (12, 34, 56)).save(images / "sample.jpg", "JPEG")
    mapping = dataset / "ScienceQA_MC_mapping.jsonl"
    mapping.write_text(
        json.dumps(
            {
                "image_filename": "images/sample.jpg",
                "question": "Pick one.",
                "answers": ["A"],
                "type": "multiple_choice",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    model = tmp_path / "model"
    clip = tmp_path / "clip"
    model.mkdir()
    clip.mkdir()
    (model / "model.safetensors").write_bytes(b"model")
    (clip / "model.safetensors").write_bytes(b"clip")
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "transformers.py").write_text(
        "class _NoModel:\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, *args, **kwargs):\n"
        "        raise RuntimeError('MODEL_LOAD_SENTINEL')\n"
        "AutoProcessor = _NoModel\n"
        "LlavaForConditionalGeneration = _NoModel\n"
        "CLIPVisionModel = _NoModel\n"
        "CLIPImageProcessor = _NoModel\n",
        encoding="utf-8",
    )
    locations = {
        "data": data_root,
        "model": model,
        "clip": clip,
        "output": tmp_path / "output",
        "cache": tmp_path / "cache",
        "stub": stub,
    }
    return locations


def _command(case, locations):
    extra = [item.format(**locations) for item in case["extra"]]
    return [
        sys.executable,
        "-m",
        case["module"],
        "--method",
        "VisionZIP",
        "--dataset",
        "ScienceQA_MC",
        "--limit",
        "1",
        "--seed",
        "0",
        "--model-path",
        str(locations["model"]),
        "--output-root",
        str(locations["output"]),
        "--dataset-root",
        str(locations["data"]),
        *extra,
    ]


def _run(command, locations):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join(
        [str(locations["stub"]), str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _initialize_contract_and_header(case, locations):
    command = _command(case, locations)
    initial = _run(command, locations)
    assert initial.returncode != 0, initial.stdout
    assert "MODEL_LOAD_SENTINEL" in initial.stdout
    result = locations["output"] / case["relative"]
    assert result.is_file()
    meta = result.with_suffix(".meta.json")
    contract = json.loads(meta.read_text(encoding="utf-8"))
    assert contract["result_schema"]["name"] == "llava_answer_score_v2"
    with result.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == [
        "Image_ID",
        "Question",
        *case["scores"],
        *(f"{score}{RAW_ANSWER_SUFFIX}" for score in case["scores"]),
    ]
    return command, result, header


@pytest.mark.parametrize("case_name", CASES)
def test_all_four_entrypoints_reject_legacy_score_only_resume_before_model_load(
    tmp_path, case_name
):
    locations = _prepare(tmp_path)
    case = CASES[case_name]
    command, result, _ = _initialize_contract_and_header(case, locations)
    with result.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Image_ID", "Question", *case["scores"]])
        writer.writerow(
            ["images/sample.jpg", '"Pick one."', *("0.00" for _ in case["scores"])]
        )
    if case_name == "caa":
        command = [*command, "--cache-only"]
    resumed = _run(command, locations)
    assert resumed.returncode != 0
    assert "unexpected LLaVA answer+score v2 CSV header" in resumed.stdout
    assert "MODEL_LOAD_SENTINEL" not in resumed.stdout


def test_objective_resume_rejects_all_zero_tamper_then_accepts_valid_v2(tmp_path):
    locations = _prepare(tmp_path)
    case = CASES["objective"]
    command, result, header = _initialize_contract_and_header(case, locations)

    def write_scores(score: str):
        with result.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerow(
                [
                    "images/sample.jpg",
                    '"Pick one."',
                    *(score for _ in case["scores"]),
                    *("Option A." for _ in case["scores"]),
                ]
            )

    write_scores("0.00")
    tampered = _run(command, locations)
    assert tampered.returncode != 0
    assert "score/raw-answer mismatch" in tampered.stdout
    assert "MODEL_LOAD_SENTINEL" not in tampered.stdout

    write_scores("1.00")
    valid = _run(command, locations)
    assert valid.returncode == 0, valid.stdout
    assert "精确恢复" in valid.stdout
    assert "MODEL_LOAD_SENTINEL" not in valid.stdout
