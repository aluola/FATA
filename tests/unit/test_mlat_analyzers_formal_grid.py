from __future__ import annotations

import numpy as np
import pytest

from fata.detection.mlatd import analyze_mlat_attack_ablation as attack_analysis
from fata.detection.mlatd import analyze_mlat_full_grid as full_analysis
from fata.detection.mlatd.check_mlat_full_grid import (
    ATTACKS,
    DATASETS,
    METHODS,
    practical_token_budget,
)


def _grid(attacks, total=2):
    data = {}
    for method in METHODS:
        for dataset in DATASETS:
            ids = np.asarray([f"{dataset}/image-{index}.jpg" for index in range(total)])
            for attack in attacks:
                data[(method, dataset, attack, practical_token_budget(method, dataset))] = {
                    "sample_idx": np.arange(total, dtype=np.int64),
                    "label": np.full(
                        total, 0 if attack in {"clean_clip", "random_clip"} else 1,
                        dtype=np.int8,
                    ),
                    "image_id": ids.copy(),
                    "contract": {
                        "llava_model": {"digest": "model"},
                        "clip_model": {"digest": "clip"},
                        "dataset_mapping_sha256": dataset,
                        "dataset_images": {"aggregate": dataset},
                    },
                }
    return data


def test_full_analyzer_requires_exact_64_cell_grid_and_aligned_ids():
    data = _grid(ATTACKS)
    full_analysis.validate_formal_grid(data, 2)
    data.pop(next(iter(data)))
    with pytest.raises(RuntimeError, match="grid mismatch"):
        full_analysis.validate_formal_grid(data, 2)


def test_attack_analyzer_requires_exact_96_cell_grid_and_same_cohort():
    attacks = ("clean_clip", "random_clip", "base", "fata", "cage", "caa")
    data = _grid(attacks)
    attack_analysis.validate_formal_grid(data, 2)
    key = ("VisionZIP", "TextVQA_Open", "caa", 64)
    data[key]["image_id"] = np.asarray(["wrong-a", "wrong-b"])
    with pytest.raises(RuntimeError, match="image identities differ"):
        attack_analysis.validate_formal_grid(data, 2)


@pytest.mark.parametrize("analyzer", (full_analysis, attack_analysis))
def test_mlat_direction_fit_rejects_degenerate_features(analyzer):
    constant = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="degenerate attack direction"):
        analyzer.fit_direction(constant, constant)


@pytest.mark.parametrize(
    "standardizer",
    (full_analysis.standardize_by_train_neg, attack_analysis.standardize),
)
def test_mlat_stage_standardization_rejects_zero_variance(standardizer):
    constant = np.ones(4, dtype=np.float32)
    with pytest.raises(RuntimeError, match="degenerate negative stage"):
        standardizer(constant, constant)


@pytest.mark.parametrize("relationship", ("equal", "child", "parent"))
def test_attack_analyzer_refuses_output_over_cache_tree(tmp_path, relationship):
    cache_root = tmp_path / "cache"
    if relationship == "equal":
        output = cache_root
    elif relationship == "child":
        output = cache_root / "analysis"
    else:
        output = tmp_path
    with pytest.raises(ValueError, match="overlaps protected"):
        attack_analysis.main(
            [
                "--original-glob", str(tmp_path / "missing-direct-*.npz"),
                "--ablation-glob", str(tmp_path / "missing-cache-*.npz"),
                "--cache-root", str(cache_root),
                "--output-dir", str(output),
                "--train_start", "0", "--train_end", "1",
                "--test_start", "1", "--test_end", "2",
                "--total-limit", "2", "--seed", "0",
            ]
        )


@pytest.mark.parametrize("analyzer", (full_analysis, attack_analysis))
@pytest.mark.parametrize(
    "total,train_end,test_start,test_end",
    (
        (3, 1, 1, 3),
        (1000, 999, 999, 1000),
        (1000, 400, 400, 1000),
    ),
)
def test_formal_analyzers_require_exact_even_half_split(
    tmp_path, analyzer, total, train_end, test_start, test_end
):
    if analyzer is full_analysis:
        arguments = [
            "--input-glob", str(tmp_path / "missing-*.npz"),
            "--output-dir", str(tmp_path / "output"),
        ]
    else:
        arguments = [
            "--original-glob", str(tmp_path / "missing-direct-*.npz"),
            "--ablation-glob", str(tmp_path / "missing-cache-*.npz"),
            "--cache-root", str(tmp_path / "cache"),
            "--output-dir", str(tmp_path / "output"),
        ]
    arguments.extend(
        [
            "--train_start", "0",
            "--train_end", str(train_end),
            "--test_start", str(test_start),
            "--test_end", str(test_end),
            "--total-limit", str(total),
            "--seed", "0",
        ]
    )
    with pytest.raises(SystemExit) as error:
        analyzer.main(arguments)
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()
