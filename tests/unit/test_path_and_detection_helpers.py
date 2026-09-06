from __future__ import annotations

import pytest

from fata.detection.feature_squeezing.feature_squeezing_utils import detection_label
from fata.utils.paths import (
    PathConfig,
    assert_no_output_file_collision,
    assert_output_separate,
    create_owned_output_directory,
    metadata_sidecar_path,
    resolve_attack_cache_root,
    resolve_dataset_relative_path,
    resolve_owned_output_path,
    safe_filename_component,
)


@pytest.mark.parametrize("attack", ("clean", "random"))
def test_negative_detection_labels(attack):
    assert detection_label(attack) == 0


@pytest.mark.parametrize("attack", ("base", "fata", "cage", "caa"))
def test_positive_detection_labels(attack):
    assert detection_label(attack) == 1


def test_unknown_detection_label_fails_closed():
    with pytest.raises(ValueError):
        detection_label("unknown")


def test_attack_cache_prefers_explicit_root(tmp_path):
    assert resolve_attack_cache_root(
        explicit_cache_root=tmp_path / "explicit",
        output_root=tmp_path / "root",
        explicit_output_dir=tmp_path / "results",
    ) == (tmp_path / "explicit").resolve()


def test_attack_cache_falls_back_from_explicit_output_dir(tmp_path):
    result_dir = tmp_path / "detector" / "results"
    assert resolve_attack_cache_root(
        explicit_cache_root=None,
        output_root=None,
        explicit_output_dir=result_dir,
    ) == (tmp_path / "detector" / "adversarial_images").resolve()


def test_attack_cache_requires_a_base():
    with pytest.raises(ValueError):
        resolve_attack_cache_root(
            explicit_cache_root=None,
            output_root=None,
            explicit_output_dir=None,
        )


def test_derived_attack_cache_rejects_child_symlink(tmp_path):
    output_root = tmp_path / "output"
    external = tmp_path / "external"
    output_root.mkdir()
    external.mkdir()
    (output_root / "adversarial_images").symlink_to(
        external, target_is_directory=True
    )
    with pytest.raises(ValueError, match="symbolic link"):
        resolve_attack_cache_root(
            explicit_cache_root=None,
            output_root=output_root,
            explicit_output_dir=None,
        )


def test_dataset_relative_path_accepts_nested_image(tmp_path):
    assert resolve_dataset_relative_path(
        tmp_path, "images/TextVQA_Open_0000.jpg"
    ) == (tmp_path / "images" / "TextVQA_Open_0000.jpg").resolve()


@pytest.mark.parametrize(
    "value",
    ("", "/absolute.jpg", "../escape.jpg", "images/../escape.jpg", "./a.jpg", "a\\b.jpg", "a//b.jpg"),
)
def test_dataset_relative_path_rejects_escape_or_noncanonical_path(tmp_path, value):
    with pytest.raises(ValueError):
        resolve_dataset_relative_path(tmp_path, value)


def test_safe_filename_component():
    assert safe_filename_component("formal-v6.1") == "formal-v6.1"
    for value in ("", ".", "..", "../escape", "/absolute", "a/b", "bad name"):
        with pytest.raises(ValueError):
            safe_filename_component(value)


def test_metadata_sidecar_replaces_only_the_result_suffix(tmp_path):
    result = tmp_path / "parent.csv" / "nested.csv" / "result.csv"
    assert metadata_sidecar_path(result) == (
        tmp_path / "parent.csv" / "nested.csv" / "result.meta.json"
    )
    with pytest.raises(ValueError, match="must end in .csv"):
        metadata_sidecar_path(tmp_path / "result.json")


def test_exact_output_file_collision_is_rejected(tmp_path):
    stats = tmp_path / "results.csv"
    with pytest.raises(ValueError, match="would overwrite protected fitted stats"):
        assert_no_output_file_collision(
            {"result": stats},
            {"fitted stats": stats},
        )
    with pytest.raises(ValueError, match="duplicate output file paths"):
        assert_no_output_file_collision(
            {"result": stats, "metadata": stats},
            {},
        )


def test_output_separation_rejects_both_overlap_directions(tmp_path):
    data = tmp_path / "data"
    with pytest.raises(ValueError, match="overlaps protected data"):
        assert_output_separate(data / "results", {"data": data})
    with pytest.raises(ValueError, match="overlaps protected data"):
        assert_output_separate(tmp_path, {"data": data})
    assert_output_separate(tmp_path / "results", {"data": data})


def test_path_config_rejects_output_inside_input(tmp_path):
    model = tmp_path / "model"
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    for path in (model, data, cache):
        path.mkdir()
    config = PathConfig(model, data, data / "results", cache)
    with pytest.raises(ValueError, match="data_root"):
        config.validate(create_output=True)


def test_owned_output_path_creates_real_directory_and_file_path(tmp_path):
    directory = create_owned_output_directory(tmp_path, "llava/main")
    assert directory == (tmp_path / "llava" / "main").resolve()
    assert resolve_owned_output_path(tmp_path, "llava/main/result.csv") == (
        directory / "result.csv"
    )


@pytest.mark.parametrize("component", ("llava", "main", "result.csv"))
def test_owned_output_path_rejects_child_symlink(tmp_path, component):
    external = tmp_path / "external"
    root = tmp_path / "output"
    external.mkdir()
    root.mkdir()
    current = root
    for name in ("llava", "main", "result.csv"):
        child = current / name
        if name == component:
            child.symlink_to(external, target_is_directory=name != "result.csv")
            break
        if name != "result.csv":
            child.mkdir()
        current = child
    with pytest.raises(ValueError, match="symbolic link"):
        resolve_owned_output_path(root, "llava/main/result.csv")


@pytest.mark.parametrize("value", ("", "../escape", "/absolute", "a//b", "a\\b"))
def test_owned_output_path_rejects_unsafe_relative_value(tmp_path, value):
    with pytest.raises(ValueError, match="unsafe output-relative"):
        resolve_owned_output_path(tmp_path, value)
