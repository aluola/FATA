from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fata.runtimes.llava.attack_cache_io import (
    attack_cache_contract_path,
    attack_namespace,
    attack_image_exists,
    attack_image_path,
    baseline_attack_contract_extra,
    load_attack_image,
    save_attack_image,
)
from fata.utils.run_contract import sha256_file


BASE = {
    "method": "shared",
    "dataset": "TextVQA_Open",
    "attack": "fata_deadbeef",
}


def test_nested_mapping_path_is_preserved_and_lossless(tmp_path):
    expected = (
        tmp_path
        / "fata_deadbeef"
        / "TextVQA_Open"
        / "shared"
        / "images"
        / "TextVQA_Open_0000.jpg.png"
    ).resolve()
    kwargs = {
        "cache_root": tmp_path,
        "image_filename": "images/TextVQA_Open_0000.jpg",
        **BASE,
    }
    assert attack_image_path(**kwargs) == expected

    image = Image.new("RGB", (2, 2), (17, 33, 65))
    assert save_attack_image(image=image, **kwargs) == expected
    assert expected.with_name(expected.name + ".sha256").is_file()
    assert attack_image_exists(**kwargs)
    assert load_attack_image(**kwargs).getpixel((0, 0)) == (17, 33, 65)

    expected.write_bytes(b"valid-looking replacement is not trusted")
    assert not attack_image_exists(**kwargs)


def test_attack_namespace_locks_cache_changing_parameters():
    assert attack_namespace(
        "caa", seed=0, eps_255=2.0, alpha_255=1.0, steps=100
    ) == "caa_eps2_a1_s100_seed0"


def test_baseline_objective_contract_structures_every_pixel_affecting_constant():
    caa = baseline_attack_contract_extra("caa", max_input_tokens=0)
    assert caa == {
        "max_input_tokens": 0,
        "objective": {
            "target_layer": 1,
            "least_important_region_fraction": 0.30,
            "weights": {
                "bpr_inter": 10.0,
                "bpr_intra": 10.0,
                "semantic": 2.0,
                "question_answer": 5.0,
            },
        },
    }
    cage = baseline_attack_contract_extra("cage", max_input_tokens=0)
    assert cage["objective"] == {
        "lambda_cage": 0.005,
        "k_min": 16,
        "k_max": 192,
    }
    with pytest.raises(ValueError):
        baseline_attack_contract_extra("unknown")


@pytest.mark.parametrize("format_name,mode", (("JPEG", "RGB"), ("PNG", "L")))
def test_cache_gate_rejects_disguised_jpeg_and_non_rgb_png(
    tmp_path, format_name, mode
):
    kwargs = {"cache_root": tmp_path, "image_filename": "sample.jpg", **BASE}
    path = attack_image_path(**kwargs)
    path.parent.mkdir(parents=True)
    Image.new(mode, (2, 2), 17 if mode == "L" else (17, 33, 65)).save(
        path, format=format_name
    )
    path.with_name(path.name + ".sha256").write_text(
        sha256_file(path) + "\n", encoding="ascii"
    )
    assert not attack_image_exists(**kwargs)
    with pytest.raises(RuntimeError, match="missing, corrupt, or unhashed"):
        load_attack_image(**kwargs)


@pytest.mark.parametrize(
    "value",
    ("", "/absolute.jpg", "../escape.jpg", "images/../escape.jpg", "./a.jpg", "a\\b.jpg", "a//b.jpg", "bad name.jpg"),
)
def test_unsafe_mapping_paths_are_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        attack_image_path(cache_root=tmp_path, image_filename=value, **BASE)


@pytest.mark.parametrize("linked_component", ("attack", "dataset", "method", "images"))
def test_cache_paths_reject_intermediate_symlink_escape(tmp_path, linked_component):
    cache_root = tmp_path / "cache"
    external = tmp_path / "external"
    cache_root.mkdir()
    external.mkdir()
    components = ["fata_deadbeef", "TextVQA_Open", "shared", "images"]
    parent = cache_root
    for index, component in enumerate(components):
        label = ("attack", "dataset", "method", "images")[index]
        child = parent / component
        if label == linked_component:
            child.symlink_to(external, target_is_directory=True)
            break
        child.mkdir()
        parent = child

    kwargs = {
        "cache_root": cache_root,
        "image_filename": "images/TextVQA_Open_0000.jpg",
        **BASE,
    }
    with pytest.raises(ValueError, match="symbolic link"):
        attack_image_path(**kwargs)
    with pytest.raises(ValueError, match="symbolic link"):
        save_attack_image(image=Image.new("RGB", (2, 2)), **kwargs)
    assert not any(external.iterdir())


def test_cache_contract_path_rejects_symlink_escape(tmp_path):
    cache_root = tmp_path / "cache"
    external = tmp_path / "external"
    cache_root.mkdir()
    external.mkdir()
    (cache_root / "fata_deadbeef").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        attack_cache_contract_path(
            cache_root=cache_root,
            attack="fata_deadbeef",
            dataset="TextVQA_Open",
            method="shared",
        )
    assert not any(external.iterdir())


def test_cache_path_rejects_existing_output_symlink(tmp_path):
    external = tmp_path / "external.png"
    external.write_bytes(b"sentinel")
    output = (
        tmp_path
        / "fata_deadbeef"
        / "TextVQA_Open"
        / "shared"
        / "images"
        / "TextVQA_Open_0000.jpg.png"
    )
    output.parent.mkdir(parents=True)
    output.symlink_to(external)
    kwargs = {
        "cache_root": tmp_path,
        "image_filename": "images/TextVQA_Open_0000.jpg",
        **BASE,
    }
    with pytest.raises(ValueError, match="symbolic link"):
        attack_image_path(**kwargs)
    assert external.read_bytes() == b"sentinel"


def test_cache_write_rejects_digest_symlink(tmp_path):
    kwargs = {
        "cache_root": tmp_path,
        "image_filename": "images/TextVQA_Open_0000.jpg",
        **BASE,
    }
    output = attack_image_path(**kwargs)
    output.parent.mkdir(parents=True)
    external = tmp_path / "digest-sentinel"
    external.write_text("sentinel", encoding="utf-8")
    output.with_name(output.name + ".sha256").symlink_to(external)

    with pytest.raises(ValueError, match="symbolic link"):
        save_attack_image(image=Image.new("RGB", (2, 2)), **kwargs)
    assert external.read_text(encoding="utf-8") == "sentinel"
