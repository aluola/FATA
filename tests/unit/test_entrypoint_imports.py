"""Public run modules must be safe to import without parsing CLI or loading models."""

from __future__ import annotations

import importlib
import sys

import pytest

from fata.utils import module_runner


PUBLIC_WRAPPERS = (
    "fata.runtimes.llava.run_llava_fata",
    "fata.runtimes.llava.run_llava_caa",
    "fata.runtimes.llava.run_llava_cage",
    "fata.runtimes.llava.run_llava_objective_ablation",
    "fata.detection.feature_squeezing.run_feature_squeezing",
    "fata.detection.mahalanobis.run_mahalanobis",
)


@pytest.mark.parametrize("module_name", PUBLIC_WRAPPERS)
def test_public_wrapper_import_has_no_legacy_execution(module_name: str):
    implementation = module_name.rsplit(".", 1)[0] + "._" + module_name.rsplit(".", 1)[1] + "_impl"
    sys.modules.pop(implementation, None)
    module = importlib.import_module(module_name)
    assert callable(module.main)
    assert implementation not in sys.modules


def test_module_runner_preserves_public_argv_zero(monkeypatch):
    observed = {}

    def fake_run_module(module_name, *, run_name, alter_sys):
        observed.update(
            module_name=module_name,
            run_name=run_name,
            alter_sys=alter_sys,
        )

    monkeypatch.setattr(module_runner.runpy, "run_module", fake_run_module)
    assert module_runner.run_implementation("fata.example._private_impl") == 0
    assert observed == {
        "module_name": "fata.example._private_impl",
        "run_name": "__main__",
        "alter_sys": False,
    }
