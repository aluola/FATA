"""Helpers for exposing audited legacy scripts as import-safe module CLIs."""

from __future__ import annotations

import runpy


def run_implementation(module_name: str) -> int:
    """Execute a package module only after its public wrapper's ``main`` runs."""

    # Keep ``sys.argv[0]`` owned by the public wrapper.  ``alter_sys=True``
    # temporarily replaces it with the private implementation path, which in
    # turn makes argparse advertise ``_run_*_impl.py`` even when reviewers ran
    # the documented public ``python -m fata...`` entrypoint.  Relative imports
    # still resolve from the implementation's module spec without mutating
    # ``sys.argv`` or ``sys.modules['__main__']``.
    runpy.run_module(module_name, run_name="__main__", alter_sys=False)
    return 0
