"""Validate the four centralized FATA path variables."""

from __future__ import annotations

import argparse
import json

from fata.utils.paths import PathConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-output", action="store_true")
    parser.add_argument("--skip-input-existence", action="store_true")
    args = parser.parse_args(argv)
    config = PathConfig.from_environment()
    config.validate(require_inputs=not args.skip_input_existence, create_output=args.create_output)
    print(json.dumps(config.as_environment(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
