"""Import-safe CLI wrapper for the Feature Squeezing experiment."""

from fata.utils.module_runner import run_implementation


def main() -> int:
    return run_implementation(
        "fata.detection.feature_squeezing._run_feature_squeezing_impl"
    )


if __name__ == "__main__":
    raise SystemExit(main())
