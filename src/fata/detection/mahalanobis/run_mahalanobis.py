"""Import-safe CLI wrapper for the Mahalanobis detector experiment."""

from fata.utils.module_runner import run_implementation


def main() -> int:
    return run_implementation(
        "fata.detection.mahalanobis._run_mahalanobis_impl"
    )


if __name__ == "__main__":
    raise SystemExit(main())
