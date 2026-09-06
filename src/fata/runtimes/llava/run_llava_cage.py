"""Import-safe CLI wrapper for the LLaVA CAGE benchmark."""

from fata.utils.module_runner import run_implementation


def main() -> int:
    return run_implementation("fata.runtimes.llava._run_llava_cage_impl")


if __name__ == "__main__":
    raise SystemExit(main())
