"""Report why the formal four-compressor InternVL rerun is blocked."""

from __future__ import annotations

import argparse
import json


BLOCKER = {
    "status": "blocked_pending_author_decision_and_rerun",
    "requested_protocol": ["VisionZIP", "VisPruner", "PruMerge", "FlowCut"],
    "historical_attack_objective": ["VisionZIP", "VisPruner", "DivPrune", "PruMerge", "FlowCut"],
    "historical_result_operation": "post-hoc filter evaluation rows to four compressors",
    "reason": (
        "Removing DivPrune changes clean references, post-compression disruption, "
        "cutoff ranking, family averaging, and the diversity objective. No audited "
        "artifact proves that the paper aggregate came from a four-compressor attack."
    ),
    "required_resolution": (
        "Choose and document a four-compressor diversity surrogate, rerun V5-E with "
        "epsilon=2/255, alpha=0.5/255, 100 steps, seed=42, then replace the summaries."
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    if args.json:
        print(json.dumps(BLOCKER, indent=2, sort_keys=True))
    else:
        print("InternVL formal attack: BLOCKED")
        print(BLOCKER["reason"])
        print("Required resolution:", BLOCKER["required_resolution"])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
