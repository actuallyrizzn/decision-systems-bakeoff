#!/usr/bin/env python3
"""Run the Flybrain arm on one locked task (frozen graph + ridge head)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_bakeoff.flybrain.pipeline import run_flybrain_task  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--glove", type=Path, required=True, help="Path to glove.6B.100d.txt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", required=True, choices=["sst2", "clinc10", "clinc150", "bugsev"])
    ap.add_argument("--cfg", type=Path, default=None, help="Override configs/flybrain_best.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-verify", action="store_true", help="Dangerous; for smoke fixtures only")
    args = ap.parse_args()
    summary = run_flybrain_task(
        task=args.task,
        data_root=args.data,
        glove_path=args.glove,
        out_dir=args.out,
        cfg_path=args.cfg,
        limit=args.limit,
        skip_verify=args.skip_verify,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
