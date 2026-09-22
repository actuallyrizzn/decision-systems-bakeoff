#!/usr/bin/env python3
"""Merge arm summaries into one bakeoff report JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True, help="Directory of *_summary.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    summaries = []
    for path in sorted(args.runs.glob("*_summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    report = {
        "n_summaries": len(summaries),
        "summaries": summaries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(summaries)} summaries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
