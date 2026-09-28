#!/usr/bin/env python3
"""Ingest run summaries into the decision-matrix DB (Benchlab).

Examples:
    # one summary file
    python scripts/matrix_ingest.py runs/fair-2026-09-28/julia_emotion_summary.json \
        --run-id fair-2026-09-28 --suite fair --host ngram

    # a whole run directory of *_summary.json files
    python scripts/matrix_ingest.py runs/fair-2026-09-28 --run-id fair-2026-09-28 --suite fair

    # print the current board
    python scripts/matrix_ingest.py --board --suite fair
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from decision_bakeoff.matrix import board_table, ingest_summary, open_db  # noqa: E402

DEFAULT_DB = Path(__file__).resolve().parent.parent / "runs" / "matrix.db"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="summary JSON file or directory of *_summary.json")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="matrix SQLite path")
    ap.add_argument("--run-id", help="run identifier (default: source directory name)")
    ap.add_argument("--suite", help="suite tag (original | fair | benchlab-N)")
    ap.add_argument("--fidelity", default="full", help="full | pilot-100 | smoke | ...")
    ap.add_argument("--host", help="host the run executed on")
    ap.add_argument("--board", action="store_true", help="print the latest-cells board and exit")
    ap.add_argument("--kpi", default="accuracy", help="KPI for --board (accuracy|brier|median_seconds|...)")
    args = ap.parse_args()

    conn = open_db(args.db)

    if args.board:
        print(board_table(conn, suite=args.suite, kpi=args.kpi))
        return 0

    if not args.source:
        ap.error("source required unless --board")

    src = Path(args.source)
    files = sorted(src.glob("*_summary.json")) if src.is_dir() else [src]
    if not files:
        print(f"no *_summary.json under {src}", file=sys.stderr)
        return 1

    run_id = args.run_id or (src.name if src.is_dir() else src.parent.name)
    total = 0
    for f in files:
        total += ingest_summary(
            conn, f, run_id=run_id, suite=args.suite, fidelity=args.fidelity, host=args.host
        )
        print(f"ingested {f.name} (run_id={run_id})")
    print(f"{total} cell(s) -> {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
