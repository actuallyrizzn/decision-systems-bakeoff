#!/usr/bin/env python3
"""Verify frozen test.tsv files against lockfile.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_bakeoff.data import load_lockfile, verify_task_test  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Task id (repeatable). Default: all in lockfile.",
    )
    args = ap.parse_args()
    lock = load_lockfile()
    tasks = args.tasks or sorted(lock["tasks"])
    failed = 0
    for task in tasks:
        try:
            digest = verify_task_test(task, args.data, lock)
            print(f"OK  {task}  {digest}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {task}  {exc}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
