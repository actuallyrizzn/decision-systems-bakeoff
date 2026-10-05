#!/usr/bin/env python3
"""Verify frozen test files against lockfile.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_bakeoff.data import (  # noqa: E402
    load_lockfile,
    verify_task_test,
    verify_typed_parquet,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument(
        "--suite",
        choices=["all", "original", "fair", "locked_additions"],
        default="all",
        help="Which lockfile suite to verify",
    )
    ap.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Task id (repeatable). Default: suite selection.",
    )
    args = ap.parse_args()
    lock = load_lockfile()
    if args.tasks:
        tasks = args.tasks
    elif args.suite == "all":
        tasks = sorted(lock["tasks"])
    else:
        tasks = list(lock["suites"][args.suite]["tasks"])
    failed = 0
    for task in tasks:
        try:
            if task == "typed_decisions":
                digest = verify_typed_parquet(args.data, lock)
            else:
                # original suite files may live in a different data root than fair
                digest = verify_task_test(task, args.data, lock)
            print(f"OK  {task}  {digest}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {task}  {exc}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
