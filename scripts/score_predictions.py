#!/usr/bin/env python3
"""Re-score a predictions JSONL against frozen gold (or use embedded gold)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_bakeoff.data import (  # noqa: E402
    class_names,
    gold_name,
    load_labels,
    load_lockfile,
    load_tsv,
    verify_task_test,
)
from decision_bakeoff.score import load_predictions_jsonl, score_rows  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--task", required=True, choices=["sst2", "clinc10", "clinc150", "bugsev"])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--arm", default="unknown")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    lock = load_lockfile()
    if not args.skip_verify:
        verify_task_test(args.task, args.data, lock)

    task_dir = args.data / args.task
    gold_rows = load_tsv(task_dir / "test.tsv")
    labels = load_labels(task_dir)
    names = class_names(args.task, labels)
    preds = load_predictions_jsonl(args.predictions)

    # Prefer embedded gold; else align by index / id
    scored = []
    by_id = {r["id"]: r for r in gold_rows}
    for p in preds:
        if "gold" not in p:
            g = by_id.get(str(p.get("id")))
            if g is None and "i" in p:
                g = gold_rows[int(p["i"])]
            if g is None:
                raise SystemExit(f"cannot resolve gold for prediction {p!r}")
            p = dict(p)
            p["gold"] = gold_name(args.task, g["label"], labels)
        scored.append(p)

    summary = score_rows(scored, names=names, n_planned=int(lock["tasks"][args.task]["rows_test"]))
    summary["arm"] = args.arm
    summary["task"] = args.task
    summary["lockfile_bakeoff_id"] = lock["bakeoff_id"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
