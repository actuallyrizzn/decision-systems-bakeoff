#!/usr/bin/env python3
"""Build a tiny prediction file from smoke sst2 gold and score it."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_bakeoff.data import class_names, gold_name, load_labels, load_tsv  # noqa: E402
from decision_bakeoff.score import score_rows  # noqa: E402


def main() -> int:
    task_dir = ROOT / "fixtures" / "smoke" / "sst2"
    rows = load_tsv(task_dir / "test.tsv")[:5]
    labels = load_labels(task_dir)
    names = class_names("sst2", labels)
    preds = []
    for i, row in enumerate(rows):
        gold = gold_name("sst2", row["label"], labels)
        # perfect predictions for plumbing check
        probs = {n: (1.0 if n == gold else 0.0) for n in names}
        preds.append(
            {
                "i": i,
                "id": row["id"],
                "gold": gold,
                "choice": gold,
                "probabilities": probs,
                "seconds": 0.001,
            }
        )
    summary = score_rows(preds, names=names, n_planned=len(rows))
    assert summary["accuracy"] == 1.0
    assert summary["brier"] == 0.0
    print(json.dumps(summary, indent=2))
    print("smoke-score OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
