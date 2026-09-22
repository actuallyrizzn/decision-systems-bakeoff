"""Score prediction JSONL against frozen gold."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def brier(probs: dict[str, float], gold: str, names: list[str]) -> float:
    total = 0.0
    for name in names:
        p = float(probs.get(name, 0.0))
        y = 1.0 if name == gold else 0.0
        total += (p - y) ** 2
    return total


def score_rows(
    rows: list[dict[str, Any]],
    *,
    names: list[str],
    n_planned: int | None = None,
) -> dict[str, Any]:
    if not rows:
        return {
            "n_scored": 0,
            "n_planned": n_planned,
            "accuracy": None,
            "brier": None,
            "median_seconds": None,
            "finishability": 0.0 if n_planned else None,
        }
    correct = sum(1 for r in rows if r.get("hit") or r.get("choice") == r.get("gold"))
    # recompute hit/brier if only choice+probs+gold present
    briars: list[float] = []
    latencies: list[float] = []
    hits = 0
    for r in rows:
        gold = str(r["gold"])
        choice = str(r["choice"])
        probs = {str(k): float(v) for k, v in (r.get("probabilities") or {}).items()}
        hit = choice == gold
        if hit:
            hits += 1
        briars.append(brier(probs, gold, names) if "brier" not in r else float(r["brier"]))
        if r.get("seconds") is not None:
            latencies.append(float(r["seconds"]))
        elif r.get("ms") is not None:
            latencies.append(float(r["ms"]) / 1000.0)
    n = len(rows)
    planned = n_planned if n_planned is not None else n
    return {
        "n_scored": n,
        "n_planned": planned,
        "accuracy": hits / n,
        "brier": sum(briars) / n,
        "median_seconds": float(statistics.median(latencies)) if latencies else None,
        "finishability": n / planned if planned else None,
    }


def load_predictions_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows
