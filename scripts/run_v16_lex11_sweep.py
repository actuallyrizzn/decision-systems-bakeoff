#!/usr/bin/env python3
"""v16.2 lex fast sweep — regex-only fire stats on wild teacher sets (no Laya).

Fires lexicon_hits_v16_1 on every row of a teacher dataset, reports fire rate,
per-hub fire counts and precision, and dumps every wrong fire for autopsy.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from decision_bakeoff.data import load_labels, load_tsv  # noqa: E402
from run_v16_lex11 import lexicon_hits_v16_1  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.data)
    rows = load_tsv(args.data / "test.tsv")

    n = 0
    lex_fire: Counter = Counter()
    lex_fire_correct: Counter = Counter()
    wrong_fires: list[dict] = []
    with (out / "sweep_rows.jsonl").open("w") as fh:
        for row in rows:
            gold = labels[int(row["label"])]
            n += 1
            h = lexicon_hits_v16_1(row["text"])
            hub = h[0][1] if h else None
            if hub:
                lex_fire[hub] += 1
                ok = hub == gold
                lex_fire_correct[hub] += int(ok)
                if not ok:
                    wrong_fires.append({
                        "id": row["id"], "gold": gold, "fired": hub,
                        "hint": h[0][0], "text": row["text"][:240],
                    })
            fh.write(json.dumps({
                "id": row["id"], "gold": gold, "v16_2_lex_hit": hub,
            }) + "\n")

    fires = sum(lex_fire.values())
    scores = {
        "tag": f"v16_2_lex_sweep:{args.data.name}",
        "n": n,
        "fire_rate": round(fires / n, 4),
        "fire_precision": round(sum(lex_fire_correct.values()) / max(1, fires), 4),
        "fire_by_hub": dict(lex_fire),
        "fire_correct_by_hub": dict(lex_fire_correct),
        "wrong_fires": wrong_fires,
    }
    (out / "sweep_scores.json").write_text(json.dumps(scores, indent=1))
    print(json.dumps({k: scores[k] for k in ("tag", "n", "fire_rate", "fire_precision")}, indent=1))
    print("fire_by_hub", dict(lex_fire))
    print("fire_correct_by_hub", dict(lex_fire_correct))
    print(f"wrong fires: {len(wrong_fires)}")
    for w in wrong_fires:
        print(f"  {w['id']} gold={w['gold']} fired={w['fired']} hint={w['hint']} :: {w['text'][:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
