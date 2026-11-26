#!/usr/bin/env python3
"""v16 teacher-scale confirm — do v16 lex + hybrids transfer to wild teacher sets?

Arms compared per row: instr_v2, force_lex_r5 (baselines),
  v16lex_else_confuse_x (gold 1.0), v16lex_else_v5 (gold 0.97).
Also logs per-hub lex fire stats (does new-hub lex fire on wild text, and is it right?).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from decision_bakeoff.data import load_labels, load_tsv  # noqa: E402
from decision_bakeoff.laya_arm import LayaArm  # noqa: E402
from run_failure_pattern_round5 import lexicon_hits as lexicon_hits_r5  # noqa: E402
from run_failure_pattern_round6 import (  # noqa: E402
    decide,
    safe_path,
    _margin,
    _top2,
)
from run_failure_pattern_round9 import apply_confuse_expanded  # noqa: E402
from run_v16_lex11 import lexicon_hits_v16_1 as lexicon_hits_v16  # noqa: E402
from run_v16_cycle2 import build_q_v5  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.data)
    rows = load_tsv(args.data / "test.tsv")
    if args.limit:
        rows = rows[: args.limit]
    q_v2 = json.loads((REPO / "questions" / "centa_topics_instr_v2.json").read_text())
    q_v3 = json.loads((REPO / "questions" / "centa_topics_instr_v3.json").read_text())
    q_v5 = build_q_v5(q_v3)
    criteria = dict(q_v2["topic"]["criteria"])

    os.environ.setdefault("USE_TF", "0")
    print(f"loading Laya… data={args.data.name} n={len(rows)}", flush=True)
    arm = LayaArm()
    print("Laya ready", flush=True)

    hit: Counter = Counter()
    n = 0
    t0 = time.perf_counter()
    lex_fire: Counter = Counter()
    lex_fire_correct: Counter = Counter()
    hub_lex_fire: dict[str, Counter] = defaultdict(Counter)
    with (out / "confirm_rows.jsonl").open("w") as fh:
        for row in rows:
            gold = labels[int(row["label"])]
            state = row["text"]
            n += 1
            c2, _, p2, _ = decide(arm, state, q_v2, 0.3)
            m2 = _margin(p2)
            t2 = _top2(p2)
            hit["instr_v2"] += int(c2 == gold)

            c3, _, p3, _ = decide(arm, state, q_v3, 0.3)
            t3 = _top2(p3)
            c5, _, p5, _ = decide(arm, state, q_v5, 0.3)

            lex_r5 = lexicon_hits_r5(state)
            lex_v16 = lexicon_hits_v16(state)
            c_safe_r5, _ = safe_path(arm, state, criteria, c2, m2, t2, lex_r5)
            c_r5 = lex_r5[0][1] if lex_r5 else c_safe_r5
            hit["force_lex_r5"] += int(c_r5 == gold)

            c_cf, _ = apply_confuse_expanded(arm, state, criteria, c3, t3, p3)
            ch_conf = lex_v16[0][1] if lex_v16 else c_cf
            ch_v5 = lex_v16[0][1] if lex_v16 else c5
            hit["v16lex_else_confuse_x"] += int(ch_conf == gold)
            hit["v16lex_else_v5"] += int(ch_v5 == gold)

            if lex_v16:
                hub = lex_v16[0][1]
                lex_fire[hub] += 1
                hub_lex_fire[gold][hub] += 1
                lex_fire_correct[hub] += int(hub == gold)

            fh.write(json.dumps({
                "id": row["id"], "gold": gold, "instr_v2": c2, "force_lex_r5": c_r5,
                "v16lex_else_confuse_x": ch_conf, "v16lex_else_v5": ch_v5,
                "v16_lex_hit": lex_v16[0][1] if lex_v16 else None,
            }) + "\n")
            if n % 100 == 0:
                acc = {k: round(v / n, 3) for k, v in hit.items()}
                print(f"[{n}/{len(rows)}] {acc}", flush=True)

    scores = {
        "tag": f"v16_2_confirm:{args.data.name}",
        "n": n,
        "accuracies": {k: round(v / n, 4) for k, v in hit.items()},
        "counts": dict(hit),
        "lex_v16_fire_by_hub": dict(lex_fire),
        "lex_v16_fire_correct_by_hub": dict(lex_fire_correct),
        "lex_v16_fire_rate": round(sum(lex_fire.values()) / n, 4),
        "lex_v16_fire_precision": round(
            sum(lex_fire_correct.values()) / max(1, sum(lex_fire.values())), 4
        ),
        "gold_x_fired_hub": {g: dict(c) for g, c in hub_lex_fire.items()},
        "wall_s": time.perf_counter() - t0,
        "complete": True,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "confirm_scores.json").write_text(json.dumps(scores, indent=1))
    print(json.dumps(scores["accuracies"], indent=1))
    print("lex fire rate", scores["lex_v16_fire_rate"], "precision", scores["lex_v16_fire_precision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
