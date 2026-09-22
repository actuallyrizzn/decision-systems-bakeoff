#!/usr/bin/env python3
"""Run one arm (jev|laya) on one locked task. Writes summary + rows JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# allow `python scripts/run_arm.py` without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_bakeoff.data import (  # noqa: E402
    class_names,
    gold_name,
    load_labels,
    load_lockfile,
    load_tsv,
    questions_for,
    verify_task_test,
)
from decision_bakeoff.jev import call_jev, extract_choice  # noqa: E402
from decision_bakeoff.laya_arm import LayaArm  # noqa: E402
from decision_bakeoff.score import brier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="Data root containing <task>/test.tsv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", required=True, choices=["sst2", "clinc10", "clinc150", "bugsev"])
    ap.add_argument("--arm", required=True, choices=["jev", "laya"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-at", type=int, default=0)
    ap.add_argument("--min-interval", type=float, default=0.35)
    ap.add_argument("--usd-stop", type=float, default=None)
    ap.add_argument("--skip-verify", action="store_true", help="Dangerous; for smoke fixtures only")
    args = ap.parse_args()

    lock = load_lockfile()
    if not args.skip_verify:
        verify_task_test(args.task, args.data, lock)
    else:
        print("WARNING: --skip-verify set; not a locked run", file=sys.stderr)

    task_dir = args.data / args.task
    rows = load_tsv(task_dir / "test.tsv")
    if args.start_at:
        rows = rows[args.start_at :]
    labels = load_labels(task_dir)
    questions, qid = questions_for(args.task, labels)
    names = class_names(args.task, labels)

    price = float(lock["jev_pricing"]["input_usd_per_mtok"])
    usd_stop = args.usd_stop
    if usd_stop is None:
        usd_stop = float(lock["jev_pricing"]["usd_stop_default"])

    api_key = None
    laya = None
    if args.arm == "jev":
        api_key = os.environ.get("VENICE_JEV_AB_API_KEY") or os.environ.get("VENICE_API_KEY")
        if not api_key:
            print("Set VENICE_JEV_AB_API_KEY or VENICE_API_KEY", file=sys.stderr)
            return 2
    else:
        os.environ.setdefault("USE_TF", "0")
        multilingual = args.task == "clinc150"
        laya = LayaArm(
            multilingual=multilingual,
            max_len=4096 if multilingual else None,
            head_max_len=2048 if multilingual else None,
        )

    subset = rows[: args.limit] if args.limit else rows
    results = []
    latencies: list[float] = []
    briars: list[float] = []
    correct = 0
    input_tokens = 0
    usd = 0.0
    stopped = None
    balance_usd = None

    for i, row in enumerate(subset):
        state = row["text"]
        gold = gold_name(args.task, row["label"], labels)
        it = 0
        try:
            if args.arm == "jev":
                assert api_key
                payload, elapsed, headers = call_jev(
                    api_key=api_key, state=state, questions=questions
                )
                bal = headers.get("x-venice-balance-usd")
                if bal is not None:
                    try:
                        balance_usd = float(bal)
                    except ValueError:
                        pass
                usage = payload.get("usage") or {}
                it = int(usage.get("input_tokens") or 0)
                input_tokens += it
                usd = input_tokens * price / 1_000_000.0
                if args.min_interval > 0:
                    time.sleep(args.min_interval)
            else:
                assert laya is not None
                payload, elapsed = laya.decide(state, questions)
            choice, probs = extract_choice(payload, qid)
        except Exception as exc:  # noqa: BLE001
            stopped = f"error at row {args.start_at + i}: {type(exc).__name__}: {exc}"
            break

        hit = choice == gold
        if hit:
            correct += 1
        br = brier(probs, gold, names)
        briars.append(br)
        latencies.append(elapsed)
        results.append(
            {
                "i": args.start_at + i,
                "id": row.get("id"),
                "gold": gold,
                "choice": choice,
                "hit": hit,
                "brier": br,
                "seconds": elapsed,
                "input_tokens": it if args.arm == "jev" else None,
                "probabilities": probs,
            }
        )
        if args.arm == "jev" and usd > usd_stop:
            stopped = f"usd_stop ${usd:.6f} > ${usd_stop}"
            break
        if (i + 1) % 25 == 0 or i == 0:
            acc = correct / (i + 1)
            med = sorted(latencies)[len(latencies) // 2]
            print(
                f"[{args.arm} {args.task}] {args.start_at + i + 1} "
                f"acc={acc:.4f} usd={usd:.6f} med_s={med:.3f}",
                flush=True,
            )

    n = len(results)
    import statistics

    summary = {
        "arm": args.arm,
        "task": args.task,
        "n_scored": n,
        "n_planned": len(subset),
        "index_offset": args.start_at,
        "accuracy": (correct / n) if n else None,
        "brier": (sum(briars) / n) if n else None,
        "median_seconds": float(statistics.median(latencies)) if latencies else None,
        "input_tokens": input_tokens if args.arm == "jev" else None,
        "usd": usd if args.arm == "jev" else None,
        "stopped": stopped,
        "balance_usd_last": balance_usd,
        "qid": qid,
        "lockfile_bakeoff_id": lock["bakeoff_id"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f"_from{args.start_at}" if args.start_at else ""
    (args.out / f"{args.arm}_{args.task}{suffix}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / f"{args.arm}_{args.task}{suffix}_rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + ("\n" if results else ""),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if stopped is None or n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
