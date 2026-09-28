#!/usr/bin/env python3
"""Run Julia-1 cold on locked bakeoff tasks (excludes clinc150 — >20 options).

Writes STATUS.json for live observation and per-task summary/rows JSONL.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
from decision_bakeoff.jev import extract_choice  # noqa: E402
from decision_bakeoff.julia_arm import JuliaArm  # noqa: E402
from decision_bakeoff.score import brier  # noqa: E402

COMPATIBLE = ("sst2", "clinc10", "bugsev")
EXCLUDED = ("clinc150",)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "updated_at": utc_now()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_one(
    *,
    arm: JuliaArm,
    task: str,
    data_root: Path,
    out_dir: Path,
    status_path: Path,
    status_base: dict,
    limit: int | None,
    start_at: int,
) -> dict:
    lock = load_lockfile()
    verify_task_test(task, data_root, lock)
    task_dir = data_root / task
    rows = load_tsv(task_dir / "test.tsv")
    if start_at:
        rows = rows[start_at:]
    labels = load_labels(task_dir)
    questions, qid = questions_for(task, labels)
    names = class_names(task, labels)
    # Julia native max 20 options
    n_opts = len(questions[qid].get("criteria") or {})
    if n_opts > 20:
        raise SystemExit(f"REFUSE {task}: {n_opts} options > Julia native max 20")

    subset = rows[:limit] if limit else rows
    results = []
    latencies: list[float] = []
    briars: list[float] = []
    correct = 0
    stopped = None
    t_task0 = time.perf_counter()

    for i, row in enumerate(subset):
        state = row["text"]
        gold = gold_name(task, row["label"], labels)
        try:
            payload, elapsed = arm.decide(state, questions)
            choice, probs = extract_choice(payload, qid)
        except Exception as exc:  # noqa: BLE001
            stopped = f"error at row {start_at + i}: {type(exc).__name__}: {exc}"
            break

        hit = choice == gold
        if hit:
            correct += 1
        br = brier(probs, gold, names)
        briars.append(br)
        latencies.append(elapsed)
        results.append(
            {
                "i": start_at + i,
                "id": row.get("id"),
                "gold": gold,
                "choice": choice,
                "hit": hit,
                "brier": br,
                "seconds": elapsed,
                "probabilities": probs,
            }
        )
        if (i + 1) % 25 == 0 or i == 0 or i + 1 == len(subset):
            acc = correct / (i + 1)
            med = sorted(latencies)[len(latencies) // 2]
            write_status(
                status_path,
                {
                    **status_base,
                    "phase": "running",
                    "current_task": task,
                    "progress": {
                        "task": task,
                        "done": i + 1,
                        "planned": len(subset),
                        "index": start_at + i,
                        "accuracy": round(acc, 4),
                        "median_seconds": round(med, 4),
                        "elapsed_task_seconds": round(time.perf_counter() - t_task0, 1),
                    },
                },
            )
            print(
                f"[julia {task}] {start_at + i + 1}/{start_at + len(subset)} "
                f"acc={acc:.4f} med_s={med:.3f}",
                flush=True,
            )

    n = len(results)
    summary = {
        "arm": "julia",
        "model": "SupersonicLabs/Julia-1",
        "task": task,
        "n_scored": n,
        "n_planned": len(subset),
        "index_offset": start_at,
        "accuracy": (correct / n) if n else None,
        "brier": (sum(briars) / n) if n else None,
        "median_seconds": float(statistics.median(latencies)) if latencies else None,
        "stopped": stopped,
        "qid": qid,
        "lockfile_bakeoff_id": lock["bakeoff_id"],
        "wall_seconds": round(time.perf_counter() - t_task0, 1),
        "finished_at": utc_now(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"julia_{task}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / f"julia_{task}_rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + ("\n" if results else ""),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--status", type=Path, required=True, help="Live STATUS.json path")
    ap.add_argument(
        "--tasks",
        default="sst2,clinc10,bugsev",
        help="Comma list; clinc150 refused",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-at", type=int, default=0)
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t in EXCLUDED or t == "clinc150":
            raise SystemExit(
                f"REFUSE task {t}: Julia native max 20 options; excluded from this arm"
            )
        if t not in COMPATIBLE:
            raise SystemExit(f"unknown/unsupported task {t}")

    status_base = {
        "run_id": f"julia-bakeoff-{utc_now()}",
        "arm": "julia",
        "model": "SupersonicLabs/Julia-1",
        "host_hint": "ngram",
        "tasks_planned": tasks,
        "tasks_excluded": list(EXCLUDED),
        "exclude_reason": "Julia accepts at most 20 options per call; clinc150 has 151",
        "completed": [],
        "observable": {
            "status_json": str(args.status),
            "out_dir": str(args.out),
            "how_to_check": "cat STATUS.json; or ask Otto for a status check",
        },
    }
    write_status(args.status, {**status_base, "phase": "loading_model", "current_task": None})

    print(f"loading Julia-1 from {args.model_dir}", flush=True)
    arm = JuliaArm(args.model_dir, device="cpu", max_length=1024, head_length=512)
    write_status(args.status, {**status_base, "phase": "model_ready", "current_task": None})

    completed: list[dict] = []
    for task in tasks:
        write_status(
            args.status,
            {
                **status_base,
                "phase": "running",
                "current_task": task,
                "completed": completed,
                "progress": {"task": task, "done": 0, "planned": "?", "note": "starting"},
            },
        )
        summary = run_one(
            arm=arm,
            task=task,
            data_root=args.data,
            out_dir=args.out,
            status_path=args.status,
            status_base={**status_base, "completed": completed},
            limit=args.limit,
            start_at=args.start_at if task == tasks[0] else 0,
        )
        completed.append(summary)
        status_base = {**status_base, "completed": completed}

    write_status(
        args.status,
        {
            **status_base,
            "phase": "done",
            "current_task": None,
            "progress": None,
            "completed": completed,
        },
    )
    # compact results markdown for humans
    lines = [
        f"# Julia-1 bakeoff results",
        f"",
        f"Finished: {utc_now()}",
        f"Excluded: clinc150 (too many menu options for Julia).",
        f"",
        f"| Test | Rows scored | Accuracy | Brier | Median s |",
        f"|---|---:|---:|---:|---:|",
    ]
    for s in completed:
        lines.append(
            f"| {s['task']} | {s['n_scored']}/{s['n_planned']} | "
            f"{s['accuracy']:.4f} | {s['brier']:.4f} | {s['median_seconds']:.4f} |"
        )
    (args.out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
