#!/usr/bin/env python3
"""Unattended fair-suite chain: classification tasks then typed-decisions.

Default arm=julia. Writes STATUS.json for live observation.
Classification tasks: agnews, emotion, massive_scenario_en.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
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
from decision_bakeoff.jev import call_jev, extract_choice  # noqa: E402
from decision_bakeoff.julia_arm import JuliaArm  # noqa: E402
from decision_bakeoff.laya_arm import LayaArm  # noqa: E402
from decision_bakeoff.score import brier  # noqa: E402

FAIR_CLASSIFICATION = ("agnews", "emotion", "massive_scenario_en")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "updated_at": utc_now()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _empty_meter() -> dict:
    return {"input_tokens": None, "output_tokens": None, "balance_usd": None}


def run_classification(
    *,
    arm_name: str,
    model: str | None,
    decide_fn,
    task: str,
    data_root: Path,
    out_dir: Path,
    status_path: Path,
    status_base: dict,
    limit: int | None,
    min_interval: float,
    start_at: int = 0,
    row_retries: int = 5,
    usd_stop: float | None = None,
) -> dict:
    """decide_fn(state, questions) → (payload, elapsed, meter).

    meter keys (full fidelity with run_arm.py): input_tokens, output_tokens,
    balance_usd. Non-Jev arms pass nulls.
    """
    lock = load_lockfile()
    verify_task_test(task, data_root, lock)
    task_dir = data_root / task
    rows = load_tsv(task_dir / "test.tsv")
    labels = load_labels(task_dir)
    questions, qid = questions_for(task, labels)
    names = class_names(task, labels)
    price = float(lock["jev_pricing"]["input_usd_per_mtok"])
    if usd_stop is None and arm_name == "jev":
        usd_stop = float(lock["jev_pricing"].get("usd_stop_fair_default")
                         or lock["jev_pricing"]["usd_stop_default"])

    # Resume: keep prior scored rows if present and start_at > 0
    prior_path = out_dir / f"{arm_name}_{task}_rows.jsonl"
    results: list[dict] = []
    if start_at and prior_path.is_file():
        for line in prior_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r.get("i", -1)) < start_at:
                results.append(r)
        results.sort(key=lambda r: int(r["i"]))
    subset = rows[start_at:]
    if limit is not None:
        subset = subset[:limit]
    n_planned = start_at + len(subset)
    latencies: list[float] = [float(r["seconds"]) for r in results]
    briars: list[float] = [float(r["brier"]) for r in results]
    correct = sum(1 for r in results if r.get("hit"))
    input_tokens = sum(int(r["input_tokens"]) for r in results if r.get("input_tokens") is not None)
    output_tokens = sum(
        int(r["output_tokens"]) for r in results if r.get("output_tokens") is not None
    )
    usd = (input_tokens * price / 1_000_000.0) if arm_name == "jev" else 0.0
    balance_usd = None
    stopped = None
    t_task0 = time.perf_counter()
    last = 0.0

    for offset, row in enumerate(subset):
        i = start_at + offset
        if min_interval and arm_name == "jev":
            wait = min_interval - (time.perf_counter() - last)
            if wait > 0:
                time.sleep(wait)
        gold = gold_name(task, row["label"], labels)
        payload = None
        elapsed = 0.0
        choice = ""
        probs: dict = {}
        meter = _empty_meter()
        last_exc: Exception | None = None
        for attempt in range(row_retries):
            try:
                payload, elapsed, meter = decide_fn(row["text"], questions)
                last = time.perf_counter()
                choice, probs = extract_choice(payload, qid)
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(min(30.0, 2.0 ** attempt))
        if last_exc is not None:
            stopped = f"error at row {i}: {type(last_exc).__name__}: {last_exc}"
            break
        it = meter.get("input_tokens")
        ot = meter.get("output_tokens")
        if meter.get("balance_usd") is not None:
            balance_usd = meter["balance_usd"]
        if it is not None:
            input_tokens += int(it)
            usd = input_tokens * price / 1_000_000.0
        if ot is not None:
            output_tokens += int(ot)
        hit = choice == gold
        if hit:
            correct += 1
        br = brier(probs, gold, names)
        briars.append(br)
        latencies.append(elapsed)
        results.append(
            {
                "i": i,
                "id": row.get("id"),
                "gold": gold,
                "choice": choice,
                "hit": hit,
                "brier": br,
                "seconds": elapsed,
                "input_tokens": it if arm_name == "jev" else None,
                "output_tokens": ot if arm_name == "jev" else None,
                "probabilities": probs,
            }
        )
        if arm_name == "jev" and usd_stop is not None and usd > usd_stop:
            stopped = f"usd_stop ${usd:.6f} > ${usd_stop}"
            break
        done = len(results)
        if done % 50 == 0 or offset == 0 or offset + 1 == len(subset):
            acc = correct / done
            med = sorted(latencies)[len(latencies) // 2]
            write_status(
                status_path,
                {
                    **status_base,
                    "phase": "running",
                    "current_task": task,
                    "tasks_planned_all_arms": ["julia", "jev", "laya", "flybrain"],
                    "progress": {
                        "task": task,
                        "done": done,
                        "planned": n_planned,
                        "index": i,
                        "accuracy": round(acc, 4),
                        "median_seconds": round(med, 4),
                        "finishability": round(done / n_planned, 4) if n_planned else None,
                        "input_tokens": input_tokens if arm_name == "jev" else None,
                        "usd": round(usd, 6) if arm_name == "jev" else None,
                        "balance_usd_last": balance_usd,
                        "elapsed_task_seconds": round(time.perf_counter() - t_task0, 1),
                    },
                },
            )
            usd_bit = f" usd={usd:.6f}" if arm_name == "jev" else ""
            print(
                f"[{arm_name} {task}] {done}/{n_planned} "
                f"acc={acc:.4f}{usd_bit} med_s={med:.3f}",
                flush=True,
            )

    n = len(results)
    summary = {
        "arm": arm_name,
        "model": model,
        "task": task,
        "n_scored": n,
        "n_planned": n_planned,
        "finishability": (n / n_planned) if n_planned else None,
        "index_offset": start_at,
        "accuracy": (correct / n) if n else None,
        "brier": (sum(briars) / n) if n else None,
        "median_seconds": float(statistics.median(latencies)) if latencies else None,
        "input_tokens": input_tokens if arm_name == "jev" else None,
        "output_tokens": output_tokens if arm_name == "jev" else None,
        "usd": usd if arm_name == "jev" else None,
        "stopped": stopped,
        "balance_usd_last": balance_usd,
        "qid": qid,
        "lockfile_bakeoff_id": lock.get("bakeoff_id"),
        "lockfile_fair_id": lock.get("fair_bakeoff_id"),
        "wall_seconds": round(time.perf_counter() - t_task0, 1),
        "finished_at": utc_now(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{arm_name}_{task}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / f"{arm_name}_{task}_rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + ("\n" if results else ""),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--status", type=Path, required=True)
    ap.add_argument("--arm", default="julia", choices=["julia", "jev", "laya"])
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--tasks", default="agnews,emotion,massive_scenario_en,typed_decisions")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-at", type=int, default=0, help="Resume classification from this index")
    ap.add_argument("--skip-typed", action="store_true")
    ap.add_argument("--usd-stop", type=float, default=None)
    ap.add_argument("--min-interval", type=float, default=0.35)
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Do not abort the chain if a task stops early",
    )
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    status_base = {
        "run_id": f"fair-{args.arm}-{utc_now()}",
        "suite": "fair",
        "fair_bakeoff_id": "julia-fair-turf-2026-09",
        "arm": args.arm,
        "tasks_planned": tasks,
        "tasks_planned_all_arms": ["julia", "jev", "laya", "flybrain"],
        "completed": [],
        "observable": {
            "status_json": str(args.status),
            "out_dir": str(args.out),
            "how_to_check": "ask Otto for a status check",
        },
    }
    write_status(args.status, {**status_base, "phase": "loading", "current_task": None})

    model_name: str | None = None
    decide_fn = None
    if args.arm == "julia":
        if not args.model_dir:
            raise SystemExit("--model-dir required for julia")
        print(f"loading Julia from {args.model_dir}", flush=True)
        julia = JuliaArm(args.model_dir, device="cpu", max_length=1024, head_length=512)
        model_name = "SupersonicLabs/Julia-1"

        def decide_fn(state, questions):
            payload, elapsed = julia.decide(state, questions)
            return payload, elapsed, _empty_meter()

    elif args.arm == "laya":
        os.environ.setdefault("USE_TF", "0")
        laya = LayaArm(multilingual=False)
        model_name = "convaiinnovations/laya"

        def decide_fn(state, questions):
            payload, elapsed = laya.decide(state, questions)
            return payload, elapsed, _empty_meter()

    else:
        api_key = os.environ.get("VENICE_JEV_AB_API_KEY") or os.environ.get("VENICE_API_KEY")
        if not api_key:
            raise SystemExit("Set VENICE_JEV_AB_API_KEY or VENICE_API_KEY")
        model_name = "jev-latest"

        def decide_fn(state, questions):
            payload, elapsed, headers = call_jev(
                api_key=api_key, state=state, questions=questions
            )
            usage = payload.get("usage") or {}
            it = int(usage.get("input_tokens") or 0)
            ot = int(usage.get("output_tokens") or 0)
            # header fallbacks (some Venice responses put meters here)
            if not it:
                for hk in ("x-venice-prompt-tokens", "x-prompt-tokens"):
                    if headers.get(hk):
                        try:
                            it = int(float(headers[hk]))
                        except ValueError:
                            pass
                        break
            if not ot:
                for hk in ("x-venice-completion-tokens", "x-completion-tokens"):
                    if headers.get(hk):
                        try:
                            ot = int(float(headers[hk]))
                        except ValueError:
                            pass
                        break
            bal = None
            raw_bal = headers.get("x-venice-balance-usd")
            if raw_bal is not None:
                try:
                    bal = float(raw_bal)
                except ValueError:
                    pass
            return (
                payload,
                elapsed,
                {"input_tokens": it, "output_tokens": ot, "balance_usd": bal},
            )

    completed: list[dict] = []
    for task in tasks:
        if task == "typed_decisions":
            if args.skip_typed:
                continue
            write_status(
                args.status,
                {
                    **status_base,
                    "phase": "running",
                    "current_task": task,
                    "completed": completed,
                },
            )
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "run_typed_decisions.py"),
                "--data",
                str(args.data),
                "--out",
                str(args.out),
                "--arm",
                args.arm,
                "--status",
                str(args.status),
            ]
            if args.arm == "julia":
                cmd += ["--model-dir", str(args.model_dir)]
            if args.limit:
                cmd += ["--limit-cases", str(args.limit)]
            if args.usd_stop is not None:
                cmd += ["--usd-stop", str(args.usd_stop)]
            rc = subprocess.call(cmd)
            summary_path = args.out / f"{args.arm}_typed_decisions_summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                completed.append(summary)
                status_base = {**status_base, "completed": completed}
            if rc != 0 and not args.continue_on_error:
                write_status(
                    args.status,
                    {
                        **status_base,
                        "phase": "error",
                        "current_task": task,
                        "note": f"typed_decisions exit {rc}",
                    },
                )
                return rc
            continue

        if task not in FAIR_CLASSIFICATION:
            raise SystemExit(f"unsupported fair task {task}")
        write_status(
            args.status,
            {
                **status_base,
                "phase": "running",
                "current_task": task,
                "completed": completed,
            },
        )
        summary = run_classification(
            arm_name=args.arm,
            model=model_name,
            decide_fn=decide_fn,
            task=task,
            data_root=args.data,
            out_dir=args.out,
            status_path=args.status,
            status_base={**status_base, "completed": completed},
            limit=args.limit,
            min_interval=args.min_interval if args.arm == "jev" else 0.0,
            start_at=args.start_at if task == tasks[0] else 0,
            usd_stop=args.usd_stop,
        )
        completed.append(summary)
        status_base = {**status_base, "completed": completed}
        if summary.get("stopped") and not args.continue_on_error:
            write_status(
                args.status,
                {
                    **status_base,
                    "phase": "error",
                    "current_task": task,
                    "note": summary.get("stopped"),
                },
            )
            return 1

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
    lines = [
        f"# Fair-turf bakeoff results — {args.arm}",
        "",
        f"Finished: {utc_now()}",
        "",
        "| Test | Rows | Finish | Accuracy | Brier | Median s | USD | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in completed:
        fin = s.get("finishability")
        fin_s = f"{fin:.4f}" if isinstance(fin, (int, float)) else ""
        usd = s.get("usd") if s.get("usd") is not None else s.get("usd_spent_est")
        usd_s = f"{usd:.6f}" if isinstance(usd, (int, float)) else ""
        if s["task"] == "typed_decisions":
            lines.append(
                f"| typed_decisions | {s.get('n_scored')}/{s.get('n_planned')} | "
                f"{fin_s} | {s.get('accuracy'):.4f} | {s.get('brier'):.4f} | "
                f"{s.get('median_seconds_per_case')} | {usd_s} | "
                f"by_type={s.get('by_type_accuracy')} |"
            )
        else:
            lines.append(
                f"| {s['task']} | {s['n_scored']}/{s['n_planned']} | "
                f"{fin_s} | {s['accuracy']:.4f} | {s['brier']:.4f} | "
                f"{s['median_seconds']:.4f} | {usd_s} | |"
            )
    (args.out / f"RESULTS_{args.arm}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
