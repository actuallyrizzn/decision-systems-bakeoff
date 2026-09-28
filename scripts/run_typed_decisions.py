#!/usr/bin/env python3
"""Run typed-decisions (LocalLLaMA pinned parquet) for julia|jev|laya|flybrain.

Scoring matches Julia's published harness: argmax over probabilities for all
three question types (do not round score expected index).

Important: Julia/Jev typed APIs require score criteria as an *ordered list*,
not a dict of index→label. Choice/noul stay as dicts.

Flybrain: reservoir zero-shot (cosine context↔option) — no ridge head; needs --glove.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_bakeoff.data import load_lockfile, sha256_file, verify_typed_parquet  # noqa: E402
from decision_bakeoff.jev import call_jev, extract_typed_answer  # noqa: E402
from decision_bakeoff.julia_arm import JuliaArm  # noqa: E402
from decision_bakeoff.laya_arm import LayaArm  # noqa: E402
from decision_bakeoff.score import brier  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def state_to_text(state: object) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def build_typed_question(
    kind: str, instructions: str, criteria: object
) -> tuple[dict[str, Any], list[str]]:
    """Return (question object for API, label keys in gold/argmax order)."""
    if kind == "score":
        if isinstance(criteria, list):
            labels = [str(v) for v in criteria]
        elif isinstance(criteria, dict):
            # recover order if someone already dict-ified indices
            keys_sorted = sorted(
                criteria.keys(),
                key=lambda k: int(k) if str(k).isdigit() else str(k),
            )
            labels = [str(criteria[k]) for k in keys_sorted]
        else:
            raise ValueError(f"score criteria must be list, got {type(criteria)}")
        if not 2 <= len(labels) <= 20:
            raise ValueError(f"score rubric length {len(labels)} out of 2–20")
        keys = [str(i) for i in range(len(labels))]
        return {
            "type": "score",
            "instructions": instructions,
            "criteria": labels,  # MUST stay a list for Julia typed.py
        }, keys

    if kind == "noul":
        if criteria is None:
            criteria = {"false": "false", "true": "true"}
        if not isinstance(criteria, dict) or set(criteria) != {"false", "true"}:
            raise ValueError("noul criteria must map false/true")
        keys = ["false", "true"]
        return {
            "type": "noul",
            "instructions": instructions,
            "criteria": {k: str(criteria[k]) for k in keys},
        }, keys

    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("choice criteria must be a nonempty dict")
        keys = list(criteria.keys())
        return {
            "type": "choice",
            "instructions": instructions,
            "criteria": {str(k): str(v) for k, v in criteria.items()},
        }, [str(k) for k in keys]

    raise ValueError(f"unknown typed question type {kind}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="Root containing typed_decisions/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", required=True, choices=["julia", "jev", "laya", "flybrain"])
    ap.add_argument("--model-dir", type=Path, default=None, help="Julia checkpoint dir")
    ap.add_argument("--glove", type=Path, default=None, help="GloVe path (required for flybrain)")
    ap.add_argument("--cfg", type=Path, default=None, help="Override flybrain_best.json")
    ap.add_argument("--limit-cases", type=int, default=None)
    ap.add_argument("--status", type=Path, default=None)
    ap.add_argument("--usd-stop", type=float, default=None)
    ap.add_argument("--min-interval", type=float, default=0.35)
    args = ap.parse_args()

    lock = load_lockfile()
    verify_typed_parquet(args.data, lock)
    parquet_path = args.data / "typed_decisions" / "test.parquet"

    import pyarrow.parquet as pq

    cases = pq.read_table(parquet_path).to_pylist()
    if args.limit_cases:
        cases = cases[: args.limit_cases]

    arm_obj = None
    api_key = None
    model_name: str | None = None
    fly_meta: dict[str, Any] = {}
    if args.arm == "julia":
        if not args.model_dir:
            raise SystemExit("--model-dir required for julia")
        arm_obj = JuliaArm(args.model_dir, device="cpu", max_length=1024, head_length=512)
        model_name = "SupersonicLabs/Julia-1"
    elif args.arm == "jev":
        api_key = os.environ.get("VENICE_JEV_AB_API_KEY") or os.environ.get("VENICE_API_KEY")
        if not api_key:
            raise SystemExit("Set VENICE_JEV_AB_API_KEY or VENICE_API_KEY")
        model_name = "jev-latest"
    elif args.arm == "flybrain":
        if not args.glove:
            raise SystemExit("--glove required for flybrain")
        from decision_bakeoff.flybrain.typed_arm import FlybrainTypedArm  # noqa: E402

        arm_obj = FlybrainTypedArm.from_typed_parquet(
            parquet_path=parquet_path,
            glove_path=args.glove,
            cfg_path=args.cfg,
        )
        model_name = "flybrain-typed-zeroshot"
        fly_meta = {
            "cfg_key": arm_obj.cfg_key,
            "glove_coverage": arm_obj.glove_coverage,
            "tokenizer_fingerprint": arm_obj.tokenizer_fingerprint,
            "packet_cap": arm_obj.cap,
        }
    else:
        os.environ.setdefault("USE_TF", "0")
        arm_obj = LayaArm(multilingual=False)
        model_name = "convaiinnovations/laya"

    price = float(lock["jev_pricing"]["input_usd_per_mtok"])
    usd_stop = args.usd_stop
    if usd_stop is None and args.arm == "jev":
        usd_stop = float(lock["jev_pricing"].get("usd_stop_fair_default", 2.0))
    input_tokens = 0
    output_tokens = 0
    usd = 0.0
    balance_usd = None

    results = []
    latencies: list[float] = []
    by_type_correct: dict[str, list[int]] = {"choice": [], "score": [], "noul": []}
    correct = 0
    planned = 0
    stopped = None
    t0 = time.perf_counter()
    last_call = 0.0
    cases_scored = 0

    for ci, case in enumerate(cases):
        state = json.loads(case["state"]) if isinstance(case["state"], str) else case["state"]
        gold = json.loads(case["gold"]) if isinstance(case["gold"], str) else case["gold"]
        questions_raw = (
            json.loads(case["questions"])
            if isinstance(case["questions"], str)
            else case["questions"]
        )
        state_text = state_to_text(state)
        questions: dict = {}
        meta_q: list[tuple[str, str, list[str], str]] = []
        for qid, q in questions_raw.items():
            kind = q["type"]
            qobj, keys = build_typed_question(kind, q["instructions"], q.get("criteria"))
            questions[qid] = qobj
            gold_label = str(gold[qid]["label"])
            meta_q.append((qid, kind, keys, gold_label))
            planned += 1

        case_it: int | None = None
        case_ot: int | None = None
        if args.arm == "jev":
            wait = args.min_interval - (time.perf_counter() - last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                payload, elapsed, headers = call_jev(
                    api_key=api_key, state=state_text, questions=questions
                )
            except Exception as exc:  # noqa: BLE001
                stopped = f"jev error case {case['id']}: {type(exc).__name__}: {exc}"
                break
            last_call = time.perf_counter()
            usage = payload.get("usage") or {}
            case_it = int(usage.get("input_tokens") or 0)
            case_ot = int(usage.get("output_tokens") or 0)
            if not case_it:
                for hk in ("x-venice-prompt-tokens", "x-prompt-tokens"):
                    if headers.get(hk):
                        try:
                            case_it = int(float(headers[hk]))
                        except ValueError:
                            pass
                        break
            if not case_ot:
                for hk in ("x-venice-completion-tokens", "x-completion-tokens"):
                    if headers.get(hk):
                        try:
                            case_ot = int(float(headers[hk]))
                        except ValueError:
                            pass
                        break
            raw_bal = headers.get("x-venice-balance-usd")
            if raw_bal is not None:
                try:
                    balance_usd = float(raw_bal)
                except ValueError:
                    pass
            input_tokens += case_it
            output_tokens += case_ot
            usd = input_tokens * price / 1_000_000.0
            if usd_stop is not None and usd > usd_stop:
                stopped = f"usd_stop ${usd:.6f} > ${usd_stop}"
        elif args.arm == "julia":
            try:
                payload, elapsed = arm_obj.decide(state_text, questions)
            except Exception as exc:  # noqa: BLE001
                try:
                    payload, elapsed = arm_obj.decide(state, questions)  # type: ignore[arg-type]
                except Exception as exc2:  # noqa: BLE001
                    stopped = f"julia error case {case['id']}: {exc} / {exc2}"
                    break
        else:
            # laya + flybrain
            try:
                payload, elapsed = arm_obj.decide(state_text, questions)
            except Exception as exc:  # noqa: BLE001
                stopped = f"{args.arm} error case {case['id']}: {type(exc).__name__}: {exc}"
                break

        for qi, (qid, kind, keys, gold_label) in enumerate(meta_q):
            try:
                choice, probs = extract_typed_answer(payload, qid, kind)
            except Exception as exc:  # noqa: BLE001
                stopped = f"extract {case['id']}:{qid}: {exc}"
                choice, probs = "", {}
            hit = choice == gold_label
            if hit:
                correct += 1
            by_type_correct[kind].append(1 if hit else 0)
            br = brier(probs, gold_label, list(keys)) if probs else None
            # Attribute case meters to the first question row only (avoid double-count sums)
            results.append(
                {
                    "i": len(results),
                    "case_id": case["id"],
                    "qid": qid,
                    "type": kind,
                    "gold": gold_label,
                    "choice": choice,
                    "hit": hit,
                    "brier": br,
                    "seconds": elapsed / max(1, len(meta_q)),
                    "input_tokens": case_it if args.arm == "jev" and qi == 0 else None,
                    "output_tokens": case_ot if args.arm == "jev" and qi == 0 else None,
                    "probabilities": probs,
                }
            )
        cases_scored = ci + 1
        latencies.append(elapsed)
        if args.status and ((ci + 1) % 10 == 0 or ci == 0 or ci + 1 == len(cases)):
            prior = {}
            if args.status.is_file():
                try:
                    prior = json.loads(args.status.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    prior = {}
            prior.update(
                {
                    "phase": "running",
                    "current_task": "typed_decisions",
                    "arm": args.arm,
                    "tasks_planned_all_arms": ["julia", "jev", "laya", "flybrain"],
                    "progress": {
                        "cases_done": ci + 1,
                        "cases_planned": len(cases),
                        "questions_scored": len(results),
                        "n_planned": planned,
                        "accuracy": (correct / len(results)) if results else None,
                        "finishability": (len(results) / planned) if planned else None,
                        "input_tokens": input_tokens if args.arm == "jev" else None,
                        "usd": round(usd, 6) if args.arm == "jev" else None,
                        "balance_usd_last": balance_usd,
                        "median_seconds_per_case": (
                            float(statistics.median(latencies)) if latencies else None
                        ),
                    },
                    "updated_at": utc_now(),
                }
            )
            args.status.write_text(json.dumps(prior, indent=2) + "\n", encoding="utf-8")
        if stopped and ("usd_stop" in stopped or "extract" in stopped):
            break

    n = len(results)
    briars = [r["brier"] for r in results if r["brier"] is not None]
    summary = {
        "arm": args.arm,
        "model": model_name,
        "task": "typed_decisions",
        "n_scored": n,
        "n_planned": planned,
        "finishability": (n / planned) if planned else None,
        "index_offset": 0,
        "n_cases": cases_scored,
        "n_cases_planned": len(cases),
        "accuracy": (correct / n) if n else None,
        "brier": (sum(briars) / len(briars)) if briars else None,
        "median_seconds_per_case": float(statistics.median(latencies)) if latencies else None,
        "by_type_accuracy": {
            k: (sum(v) / len(v) if v else None) for k, v in by_type_correct.items()
        },
        "input_tokens": input_tokens if args.arm == "jev" else None,
        "output_tokens": output_tokens if args.arm == "jev" else None,
        "usd": usd if args.arm == "jev" else None,
        "usd_spent_est": usd if args.arm == "jev" else None,  # alias kept for older readers
        "stopped": stopped,
        "balance_usd_last": balance_usd,
        "parquet_sha256": sha256_file(parquet_path),
        "lockfile_bakeoff_id": lock.get("bakeoff_id"),
        "lockfile_fair_id": lock.get("fair_bakeoff_id"),
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "finished_at": utc_now(),
        **fly_meta,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.arm}_typed_decisions_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / f"{args.arm}_typed_decisions_rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + ("\n" if results else ""),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not stopped else 1


if __name__ == "__main__":
    raise SystemExit(main())
