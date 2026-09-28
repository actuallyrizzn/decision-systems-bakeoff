#!/usr/bin/env python3
"""Pull ngram fair-STATUS.json and refresh the Tasks fair-results doc."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

DOC_ID = int(os.environ.get("FAIR_BAKEOFF_DOC_ID", "0"))
PASS = Path.home() / ".ssh" / "tasks-dsc-ottovernal.pass"
POLL = float(os.environ.get("FAIR_STATUS_POLL", "120"))
STATUS_REMOTE = os.environ.get(
    "FAIR_STATUS_REMOTE", "~/julia-bakeoff/fair-STATUS.json"
)


def load_env() -> dict[str, str]:
    env = dict(os.environ)
    text = PASS.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def ssh_cat_status() -> dict:
    cmd = [
        "sshpass",
        "-f",
        str(Path.home() / ".ssh" / "athena-moya.pass"),
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "moya",
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "rizzn@192.168.1.225",
        f"cat {STATUS_REMOTE} 2>/dev/null || echo '{{}}'",
    ]
    out = subprocess.check_output(cmd, text=True, timeout=60)
    return json.loads(out or "{}")


def render_body(st: dict) -> str:
    completed = [c for c in (st.get("completed") or []) if isinstance(c, dict)]
    prog = st.get("progress") or {}
    queue = st.get("tasks_planned_all_arms") or ["julia", "jev", "laya", "flybrain"]
    rows = [
        "# Fair-turf bakeoff — live results",
        "",
        "**Suite:** `julia-fair-turf-2026-09` · AG News · Emotion · MASSIVE en scenario · typed-decisions",
        f"**Successive queue:** {' → '.join(queue)}",
        "",
        "Original-suite Julia cold (weak): [Doc #1423](https://tasks.decisionsciencecorp.com/admin/doc.php?id=1423).",
        "",
        "## Status (machine-updated)",
        "",
        "```",
        f"phase: {st.get('phase')}",
        f"arm: {st.get('arm')}",
        f"current_task: {st.get('current_task')}",
        f"updated_at: {st.get('updated_at')}",
        f"queue: {' → '.join(queue)}",
        f"progress: {json.dumps(prog, sort_keys=True)}",
        "```",
        "",
        "Ask Otto anytime for a status check. Source: `ngram:~/julia-bakeoff/fair-STATUS.json`.",
        "",
        "## Results table",
        "",
        "| Test | Arm | Rows | Accuracy | Brier | Median s | Notes |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    order = ("agnews", "emotion", "massive_scenario_en", "typed_decisions")
    # One row per completed (task, arm) — do not collapse arms.
    seen = set()
    for s in completed:
        task = s.get("task")
        arm = s.get("arm", "?")
        key = (task, arm)
        if key in seen:
            continue
        seen.add(key)
        if task == "typed_decisions":
            note = json.dumps(s.get("by_type_accuracy") or {})
            med = s.get("median_seconds_per_case")
            rows.append(
                f"| {task} | {arm} | {s.get('n_scored')}/{s.get('n_planned')} | "
                f"{(s.get('accuracy') or 0):.4f} | {(s.get('brier') or 0):.4f} | {med} | {note} |"
            )
        else:
            note = s.get("stopped") or ""
            acc = s.get("accuracy")
            br = s.get("brier")
            med = s.get("median_seconds")
            rows.append(
                f"| {task} | {arm} | {s.get('n_scored')}/{s.get('n_planned')} | "
                f"{acc:.4f} | {br:.4f} | {med:.4f} | {note} |"
                if acc is not None and br is not None and med is not None
                else f"| {task} | {arm} | {s.get('n_scored')}/{s.get('n_planned')} | — | — | — | {note} |"
            )
    cur = st.get("current_task")
    if cur and (cur, st.get("arm")) not in seen:
        rows.append(
            f"| {cur} | {st.get('arm')} | in progress… | — | — | — | "
            f"{prog.get('done', '?')}/{prog.get('planned', '?')} |"
        )
    for task in order:
        if not any(t == task for t, _a in seen) and cur != task:
            rows.append(f"| {task} | — | queued | — | — | — | |")
    rows += [
        "",
        "## Narrative scratch (refine later)",
        "",
        "_Include original-suite Julia miss (Doc #1423) then this fair rematch vs Jev/Laya/Flybrain._",
        "",
        "## Artifacts",
        "",
        "- `ngram:~/julia-bakeoff/fair-STATUS.json`",
        "- `ngram:~/julia-bakeoff/fair-runs/`",
        "- `ngram:~/julia-bakeoff/fair-successive.log`",
        "- Protocol: `docs/FAIR_BAKEOFF.md`",
        "",
    ]
    return "\n".join(rows)


def update_doc(env: dict, body: str) -> None:
    import urllib.request

    base = env["TASKS_DSC_BASE_URL"].rstrip("/")
    key = env["TASKS_DSC_OTTOVERNAL_API_KEY"]
    payload = json.dumps({"id": DOC_ID, "body": body}).encode()
    req = urllib.request.Request(
        f"{base}/api/update-document.php",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "X-API-Key": key},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        resp.read()


def main() -> int:
    if DOC_ID <= 0:
        print("Set FAIR_BAKEOFF_DOC_ID", file=sys.stderr)
        return 2
    env = load_env()
    once = "--once" in sys.argv
    while True:
        try:
            st = ssh_cat_status()
            if st:
                update_doc(env, render_body(st))
                print(
                    f"synced phase={st.get('phase')} task={st.get('current_task')}",
                    flush=True,
                )
                if st.get("phase") in ("done", "error"):
                    return 0 if st.get("phase") == "done" else 1
        except Exception as exc:  # noqa: BLE001
            print(f"sync warn: {exc}", flush=True)
        if once:
            return 0
        time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())
