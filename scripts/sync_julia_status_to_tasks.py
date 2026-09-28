#!/usr/bin/env python3
"""Pull ngram ~/julia-bakeoff/STATUS.json and refresh the Tasks live-results doc."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

DOC_ID = int(os.environ.get("JULIA_BAKEOFF_DOC_ID", "0"))
PASS = Path.home() / ".ssh" / "tasks-dsc-ottovernal.pass"
POLL = float(os.environ.get("JULIA_STATUS_POLL", "120"))


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
        "cat ~/julia-bakeoff/STATUS.json 2>/dev/null || echo '{}'",
    ]
    out = subprocess.check_output(cmd, text=True, timeout=60)
    return json.loads(out or "{}")


def render_body(st: dict) -> str:
    completed = st.get("completed") or []
    prog = st.get("progress") or {}
    rows = [
        "# Julia-1 bakeoff arm — live results",
        "",
        "**Arm:** Supersonic Labs Julia-1 (cold, CPU on ngram) · **Bakeoff lock:** flybrain-jev-laya-2026-09",
        "",
        "## Status (machine-updated)",
        "",
        "```",
        f"phase: {st.get('phase')}",
        f"current_task: {st.get('current_task')}",
        f"updated_at: {st.get('updated_at')}",
        f"progress: {json.dumps(prog, sort_keys=True)}",
        "```",
        "",
        "Ask Otto anytime for a status check. Source of truth: `ngram:~/julia-bakeoff/STATUS.json`.",
        "",
        "## Scope",
        "",
        "| Test | Run? | Why |",
        "|---|---|---|",
        "| sst2 (872) | yes | 2 answers |",
        "| clinc10 (300) | yes | 10 answers |",
        "| bugsev (2000) | yes | 3 answers |",
        "| clinc150 (5500) | **no** | 151 answers; Julia max ~20 per call |",
        "",
        "Training: **none** (cold).",
        "",
        "## Results table",
        "",
        "| Test | Rows | Accuracy | Brier | Median s | Wall s | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    by = {c["task"]: c for c in completed if isinstance(c, dict)}
    for task in ("sst2", "clinc10", "bugsev"):
        if task in by:
            s = by[task]
            note = s.get("stopped") or ""
            rows.append(
                f"| {task} | {s.get('n_scored')}/{s.get('n_planned')} | "
                f"{s.get('accuracy'):.4f} | {s.get('brier'):.4f} | "
                f"{s.get('median_seconds'):.4f} | {s.get('wall_seconds')} | {note} |"
            )
        elif st.get("current_task") == task:
            rows.append(
                f"| {task} | in progress… | — | — | — | — | {prog.get('done','?')}/{prog.get('planned','?')} |"
            )
        else:
            rows.append(f"| {task} | queued | — | — | — | — | |")
    rows += [
        "",
        "## Narrative scratch (refine later)",
        "",
        "_Fill after the three benches finish. Compare to published Flybrain / Jev / Laya board; do not overclaim._",
        "",
        "## Artifacts",
        "",
        "- `ngram:~/julia-bakeoff/STATUS.json`",
        "- `ngram:~/julia-bakeoff/runs/`",
        "- Model: `SupersonicLabs/Julia-1` (Apache 2.0)",
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
    global DOC_ID
    if DOC_ID <= 0:
        print("Set JULIA_BAKEOFF_DOC_ID", file=sys.stderr)
        return 2
    env = load_env()
    once = "--once" in sys.argv
    while True:
        try:
            st = ssh_cat_status()
            if st:
                body = render_body(st)
                update_doc(env, body)
                print(f"synced phase={st.get('phase')} task={st.get('current_task')}", flush=True)
                if st.get("phase") == "done":
                    return 0
                if st.get("phase") == "error":
                    return 1
        except Exception as exc:  # noqa: BLE001
            print(f"sync warn: {exc}", flush=True)
        if once:
            return 0
        time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())
