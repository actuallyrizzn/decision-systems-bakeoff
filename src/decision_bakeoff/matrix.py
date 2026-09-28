"""Decision-matrix store — the Benchlab's canonical results DB.

Every locked run lands here as cells: arm x task x KPIs. This store is what
Phase 2 (MoEjev public router) reads as its lookup table, so cells are
append-only and idempotent on (arm, task, run_id): re-ingesting the same run
updates the same row, never duplicates.

SQLite on purpose: zero-infra, diffable dumps, lives next to the repo.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS cells (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL,          -- which run produced this cell (dir name / queue job id)
    arm TEXT NOT NULL,             -- jev | laya | flybrain | julia | ...
    model TEXT,                    -- concrete checkpoint / endpoint model
    task TEXT NOT NULL,            -- sst2 | agnews | emotion | ...
    suite TEXT,                    -- original | fair | benchlab-<n>
    qid TEXT,                      -- question-family id from lockfile/registry
    accuracy REAL,
    brier REAL,
    median_seconds REAL,
    n_scored INTEGER,
    n_planned INTEGER,
    finishability REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    usd REAL,                      -- metered spend for this cell (hosted arms)
    stopped TEXT,                  -- cost-stop reason, if any
    fidelity TEXT NOT NULL DEFAULT 'full',  -- full | pilot-100 | smoke | ...
    host TEXT,                     -- ngram | newdev | moya | ...
    wall_seconds REAL,
    finished_at TEXT,              -- utc timestamp from the run summary
    ingested_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(arm, task, run_id)
);

CREATE TABLE IF NOT EXISTS ingest_log (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,          -- file/dir ingested
    cells INTEGER NOT NULL,        -- rows written (insert or update)
    ingested_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cells_arm_task ON cells(arm, task);
CREATE INDEX IF NOT EXISTS idx_cells_task ON cells(task);
"""

SUMMARY_FIELDS = (
    "arm", "model", "task", "qid", "accuracy", "brier", "median_seconds",
    "n_scored", "n_planned", "finishability", "input_tokens", "output_tokens",
    "usd", "stopped", "wall_seconds", "finished_at",
)


def open_db(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def cell_from_summary(
    summary: dict[str, Any],
    *,
    run_id: str,
    suite: str | None = None,
    fidelity: str = "full",
    host: str | None = None,
) -> dict[str, Any]:
    """Map a run summary (see scripts/run_fair_chain.py) onto a cell row."""
    missing = [k for k in ("arm", "task") if not summary.get(k)]
    if missing:
        raise ValueError(f"summary missing required keys: {missing}")
    cell = {k: summary.get(k) for k in SUMMARY_FIELDS}
    cell.update(
        run_id=run_id,
        suite=suite,
        fidelity=fidelity,
        host=host,
    )
    return cell


def upsert_cell(conn: sqlite3.Connection, cell: dict[str, Any]) -> None:
    cols = [
        "run_id", "arm", "model", "task", "suite", "qid", "accuracy", "brier",
        "median_seconds", "n_scored", "n_planned", "finishability",
        "input_tokens", "output_tokens", "usd", "stopped", "fidelity", "host",
        "wall_seconds", "finished_at",
    ]
    placeholders = ", ".join(f":{c}" for c in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in ("arm", "task", "run_id"))
    conn.execute(
        f"INSERT INTO cells ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(arm, task, run_id) DO UPDATE SET {updates}",
        cell,
    )


def ingest_summary(
    conn: sqlite3.Connection,
    summary_path: str | Path,
    *,
    run_id: str,
    suite: str | None = None,
    fidelity: str = "full",
    host: str | None = None,
) -> int:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    cell = cell_from_summary(summary, run_id=run_id, suite=suite, fidelity=fidelity, host=host)
    upsert_cell(conn, cell)
    conn.execute(
        "INSERT INTO ingest_log (source, cells) VALUES (?, 1)", (str(summary_path),)
    )
    conn.commit()
    return 1


def board(conn: sqlite3.Connection, suite: str | None = None, fidelity: str = "full") -> list[dict[str, Any]]:
    """Latest cell per (arm, task) — the public board shape."""
    where = ["fidelity = ?"]
    args: list[Any] = [fidelity]
    if suite:
        where.append("suite = ?")
        args.append(suite)
    rows = conn.execute(
        f"""
        SELECT c.* FROM cells c
        JOIN (
            SELECT arm, task, MAX(finished_at) AS latest
            FROM cells WHERE {' AND '.join(where)}
            GROUP BY arm, task
        ) latest ON latest.arm = c.arm AND latest.task = c.task
                AND latest.latest = c.finished_at
        WHERE {' AND '.join(f'c.{w}' for w in where)}
        ORDER BY c.task, c.arm
        """,
        args + args,
    ).fetchall()
    return [dict(r) for r in rows]


def board_table(conn: sqlite3.Connection, suite: str | None = None, kpi: str = "accuracy") -> str:
    """Markdown table: rows = task, cols = arm, values = chosen KPI."""
    cells = board(conn, suite=suite)
    tasks = sorted({c["task"] for c in cells})
    arms = sorted({c["arm"] for c in cells})
    by = {(c["task"], c["arm"]): c for c in cells}
    head = "| Task | " + " | ".join(arms) + " |"
    sep = "|---|" + "---:|" * len(arms)
    lines = [head, sep]
    for t in tasks:
        row = [t]
        for a in arms:
            v = by.get((t, a), {}).get(kpi)
            row.append(f"{v:.3f}" if isinstance(v, float) else ("—" if v is None else str(v)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
