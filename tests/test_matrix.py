"""Matrix store: idempotent ingest + board shape."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from decision_bakeoff.matrix import board, board_table, ingest_summary, open_db  # noqa: E402


def _summary(**kw):
    base = {
        "arm": "julia",
        "model": "SupersonicLabs/Julia-1",
        "task": "emotion",
        "qid": "affect6",
        "accuracy": 0.799,
        "brier": 0.31,
        "median_seconds": 0.42,
        "n_scored": 2000,
        "n_planned": 2000,
        "finishability": 1.0,
        "usd": None,
        "stopped": None,
        "wall_seconds": 900.0,
        "finished_at": "2026-09-28T00:00:00Z",
    }
    base.update(kw)
    return base


def test_ingest_idempotent(tmp_path):
    db = tmp_path / "matrix.db"
    f = tmp_path / "julia_emotion_summary.json"
    f.write_text(json.dumps(_summary()))

    conn = open_db(db)
    ingest_summary(conn, f, run_id="run-a", suite="fair", host="ngram")
    ingest_summary(conn, f, run_id="run-a", suite="fair", host="ngram")  # re-ingest

    rows = conn.execute("SELECT COUNT(*) AS n FROM cells").fetchone()
    assert rows["n"] == 1, "re-ingest must update in place, never duplicate"

    # updated value lands on the same row
    f.write_text(json.dumps(_summary(accuracy=0.801)))
    ingest_summary(conn, f, run_id="run-a", suite="fair", host="ngram")
    cell = conn.execute("SELECT accuracy FROM cells WHERE run_id='run-a'").fetchone()
    assert abs(cell["accuracy"] - 0.801) < 1e-9


def test_board_latest_wins(tmp_path):
    db = tmp_path / "matrix.db"
    f = tmp_path / "s.json"
    conn = open_db(db)
    f.write_text(json.dumps(_summary(accuracy=0.5, finished_at="2026-09-27T00:00:00Z")))
    ingest_summary(conn, f, run_id="run-old", suite="fair")
    f.write_text(json.dumps(_summary(accuracy=0.799, finished_at="2026-09-28T00:00:00Z")))
    ingest_summary(conn, f, run_id="run-new", suite="fair")

    cells = board(conn, suite="fair")
    assert len(cells) == 1
    assert abs(cells[0]["accuracy"] - 0.799) < 1e-9
    assert cells[0]["run_id"] == "run-new"


def test_board_table_renders(tmp_path):
    db = tmp_path / "matrix.db"
    f = tmp_path / "s.json"
    conn = open_db(db)
    f.write_text(json.dumps(_summary()))
    ingest_summary(conn, f, run_id="run-a", suite="fair")
    f.write_text(json.dumps(_summary(arm="jev", accuracy=0.59)))
    ingest_summary(conn, f, run_id="run-a", suite="fair")
    table = board_table(conn, suite="fair")
    assert "| Task | jev | julia |" in table
    assert "| emotion | 0.590 | 0.799 |" in table


def test_fidelity_filter(tmp_path):
    db = tmp_path / "matrix.db"
    f = tmp_path / "s.json"
    conn = open_db(db)
    f.write_text(json.dumps(_summary(accuracy=0.94, n_scored=100, n_planned=100)))
    ingest_summary(conn, f, run_id="pilot", suite="fair", fidelity="pilot-100")
    f.write_text(json.dumps(_summary(accuracy=0.799)))
    ingest_summary(conn, f, run_id="full", suite="fair", fidelity="full")

    full = board(conn, suite="fair", fidelity="full")
    assert len(full) == 1 and full[0]["run_id"] == "full"
