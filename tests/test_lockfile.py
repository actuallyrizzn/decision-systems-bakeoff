from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_bakeoff.data import load_lockfile, questions_for


def test_lockfile_tasks():
    lock = load_lockfile()
    assert lock["bakeoff_id"] == "flybrain-jev-laya-2026-09"
    assert set(lock["tasks"]) == {"sst2", "clinc10", "clinc150", "bugsev"}


def test_questions_sst2():
    q, qid = questions_for("sst2", None)
    assert qid == "sentiment"
    assert set(q["sentiment"]["criteria"]) == {"negative", "positive"}
