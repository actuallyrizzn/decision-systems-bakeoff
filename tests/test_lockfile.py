from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_bakeoff.data import load_lockfile, questions_for


def test_lockfile_tasks():
    lock = load_lockfile()
    assert lock["bakeoff_id"] == "flybrain-jev-laya-2026-09"
    assert lock["fair_bakeoff_id"] == "julia-fair-turf-2026-09"
    assert set(lock["suites"]["original"]["tasks"]) == {
        "sst2",
        "clinc10",
        "clinc150",
        "bugsev",
    }
    assert set(lock["suites"]["fair"]["tasks"]) == {
        "agnews",
        "emotion",
        "massive_scenario_en",
        "typed_decisions",
    }
    for t in lock["suites"]["original"]["tasks"]:
        assert t in lock["tasks"]
    for t in lock["suites"]["fair"]["tasks"]:
        assert t in lock["tasks"]


def test_questions_sst2():
    q, qid = questions_for("sst2", None)
    assert qid == "sentiment"
    assert set(q["sentiment"]["criteria"]) == {"negative", "positive"}


def test_questions_fair():
    q, qid = questions_for("agnews", ["world", "sports", "business", "sci_tech"])
    assert qid == "topic"
    assert len(q["topic"]["criteria"]) == 4
    q, qid = questions_for(
        "emotion", ["sadness", "joy", "love", "anger", "fear", "surprise"]
    )
    assert qid == "emotion"
    assert len(q["emotion"]["criteria"]) == 6
