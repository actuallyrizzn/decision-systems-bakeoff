"""Typed question builder — score criteria must remain a list."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# import from run_typed_decisions without executing main
import importlib.util

spec = importlib.util.spec_from_file_location(
    "run_typed_decisions", ROOT / "scripts" / "run_typed_decisions.py"
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_score_criteria_stays_list():
    q, keys = mod.build_typed_question(
        "score",
        "How risky?",
        ["Benign", "Low", "Moderate", "High"],
    )
    assert q["type"] == "score"
    assert isinstance(q["criteria"], list)
    assert q["criteria"] == ["Benign", "Low", "Moderate", "High"]
    assert keys == ["0", "1", "2", "3"]


def test_choice_criteria_dict():
    q, keys = mod.build_typed_question(
        "choice", "Pick", {"a": "A desc", "b": "B desc"}
    )
    assert isinstance(q["criteria"], dict)
    assert keys == ["a", "b"]


def test_noul_criteria_dict():
    q, keys = mod.build_typed_question("noul", "Yes?", None)
    assert q["criteria"] == {"false": "false", "true": "true"}
    assert keys == ["false", "true"]
