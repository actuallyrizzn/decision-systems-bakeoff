"""Load lockfile, frozen TSVs, and decision questions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKFILE_PATH = REPO_ROOT / "lockfile.json"
QUESTIONS_DIR = REPO_ROOT / "questions"

SST2_GOLD = {"0": "negative", "1": "positive"}
BUGSEV_GOLD = {"0": "low", "1": "normal", "2": "high"}


def load_lockfile(path: Path | None = None) -> dict[str, Any]:
    p = path or LOCKFILE_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tsv(path: Path) -> list[dict[str, str]]:
    """Tab-split loader (not csv) — CLINC text can contain quotes."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "id\ttext\tlabel":
        raise ValueError(f"bad header in {path}")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if not line:
            continue
        row_id, text, label = line.split("\t")
        rows.append({"id": row_id, "text": text, "label": label})
    return rows


def load_labels(task_dir: Path) -> list[str] | None:
    p = task_dir / "labels.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def verify_task_test(task: str, data_root: Path, lock: dict[str, Any] | None = None) -> str:
    lock = lock or load_lockfile()
    meta = lock["tasks"][task]
    test_path = data_root / task / "test.tsv"
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
    got = sha256_file(test_path)
    want = meta["test_sha256"]
    if got != want:
        raise ValueError(f"REFUSE {test_path}: sha256 {got} != lockfile {want}")
    rows = load_tsv(test_path)
    if len(rows) != int(meta["rows_test"]):
        raise ValueError(
            f"REFUSE {test_path}: rows {len(rows)} != lockfile {meta['rows_test']}"
        )
    return got


FAIR_INDEXED = (
    "agnews",
    "emotion",
    "massive_scenario_en",
    "clinc10",
    "clinc150",
    "banking77",
)


def gold_name(task: str, label: str, labels: list[str] | None) -> str:
    if task == "sst2":
        return SST2_GOLD[label]
    if task == "bugsev":
        return BUGSEV_GOLD[label]
    if task in FAIR_INDEXED:
        assert labels is not None
        return labels[int(label)]
    raise ValueError(f"unknown task {task}")


def class_names(task: str, labels: list[str] | None) -> list[str]:
    if task == "sst2":
        return ["negative", "positive"]
    if task == "bugsev":
        return ["low", "normal", "high"]
    assert labels is not None
    return list(labels)


def questions_for(task: str, labels: list[str] | None) -> tuple[dict[str, Any], str]:
    if task in ("clinc150", "banking77"):
        if not labels:
            raise ValueError(f"{task} needs labels.json")
        criteria = {name: name for name in labels}
        qid = "intent"
        instructions = (
            "Which banking intent does this customer message express?"
            if task == "banking77"
            else "Which of these intents does the utterance express?"
        )
        return {
            qid: {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        }, qid
    if task == "typed_decisions":
        raise ValueError("typed_decisions uses per-case questions; see run_typed_decisions.py")
    path = QUESTIONS_DIR / f"{task}.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    qid = next(iter(obj))
    # Prefer frozen labels.json order when present (massive / agnews / emotion).
    if labels and qid in obj and isinstance(obj[qid].get("criteria"), dict):
        crit = obj[qid]["criteria"]
        missing = [n for n in labels if n not in crit]
        if missing:
            raise ValueError(f"{task} questions missing criteria for {missing}")
    return obj, qid


def verify_typed_parquet(data_root: Path, lock: dict[str, Any] | None = None) -> str:
    lock = lock or load_lockfile()
    meta = lock["tasks"]["typed_decisions"]
    path = data_root / "typed_decisions" / "test.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    got = sha256_file(path)
    want = meta["test_sha256"]
    if got != want:
        raise ValueError(f"REFUSE {path}: sha256 {got} != lockfile {want}")
    return got
