#!/usr/bin/env python3
"""Freeze the fair-turf bakeoff tasks (AG News, Emotion, MASSIVE en scenario, typed-decisions).

Writes TSV + labels under --out (default: data/fair/) and prints sha256 digests for lockfile.
Does not mutate lockfile.json — paste digests after a clean freeze.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

AGNEWS_LABELS = ["world", "sports", "business", "sci_tech"]
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
TYPED_REV = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
TYPED_SHA = "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c"
TYPED_REMOTE = "all/test-00000-of-00001.parquet"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tsv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["id\ttext\tlabel"]
    for rid, text, label in rows:
        text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
        lines.append(f"{rid}\t{text}\t{label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def freeze_agnews(out: Path) -> dict:
    train_p = hf_hub_download(
        "fancyzhx/ag_news", "data/train-00000-of-00001.parquet", repo_type="dataset"
    )
    test_p = hf_hub_download(
        "fancyzhx/ag_news", "data/test-00000-of-00001.parquet", repo_type="dataset"
    )
    train = pq.read_table(train_p).to_pylist()
    test = pq.read_table(test_p).to_pylist()
    # Deterministic valid carve: last 5_000 of train (sorted by original order).
    valid = train[-5000:]
    train_fit = train[:-5000]
    task = out / "agnews"
    task.mkdir(parents=True, exist_ok=True)
    (task / "labels.json").write_text(
        json.dumps(AGNEWS_LABELS, indent=2) + "\n", encoding="utf-8"
    )

    def pack(rows: list[dict], prefix: str) -> list[tuple[str, str, str]]:
        packed = []
        for i, r in enumerate(rows):
            packed.append((f"{prefix}{i}", r["text"], str(int(r["label"]))))
        return packed

    write_tsv(task / "train.tsv", pack(train_fit, "tr"))
    write_tsv(task / "valid.tsv", pack(valid, "va"))
    write_tsv(task / "test.tsv", pack(test, "te"))
    return {
        "task": "agnews",
        "rows_test": len(test),
        "test_sha256": sha256_file(task / "test.tsv"),
        "rows_train": len(train_fit),
        "rows_valid": len(valid),
    }


def freeze_emotion(out: Path) -> dict:
    train_p = hf_hub_download(
        "dair-ai/emotion", "split/train-00000-of-00001.parquet", repo_type="dataset"
    )
    valid_p = hf_hub_download(
        "dair-ai/emotion", "split/validation-00000-of-00001.parquet", repo_type="dataset"
    )
    test_p = hf_hub_download(
        "dair-ai/emotion", "split/test-00000-of-00001.parquet", repo_type="dataset"
    )
    train = pq.read_table(train_p).to_pylist()
    valid = pq.read_table(valid_p).to_pylist()
    test = pq.read_table(test_p).to_pylist()
    task = out / "emotion"
    task.mkdir(parents=True, exist_ok=True)
    (task / "labels.json").write_text(
        json.dumps(EMOTION_LABELS, indent=2) + "\n", encoding="utf-8"
    )

    def pack(rows: list[dict], prefix: str) -> list[tuple[str, str, str]]:
        return [
            (f"{prefix}{i}", r["text"], str(int(r["label"]))) for i, r in enumerate(rows)
        ]

    write_tsv(task / "train.tsv", pack(train, "tr"))
    write_tsv(task / "valid.tsv", pack(valid, "va"))
    write_tsv(task / "test.tsv", pack(test, "te"))
    return {
        "task": "emotion",
        "rows_test": len(test),
        "test_sha256": sha256_file(task / "test.tsv"),
        "rows_train": len(train),
        "rows_valid": len(valid),
    }


def _massive_jsonl(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def freeze_massive_en(out: Path) -> dict:
    train_p = Path(
        hf_hub_download(
            "mteb/amazon_massive_scenario", "train/en.json.gz", repo_type="dataset"
        )
    )
    valid_p = Path(
        hf_hub_download(
            "mteb/amazon_massive_scenario", "validation/en.json.gz", repo_type="dataset"
        )
    )
    test_p = Path(
        hf_hub_download(
            "mteb/amazon_massive_scenario", "test/en.json.gz", repo_type="dataset"
        )
    )
    train = _massive_jsonl(train_p)
    valid = _massive_jsonl(valid_p)
    test = _massive_jsonl(test_p)
    labels = sorted({r["label"] for r in train + valid + test})
    assert len(labels) <= 20, labels
    label_to_idx = {name: i for i, name in enumerate(labels)}
    task = out / "massive_scenario_en"
    task.mkdir(parents=True, exist_ok=True)
    (task / "labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")

    def pack(rows: list[dict], prefix: str) -> list[tuple[str, str, str]]:
        packed = []
        for i, r in enumerate(rows):
            rid = str(r.get("id", f"{prefix}{i}"))
            packed.append((f"{prefix}{rid}", r["text"], str(label_to_idx[r["label"]])))
        return packed

    write_tsv(task / "train.tsv", pack(train, "tr"))
    write_tsv(task / "valid.tsv", pack(valid, "va"))
    write_tsv(task / "test.tsv", pack(test, "te"))
    return {
        "task": "massive_scenario_en",
        "rows_test": len(test),
        "test_sha256": sha256_file(task / "test.tsv"),
        "rows_train": len(train),
        "rows_valid": len(valid),
        "n_labels": len(labels),
        "labels": labels,
    }


def freeze_typed(out: Path) -> dict:
    remote = hf_hub_download(
        "LocalLLaMA/typed-decisions",
        TYPED_REMOTE,
        repo_type="dataset",
        revision=TYPED_REV,
    )
    task = out / "typed_decisions"
    task.mkdir(parents=True, exist_ok=True)
    dest = task / "test.parquet"
    shutil.copy2(remote, dest)
    got = sha256_file(dest)
    if got != TYPED_SHA:
        raise SystemExit(f"typed parquet sha mismatch: {got} != {TYPED_SHA}")
    n = pq.read_table(dest).num_rows
    meta = {
        "dataset": "LocalLLaMA/typed-decisions",
        "revision": TYPED_REV,
        "remote_path": TYPED_REMOTE,
        "rows_cases": n,
        "test_sha256": got,
        "note": "400 cases / 2000 questions (choice+score+noul); not a TSV task",
    }
    (task / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"task": "typed_decisions", "rows_test": n, "test_sha256": got}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "fair",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reports = [
        freeze_agnews(args.out),
        freeze_emotion(args.out),
        freeze_massive_en(args.out),
        freeze_typed(args.out),
    ]
    summary_path = args.out / "FREEZE_REPORT.json"
    summary_path.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reports, indent=2))
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
