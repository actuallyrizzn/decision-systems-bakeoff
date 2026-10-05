#!/usr/bin/env python3
"""Freeze Banking77 (full 77 intents) for the locked sibling suite.

Source: Hugging Face ``mteb/banking77`` parquet (same PolyAI labels/splits:
train 10003 / test 3080). We carve a deterministic valid set as the last 1003
train rows (original parquet order) so Flybrain ridge has a val split.
The test TSV is the official test — never resampled.

Mark lock 2026-10-05: full 77-way, no option shortlist (Tasks Doc #1466 / #4846).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_commits
import pyarrow.parquet as pq

BANKING77_REPO = "mteb/banking77"
VALID_TAIL = 1003

# Filled after first freeze against a pinned commit; keep in lockfile too.
DEFAULT_REV = "18072d2685ea682290f7b8924d94c62acc19c0b2"


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


def _label_names_from_table(table) -> list[str] | None:
    pandas_meta = table.schema.pandas_metadata or {}
    for col in pandas_meta.get("columns", []):
        if col.get("name") != "label":
            continue
        extra = col.get("metadata") or {}
        hf = extra.get("huggingface") or extra
        feat = hf.get("info") or hf.get("feature") or hf
        if isinstance(feat, dict):
            if "names" in feat:
                return list(feat["names"])
            inner = feat.get("feature") or {}
            if isinstance(inner, dict) and "names" in inner:
                return list(inner["names"])
    return None


def freeze_banking77(out: Path, revision: str | None = None) -> dict:
    rev = revision or DEFAULT_REV
    kwargs = {"repo_type": "dataset"}
    if rev:
        kwargs["revision"] = rev
    train_p = hf_hub_download(
        BANKING77_REPO, "data/train-00000-of-00001.parquet", **kwargs
    )
    test_p = hf_hub_download(
        BANKING77_REPO, "data/test-00000-of-00001.parquet", **kwargs
    )
    train_table = pq.read_table(train_p)
    test_table = pq.read_table(test_p)
    train = train_table.to_pylist()
    test = test_table.to_pylist()

    names = _label_names_from_table(train_table)
    if not names:
        # mteb parquet: int `label` + string `label_text`
        by_idx: dict[int, str] = {}
        for r in train + test:
            if "label_text" in r and r["label_text"] is not None:
                by_idx[int(r["label"])] = str(r["label_text"])
        if by_idx:
            names = [by_idx[i] for i in range(max(by_idx) + 1)]
        else:
            meta = train_table.schema.metadata or {}
            raw = meta.get(b"huggingface")
            if raw:
                info = json.loads(raw.decode("utf-8"))
                feat = (info.get("info") or {}).get("features") or {}
                lab = feat.get("label") or {}
                names = list(lab.get("names") or [])
    if not names or len(names) != 77:
        raise SystemExit(f"expected 77 Banking77 labels, got {names!r}")

    name_to_idx = {n: i for i, n in enumerate(names)}
    if VALID_TAIL >= len(train):
        raise SystemExit("valid tail too large for train")
    valid = train[-VALID_TAIL:]
    train_fit = train[:-VALID_TAIL]

    task = out / "banking77"
    task.mkdir(parents=True, exist_ok=True)
    (task / "labels.json").write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")

    def pack(rows: list[dict], prefix: str) -> list[tuple[str, str, str]]:
        packed = []
        for i, r in enumerate(rows):
            lab = r["label"]
            if isinstance(lab, str):
                idx = name_to_idx[lab]
            else:
                idx = int(lab)
            packed.append((f"{prefix}{i}", r["text"], str(idx)))
        return packed

    write_tsv(task / "train.tsv", pack(train_fit, "tr"))
    write_tsv(task / "valid.tsv", pack(valid, "va"))
    write_tsv(task / "test.tsv", pack(test, "te"))

    # Record the Hub commit we actually pulled when revision was omitted.
    if not rev:
        commits = list_repo_commits(BANKING77_REPO, repo_type="dataset")
        rev = commits[0].commit_id if commits else "unknown"

    meta = {
        "task": "banking77",
        "dataset": BANKING77_REPO,
        "revision": rev,
        "n_labels": 77,
        "rows_train": len(train_fit),
        "rows_valid": len(valid),
        "rows_test": len(test),
        "test_sha256": sha256_file(task / "test.tsv"),
        "valid_carve": f"last {VALID_TAIL} of official train, original parquet order",
        "lock": "full 77-way; no shortlist (Doc #1466)",
    }
    (task / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "fair",
    )
    ap.add_argument("--revision", default=None, help="Hub commit to pin")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = freeze_banking77(args.out, revision=args.revision)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
