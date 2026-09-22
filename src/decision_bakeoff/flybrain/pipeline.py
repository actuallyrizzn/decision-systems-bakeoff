"""End-to-end Flybrain decision arm: frozen graph + ridge head."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from decision_bakeoff.flybrain.features import encode_packets, pooled_states
from decision_bakeoff.flybrain.readout import fit_ridge_classes, predict_proba
from decision_bakeoff.flybrain.reservoir import build_reservoir_from_arm
from decision_bakeoff.flybrain.tokenizer import Tokenizer
from decision_bakeoff.flybrain.vectors import build_embed, load_glove

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CFG = REPO_ROOT / "configs" / "flybrain_best.json"


def load_fly_cfg(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or DEFAULT_CFG).read_text(encoding="utf-8"))


def _texts_labels(tsv: Path) -> tuple[list[str], np.ndarray, list[str]]:
    lines = tsv.read_text(encoding="utf-8").splitlines()
    texts: list[str] = []
    labels: list[int] = []
    ids: list[str] = []
    for line in lines[1:]:
        if not line:
            continue
        row_id, text, label = line.split("\t")
        ids.append(row_id)
        texts.append(text)
        labels.append(int(label))
    return texts, np.asarray(labels, dtype=np.int64), ids


def run_flybrain_task(
    *,
    task: str,
    data_root: Path,
    glove_path: Path,
    out_dir: Path,
    cfg_path: Path | None = None,
    limit: int | None = None,
    skip_verify: bool = False,
) -> dict[str, Any]:
    """Fit ridge head on train/val; score test; write bakeoff JSONL + summary."""
    from decision_bakeoff.data import class_names, gold_name, load_labels, verify_task_test
    from decision_bakeoff.score import brier

    if not skip_verify:
        verify_task_test(task, data_root)
    fly_lock = load_fly_cfg(cfg_path)
    arm = fly_lock["tasks"][task]
    cap = int(fly_lock["packet_cap"])
    task_dir = data_root / task

    train_texts, y_train, _ = _texts_labels(task_dir / "train.tsv")
    valid_texts, y_valid, _ = _texts_labels(task_dir / "valid.tsv")
    test_texts, y_test, test_ids = _texts_labels(task_dir / "test.tsv")
    if limit is not None:
        test_texts = test_texts[:limit]
        y_test = y_test[:limit]
        test_ids = test_ids[:limit]

    tok = Tokenizer.build(train_texts)
    glove = load_glove(glove_path, set(tok.token_to_id))
    embed, coverage = build_embed(
        tok, glove, int(arm["inject_count"]), int(arm["seed"])
    )

    res = build_reservoir_from_arm(
        leak=float(arm["leak"]),
        steps=int(arm["steps"]),
        radius=float(arm["radius"]),
        inject_count=int(arm["inject_count"]),
        input_scale=float(arm["input_scale"]),
        seed=int(arm["seed"]),
        vocab_size=tok.size,
        embed=embed,
    )
    pooling = str(arm["pooling"])

    train_seqs, _ = encode_packets(tok, train_texts, cap=cap)
    valid_seqs, _ = encode_packets(tok, valid_texts, cap=cap)
    test_seqs, trunc = encode_packets(tok, test_texts, cap=cap)

    x_train = pooled_states(res, train_seqs, pooling=pooling)
    x_valid = pooled_states(res, valid_seqs, pooling=pooling)

    n_classes = int(y_train.max()) + 1
    head, fit_info = fit_ridge_classes(
        x_train, y_train, n_classes, x_valid=x_valid, y_valid=y_valid
    )

    # Per-row latency on test (rebuild features one-by-one is expensive;
    # measure batched then divide — also time each packet for median).
    latencies: list[float] = []
    feats: list[np.ndarray] = []
    for seq in test_seqs:
        t0 = time.perf_counter()
        feat = pooled_states(res, [seq], pooling=pooling)
        latencies.append(time.perf_counter() - t0)
        feats.append(feat[0])
    x_test = np.stack(feats).astype(np.float32)
    proba = predict_proba(head, x_test)

    labels_json = load_labels(task_dir)
    names = class_names(task, labels_json)
    # class index -> name
    # gold_name expects string label index from tsv
    rows_out = []
    correct = 0
    briars = []
    for i, (row_id, y, secs) in enumerate(zip(test_ids, y_test, latencies)):
        gold = gold_name(task, str(int(y)), labels_json)
        pred_i = int(proba[i].argmax())
        choice = names[pred_i]
        probs = {names[j]: float(proba[i, j]) for j in range(len(names))}
        hit = choice == gold
        if hit:
            correct += 1
        br = brier(probs, gold, names)
        briars.append(br)
        rows_out.append(
            {
                "i": i,
                "id": row_id,
                "gold": gold,
                "choice": choice,
                "hit": hit,
                "brier": br,
                "seconds": secs,
                "probabilities": probs,
            }
        )

    import statistics

    n = len(rows_out)
    summary = {
        "arm": "flybrain",
        "task": task,
        "n_scored": n,
        "n_planned": n,
        "accuracy": correct / n if n else None,
        "brier": sum(briars) / n if n else None,
        "median_seconds": float(statistics.median(latencies)) if latencies else None,
        "cfg_key": arm["cfg_key"],
        "glove_coverage": coverage,
        "tokenizer_fingerprint": tok.fingerprint,
        "fit": fit_info,
        "packet_cap": cap,
        "test_truncated_fraction": trunc,
        "stopped": None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"flybrain_{task}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / f"flybrain_{task}_rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows_out) + ("\n" if rows_out else ""),
        encoding="utf-8",
    )
    return summary
