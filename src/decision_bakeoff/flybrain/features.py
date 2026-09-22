"""Pooled reservoir states for decision packets."""

from __future__ import annotations

import numpy as np

from decision_bakeoff.flybrain.tokenizer import BOS, Tokenizer


def pooled_states(
    res,
    seqs: list[list[int]],
    *,
    pooling: str,
    batch_lines: int = 512,
) -> np.ndarray:
    if pooling not in {"last", "mean", "last+mean"}:
        raise ValueError(f"bad pooling {pooling}")
    if not seqs:
        width = res.n * (2 if pooling == "last+mean" else 1)
        return np.zeros((0, width), dtype=np.float32)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
    ordered = [seqs[i] for i in order]
    blocks: list[np.ndarray] = []
    for b0 in range(0, len(ordered), batch_lines):
        chunk = ordered[b0 : b0 + batch_lines]
        batch = len(chunk)
        length = max(len(seq) for seq in chunk)
        ids = np.zeros((batch, length), dtype=np.int64)
        lens = np.array([len(seq) for seq in chunk], dtype=np.int64)
        for i, seq in enumerate(chunk):
            ids[i, : len(seq)] = seq
        state = np.zeros((res.n, batch), dtype=np.float32)
        total = np.zeros((res.n, batch), dtype=np.float32)
        for t in range(length):
            active = t < lens
            if not active.any():
                break
            if bool(active.all()):
                state = res.step(state, res.drive(ids[:, t]))
                total += state
            else:
                idx = np.flatnonzero(active)
                nxt = res.step(state[:, idx], res.drive(ids[idx, t]))
                state[:, idx] = nxt
                total[:, idx] += nxt
        last = state.T
        mean = (total / np.maximum(lens, 1)).T
        if pooling == "last":
            blocks.append(last)
        elif pooling == "mean":
            blocks.append(mean)
        else:
            blocks.append(np.concatenate([last, mean], axis=1))
    out = np.concatenate(blocks, axis=0).astype(np.float32)
    inv = np.empty(len(order), dtype=np.int64)
    inv[np.asarray(order, dtype=np.int64)] = np.arange(len(order), dtype=np.int64)
    return out[inv]


def encode_packets(
    tok: Tokenizer,
    texts: list[str],
    *,
    cap: int = 64,
) -> tuple[list[list[int]], float]:
    if cap < 1:
        raise ValueError("cap must be positive")
    bos = tok.token_to_id[BOS]
    seqs: list[list[int]] = []
    truncated = 0
    for text in texts:
        ids = tok.encode(text)
        if len(ids) > cap - 1:
            truncated += 1
        seqs.append([bos, *ids[: cap - 1]])
    fraction = truncated / len(texts) if texts else 0.0
    return seqs, fraction
