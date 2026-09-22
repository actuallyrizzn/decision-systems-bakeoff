"""GloVe load and the fixed projection onto inject neurons.

embed_dim is always equal to inject_count. FlyBrain.inject_token maps
vec[i % len(vec)] onto inject neuron i, and FastReservoir.inject_m does the
same. With dim == inject_count every inject neuron gets exactly one dimension.
Do not set embed_dim to 100.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from decision_bakeoff.flybrain.tokenizer import BOS, EOS, PAD, SEP, UNK, Tokenizer

__all__ = ["build_embed", "load_glove"]

_SPECIALS = (PAD, UNK, BOS, EOS, SEP)
_GLOVE_DIM = 100


def load_glove(path: str | Path, vocab: set[str]) -> dict[str, np.ndarray]:
    """Stream a GloVe text file and keep only ``vocab`` (matched lowercase)."""
    wanted = {word.lower() for word in vocab}
    found: dict[str, np.ndarray] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if len(found) == len(wanted):
                break
            word, _sep, rest = line.partition(" ")
            key = word.lower()
            if key not in wanted or key in found:
                continue
            values = np.fromstring(rest, dtype=np.float32, sep=" ")
            if values.size != _GLOVE_DIM:
                continue
            found[key] = values
    return found


def build_embed(
    tok: Tokenizer,
    glove: dict[str, np.ndarray],
    inject_count: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Return ``(embed, coverage)``.

    ``embed`` is ``(tok.size, inject_count)`` float32. Specials are zero.
    Out-of-vocabulary rows are the mean of the found GloVe vectors, projected
    with the same matrix. ``coverage`` is found words over non-special tokens.
    """
    if inject_count < 1:
        raise ValueError("inject_count must be positive")
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(_GLOVE_DIM)
    projection = rng.normal(0.0, scale, size=(_GLOVE_DIM, inject_count)).astype(np.float32)
    found_vecs = [vec for word, vec in glove.items() if word in tok.token_to_id and word not in _SPECIALS]
    if found_vecs:
        mean_vec = np.mean(np.stack(found_vecs), axis=0).astype(np.float32)
    else:
        mean_vec = np.zeros(_GLOVE_DIM, dtype=np.float32)
    oov = mean_vec @ projection
    embed = np.tile(oov, (tok.size, 1)).astype(np.float32)
    n_found = 0
    for word, index in tok.token_to_id.items():
        if word in _SPECIALS:
            embed[index] = 0.0
            continue
        vec = glove.get(word)
        if vec is None:
            continue
        embed[index] = vec @ projection
        n_found += 1
    denom = max(1, tok.size - len(_SPECIALS))
    return embed, n_found / denom
