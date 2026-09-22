"""Frozen-wiring recurrent core + Level A thin readout (NumPy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from decision_bakeoff.flybrain.connectome import Connectome, load_connectome, scramble, spectral_radius


def _scale_weights(weights: np.ndarray, radius: float, *, seed: int) -> np.ndarray:
    current = spectral_radius(weights, seed=seed)
    if current <= 0:
        return weights.astype(np.float32)
    return (weights * (radius / current)).astype(np.float32)


@dataclass
class FlyBrain:
    """Leaky tanh reservoir over a fixed (or scrambled) wiring matrix.

    Level A: ``W`` frozen. Token embeddings + readout are the thin translator.
    Synapses are stored sparse (pre/post/val) so steps stay cheap on CPU.
    """

    n_neurons: int
    syn_pre: np.ndarray
    syn_post: np.ndarray
    syn_val: np.ndarray
    inject: np.ndarray  # neuron indices that receive input
    embed: np.ndarray  # (vocab, embed_dim)
    readout: np.ndarray  # (n_neurons, vocab)
    leak: float = 0.5
    steps: int = 2
    input_scale: float = 1.0

    def __post_init__(self) -> None:
        n = self.n_neurons
        if self.readout.shape[0] != n:
            raise ValueError("readout rows must match neuron count")
        if self.embed.shape[1] < 1:
            raise ValueError("embed dim must be positive")
        if not 0.0 < self.leak <= 1.0:
            raise ValueError("leak must be in (0, 1]")
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if len(self.syn_pre) != len(self.syn_post) or len(self.syn_pre) != len(self.syn_val):
            raise ValueError("synapse arrays must align")
        self._state = np.zeros(n, dtype=np.float32)

    @property
    def size(self) -> int:
        return self.n_neurons

    @property
    def vocab_size(self) -> int:
        return int(self.embed.shape[0])

    def reset(self) -> None:
        self._state.fill(0.0)

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    def _recurrent(self, x: np.ndarray) -> np.ndarray:
        acc = np.zeros(self.n_neurons, dtype=np.float32)
        np.add.at(acc, self.syn_post, self.syn_val * x[self.syn_pre])
        return acc

    def inject_token(self, token_id: int) -> np.ndarray:
        """Add embedding onto inject neurons, run ``steps``, return state."""
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError("token_id out of range")
        drive = np.zeros(self.n_neurons, dtype=np.float32)
        vec = self.embed[token_id] * self.input_scale
        pool = self.inject
        for i, neuron in enumerate(pool):
            drive[int(neuron)] += float(vec[i % len(vec)])
        alpha = self.leak
        x = self._state
        for _ in range(self.steps):
            x = (1.0 - alpha) * x + alpha * np.tanh(drive + self._recurrent(x))
        self._state = x.astype(np.float32)
        return self.state

    def logits(self) -> np.ndarray:
        out = self._state @ self.readout
        bias = getattr(self, "_logit_bias", None)
        if bias is not None:
            out = out + bias
        return out

    def next_token_probs(self) -> np.ndarray:
        logits = self.logits().astype(np.float64)
        logits -= logits.max()
        ex = np.exp(logits)
        return ex / (ex.sum() or 1.0)


def _sparse_from_dense(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre, post = np.nonzero(weights)
    return (
        pre.astype(np.int32),
        post.astype(np.int32),
        weights[pre, post].astype(np.float32),
    )


def pick_inject_neurons(connectome: Connectome, *, count: int = 64, seed: int = 0) -> np.ndarray:
    """Prefer sensory cells; fall back to a seeded sample of all neurons."""
    sensory = connectome.neurons_of_type("sensory")
    rng = np.random.default_rng(seed)
    if len(sensory) >= count:
        return np.sort(rng.choice(sensory, size=count, replace=False))
    if len(sensory):
        need = count - len(sensory)
        others = np.setdiff1d(np.arange(connectome.size), sensory, assume_unique=False)
        extra = rng.choice(others, size=min(need, len(others)), replace=False)
        return np.sort(np.concatenate([sensory, extra]))
    return np.sort(rng.choice(connectome.size, size=min(count, connectome.size), replace=False))


def build_fly_brain(
    *,
    vocab_size: int,
    embed_dim: int = 32,
    radius: float = 0.9,
    leak: float = 0.5,
    steps: int = 2,
    inject_count: int = 64,
    input_scale: float = 1.0,
    seed: int = 0,
    scrambled: bool = False,
    connectome: Connectome | None = None,
) -> FlyBrain:
    conn = connectome or load_connectome()
    weights = scramble(conn.weights, seed=seed) if scrambled else conn.weights.copy()
    weights = _scale_weights(weights, radius, seed=seed)
    pre, post, val = _sparse_from_dense(weights)
    rng = np.random.default_rng(seed)
    inject = pick_inject_neurons(conn, count=inject_count, seed=seed)
    embed = rng.normal(0.0, 0.1, size=(vocab_size, embed_dim)).astype(np.float32)
    readout = rng.normal(0.0, 0.01, size=(conn.size, vocab_size)).astype(np.float32)
    return FlyBrain(
        n_neurons=conn.size,
        syn_pre=pre,
        syn_post=post,
        syn_val=val,
        inject=inject,
        embed=embed,
        readout=readout,
        leak=leak,
        steps=steps,
        input_scale=input_scale,
    )


@dataclass
class NoFlyPredictor:
    """Capacity-matched control: last-K token embeddings → linear vocab logits.

    No recurrent connectome. Same embed dim and vocab as the fly path.
    """

    embed: np.ndarray  # (vocab, dim)
    window: np.ndarray  # (K * dim, vocab) readout
    window_k: int = 3

    def __post_init__(self) -> None:
        if self.embed.ndim != 2:
            raise ValueError("embed must be 2d")
        expect = self.window_k * self.embed.shape[1]
        if self.window.shape[0] != expect:
            raise ValueError("window rows must be window_k * embed_dim")

    @property
    def vocab_size(self) -> int:
        return int(self.embed.shape[0])

    def features(self, token_ids: Sequence[int]) -> np.ndarray:
        dim = self.embed.shape[1]
        feats = np.zeros(self.window_k * dim, dtype=np.float32)
        recent = list(token_ids)[-self.window_k :]
        # left-pad
        while len(recent) < self.window_k:
            recent = [0] + recent
        for i, tid in enumerate(recent):
            tid = int(tid) % self.vocab_size
            feats[i * dim : (i + 1) * dim] = self.embed[tid]
        return feats

    def logits(self, token_ids: Sequence[int]) -> np.ndarray:
        out = self.features(token_ids) @ self.window
        bias = getattr(self, "_logit_bias", None)
        if bias is not None:
            out = out + bias
        return out

    def probs(self, token_ids: Sequence[int]) -> np.ndarray:
        logits = self.logits(token_ids).astype(np.float64)
        logits -= logits.max()
        ex = np.exp(logits)
        return ex / (ex.sum() or 1.0)


def build_no_fly(
    *,
    vocab_size: int,
    embed_dim: int = 32,
    window_k: int = 3,
    seed: int = 0,
) -> NoFlyPredictor:
    rng = np.random.default_rng(seed + 99)
    embed = rng.normal(0.0, 0.1, size=(vocab_size, embed_dim)).astype(np.float32)
    window = rng.normal(0.0, 0.01, size=(window_k * embed_dim, vocab_size)).astype(np.float32)
    return NoFlyPredictor(embed=embed, window=window, window_k=window_k)


def param_budget_fly_io(brain: FlyBrain) -> int:
    """Trainable Level-A-ish count: embeddings + readout (not W)."""
    return int(brain.embed.size + brain.readout.size)


def param_budget_no_fly(pred: NoFlyPredictor) -> int:
    return int(pred.embed.size + pred.window.size)
