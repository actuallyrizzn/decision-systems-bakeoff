"""Batched sparse-CSR reservoir over the frozen larva connectome."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decision_bakeoff.flybrain.brain import FlyBrain, build_fly_brain
from decision_bakeoff.flybrain.connectome import Connectome, load_connectome

_CONN: Connectome | None = None


def connectome() -> Connectome:
    global _CONN
    if _CONN is None:
        _CONN = load_connectome()
    return _CONN


@dataclass
class ReservoirCfg:
    embed_dim: int = 32
    radius: float = 0.9
    leak: float = 0.5
    steps: int = 2
    inject_count: int = 64
    input_scale: float = 1.0
    seed: int = 0
    scrambled: bool = False


def wiring_matrix(brain: FlyBrain):
    from scipy import sparse

    n = brain.n_neurons
    return sparse.csr_matrix(
        (
            brain.syn_val.astype(np.float32),
            (brain.syn_post.astype(np.int64), brain.syn_pre.astype(np.int64)),
        ),
        shape=(n, n),
    )


class FastReservoir:
    """Batched sparse-CSR twin of FlyBrain.inject_token."""

    def __init__(self, brain: FlyBrain):
        self.brain = brain
        n = brain.n_neurons
        self.wt = wiring_matrix(brain)
        dim = brain.embed.shape[1]
        m = np.zeros((n, dim), dtype=np.float32)
        for i, neuron in enumerate(brain.inject):
            m[int(neuron), i % dim] += 1.0
        self.inject_m = m
        self.n = n

    def drive(self, tokens: np.ndarray) -> np.ndarray:
        e = self.brain.embed[tokens].astype(np.float32) * self.brain.input_scale
        return self.inject_m @ e.T

    def step(self, x: np.ndarray, drive: np.ndarray) -> np.ndarray:
        a = self.brain.leak
        steps = int(self.brain.steps)
        if steps == 1:
            return (1.0 - a) * x + a * np.tanh(drive + self.wt @ x)
        for _ in range(steps):
            x = (1.0 - a) * x + a * np.tanh(drive + self.wt @ x)
        return x


def build_brain(cfg: ReservoirCfg, vocab_size: int) -> FlyBrain:
    return build_fly_brain(
        vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        radius=cfg.radius,
        leak=cfg.leak,
        steps=cfg.steps,
        inject_count=cfg.inject_count,
        input_scale=cfg.input_scale,
        seed=cfg.seed,
        scrambled=cfg.scrambled,
        connectome=connectome(),
    )


def build_reservoir_from_arm(
    *,
    leak: float,
    steps: int,
    radius: float,
    inject_count: int,
    input_scale: float,
    seed: int,
    vocab_size: int,
    embed: np.ndarray,
) -> FastReservoir:
    if embed.shape != (vocab_size, inject_count):
        raise ValueError(
            f"embed {embed.shape} != ({vocab_size}, {inject_count}); "
            "embed_dim must equal inject_count"
        )
    cfg = ReservoirCfg(
        embed_dim=inject_count,
        radius=radius,
        leak=leak,
        steps=steps,
        inject_count=inject_count,
        input_scale=input_scale,
        seed=seed,
        scrambled=False,
    )
    brain = build_brain(cfg, vocab_size)
    brain.embed = embed.astype(np.float32).copy()
    return FastReservoir(brain)
