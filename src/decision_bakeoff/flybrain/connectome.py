"""Load a fly wiring graph as a weight matrix. Layout is frozen data."""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
EDGES_FILE = DATA_DIR / "fly_larva_edges.csv.gz"
NODES_FILE = DATA_DIR / "fly_larva_nodes.csv.gz"
EDGE_TYPES = frozenset({"ad", "aa", "dd", "da"})


@dataclass(frozen=True)
class Connectome:
    """Synapse-count matrix ``W[pre, post]`` plus one cell type per neuron."""

    weights: np.ndarray
    cell_types: tuple[str, ...]

    @property
    def size(self) -> int:
        return int(self.weights.shape[0])

    @property
    def edge_count(self) -> int:
        return int(np.count_nonzero(self.weights))

    def neurons_of_type(self, cell_type: str) -> np.ndarray:
        """Indices whose ``cell_type`` matches (exact, case-sensitive)."""
        return np.array(
            [i for i, kind in enumerate(self.cell_types) if kind == cell_type],
            dtype=np.int64,
        )


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return open(path, "rt", newline="", encoding="utf-8")


def load_cell_types(path: Path | None = None) -> tuple[str, ...]:
    nodes_path = path or NODES_FILE
    kinds: dict[int, str] = {}
    with _open_text(nodes_path) as handle:
        for row in csv.DictReader(handle):
            kinds[int(row["index"])] = row.get("cell_type", "") or ""
    if not kinds:
        raise ValueError(f"{nodes_path} has no neurons")
    size = max(kinds) + 1
    return tuple(kinds.get(i, "") for i in range(size))


def load_adjacency(
    path: Path | None = None,
    *,
    size: int | None = None,
    edge_types: frozenset[str] | set[str] | None = None,
) -> np.ndarray:
    """Dense ``W[pre, post] = synapse count``. Filter by ``etype`` if asked."""
    edges_path = path or EDGES_FILE
    keep = frozenset(edge_types) if edge_types is not None else None
    if keep is not None and not keep <= EDGE_TYPES:
        raise ValueError(f"unknown edge types: {sorted(keep - EDGE_TYPES)}")
    sources: list[int] = []
    targets: list[int] = []
    counts: list[float] = []
    with _open_text(edges_path) as handle:
        for row in csv.DictReader(handle):
            if keep is not None and row.get("etype", "") not in keep:
                continue
            sources.append(int(row["source"]))
            targets.append(int(row["target"]))
            counts.append(float(row["count"]))
    if not sources:
        raise ValueError(f"{edges_path} has no edges")
    n = size if size is not None else max(max(sources), max(targets)) + 1
    if n <= max(max(sources), max(targets)):
        raise ValueError("size is smaller than the largest neuron index")
    weights = np.zeros((n, n), dtype=np.float32)
    np.add.at(weights, (np.array(sources), np.array(targets)), np.array(counts, dtype=np.float32))
    return weights


def load_connectome(
    edges: Path | None = None,
    nodes: Path | None = None,
    *,
    edge_types: frozenset[str] | set[str] | None = None,
) -> Connectome:
    cell_types = load_cell_types(nodes)
    weights = load_adjacency(edges, size=len(cell_types), edge_types=edge_types)
    return Connectome(weights=weights, cell_types=cell_types)


def scramble(weights: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Control graph: same weights, targets shuffled. Kills the fly's wiring layout."""
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be square")
    rng = np.random.default_rng(seed)
    pre, post = np.nonzero(weights)
    values = weights[pre, post]
    shuffled_post = rng.permutation(post)
    out = np.zeros_like(weights)
    np.add.at(out, (pre, shuffled_post), values)
    return out


def spectral_radius(weights: np.ndarray, *, iterations: int = 60, seed: int = 0) -> float:
    """Largest |eigenvalue| by power iteration. Good enough to scale ``W``."""
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be square")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    rng = np.random.default_rng(seed)
    n = weights.shape[0]
    vec = rng.standard_normal(n).astype(np.float64)
    vec /= np.linalg.norm(vec) or 1.0
    matrix = weights.astype(np.float64)
    radius = 0.0
    for _ in range(iterations):
        nxt = matrix @ vec
        norm = float(np.linalg.norm(nxt))
        if norm == 0.0:
            return 0.0
        radius = norm
        vec = nxt / norm
    return radius
