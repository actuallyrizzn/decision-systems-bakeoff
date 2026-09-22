"""Ridge and logistic class heads, temperature, and metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Head",
    "fit_logistic",
    "fit_ridge_classes",
    "fit_temperature",
    "metrics",
    "off_by_one_acc",
    "predict_proba",
    "standardize",
    "top_k_acc",
]

_CHUNK = 2000


@dataclass
class Head:
    w: np.ndarray
    b: np.ndarray
    temperature: float
    kind: str
    lam: float
    mu: np.ndarray
    sd: np.ndarray


def standardize(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x_train.astype(np.float64).mean(axis=0)
    sd = x_train.astype(np.float64).std(axis=0)
    sd = np.maximum(sd, 1e-6)
    return mu, sd


def _apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x.astype(np.float64) - mu) / sd


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(z)
    return ex / ex.sum(axis=1, keepdims=True)


def _nll(proba: np.ndarray, y: np.ndarray) -> float:
    picked = proba[np.arange(len(y)), y]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def fit_temperature(logits_valid: np.ndarray, y_valid: np.ndarray, grid: list[float] | tuple[float, ...]) -> float:
    best_t = float(grid[0])
    best = float("inf")
    for temp in grid:
        score = _nll(_softmax(logits_valid / float(temp)), y_valid)
        if score < best:
            best = score
            best_t = float(temp)
    return best_t


def fit_ridge_classes(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    ridges: tuple[float, ...] = (0.1, 1, 10, 100),
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    temperatures: tuple[float, ...] = (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8),
    chunk: int = _CHUNK,
) -> tuple[Head, dict]:
    mu, sd = standardize(x)
    xs = _apply(x, mu, sd)
    feat = xs.shape[1]
    gram = np.zeros((feat + 1, feat + 1), dtype=np.float64)
    xty = np.zeros((feat + 1, n_classes), dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    for start in range(0, len(xs), chunk):
        block = xs[start : start + chunk]
        target = y[start : start + chunk]
        bias = np.concatenate([block, np.ones((len(block), 1))], axis=1)
        gram += bias.T @ bias
        for cls in range(n_classes):
            mask = target == cls
            if mask.any():
                xty[:, cls] += bias[mask].sum(axis=0)
    eye = np.eye(feat + 1)
    eye[-1, -1] = 0.0
    xv = _apply(x_valid, mu, sd)
    yv = np.asarray(y_valid, dtype=np.int64)
    best: tuple[float, Head] | None = None
    for lam in ridges:
        solved = np.linalg.solve(gram + lam * eye, xty)
        w = solved[:-1]
        b = solved[-1]
        logits = xv @ w + b
        temp = fit_temperature(logits, yv, temperatures)
        score = _nll(_softmax(logits / temp), yv)
        head = Head(
            w=w.astype(np.float32),
            b=b.astype(np.float32),
            temperature=temp,
            kind="ridge",
            lam=float(lam),
            mu=mu.astype(np.float32),
            sd=sd.astype(np.float32),
        )
        if best is None or score < best[0]:
            best = (score, head)
    assert best is not None
    return best[1], {"valid_nll": best[0], "ridge": best[1].lam, "temperature": best[1].temperature}


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cs: tuple[float, ...] = (0.1, 1, 10),
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    temperatures: tuple[float, ...] = (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8),
) -> Head:
    from sklearn.linear_model import LogisticRegression

    mu, sd = standardize(x)
    xs = _apply(x, mu, sd)
    xv = _apply(x_valid, mu, sd)
    y = np.asarray(y, dtype=np.int64)
    yv = np.asarray(y_valid, dtype=np.int64)
    best: tuple[float, Head] | None = None
    for c_value in cs:
        clf = LogisticRegression(C=c_value, max_iter=2000)
        clf.fit(xs, y)
        classes = [int(c) for c in clf.classes_]
        if clf.coef_.shape[0] == 1:
            w = np.zeros((xs.shape[1], max(classes) + 1), dtype=np.float64)
            b = np.zeros(max(classes) + 1, dtype=np.float64)
            w[:, classes[1]] = clf.coef_.ravel()
            b[classes[1]] = float(clf.intercept_[0])
        else:
            w = np.zeros((xs.shape[1], max(classes) + 1), dtype=np.float64)
            b = np.zeros(max(classes) + 1, dtype=np.float64)
            w[:, classes] = clf.coef_.T
            b[classes] = clf.intercept_
        logits = xv @ w + b
        temp = fit_temperature(logits, yv, temperatures)
        score = _nll(_softmax(logits / temp), yv)
        head = Head(
            w=w.astype(np.float32),
            b=b.astype(np.float32),
            temperature=temp,
            kind="logistic",
            lam=float(c_value),
            mu=mu.astype(np.float32),
            sd=sd.astype(np.float32),
        )
        if best is None or score < best[0]:
            best = (score, head)
    assert best is not None
    return best[1]


def predict_proba(head: Head, x: np.ndarray) -> np.ndarray:
    xs = _apply(x, head.mu.astype(np.float64), head.sd.astype(np.float64))
    logits = xs @ head.w.astype(np.float64) + head.b.astype(np.float64)
    return _softmax(logits / head.temperature)


def top_k_acc(proba: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """Fraction of rows whose true label is among the top-k predicted classes."""
    y = np.asarray(y, dtype=np.int64)
    if len(y) == 0:
        return 0.0
    k = min(int(k), proba.shape[1])
    top = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]
    return float(np.any(top == y[:, None], axis=1).mean())


def off_by_one_acc(proba: np.ndarray, y: np.ndarray) -> float:
    """Ordinal accuracy: |argmax(p) - y| <= 1 (Bugzilla severity)."""
    y = np.asarray(y, dtype=np.int64)
    if len(y) == 0:
        return 0.0
    pred = proba.argmax(axis=1)
    return float((np.abs(pred - y) <= 1).mean())


def metrics(proba: np.ndarray, y: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int64)
    n_classes = proba.shape[1]
    pred = proba.argmax(axis=1)
    acc = float((pred == y).mean()) if len(y) else 0.0
    f1s: list[float] = []
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, guess in zip(y, pred):
        confusion[int(truth), int(guess)] += 1
    for cls in range(n_classes):
        tp = int(confusion[cls, cls])
        fp = int(confusion[:, cls].sum() - tp)
        fn = int(confusion[cls, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    one_hot = np.eye(n_classes)[y]
    brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))
    conf = proba.max(axis=1)
    correct = (pred == y).astype(np.float64)
    bins = []
    ece = 0.0
    for i in range(10):
        lo = i / 10
        hi = (i + 1) / 10
        if i == 9:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum())
        mean_conf = float(conf[mask].mean()) if count else 0.0
        mean_acc = float(correct[mask].mean()) if count else 0.0
        bins.append([lo, hi, count, mean_conf, mean_acc])
        if count:
            ece += (count / len(y)) * abs(mean_acc - mean_conf)
    return {
        "acc": acc,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "brier": brier,
        "ece": float(ece),
        "nll": _nll(proba, y) if len(y) else 0.0,
        "n": int(len(y)),
        "confusion": confusion.tolist(),
        "reliability": bins,
        "top3_acc": top_k_acc(proba, y, k=3),
        "off_by_one_acc": off_by_one_acc(proba, y),
    }
