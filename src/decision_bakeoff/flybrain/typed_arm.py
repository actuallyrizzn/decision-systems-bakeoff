"""Flybrain typed-decisions arm: reservoir zero-shot over per-question menus.

No ridge head — typed has no train split. For each question we encode the
(state + instructions) context and each option label, score by cosine
similarity of pooled reservoir states, then softmax (Julia harness = argmax).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from decision_bakeoff.flybrain.features import encode_packets, pooled_states
from decision_bakeoff.flybrain.pipeline import load_fly_cfg
from decision_bakeoff.flybrain.reservoir import build_reservoir_from_arm
from decision_bakeoff.flybrain.tokenizer import Tokenizer
from decision_bakeoff.flybrain.vectors import build_embed, load_glove


def _state_to_text(state: object) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def _option_labels(qobj: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (keys, label_texts) in harness order."""
    kind = qobj["type"]
    criteria = qobj["criteria"]
    if kind == "score":
        labels = [str(v) for v in criteria]
        keys = [str(i) for i in range(len(labels))]
        return keys, labels
    if kind == "noul":
        keys = ["false", "true"]
        return keys, [str(criteria[k]) for k in keys]
    if kind == "choice":
        keys = [str(k) for k in criteria.keys()]
        return keys, [str(criteria[k]) for k in keys]
    raise ValueError(f"unknown typed type {kind}")


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores.astype(np.float64) / max(float(temperature), 1e-6)
    z = z - z.max()
    ex = np.exp(z)
    return ex / ex.sum()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class FlybrainTypedArm:
    """decide(state, questions) → Jev-shaped answers payload."""

    def __init__(
        self,
        *,
        glove_path: Path,
        vocab_texts: list[str],
        cfg_path: Path | None = None,
        task_key: str = "typed_decisions",
        temperature: float = 0.25,
    ):
        cfg = load_fly_cfg(cfg_path)
        if task_key not in cfg["tasks"]:
            raise KeyError(f"flybrain cfg missing task {task_key}")
        arm = cfg["tasks"][task_key]
        self.cap = int(cfg["packet_cap"])
        self.pooling = str(arm["pooling"])
        self.temperature = float(temperature)
        self.tok = Tokenizer.build(vocab_texts)
        glove = load_glove(glove_path, set(self.tok.token_to_id))
        embed, self.glove_coverage = build_embed(
            self.tok, glove, int(arm["inject_count"]), int(arm["seed"])
        )
        self.res = build_reservoir_from_arm(
            leak=float(arm["leak"]),
            steps=int(arm["steps"]),
            radius=float(arm["radius"]),
            inject_count=int(arm["inject_count"]),
            input_scale=float(arm["input_scale"]),
            seed=int(arm["seed"]),
            vocab_size=self.tok.size,
            embed=embed,
        )
        self.cfg_key = str(arm.get("cfg_key") or task_key)
        self.tokenizer_fingerprint = self.tok.fingerprint

    @classmethod
    def from_typed_parquet(
        cls,
        *,
        parquet_path: Path,
        glove_path: Path,
        cfg_path: Path | None = None,
    ) -> FlybrainTypedArm:
        import pyarrow.parquet as pq

        cases = pq.read_table(parquet_path).to_pylist()
        texts: list[str] = []
        for case in cases:
            state = json.loads(case["state"]) if isinstance(case["state"], str) else case["state"]
            texts.append(_state_to_text(state))
            questions = (
                json.loads(case["questions"])
                if isinstance(case["questions"], str)
                else case["questions"]
            )
            for q in questions.values():
                texts.append(str(q.get("instructions") or ""))
                crit = q.get("criteria")
                if isinstance(crit, dict):
                    texts.extend(str(v) for v in crit.values())
                elif isinstance(crit, list):
                    texts.extend(str(v) for v in crit)
        return cls(glove_path=glove_path, vocab_texts=texts, cfg_path=cfg_path)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        seqs, _ = encode_packets(self.tok, texts, cap=self.cap)
        return pooled_states(self.res, seqs, pooling=self.pooling, batch_lines=64)

    def decide(self, state: str, questions: dict[str, Any]) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        state_text = _state_to_text(state)
        answers: dict[str, Any] = {}
        for qid, qobj in questions.items():
            keys, labels = _option_labels(qobj)
            instructions = str(qobj.get("instructions") or "")
            context = f"{state_text}\n\n{instructions}"
            # One context + one vector per option label
            feats = self._embed_texts([context, *labels])
            ctx = feats[0]
            scores = np.array(
                [_cosine(ctx, feats[i + 1]) for i in range(len(labels))],
                dtype=np.float64,
            )
            probs_arr = _softmax(scores, self.temperature)
            probs = {keys[i]: float(probs_arr[i]) for i in range(len(keys))}
            choice = max(probs.items(), key=lambda kv: kv[1])[0]
            kind = qobj["type"]
            ans: dict[str, Any] = {"probabilities": probs}
            if kind == "choice":
                ans["choice"] = choice
            elif kind == "noul":
                ans["noul"] = float(probs.get("true", 0.0))
            elif kind == "score":
                ans["score"] = int(choice) if str(choice).isdigit() else choice
            answers[qid] = ans
        return {"answers": answers}, time.perf_counter() - t0
