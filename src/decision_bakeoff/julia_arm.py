"""Supersonic Labs Julia-1 decision arm (cold, CPU)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


class JuliaArm:
    """Load once; call decide() per row. Native option limit is 2–20."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cpu",
        max_length: int = 1024,
        head_length: int = 512,
        cpu_threads: int | None = 4,
    ):
        import os

        if cpu_threads is not None:
            os.environ.setdefault("JULIA_CPU_THREADS", str(cpu_threads))
        from julia import load_model  # type: ignore

        self.engine = load_model(
            str(model_dir),
            device=device,
            strict_encoding=True,
            max_length=max_length,
            head_length=head_length,
        )

    def decide(self, state: str, questions: dict[str, Any]) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        out = self.engine.predict(state=state, questions=questions)
        elapsed = time.perf_counter() - t0
        if isinstance(out, dict) and "answers" in out:
            return out, elapsed
        raise TypeError(f"unexpected Julia predict return: {type(out)!r}")
