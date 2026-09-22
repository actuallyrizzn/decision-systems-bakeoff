"""ConvAI Laya open-weight decision arm."""

from __future__ import annotations

import time
from typing import Any


class LayaArm:
    def __init__(
        self,
        *,
        multilingual: bool = False,
        max_len: int | None = None,
        head_max_len: int | None = None,
        repo_id: str = "convaiinnovations/laya",
    ):
        import laya  # type: ignore

        kwargs: dict[str, Any] = {}
        if multilingual:
            kwargs["subfolder"] = "multilingual"
        self.model = laya.load(repo_id, **kwargs)
        if max_len is not None:
            self.model.cfg["max_len"] = int(max_len)
        if head_max_len is not None:
            self.model.cfg["head_max_len"] = int(head_max_len)

    def decide(self, state: str, questions: dict[str, Any]) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        out = self.model.predict(state, questions)
        elapsed = time.perf_counter() - t0
        if isinstance(out, dict) and "answers" in out:
            return out, elapsed
        raise TypeError(f"unexpected Laya predict return: {type(out)!r}")
