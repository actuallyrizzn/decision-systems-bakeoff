"""Venice Jev decisions API client."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


def call_jev(
    *,
    api_key: str,
    state: str,
    questions: dict[str, Any],
    model: str = "jev-latest",
    endpoint: str = "https://api.venice.ai/api/v1/decisions",
    timeout: float = 120.0,
    max_retries: int = 8,
) -> tuple[dict[str, Any], float, dict[str, str]]:
    body = json.dumps(
        {"model": model, "state": state, "questions": questions}
    ).encode()
    last_err: Exception | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                headers = {k.lower(): v for k, v in resp.headers.items()}
                payload = json.loads(resp.read().decode())
            return payload, time.perf_counter() - t0, headers
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                wait = float(retry_after) if retry_after else min(60.0, 2.0 ** attempt)
            except ValueError:
                wait = min(60.0, 2.0 ** attempt)
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def extract_choice(payload: dict[str, Any], qid: str) -> tuple[str, dict[str, float]]:
    answers = payload.get("answers") or {}
    ans = answers.get(qid) or {}
    choice = str(ans.get("choice") or "")
    probs_raw = ans.get("probabilities") or {}
    probs = {str(k): float(v) for k, v in probs_raw.items()}
    return choice, probs
