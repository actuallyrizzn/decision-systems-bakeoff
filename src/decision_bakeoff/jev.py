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
    timeout: float = 180.0,
    max_retries: int = 12,
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
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            # SSL handshake timeouts / transient network — retry
            last_err = exc
            time.sleep(min(60.0, 2.0 ** attempt))
    assert last_err is not None
    raise last_err

def extract_choice(payload: dict[str, Any], qid: str) -> tuple[str, dict[str, float]]:
    """Extract a choice-typed answer (bakeoff classification tasks)."""
    answers = payload.get("answers") or {}
    ans = answers.get(qid) or {}
    choice = str(ans.get("choice") or "")
    probs_raw = ans.get("probabilities") or {}
    probs = {str(k): float(v) for k, v in probs_raw.items()}
    return choice, probs


def extract_typed_answer(
    payload: dict[str, Any], qid: str, qtype: str
) -> tuple[str, dict[str, float]]:
    """Extract choice / score / noul label for typed-decisions scoring.

    Scoring rule matches Julia's published harness: argmax over probabilities
    (for score, do not round the expected index).
    """
    answers = payload.get("answers") or {}
    ans = answers.get(qid) or {}
    probs_raw = ans.get("probabilities") or {}
    probs = {str(k): float(v) for k, v in probs_raw.items()}
    if qtype == "choice":
        if "choice" in ans and ans["choice"] is not None:
            return str(ans["choice"]), probs
    if qtype == "noul":
        if "noul" in ans and ans["noul"] is not None and not probs:
            # some APIs return only noul probability of true
            p_true = float(ans["noul"])
            probs = {"false": 1.0 - p_true, "true": p_true}
        if probs:
            return max(probs.items(), key=lambda kv: kv[1])[0], probs
    if qtype == "score":
        if probs:
            return max(probs.items(), key=lambda kv: kv[1])[0], probs
        if "score" in ans and ans["score"] is not None:
            return str(int(ans["score"])), probs
    if probs:
        return max(probs.items(), key=lambda kv: kv[1])[0], probs
    raise ValueError(f"no typed answer for {qid} type={qtype}: {ans!r}")
