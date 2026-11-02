"""
Concurrency gates — bound heavy ML operations so concurrent users don't crash
the runtime or contend on shared TensorFlow state.

Why this is needed:
  - Streamlit Cloud runs ONE Python process. Every user session is a thread
    inside that process, so they share `tf` state and the model cache.
  - TF inference under unbounded parallelism saturates a 1 CPU / 1 GB tier
    fast and the kernel OOM-kills Streamlit.

What we expose:
  predict_gate()  — bounded semaphore; caps concurrent predict loops.
  cache_lock()    — guards ml_core._model_cache against TOCTOU on first load.
  GateBusy        — raised on timeout; UI shows a clean retry message.

Tunable via env:
  MAX_CONCURRENT_PREDICTS  default 2   (per Streamlit process)
  PREDICT_TIMEOUT_SEC      default 30
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator


# ── TUNING: how many predict loops may run at once ───────────────────────────
# 2 is sized for the Streamlit Cloud free tier (1 CPU / 1 GB). TensorFlow
# inference is CPU-bound, so a third concurrent loop buys no throughput and
# pushes RSS toward the OOM killer. Raise MAX_CONCURRENT_PREDICTS only together
# with the container's CPU/RAM.
# 30 s timeout: one predict loop is ~1–3 s for a full menu, so a session that
# has waited 30 s is behind a real pile-up — better a retry message than a
# browser that looks hung.
MAX_CONCURRENT_PREDICTS = int(os.environ.get("MAX_CONCURRENT_PREDICTS", 2))
PREDICT_TIMEOUT_SEC     = float(os.environ.get("PREDICT_TIMEOUT_SEC", 30.0))


class GateBusy(Exception):
    """Raised when a gate could not be acquired within timeout."""


# ── Module-level state. Lifetime = lifetime of the Streamlit process. ──
_PREDICT_SEM = threading.BoundedSemaphore(value=MAX_CONCURRENT_PREDICTS)
_CACHE_LOCK  = threading.Lock()


@contextmanager
def predict_gate(timeout: float | None = None) -> Iterator[None]:
    """
    Cap concurrent predict-loop executions across all sessions.
    Excess users queue until a slot frees, or raise GateBusy after timeout.
    """
    t = PREDICT_TIMEOUT_SEC if timeout is None else timeout
    if not _PREDICT_SEM.acquire(timeout=t):
        raise GateBusy(
            f"Server is already serving {MAX_CONCURRENT_PREDICTS} predictions. "
            "Please retry in a few seconds."
        )
    try:
        yield
    finally:
        _PREDICT_SEM.release()


@contextmanager
def cache_lock() -> Iterator[None]:
    """
    Guard the ml_core._model_cache dict against TOCTOU on first load.
    Two threads checking `if k not in _model_cache` simultaneously must not
    both call the loader function.
    """
    with _CACHE_LOCK:
        yield


def predict_slots_in_use() -> int:
    """Best-effort: how many predict slots are currently held (informational)."""
    try:
        return MAX_CONCURRENT_PREDICTS - _PREDICT_SEM._value  # type: ignore[attr-defined]
    except Exception:
        return 0
