"""
config/llm_factory.py — LLM factory (Gemini-free edition).

Provider assignment:
  groq   → Orchestrator, Quality Gate, Responder  (14,400 req/day free)
  ollama → Coder                                   (local, unlimited)

Gemini kept as optional fallback — only used if GEMINI_API_KEY is set
and you explicitly pass provider="gemini". Otherwise ignored entirely.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from config.settings import GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, OLLAMA_BASE_URL

log = logging.getLogger("nexus.llm_factory")


# ── Token-bucket rate limiter ─────────────────────────────────────────────────

class _TokenBucket:
    def __init__(self, capacity: float, rate: float) -> None:
        self._capacity = capacity
        self._rate = rate
        self._tokens = capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
        self._last = now

    def acquire(self, timeout: float = 120.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(2.0, 1.0 / self._rate + random.uniform(0, 0.5)))


# Groq free tier: 30 req/min → 0.45 req/s; burst of 5
_BUCKETS: dict[str, _TokenBucket] = {
    "groq":   _TokenBucket(capacity=8,  rate=0.45),
    "gemini": _TokenBucket(capacity=3,  rate=0.20),
    "ollama": _TokenBucket(capacity=99, rate=99.0),
}


# ── Retry / backoff ───────────────────────────────────────────────────────────

def _with_backoff(fn, provider: str, max_retries: int = 4):
    bucket = _BUCKETS.get(provider)
    for attempt in range(max_retries + 1):
        if bucket and not bucket.acquire(timeout=120):
            raise RuntimeError(f"[{provider}] rate-limit bucket timed out")
        try:
            return fn()
        except Exception as exc:
            err = str(exc).lower()
            retryable = any(k in err for k in (
                "rate", "quota", "429", "503", "timeout",
                "resource exhausted", "overloaded",
            ))
            if not retryable or attempt == max_retries:
                log.error("[%s] failed (attempt %d): %s", provider, attempt + 1, exc)
                raise
            delay = 2.0 * (2 ** attempt) + random.uniform(0, 1.5)
            log.warning("[%s] retry %d/%d in %.1fs: %s", provider, attempt + 1, max_retries, delay, exc)
            time.sleep(delay)


# ── LLM constructors ──────────────────────────────────────────────────────────

def _make_groq(temperature: float) -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
        temperature=temperature,
        max_retries=0,
        request_timeout=60,
    )

def _make_ollama(model_name: str | None, temperature: float) -> ChatOllama:
    return ChatOllama(
        model=model_name or "llama3.1:8b",
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )

def _make_gemini(model_name: str | None, temperature: float):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model_name or GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
        max_retries=0,
        convert_system_message_to_human=True,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def get_llm(provider: str = "groq", model_name: str | None = None, temperature: float = 0.1):
    p = provider.lower()
    if p == "ollama":
        return _make_ollama(model_name, temperature)
    if p == "gemini" and GEMINI_API_KEY:
        return _make_gemini(model_name, temperature)
    return _make_groq(temperature)   # default & fallback

def invoke_with_limit(llm: Any, messages: list, provider: str = "groq") -> Any:
    return _with_backoff(lambda: llm.invoke(messages), provider=provider)

def invoke_structured(llm: Any, schema, messages: list, provider: str = "groq") -> Any:
    structured = llm.with_structured_output(schema)
    return _with_backoff(lambda: structured.invoke(messages), provider=provider)