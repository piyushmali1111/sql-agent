"""LLM interface, rate limiting, retries, and local embeddings.

Configures ChatGoogleGenerativeAI as the single LLM provider with exponential
backoff, rate limiting, token usage tracking, and safe error handling.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LLMError(Exception):
    """Base exception for LLM operations."""


class LLMRateLimitError(LLMError):
    """Raised when rate limits are exhausted after retries."""


class LLMBlockedResponseError(LLMError):
    """Raised when an LLM response is blocked by safety filters or returned empty."""


@dataclass
class LLMCallMetrics:
    total_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    def record_call(
        self,
        step_name: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self.total_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.total_latency_ms += latency_ms
        self.history.append(
            {
                "step": step_name,
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "timestamp": time.time(),
            }
        )

    def reset(self) -> None:
        self.total_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self.history.clear()


class RateLimiter:
    """Sliding-window rate limiter ensuring calls do not exceed requests_per_minute."""

    def __init__(self, requests_per_minute: int = 15):
        self.rpm = max(1, requests_per_minute)
        self.window = 60.0
        self.timestamps: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.time()
            # Discard timestamps outside the current 60s window
            self.timestamps = [t for t in self.timestamps if now - t < self.window]

            if len(self.timestamps) >= self.rpm:
                # Sleep until the oldest request falls out of the window
                sleep_duration = self.window - (now - self.timestamps[0]) + 0.1
                if sleep_duration > 0:
                    logger.info(
                        f"Rate limit throttle: sleeping {sleep_duration:.2f}s "
                        f"to stay within {self.rpm} RPM."
                    )
                    time.sleep(sleep_duration)
                now = time.time()
                self.timestamps = [t for t in self.timestamps if now - t < self.window]

            self.timestamps.append(time.time())


_global_rate_limiter = RateLimiter(requests_per_minute=settings.requests_per_minute)
_global_metrics = LLMCallMetrics()


def get_llm_metrics() -> LLMCallMetrics:
    return _global_metrics


def get_llm(
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: float = 30.0,
) -> ChatGoogleGenerativeAI:
    """Creates a ChatGoogleGenerativeAI instance.

    Uses GOOGLE_API_KEY and GEMINI_MODEL from environment / settings.
    """
    api_key = settings.google_api_key
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Please add your Gemini API key to .env "
            "or set the GOOGLE_API_KEY environment variable."
        )

    model_name = model or settings.gemini_model

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
        timeout=timeout,
    )


def invoke_with_retry(
    invocable: Callable[..., Any],
    *args: Any,
    step_name: str = "llm_call",
    max_attempts: int = 4,
    metrics: LLMCallMetrics | None = None,
    **kwargs: Any,
) -> Any:
    """Executes an LLM callable with rate limiting, exponential backoff, and metrics tracking."""
    active_metrics = metrics if metrics is not None else _global_metrics
    last_exception: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        _global_rate_limiter.acquire()
        start_time = time.perf_counter()

        try:
            response = invocable(*args, **kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Inspect token usage if available
            p_tokens = 0
            c_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                p_tokens = response.usage_metadata.get("input_tokens", 0)
                c_tokens = response.usage_metadata.get("output_tokens", 0)
            elif hasattr(response, "response_metadata"):
                meta = response.response_metadata
                usage = meta.get("usage_metadata") or meta.get("token_usage") or {}
                p_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                c_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0

            # Check for blocked or empty response
            if hasattr(response, "content"):
                if not response.content and not getattr(response, "tool_calls", None):
                    # Check finish reason
                    finish_reason = getattr(response, "response_metadata", {}).get("finish_reason", "EMPTY")
                    raise LLMBlockedResponseError(
                        f"Gemini returned an empty response (finish_reason: {finish_reason}). "
                        "The query may have triggered safety filters."
                    )

            active_metrics.record_call(step_name, latency_ms, p_tokens, c_tokens)
            return response

        except LLMBlockedResponseError:
            raise
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            last_exception = e
            err_str = str(e).lower()

            is_transient = any(
                term in err_str
                for term in [
                    "429",
                    "resource_exhausted",
                    "quota",
                    "rate limit",
                    "503",
                    "unavailable",
                    "deadline_exceeded",
                    "connection error",
                ]
            )

            if is_transient and attempt < max_attempts:
                # Exponential backoff with jitter
                sleep_secs = min(30.0, (2 ** attempt) + random.uniform(0.5, 1.5))
                logger.warning(
                    f"LLM transient error ({e}). Retrying attempt {attempt + 1}/{max_attempts} "
                    f"in {sleep_secs:.2f}s..."
                )
                time.sleep(sleep_secs)
            else:
                logger.error(f"LLM call failed permanently on attempt {attempt}: {e}")
                if is_transient:
                    raise LLMRateLimitError(
                        f"Gemini API rate limit or quota exceeded after {max_attempts} attempts: {e}"
                    ) from e
                raise LLMError(f"LLM invocation failed: {e}") from e

    raise LLMError(f"LLM call failed after {max_attempts} attempts: {last_exception}")


class LocalHuggingFaceEmbeddings(Embeddings):
    """Local HuggingFace embeddings running offline via sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return embedding.tolist()


_embeddings_cache: dict[str, LocalHuggingFaceEmbeddings] = {}
_embeddings_lock = threading.Lock()


def get_embeddings(model_name: str | None = None) -> LocalHuggingFaceEmbeddings:
    """Returns a singleton/cached instance of LocalHuggingFaceEmbeddings."""
    name = model_name or settings.embedding_model
    with _embeddings_lock:
        if name not in _embeddings_cache:
            logger.info(f"Loading local embedding model: {name}")
            _embeddings_cache[name] = LocalHuggingFaceEmbeddings(name)
        return _embeddings_cache[name]

