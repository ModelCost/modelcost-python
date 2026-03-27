"""Cost tracker with local pricing table, buffered recording, and a decorator."""

from __future__ import annotations

import functools
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

from modelcost.models.cost import ModelPricing
from modelcost.models.track import TrackRequest, TrackResponse

if TYPE_CHECKING:
    from modelcost.client import ModelCostClient
    from modelcost.session import SessionContext

logger = logging.getLogger("modelcost")

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Pricing table — loaded from sdk/common/model_pricing.json at import time,
# refreshed at runtime via GET /api/v1/pricing/models.
# ---------------------------------------------------------------------------

_PRICING_JSON_PATHS = [
    Path(__file__).resolve().parents[4] / "common" / "model_pricing.json",  # sdk/python/src/modelcost -> sdk/common
    Path(__file__).resolve().parents[5] / "sdk" / "common" / "model_pricing.json",  # alternative layout
]


def _load_bundled_pricing() -> dict[str, ModelPricing]:
    """Load pricing from the shared model_pricing.json file."""
    for path in _PRICING_JSON_PATHS:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                models = data.get("models", {})
                return {
                    name: ModelPricing(
                        provider=info["provider"],
                        input_cost_per_1k=info["input_cost_per_1k"],
                        output_cost_per_1k=info["output_cost_per_1k"],
                        cache_creation_cost_per_1k=info.get("cache_creation_cost_per_1k"),
                        cache_read_cost_per_1k=info.get("cache_read_cost_per_1k"),
                    )
                    for name, info in models.items()
                }
            except Exception:
                logger.warning("Failed to load pricing from %s", path, exc_info=True)

    logger.debug("model_pricing.json not found, using hardcoded fallback")
    return _hardcoded_fallback()


def _hardcoded_fallback() -> dict[str, ModelPricing]:
    """Fallback if model_pricing.json is unavailable."""
    return {
        "gpt-4": ModelPricing(provider="openai", input_cost_per_1k=0.03, output_cost_per_1k=0.06,
                              cache_read_cost_per_1k=0.0),
        "gpt-4-turbo": ModelPricing(provider="openai", input_cost_per_1k=0.01, output_cost_per_1k=0.03,
                                    cache_read_cost_per_1k=0.0),
        "gpt-4o": ModelPricing(provider="openai", input_cost_per_1k=0.005, output_cost_per_1k=0.015,
                               cache_read_cost_per_1k=0.0),
        "gpt-4o-mini": ModelPricing(provider="openai", input_cost_per_1k=0.00015, output_cost_per_1k=0.0006,
                                    cache_read_cost_per_1k=0.0),
        "gpt-3.5-turbo": ModelPricing(provider="openai", input_cost_per_1k=0.0015, output_cost_per_1k=0.002),
        "claude-opus-4": ModelPricing(provider="anthropic", input_cost_per_1k=0.015, output_cost_per_1k=0.075,
                                      cache_creation_cost_per_1k=0.01875, cache_read_cost_per_1k=0.0015),
        "claude-sonnet-4": ModelPricing(provider="anthropic", input_cost_per_1k=0.003, output_cost_per_1k=0.015,
                                        cache_creation_cost_per_1k=0.00375, cache_read_cost_per_1k=0.0003),
        "claude-haiku-4": ModelPricing(provider="anthropic", input_cost_per_1k=0.00025, output_cost_per_1k=0.00125,
                                       cache_creation_cost_per_1k=0.0003125, cache_read_cost_per_1k=0.000025),
        "gemini-1.5-pro": ModelPricing(provider="google", input_cost_per_1k=0.00125, output_cost_per_1k=0.005,
                                       cache_read_cost_per_1k=0.0),
        "gemini-1.5-flash": ModelPricing(provider="google", input_cost_per_1k=0.000075, output_cost_per_1k=0.0003,
                                         cache_read_cost_per_1k=0.0),
        "gemini-2.0-flash": ModelPricing(provider="google", input_cost_per_1k=0.0001, output_cost_per_1k=0.0004,
                                         cache_read_cost_per_1k=0.0),
    }


MODEL_PRICING: dict[str, ModelPricing] = _load_bundled_pricing()


def sync_pricing_from_api(client: ModelCostClient) -> None:
    """Fetch the latest pricing table from the server and update the local cache.

    Called on SDK init and periodically by the background sync timer.
    """
    try:
        resp = client._http.get(f"{client._config.base_url}/api/v1/pricing/models")
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        updated: dict[str, ModelPricing] = {}
        for entry in models:
            updated[entry["model"]] = ModelPricing(
                provider=entry["provider"],
                input_cost_per_1k=entry["input_cost_per_1k"],
                output_cost_per_1k=entry["output_cost_per_1k"],
                cache_creation_cost_per_1k=entry.get("cache_creation_cost_per_1k"),
                cache_read_cost_per_1k=entry.get("cache_read_cost_per_1k"),
            )
        if updated:
            MODEL_PRICING.clear()
            MODEL_PRICING.update(updated)
            logger.info("Synced pricing table: %d models (v%s)", len(updated), data.get("version", "?"))
    except Exception:
        logger.debug("Failed to sync pricing from API, using local table", exc_info=True)


class CostTracker:
    """Accumulates tracking events in a buffer and flushes them in batches."""

    def __init__(
        self,
        api_key: str,
        batch_size: int = 100,
    ) -> None:
        self._api_key = api_key
        self._batch_size = batch_size
        self._buffer: list[TrackRequest] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Cost calculation
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_cost(
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """Return the estimated USD cost for the given token counts.

        Returns ``0.0`` if the model is not in the local pricing table.
        When cache rates are ``None`` (model doesn't support caching),
        falls back to the input rate. When ``0.0``, charges nothing.
        """
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            logger.debug("No local pricing for model %r; returning 0.0", model)
            return 0.0
        input_cost = (input_tokens / 1000.0) * pricing.input_cost_per_1k
        output_cost = (output_tokens / 1000.0) * pricing.output_cost_per_1k

        cache_creation_cost = 0.0
        if cache_creation_tokens > 0:
            rate = (pricing.cache_creation_cost_per_1k
                    if pricing.cache_creation_cost_per_1k is not None
                    else pricing.input_cost_per_1k)
            cache_creation_cost = (cache_creation_tokens / 1000.0) * rate

        cache_read_cost = 0.0
        if cache_read_tokens > 0:
            rate = (pricing.cache_read_cost_per_1k
                    if pricing.cache_read_cost_per_1k is not None
                    else pricing.input_cost_per_1k)
            cache_read_cost = (cache_read_tokens / 1000.0) * rate

        return input_cost + output_cost + cache_creation_cost + cache_read_cost

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def record(self, request: TrackRequest) -> None:
        """Add a tracking event to the buffer.

        If the buffer reaches ``batch_size`` the caller is expected to
        call :meth:`flush` (the SDK's background timer also flushes
        periodically).
        """
        with self._lock:
            self._buffer.append(request)
            buffer_len = len(self._buffer)

        if buffer_len >= self._batch_size:
            logger.debug("Buffer reached batch size (%d); ready for flush", self._batch_size)

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def flush(self, client: ModelCostClient) -> list[TrackResponse]:
        """Send all buffered events to the API and clear the buffer."""
        with self._lock:
            to_send = list(self._buffer)
            self._buffer.clear()

        responses: list[TrackResponse] = []
        for event in to_send:
            try:
                resp = client.track(event)
                responses.append(resp)
                # Detect cost discrepancy between server and local calculation
                if resp.cost is not None:
                    local_cost = self.calculate_cost(
                        event.model, event.input_tokens, event.output_tokens,
                        event.cache_creation_tokens or 0, event.cache_read_tokens or 0,
                    )
                    if local_cost > 0 and abs(resp.cost - local_cost) > local_cost * 0.01:
                        logger.warning(
                            "Cost discrepancy for model %s: server=%.8f local=%.8f (%.1f%%)",
                            event.model, resp.cost, local_cost,
                            abs(resp.cost - local_cost) / local_cost * 100,
                        )
            except Exception:
                logger.warning("Failed to flush tracking event", exc_info=True)
        return responses

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def track_cost(
        self,
        provider: str,
        model: str,
        feature: Optional[str] = None,
        customer_id: Optional[str] = None,
        session: Optional[SessionContext] = None,
    ) -> Callable[[F], F]:
        """Return a decorator that records cost automatically.

        The decorated function is expected to return an object with a
        ``usage`` attribute containing ``prompt_tokens`` / ``input_tokens``
        and ``completion_tokens`` / ``output_tokens``.

        If *session* is provided, checks session budget/iteration limits
        before the call and updates session counters after.
        """

        def decorator(fn: F) -> F:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Session pre-check (local, sub-microsecond)
                if session is not None:
                    estimated_cost = self.calculate_cost(model, 500, 500)
                    session.pre_call_check(estimated_cost=estimated_cost)

                start = time.monotonic()
                result = fn(*args, **kwargs)
                elapsed_ms = int((time.monotonic() - start) * 1000)

                # Try to extract tokens from the response
                input_tokens = 0
                output_tokens = 0
                cache_creation_tokens = 0
                cache_read_tokens = 0
                usage = getattr(result, "usage", None)
                if usage is not None:
                    # Detect provider-specific cache token fields
                    cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
                    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

                    # OpenAI: cached tokens in prompt_tokens_details
                    details = getattr(usage, "prompt_tokens_details", None)
                    if details is not None:
                        oai_cached = getattr(details, "cached_tokens", 0) or 0
                        if oai_cached > 0:
                            cache_read_tokens = oai_cached

                    raw_input = getattr(usage, "prompt_tokens", 0) or getattr(
                        usage, "input_tokens", 0
                    )
                    output_tokens = getattr(usage, "completion_tokens", 0) or getattr(
                        usage, "output_tokens", 0
                    )
                    # Subtract cached tokens from input for OpenAI (included in prompt_tokens)
                    # Anthropic already excludes cache tokens from input_tokens
                    if details is not None and cache_read_tokens > 0:
                        input_tokens = max(0, raw_input - cache_read_tokens)
                    else:
                        input_tokens = raw_input

                # Google: check usage_metadata for cached tokens
                usage_metadata = getattr(result, "usage_metadata", None)
                if usage_metadata is not None and usage is None:
                    raw_input = getattr(usage_metadata, "prompt_token_count", 0) or 0
                    output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
                    cached = getattr(usage_metadata, "cached_content_token_count", 0) or 0
                    if cached > 0:
                        cache_read_tokens = cached
                        input_tokens = max(0, raw_input - cached)
                    else:
                        input_tokens = raw_input

                request = TrackRequest(
                    api_key=self._api_key,
                    timestamp=datetime.now(timezone.utc),
                    provider=provider,
                    model=model,
                    feature=feature or (session.feature if session else None),
                    customer_id=customer_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_creation_tokens=cache_creation_tokens if cache_creation_tokens else None,
                    cache_read_tokens=cache_read_tokens if cache_read_tokens else None,
                    latency_ms=elapsed_ms,
                )
                self.record(request)

                # Session post-recording
                if session is not None:
                    actual_cost = self.calculate_cost(model, input_tokens, output_tokens,
                                                      cache_creation_tokens, cache_read_tokens)
                    session.record_call(
                        call_type="llm",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=actual_cost,
                    )

                return result

            return wrapper  # type: ignore[return-value]

        return decorator
