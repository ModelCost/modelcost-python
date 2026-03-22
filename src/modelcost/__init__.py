"""ModelCost Python SDK — track, govern, and optimise AI model spending.

Usage::

    import modelcost

    modelcost.init(api_key="mc_...", org_id="org_123")
    wrapped = modelcost.wrap(openai_client)
    result = modelcost.scan_pii("some text with 123-45-6789")
    modelcost.shutdown()
"""

from __future__ import annotations

import logging
import threading
import uuid as _uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from modelcost._version import __version__
from modelcost.budget import BudgetManager
from modelcost.client import ModelCostClient
from modelcost.config import ModelCostConfig
from modelcost.exceptions import ConfigurationError
from modelcost.models.session import (
    CloseSessionRequest,
    CreateSessionRequest,
    RecordSessionCallRequest,
)
from modelcost.models.track import TrackRequest
from modelcost.pii import PiiResult, PiiScanner
from modelcost.providers.anthropic import AnthropicProvider
from modelcost.providers.google import GoogleProvider
from modelcost.providers.openai import OpenAIProvider
from modelcost.rate_limiter import TokenBucketRateLimiter
from modelcost.session import SessionCallRecord, SessionContext
from modelcost.tracking import CostTracker, sync_pricing_from_api

if TYPE_CHECKING:
    from modelcost.models.budget import BudgetCheckResponse, BudgetStatusResponse

logger = logging.getLogger("modelcost")

__all__ = [
    "__version__",
    "init",
    "wrap",
    "track_cost",
    "check_budget",
    "get_usage",
    "scan_pii",
    "flush",
    "shutdown",
    "start_session",
    "close_session",
    "SessionContext",
]


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

class _ModelCostSDK:
    """Internal singleton that holds initialised components."""

    def __init__(self, config: ModelCostConfig) -> None:
        self.config = config
        self.client = ModelCostClient(config)

        # Synchronous pricing sync before anything uses calculateCost
        try:
            sync_pricing_from_api(self.client)
        except Exception:
            logger.warning("Failed to sync pricing on init; cost estimates unavailable until next sync")

        self.tracker = CostTracker(
            api_key=config.api_key,
            batch_size=config.flush_batch_size,
        )
        self.budget_manager = BudgetManager(
            org_id=config.org_id,
            sync_interval=config.sync_interval_seconds,
        )
        self.pii_scanner = PiiScanner()
        self.rate_limiter = TokenBucketRateLimiter(rate=10.0, burst=20)

        # Background flush timer
        self._flush_timer: threading.Timer | None = None
        self._stopped = False
        self._start_flush_timer()

        # Background pricing sync timer (5 min interval)
        self._pricing_timer: threading.Timer | None = None
        self._start_pricing_sync_timer()

    def _start_flush_timer(self) -> None:
        if self._stopped:
            return
        self._flush_timer = threading.Timer(
            self.config.flush_interval_seconds, self._flush_tick
        )
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _flush_tick(self) -> None:
        try:
            self.tracker.flush(self.client)
        except Exception:
            logger.warning("Background flush failed", exc_info=True)
        finally:
            self._start_flush_timer()

    _PRICING_SYNC_INTERVAL_SECONDS = 300.0  # 5 minutes

    def _start_pricing_sync_timer(self) -> None:
        if self._stopped:
            return
        self._pricing_timer = threading.Timer(
            self._PRICING_SYNC_INTERVAL_SECONDS, self._pricing_sync_tick
        )
        self._pricing_timer.daemon = True
        self._pricing_timer.start()

    def _pricing_sync_tick(self) -> None:
        try:
            sync_pricing_from_api(self.client)
        except Exception:
            logger.warning("Background pricing sync failed", exc_info=True)
        finally:
            self._start_pricing_sync_timer()

    def shutdown(self) -> None:
        self._stopped = True
        if self._flush_timer is not None:
            self._flush_timer.cancel()
        if self._pricing_timer is not None:
            self._pricing_timer.cancel()
        # Final flush
        try:
            self.tracker.flush(self.client)
        except Exception:
            logger.warning("Final flush during shutdown failed", exc_info=True)
        self.client.close()


_instance: _ModelCostSDK | None = None
_init_lock = threading.Lock()


def _get_instance() -> _ModelCostSDK:
    if _instance is None:
        raise ConfigurationError("SDK not initialised. Call modelcost.init() first.")
    return _instance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init(
    *,
    api_key: str | None = None,
    org_id: str | None = None,
    environment: str = "production",
    base_url: str = "https://api.modelcost.ai",
    monthly_budget: float | None = None,
    budget_action: str = "alert",
    fail_open: bool = True,
    flush_interval_seconds: float = 5.0,
    flush_batch_size: int = 100,
    sync_interval_seconds: float = 10.0,
    content_privacy: bool = False,
) -> None:
    """Initialise the ModelCost SDK.

    If *api_key* or *org_id* are not provided they are read from the
    ``MODELCOST_API_KEY`` / ``MODELCOST_ORG_ID`` environment variables.
    """
    global _instance

    kwargs: dict[str, Any] = {
        "environment": environment,
        "base_url": base_url,
        "fail_open": fail_open,
        "flush_interval_seconds": flush_interval_seconds,
        "flush_batch_size": flush_batch_size,
        "sync_interval_seconds": sync_interval_seconds,
        "budget_action": budget_action,
        "content_privacy": content_privacy,
    }
    if monthly_budget is not None:
        kwargs["monthly_budget"] = monthly_budget

    if api_key is not None and org_id is not None:
        kwargs["api_key"] = api_key
        kwargs["org_id"] = org_id
        config = ModelCostConfig(**kwargs)
    else:
        # Fill missing values from env
        if api_key is not None:
            kwargs["api_key"] = api_key
        if org_id is not None:
            kwargs["org_id"] = org_id
        config = ModelCostConfig.from_env(**kwargs)

    with _init_lock:
        if _instance is not None:
            _instance.shutdown()
        _instance = _ModelCostSDK(config)

    logger.info("ModelCost SDK initialised (org=%s, env=%s)", config.org_id, config.environment)


def wrap(client: Any, *, feature: str | None = None, session: SessionContext | None = None) -> Any:
    """Wrap a provider client for automatic cost tracking.

    Supports OpenAI, Anthropic, and Google Generative AI clients.
    If *session* is provided, all calls through the wrapped client will
    be tracked within the session and subject to its budget/iteration limits.
    """
    sdk = _get_instance()

    # Detect provider by duck-typing
    if _has_attr_chain(client, "chat", "completions", "create"):
        provider = OpenAIProvider(
            mc_client=sdk.client,
            tracker=sdk.tracker,
            budget_manager=sdk.budget_manager,
            pii_scanner=sdk.pii_scanner,
            rate_limiter=sdk.rate_limiter,
            api_key=sdk.config.api_key,
            feature=feature,
            config=sdk.config,
            session=session,
        )
        return provider.wrap(client)

    if _has_attr_chain(client, "messages", "create"):
        provider_a = AnthropicProvider(
            mc_client=sdk.client,
            tracker=sdk.tracker,
            budget_manager=sdk.budget_manager,
            pii_scanner=sdk.pii_scanner,
            rate_limiter=sdk.rate_limiter,
            api_key=sdk.config.api_key,
            feature=feature,
            config=sdk.config,
            session=session,
        )
        return provider_a.wrap(client)

    if hasattr(client, "generate_content"):
        model_name = getattr(client, "model_name", "gemini-1.5-pro")
        provider_g = GoogleProvider(
            mc_client=sdk.client,
            tracker=sdk.tracker,
            budget_manager=sdk.budget_manager,
            pii_scanner=sdk.pii_scanner,
            rate_limiter=sdk.rate_limiter,
            api_key=sdk.config.api_key,
            feature=feature,
            model_name=model_name,
            config=sdk.config,
            session=session,
        )
        return provider_g.wrap(client)

    raise ConfigurationError(
        f"Unsupported client type: {type(client).__name__}. "
        "Supported: OpenAI, Anthropic, Google GenerativeModel."
    )


def track_cost(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    feature: str | None = None,
    customer_id: str | None = None,
    latency_ms: int | None = None,
    metadata: dict[str, Any] | None = None,
    session: SessionContext | None = None,
) -> None:
    """Manually record a cost event, optionally within a session."""
    sdk = _get_instance()

    cost = CostTracker.calculate_cost(model, input_tokens, output_tokens)

    # Session pre-check (local, sub-microsecond)
    if session is not None:
        session.pre_call_check(estimated_cost=cost)

    request = TrackRequest(
        api_key=sdk.config.api_key,
        timestamp=datetime.now(timezone.utc),
        provider=provider,
        model=model,
        feature=feature or (session.feature if session else None),
        customer_id=customer_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        metadata=metadata,
    )
    sdk.tracker.record(request)

    # Session post-recording (local, sub-microsecond)
    if session is not None:
        call_record = session.record_call(
            call_type="llm",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        _sync_session_call_async(sdk, session, call_record)


def start_session(
    *,
    feature: str | None = None,
    max_spend_usd: float | None = None,
    max_iterations: int | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> SessionContext:
    """Start a new agent session with optional spend/iteration limits.

    Returns a SessionContext that can be passed to track_cost() or wrap().
    """
    sdk = _get_instance()

    sid = session_id or str(_uuid.uuid4())

    # Create server-side record (fail-open: if unreachable, continue local-only)
    server_id: str | None = None
    try:
        resp = sdk.client.create_session(CreateSessionRequest(
            api_key=sdk.config.api_key,
            session_id=sid,
            feature=feature,
            user_id=user_id,
            max_spend_usd=max_spend_usd,
            max_iterations=max_iterations,
        ))
        server_id = resp.id
    except Exception:
        logger.warning("Failed to create server-side session; continuing local-only")

    ctx = SessionContext(
        session_id=sid,
        server_session_id=server_id,
        feature=feature,
        user_id=user_id,
        max_spend_usd=max_spend_usd,
        max_iterations=max_iterations,
    )

    logger.info(
        "Session started: %s (feature=%s, max_spend=$%s, max_iter=%s)",
        sid, feature, max_spend_usd, max_iterations,
    )
    return ctx


def close_session(session: SessionContext, *, reason: str = "completed") -> None:
    """Close a session, flush remaining state to server."""
    sdk = _get_instance()
    session.close(reason)

    if session.server_session_id and session.server_session_id != "local-only":
        try:
            sdk.client.close_session(
                session.server_session_id,
                CloseSessionRequest(
                    api_key=sdk.config.api_key,
                    status=session.status,
                    termination_reason=session._termination_reason,
                    final_spend_usd=session.current_spend_usd,
                    final_iteration_count=session.iteration_count,
                ),
            )
        except Exception:
            logger.warning("Failed to close server-side session; local state is authoritative")

    logger.info(
        "Session closed: %s (status=%s, spend=$%.4f, iterations=%d)",
        session.session_id, session.status,
        session.current_spend_usd, session.iteration_count,
    )


def check_budget(
    *,
    feature: str | None = None,
    estimated_cost: float | None = None,
) -> BudgetCheckResponse:
    """Check whether the current budget allows the planned request."""
    sdk = _get_instance()
    return sdk.budget_manager.check(
        sdk.client, feature=feature, estimated_cost=estimated_cost
    )


def get_usage() -> BudgetStatusResponse:
    """Get the current budget / usage status."""
    sdk = _get_instance()
    return sdk.client.get_budget_status(sdk.config.org_id)


def scan_pii(text: str) -> PiiResult:
    """Scan *text* for PII using local regex patterns."""
    sdk = _get_instance()
    return sdk.pii_scanner.scan(text)


def flush() -> None:
    """Flush any buffered tracking events to the API immediately."""
    sdk = _get_instance()
    sdk.tracker.flush(sdk.client)


def shutdown() -> None:
    """Flush remaining events and release resources."""
    global _instance
    with _init_lock:
        if _instance is not None:
            _instance.shutdown()
            _instance = None
    logger.info("ModelCost SDK shut down")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync_session_call_async(
    sdk: _ModelCostSDK, session: SessionContext, record: SessionCallRecord
) -> None:
    """Fire-and-forget sync of a session call to the server."""
    if not session.server_session_id or session.server_session_id == "local-only":
        return

    def _sync() -> None:
        try:
            sdk.client.record_session_call(
                session.server_session_id,  # type: ignore[arg-type]
                RecordSessionCallRequest(
                    api_key=sdk.config.api_key,
                    call_sequence=record.call_sequence,
                    call_type=record.call_type,
                    tool_name=record.tool_name,
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    cumulative_input_tokens=record.cumulative_input_tokens,
                    cost_usd=record.cost_usd,
                    cumulative_cost_usd=record.cumulative_cost_usd,
                    pii_detected=record.pii_detected,
                ),
            )
        except Exception:
            logger.debug("Failed to sync session call to server", exc_info=True)

    t = threading.Thread(target=_sync, daemon=True)
    t.start()


def _has_attr_chain(obj: Any, *attrs: str) -> bool:
    """Return True if obj.attr1.attr2... exists."""
    current = obj
    for attr in attrs:
        current = getattr(current, attr, None)
        if current is None:
            return False
    return True
