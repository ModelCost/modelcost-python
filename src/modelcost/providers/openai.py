"""OpenAI provider instrumentation."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

from modelcost.budget import BudgetManager
from modelcost.client import ModelCostClient
from modelcost.config import ModelCostConfig
from modelcost.models.governance import GovernanceSignalRequest
from modelcost.models.track import TrackRequest
from modelcost.pii import PiiScanner
from modelcost.providers.base import BaseProvider
from modelcost.rate_limiter import TokenBucketRateLimiter
from modelcost.session import SessionContext
from modelcost.tracking import CostTracker

logger = logging.getLogger("modelcost")


class _ChatCompletionsProxy:
    """Proxy for ``client.chat.completions`` that intercepts ``create()``."""

    def __init__(
        self,
        original_completions: Any,
        *,
        mc_client: ModelCostClient,
        config: Optional[ModelCostConfig],
        tracker: CostTracker,
        budget_manager: Optional[BudgetManager],
        pii_scanner: Optional[PiiScanner],
        rate_limiter: Optional[TokenBucketRateLimiter],
        api_key: str,
        feature: Optional[str],
        session: Optional[SessionContext] = None,
    ) -> None:
        self._original = original_completions
        self._mc_client = mc_client
        self._config = config
        self._tracker = tracker
        self._budget_manager = budget_manager
        self._pii_scanner = pii_scanner
        self._rate_limiter = rate_limiter
        self._api_key = api_key
        self._feature = feature
        self._session = session

    def create(self, **kwargs: Any) -> Any:
        """Intercept a chat completion request."""
        # 1. Rate limit
        if self._rate_limiter is not None:
            self._rate_limiter.allow(strict=True)

        # 2. PII / governance scan on messages
        if self._pii_scanner is not None:
            messages = kwargs.get("messages", [])
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    result = self._pii_scanner.scan(content)
                    if result.detected:
                        self._enforce_governance(content, result)

        # 3. Budget check
        model = kwargs.get("model", "unknown")
        if self._budget_manager is not None:
            estimated = CostTracker.calculate_cost(model, 500, 500)  # rough estimate
            check = self._budget_manager.check(
                self._mc_client, feature=self._feature, estimated_cost=estimated
            )
            if not check.allowed:
                from modelcost.exceptions import BudgetExceededError

                raise BudgetExceededError(
                    message=check.reason or "Budget exceeded",
                    remaining_budget=0.0,
                    scope=self._feature or "organization",
                )

        # 3b. Session pre-check (local, sub-microsecond)
        if self._session is not None:
            session_estimated = CostTracker.calculate_cost(model, 500, 500)
            self._session.pre_call_check(estimated_cost=session_estimated)

        # 4. Call original
        start = time.monotonic()
        response = self._original.create(**kwargs)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # 5. Extract usage and record
        input_tokens, output_tokens = OpenAIProvider.extract_usage_static(response)
        request = TrackRequest(
            api_key=self._api_key,
            timestamp=datetime.now(timezone.utc),
            provider="openai",
            model=model,
            feature=self._feature,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
        )
        self._tracker.record(request)

        # 6. Optimistic local spend update
        cost = CostTracker.calculate_cost(model, input_tokens, output_tokens)
        if self._budget_manager is not None:
            self._budget_manager.update_local_spend(self._feature, cost)

        # 7. Session post-recording
        if self._session is not None:
            self._session.record_call(
                call_type="llm",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )

        return response

    def _enforce_governance(self, content: str, scan_result: Any) -> None:
        """Enforce governance: metadata-only or standard server-side scanning."""
        from modelcost.exceptions import PiiDetectedError

        if self._config is not None and self._config.content_privacy:
            # Metadata-only mode: full local classification, never send raw content
            full_result = self._pii_scanner.full_scan(content)
            if full_result.detected:
                # Report signals (fire-and-forget)
                from datetime import datetime, timezone

                for v in full_result.violations:
                    try:
                        self._mc_client.report_signal(GovernanceSignalRequest(
                            organization_id=self._config.org_id,
                            violation_type=v.category,
                            violation_subtype=v.type,
                            severity=v.severity,
                            environment=self._config.environment,
                            action_taken="block",
                            was_allowed=False,
                            detected_at=datetime.now(timezone.utc).isoformat(),
                            source="metadata_only",
                            violation_count=1,
                        ))
                    except Exception:
                        pass  # fire-and-forget

                raise PiiDetectedError(
                    message="Sensitive content detected and blocked locally (metadata-only mode)",
                    detected_entities=[
                        {"type": v.category, "subtype": v.type, "severity": v.severity,
                         "start": v.start, "end": v.end}
                        for v in full_result.violations
                    ],
                    redacted_text=self._pii_scanner.redact(content),
                )
        else:
            # Standard mode: check governance policy server-side
            if self._config is not None:
                from modelcost.models.governance import GovernanceScanRequest

                gov_result = self._mc_client.scan_text(GovernanceScanRequest(
                    org_id=self._config.org_id,
                    text=content,
                    environment=self._config.environment,
                ))
                if not gov_result.is_allowed:
                    raise PiiDetectedError(
                        message="PII detected in request and blocked by policy",
                        detected_entities=[
                            {"type": v.type, "subtype": v.subtype, "severity": v.severity,
                             "start": v.start, "end": v.end}
                            for v in gov_result.violations
                        ],
                        redacted_text=gov_result.redacted_text or self._pii_scanner.redact(content),
                    )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _ChatProxy:
    """Proxy for ``client.chat`` that replaces ``.completions``."""

    def __init__(self, original_chat: Any, completions_proxy: _ChatCompletionsProxy) -> None:
        self._original = original_chat
        self.completions = completions_proxy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _OpenAIClientProxy:
    """Top-level proxy that replaces ``client.chat``."""

    def __init__(self, original_client: Any, chat_proxy: _ChatProxy) -> None:
        self._original = original_client
        self.chat = chat_proxy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class OpenAIProvider(BaseProvider):
    """Wraps an OpenAI client to add cost tracking, budgets, PII scanning."""

    def __init__(
        self,
        mc_client: ModelCostClient,
        tracker: CostTracker,
        budget_manager: Optional[BudgetManager] = None,
        pii_scanner: Optional[PiiScanner] = None,
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
        api_key: str = "",
        feature: Optional[str] = None,
        config: Optional[ModelCostConfig] = None,
        session: Optional[SessionContext] = None,
    ) -> None:
        self._mc_client = mc_client
        self._tracker = tracker
        self._budget_manager = budget_manager
        self._pii_scanner = pii_scanner
        self._rate_limiter = rate_limiter
        self._api_key = api_key
        self._feature = feature
        self._config = config
        self._session = session

    def wrap(self, client: Any) -> Any:
        """Return a proxy that instruments ``client.chat.completions.create()``."""
        completions_proxy = _ChatCompletionsProxy(
            client.chat.completions,
            mc_client=self._mc_client,
            config=self._config,
            tracker=self._tracker,
            budget_manager=self._budget_manager,
            pii_scanner=self._pii_scanner,
            rate_limiter=self._rate_limiter,
            api_key=self._api_key,
            feature=self._feature,
            session=self._session,
        )
        chat_proxy = _ChatProxy(client.chat, completions_proxy)
        return _OpenAIClientProxy(client, chat_proxy)

    def extract_usage(self, response: Any) -> Tuple[int, int]:
        return self.extract_usage_static(response)

    @staticmethod
    def extract_usage_static(response: Any) -> Tuple[int, int]:
        """Extract token counts from an OpenAI response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return (0, 0)
        return (
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )

    def get_provider_name(self) -> str:
        return "openai"

    def get_model_name(self, kwargs: dict[str, Any]) -> str:
        return str(kwargs.get("model", "unknown"))
