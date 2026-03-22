"""Google (Gemini) provider instrumentation."""

from __future__ import annotations

import contextlib
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from modelcost.models.governance import GovernanceSignalRequest
from modelcost.models.track import TrackRequest
from modelcost.providers.base import BaseProvider
from modelcost.tracking import CostTracker

if TYPE_CHECKING:
    from modelcost.budget import BudgetManager
    from modelcost.client import ModelCostClient
    from modelcost.config import ModelCostConfig
    from modelcost.pii import PiiScanner
    from modelcost.rate_limiter import TokenBucketRateLimiter
    from modelcost.session import SessionContext

logger = logging.getLogger("modelcost")


class _GoogleModelProxy:
    """Proxy for a Google ``GenerativeModel`` that intercepts ``generate_content()``."""

    def __init__(
        self,
        original_model: Any,
        *,
        mc_client: ModelCostClient,
        config: ModelCostConfig | None,
        tracker: CostTracker,
        budget_manager: BudgetManager | None,
        pii_scanner: PiiScanner | None,
        rate_limiter: TokenBucketRateLimiter | None,
        api_key: str,
        feature: str | None,
        model_name: str,
        session: SessionContext | None = None,
    ) -> None:
        self._original = original_model
        self._mc_client = mc_client
        self._config = config
        self._tracker = tracker
        self._budget_manager = budget_manager
        self._pii_scanner = pii_scanner
        self._rate_limiter = rate_limiter
        self._api_key = api_key
        self._feature = feature
        self._model_name = model_name
        self._session = session

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept a generate_content() request."""
        # 1. Rate limit
        if self._rate_limiter is not None:
            self._rate_limiter.allow(strict=True)

        # 2. PII / governance scan on text content
        if self._pii_scanner is not None and args:
            prompt = args[0]
            if isinstance(prompt, str) and prompt:
                result = self._pii_scanner.scan(prompt)
                if result.detected:
                    self._enforce_governance(prompt, result)

        # 3. Budget check
        if self._budget_manager is not None:
            estimated = CostTracker.calculate_cost(self._model_name, 500, 500)
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
            session_estimated = CostTracker.calculate_cost(self._model_name, 500, 500)
            self._session.pre_call_check(estimated_cost=session_estimated)

        # 4. Call original
        start = time.monotonic()
        response = self._original.generate_content(*args, **kwargs)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # 5. Extract usage and record
        input_tokens, output_tokens = GoogleProvider.extract_usage_static(response)
        request = TrackRequest(
            api_key=self._api_key,
            timestamp=datetime.now(timezone.utc),
            provider="google",
            model=self._model_name,
            feature=self._feature,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
        )
        self._tracker.record(request)

        # 6. Optimistic local spend update
        cost = CostTracker.calculate_cost(self._model_name, input_tokens, output_tokens)
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
            assert self._pii_scanner is not None
            full_result = self._pii_scanner.full_scan(content)
            if full_result.detected:
                from datetime import datetime, timezone

                for v in full_result.violations:
                    with contextlib.suppress(Exception):
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
                        redacted_text=gov_result.redacted_text or (self._pii_scanner.redact(content) if self._pii_scanner else content),
                    )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class GoogleProvider(BaseProvider):
    """Wraps a Google ``GenerativeModel`` to add cost tracking and governance."""

    def __init__(
        self,
        mc_client: ModelCostClient,
        tracker: CostTracker,
        budget_manager: BudgetManager | None = None,
        pii_scanner: PiiScanner | None = None,
        rate_limiter: TokenBucketRateLimiter | None = None,
        api_key: str = "",
        feature: str | None = None,
        model_name: str = "gemini-1.5-pro",
        config: ModelCostConfig | None = None,
        session: SessionContext | None = None,
    ) -> None:
        self._mc_client = mc_client
        self._tracker = tracker
        self._budget_manager = budget_manager
        self._pii_scanner = pii_scanner
        self._rate_limiter = rate_limiter
        self._api_key = api_key
        self._feature = feature
        self._model_name = model_name
        self._config = config
        self._session = session

    def wrap(self, client: Any) -> Any:
        """Return a proxy that instruments ``client.generate_content()``."""
        return _GoogleModelProxy(
            client,
            mc_client=self._mc_client,
            config=self._config,
            tracker=self._tracker,
            budget_manager=self._budget_manager,
            pii_scanner=self._pii_scanner,
            rate_limiter=self._rate_limiter,
            api_key=self._api_key,
            feature=self._feature,
            model_name=self._model_name,
            session=self._session,
        )

    def extract_usage(self, response: Any) -> tuple[int, int]:
        return self.extract_usage_static(response)

    @staticmethod
    def extract_usage_static(response: Any) -> tuple[int, int]:
        """Extract token counts from a Google Gemini response."""
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is None:
            return (0, 0)
        return (
            getattr(usage_metadata, "prompt_token_count", 0) or 0,
            getattr(usage_metadata, "candidates_token_count", 0) or 0,
        )

    def get_provider_name(self) -> str:
        return "google"

    def get_model_name(self, kwargs: dict[str, Any]) -> str:
        return self._model_name
