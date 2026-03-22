"""Asynchronous HTTP client for the ModelCost API with circuit-breaker logic."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import httpx

from modelcost._version import __version__
from modelcost.exceptions import ModelCostApiError
from modelcost.models.budget import BudgetCheckResponse, BudgetStatusResponse
from modelcost.models.governance import GovernanceScanRequest, GovernanceScanResponse
from modelcost.models.track import TrackRequest, TrackResponse

if TYPE_CHECKING:
    from modelcost.config import ModelCostConfig

logger = logging.getLogger("modelcost")

_CIRCUIT_FAILURE_THRESHOLD = 3
_CIRCUIT_OPEN_DURATION_SECONDS = 60.0


class AsyncModelCostClient:
    """Asynchronous client that communicates with the ModelCost REST API.

    Mirrors :class:`ModelCostClient` but uses ``httpx.AsyncClient``.
    """

    def __init__(self, config: ModelCostConfig) -> None:
        self._config = config
        self._http = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=5.0,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"modelcost-python/{__version__}",
            },
        )

        # Circuit-breaker state
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0

    # ------------------------------------------------------------------
    # Circuit breaker helpers
    # ------------------------------------------------------------------

    def _is_circuit_open(self) -> bool:
        if self._consecutive_failures < _CIRCUIT_FAILURE_THRESHOLD:
            return False
        if time.monotonic() >= self._circuit_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self) -> None:
        self._consecutive_failures = 0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= _CIRCUIT_FAILURE_THRESHOLD:
            self._circuit_open_until = time.monotonic() + _CIRCUIT_OPEN_DURATION_SECONDS
            logger.warning(
                "Circuit breaker opened after %d consecutive failures; "
                "will retry after %.0fs",
                self._consecutive_failures,
                _CIRCUIT_OPEN_DURATION_SECONDS,
            )

    # ------------------------------------------------------------------
    # Internal request helper
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,  # type: ignore[type-arg]
        params: dict | None = None,  # type: ignore[type-arg]
    ) -> httpx.Response:
        if self._is_circuit_open():
            raise ModelCostApiError(
                message="Circuit breaker is open — requests are temporarily blocked",
                status_code=503,
                error="circuit_open",
            )

        try:
            response = await self._http.request(method, path, json=json, params=params)
            if response.status_code >= 400:
                self._record_failure()
                body = response.json() if response.content else {}
                raise ModelCostApiError(
                    message=body.get("message", response.reason_phrase or "Unknown error"),
                    status_code=response.status_code,
                    error=body.get("error", "unknown"),
                )
            self._record_success()
            return response
        except ModelCostApiError:
            raise
        except Exception as exc:
            self._record_failure()
            if self._config.fail_open:
                logger.warning("ModelCost API request failed (fail_open=True): %s", exc)
                raise
            raise ModelCostApiError(
                message=str(exc),
                status_code=0,
                error="connection_error",
            ) from exc

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def track(self, request: TrackRequest) -> TrackResponse:
        """Send a cost-tracking event to the API."""
        try:
            resp = await self._request(
                "POST",
                "/v1/track",
                json=request.model_dump(by_alias=True, mode="json"),
            )
            return TrackResponse.model_validate(resp.json())
        except Exception:
            if self._config.fail_open:
                logger.warning("track() failed (fail_open=True); returning synthetic OK")
                return TrackResponse(status="ok")
            raise

    async def check_budget(
        self,
        org_id: str,
        feature: str | None = None,
        estimated_cost: float | None = None,
    ) -> BudgetCheckResponse:
        """Check whether a planned request is within budget."""
        params: dict[str, object] = {"org_id": org_id}
        if feature is not None:
            params["feature"] = feature
        if estimated_cost is not None:
            params["estimated_cost"] = estimated_cost
        try:
            resp = await self._request("GET", "/v1/budget/check", params=params)
            return BudgetCheckResponse.model_validate(resp.json())
        except Exception:
            if self._config.fail_open:
                logger.warning("check_budget() failed (fail_open=True); returning allowed")
                return BudgetCheckResponse(allowed=True)
            raise

    async def scan_text(self, request: GovernanceScanRequest) -> GovernanceScanResponse:
        """Scan text for governance violations (PII, etc.)."""
        try:
            resp = await self._request(
                "POST",
                "/v1/governance/scan",
                json=request.model_dump(by_alias=True, mode="json"),
            )
            return GovernanceScanResponse.model_validate(resp.json())
        except Exception:
            if self._config.fail_open:
                logger.warning("scan_text() failed (fail_open=True); returning clean")
                return GovernanceScanResponse(is_allowed=True, violations=[])
            raise

    async def get_budget_status(self, org_id: str) -> BudgetStatusResponse:
        """Retrieve the full budget status for an organisation."""
        try:
            resp = await self._request("GET", f"/v1/budget/status/{org_id}")
            return BudgetStatusResponse.model_validate(resp.json())
        except Exception:
            if self._config.fail_open:
                logger.warning("get_budget_status() failed (fail_open=True); returning empty")
                return BudgetStatusResponse(
                    policies=[],
                    total_budget_usd=0.0,
                    total_spend_usd=0.0,
                    policies_at_risk=0,
                )
            raise

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()
