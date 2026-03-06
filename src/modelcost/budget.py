"""Local budget manager with caching and optimistic updates."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

from modelcost.models.budget import BudgetCheckResponse, BudgetStatusResponse

if TYPE_CHECKING:
    from modelcost.client import ModelCostClient

logger = logging.getLogger("modelcost")


class BudgetManager:
    """Manages a local cache of budget state and syncs with the API periodically.

    Thread-safe: all access to the internal cache and timestamps is guarded
    by a :class:`threading.Lock`.
    """

    def __init__(
        self,
        org_id: str,
        sync_interval: float = 10.0,
    ) -> None:
        self._org_id = org_id
        self.sync_interval = sync_interval

        self._cache: dict[str, BudgetStatusResponse] = {}
        self._last_sync: float = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def check(
        self,
        client: ModelCostClient,
        feature: Optional[str] = None,
        estimated_cost: Optional[float] = None,
    ) -> BudgetCheckResponse:
        """Check the local cache and return whether the request is allowed.

        If the cache is stale (older than *sync_interval* seconds) a
        background sync is triggered first.
        """
        with self._lock:
            now = time.monotonic()
            if now - self._last_sync >= self.sync_interval:
                self._sync_locked(client)

            scope_key = feature or "__org__"
            status = self._cache.get(scope_key)
            if status is None:
                # No cached data for this scope — allow by default
                return BudgetCheckResponse(allowed=True)

            # Evaluate against policies
            for policy in status.policies:
                if not policy.is_active:
                    continue

                remaining = policy.budget_amount_usd - policy.current_spend_usd
                if estimated_cost is not None and remaining < estimated_cost:
                    return BudgetCheckResponse(
                        allowed=False,
                        action=policy.action,
                        throttle_percentage=(
                            policy.throttle_percentage
                            if policy.action == "throttle"
                            else None
                        ),
                        reason=(
                            f"Budget for scope '{scope_key}' would be exceeded. "
                            f"Remaining: ${remaining:.2f}, estimated cost: ${estimated_cost:.2f}"
                        ),
                    )

            return BudgetCheckResponse(allowed=True)

    def sync(self, client: ModelCostClient) -> None:
        """Fetch the latest budget status from the API (thread-safe)."""
        with self._lock:
            self._sync_locked(client)

    def update_local_spend(self, feature: Optional[str], cost: float) -> None:
        """Optimistically add *cost* to the local cached spend."""
        with self._lock:
            scope_key = feature or "__org__"
            status = self._cache.get(scope_key)
            if status is None:
                return
            # Update every active policy in-place
            for policy in status.policies:
                if policy.is_active:
                    policy.current_spend_usd += cost
                    if policy.budget_amount_usd > 0:
                        policy.spend_percentage = (
                            policy.current_spend_usd / policy.budget_amount_usd * 100.0
                        )
            status.total_spend_usd += cost

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sync_locked(self, client: ModelCostClient) -> None:
        """Must be called while holding ``self._lock``."""
        try:
            status = client.get_budget_status(self._org_id)
            # Index by feature scope — fall back to org-level key
            for policy in status.policies:
                key = policy.scope_identifier or "__org__"
                if key not in self._cache:
                    self._cache[key] = BudgetStatusResponse(
                        policies=[],
                        total_budget_usd=status.total_budget_usd,
                        total_spend_usd=status.total_spend_usd,
                        policies_at_risk=status.policies_at_risk,
                    )
                self._cache[key].policies.append(policy)

            # Also store under the org key for global lookups
            self._cache["__org__"] = status
            self._last_sync = time.monotonic()
        except Exception:
            logger.warning("Budget sync failed; will retry next interval", exc_info=True)
