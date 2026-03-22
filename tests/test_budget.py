"""Tests for modelcost.budget (BudgetManager)."""

from __future__ import annotations

from unittest.mock import MagicMock

from modelcost.budget import BudgetManager
from modelcost.models.budget import BudgetPolicy, BudgetStatusResponse


def _make_status(
    spend: float = 7234.50,
    budget: float = 10000.0,
) -> BudgetStatusResponse:
    """Helper to create a BudgetStatusResponse for testing."""
    policy = BudgetPolicy(
        id="test-policy-id",
        name="Monthly Org Budget",
        scope="organization",
        scope_identifier=None,
        budget_amount_usd=budget,
        period="monthly",
        custom_period_days=None,
        action="block",
        throttle_percentage=None,
        alert_thresholds=[50, 80, 100],
        current_spend_usd=spend,
        spend_percentage=(spend / budget * 100) if budget > 0 else 0,
        period_start="2026-02-01T00:00:00Z",
        is_active=True,
        created_at="2026-01-15T10:00:00Z",
        updated_at="2026-02-16T12:00:00Z",
    )
    return BudgetStatusResponse(
        policies=[policy],
        total_budget_usd=budget,
        total_spend_usd=spend,
        policies_at_risk=0,
    )


class TestBudgetManagerCache:
    """Tests for local cache behaviour."""

    def test_check_returns_allowed_when_within_budget(self) -> None:
        manager = BudgetManager(org_id="org_1", sync_interval=0.0)

        mock_client = MagicMock()
        mock_client.get_budget_status.return_value = _make_status(spend=5000.0, budget=10000.0)

        result = manager.check(mock_client, feature=None, estimated_cost=100.0)
        assert result.allowed is True

    def test_check_returns_blocked_when_over_budget(self) -> None:
        manager = BudgetManager(org_id="org_1", sync_interval=0.0)

        mock_client = MagicMock()
        mock_client.get_budget_status.return_value = _make_status(spend=9950.0, budget=10000.0)

        result = manager.check(mock_client, feature=None, estimated_cost=100.0)
        assert result.allowed is False
        assert result.action == "block"
        assert result.reason is not None

    def test_cache_sync_when_stale(self) -> None:
        manager = BudgetManager(org_id="org_1", sync_interval=0.0)

        mock_client = MagicMock()
        status = _make_status(spend=5000.0)
        mock_client.get_budget_status.return_value = status

        # First call triggers sync
        manager.check(mock_client, feature=None, estimated_cost=10.0)
        assert mock_client.get_budget_status.call_count == 1

        # Second call also triggers sync because interval is 0
        manager.check(mock_client, feature=None, estimated_cost=10.0)
        assert mock_client.get_budget_status.call_count == 2

    def test_local_spend_update(self) -> None:
        manager = BudgetManager(org_id="org_1", sync_interval=1000.0)

        mock_client = MagicMock()
        mock_client.get_budget_status.return_value = _make_status(spend=9000.0, budget=10000.0)

        # Force an initial sync
        manager.sync(mock_client)

        # Optimistically add spend
        manager.update_local_spend(None, 500.0)

        # Now check — remaining should be 10000 - 9500 = 500, so 600 should exceed
        result = manager.check(mock_client, feature=None, estimated_cost=600.0)
        assert result.allowed is False
