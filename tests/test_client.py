"""Tests for modelcost.client (sync client + circuit breaker)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pytest

from modelcost.client import ModelCostClient
from modelcost.exceptions import ModelCostApiError
from modelcost.models.track import TrackRequest

if TYPE_CHECKING:
    import respx

    from modelcost.config import ModelCostConfig


@pytest.fixture()
def mc_client(config: ModelCostConfig) -> ModelCostClient:
    return ModelCostClient(config)


class TestTrack:
    """Tests for the track() method."""

    def test_successful_track(
        self,
        mc_client: ModelCostClient,
        mock_client: respx.MockRouter,
        track_response_fixture: dict[str, Any],
    ) -> None:
        mock_client.post("/v1/track").respond(200, json=track_response_fixture)

        req = TrackRequest(
            api_key="mc_test_abc123def456ghi789jkl012mno345pqr678stu901vwx",
            timestamp=datetime.now(timezone.utc),
            provider="openai",
            model="gpt-4o",
            input_tokens=150,
            output_tokens=50,
        )
        resp = mc_client.track(req)
        assert resp.status == "ok"


class TestBudgetCheck:
    """Tests for the check_budget() method."""

    def test_budget_allowed(
        self,
        mc_client: ModelCostClient,
        mock_client: respx.MockRouter,
        budget_check_allowed_fixture: dict[str, Any],
    ) -> None:
        mock_client.get("/v1/budget/check").respond(200, json=budget_check_allowed_fixture)
        resp = mc_client.check_budget("org_test_123", feature="chatbot")
        assert resp.allowed is True
        assert resp.action is None

    def test_budget_blocked(
        self,
        mc_client: ModelCostClient,
        mock_client: respx.MockRouter,
        budget_check_blocked_fixture: dict[str, Any],
    ) -> None:
        mock_client.get("/v1/budget/check").respond(200, json=budget_check_blocked_fixture)
        resp = mc_client.check_budget("org_test_123", feature="chatbot")
        assert resp.allowed is False
        assert resp.action == "block"
        assert resp.reason is not None


class TestCircuitBreaker:
    """Tests for circuit-breaker behaviour."""

    def test_circuit_opens_after_three_failures(
        self,
        config: ModelCostConfig,
        mock_client: respx.MockRouter,
    ) -> None:
        # Disable fail_open so exceptions propagate
        config_strict = config.model_copy(update={"fail_open": False})
        client = ModelCostClient(config_strict)

        mock_client.post("/v1/track").respond(500, json={"error": "internal", "message": "boom"})

        req = TrackRequest(
            api_key="mc_test_abc123def456ghi789jkl012mno345pqr678stu901vwx",
            timestamp=datetime.now(timezone.utc),
            provider="openai",
            model="gpt-4o",
            input_tokens=10,
            output_tokens=5,
        )

        # First three failures should raise ModelCostApiError with status 500
        for _ in range(3):
            with pytest.raises(ModelCostApiError) as exc_info:
                client.track(req)
            assert exc_info.value.status_code == 500

        # Fourth call should get circuit-open error (503)
        with pytest.raises(ModelCostApiError) as exc_info:
            client.track(req)
        assert exc_info.value.status_code == 503
        assert exc_info.value.error == "circuit_open"

    def test_fail_open_returns_synthetic_ok(
        self,
        mc_client: ModelCostClient,
        mock_client: respx.MockRouter,
    ) -> None:
        mock_client.post("/v1/track").respond(500, json={"error": "internal", "message": "boom"})

        req = TrackRequest(
            api_key="mc_test_abc123def456ghi789jkl012mno345pqr678stu901vwx",
            timestamp=datetime.now(timezone.utc),
            provider="openai",
            model="gpt-4o",
            input_tokens=10,
            output_tokens=5,
        )

        # With fail_open=True, should get a synthetic OK
        resp = mc_client.track(req)
        assert resp.status == "ok"
