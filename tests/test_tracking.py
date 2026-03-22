"""Tests for modelcost.tracking (CostTracker)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from modelcost.models.cost import ModelPricing
from modelcost.models.track import TrackRequest, TrackResponse
from modelcost.tracking import MODEL_PRICING, CostTracker


@pytest.fixture(autouse=True)
def _seed_pricing():
    """Seed MODEL_PRICING for tests (normally populated from server API)."""
    MODEL_PRICING["gpt-4o"] = ModelPricing(
        provider="openai", input_cost_per_1k=0.0025, output_cost_per_1k=0.01
    )
    MODEL_PRICING["claude-sonnet-4"] = ModelPricing(
        provider="anthropic", input_cost_per_1k=0.003, output_cost_per_1k=0.015
    )
    yield
    MODEL_PRICING.clear()


def _make_request(**overrides: Any) -> TrackRequest:
    defaults = {
        "api_key": "mc_test_key_12345",
        "timestamp": datetime.now(timezone.utc),
        "provider": "openai",
        "model": "gpt-4o",
        "input_tokens": 100,
        "output_tokens": 50,
    }
    defaults.update(overrides)
    return TrackRequest(**defaults)


class TestCostCalculation:
    """Tests for calculate_cost()."""

    def test_gpt4o_cost(self) -> None:
        # gpt-4o: $0.0025/1k input, $0.01/1k output
        cost = CostTracker.calculate_cost("gpt-4o", 150, 50)
        expected_input = (150 / 1000) * 0.0025
        expected_output = (50 / 1000) * 0.01
        expected = expected_input + expected_output
        assert abs(cost - expected) < 1e-9

    def test_unknown_model_returns_zero(self) -> None:
        cost = CostTracker.calculate_cost("unknown-model-xyz", 1000, 1000)
        assert cost == 0.0

    def test_claude_sonnet_cost(self) -> None:
        # claude-sonnet-4: $0.003/1k input, $0.015/1k output
        cost = CostTracker.calculate_cost("claude-sonnet-4", 1000, 500)
        expected = (1000 / 1000) * 0.003 + (500 / 1000) * 0.015
        assert abs(cost - expected) < 1e-9

    def test_empty_pricing_returns_zero(self) -> None:
        MODEL_PRICING.clear()
        cost = CostTracker.calculate_cost("gpt-4o", 1000, 1000)
        assert cost == 0.0


class TestBufferAndFlush:
    """Tests for record() and flush()."""

    def test_record_adds_to_buffer(self) -> None:
        tracker = CostTracker(api_key="mc_test_key", batch_size=100)
        req = _make_request()
        tracker.record(req)
        assert tracker.buffer_size == 1

    def test_flush_sends_buffered_events(self) -> None:
        tracker = CostTracker(api_key="mc_test_key", batch_size=100)
        for _ in range(5):
            tracker.record(_make_request())

        mock_client = MagicMock()
        mock_client.track.return_value = TrackResponse(status="ok")

        responses = tracker.flush(mock_client)
        assert len(responses) == 5
        assert all(r.status == "ok" for r in responses)
        assert tracker.buffer_size == 0

    def test_flush_at_batch_size(self) -> None:
        tracker = CostTracker(api_key="mc_test_key", batch_size=3)
        for _ in range(3):
            tracker.record(_make_request())
        # Buffer should be at batch size
        assert tracker.buffer_size == 3


class TestTrackCostDecorator:
    """Tests for the track_cost() decorator."""

    def test_decorator_records_event(self) -> None:
        tracker = CostTracker(api_key="mc_test_key", batch_size=100)

        # Create a mock response with usage
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 200
        mock_usage.completion_tokens = 100

        mock_response = MagicMock()
        mock_response.usage = mock_usage

        @tracker.track_cost(provider="openai", model="gpt-4o", feature="test")
        def fake_call() -> Any:
            return mock_response

        result = fake_call()
        assert result is mock_response
        assert tracker.buffer_size == 1
