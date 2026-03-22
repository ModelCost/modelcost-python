"""Tests for modelcost.providers.openai (OpenAIProvider)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from modelcost.client import ModelCostClient
from modelcost.providers.openai import OpenAIProvider
from modelcost.tracking import CostTracker

if TYPE_CHECKING:
    from modelcost.config import ModelCostConfig


@pytest.fixture()
def tracker() -> CostTracker:
    return CostTracker(api_key="mc_test_key_12345", batch_size=100)


@pytest.fixture()
def mc_client(config: ModelCostConfig) -> MagicMock:
    """Return a mock ModelCostClient."""
    return MagicMock(spec=ModelCostClient)


@pytest.fixture()
def provider(mc_client: MagicMock, tracker: CostTracker) -> OpenAIProvider:
    return OpenAIProvider(
        mc_client=mc_client,
        tracker=tracker,
        api_key="mc_test_key_12345",
    )


class TestOpenAIProviderMeta:
    """Tests for provider metadata."""

    def test_provider_name(self, provider: OpenAIProvider) -> None:
        assert provider.get_provider_name() == "openai"

    def test_model_name_extraction(self, provider: OpenAIProvider) -> None:
        name = provider.get_model_name({"model": "gpt-4o"})
        assert name == "gpt-4o"


class TestOpenAIWrap:
    """Tests for the wrap() proxy."""

    def test_wrap_returns_proxy(self, provider: OpenAIProvider) -> None:
        # Build a mock OpenAI client structure
        mock_create = MagicMock()
        mock_completions = MagicMock()
        mock_completions.create = mock_create
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_client = MagicMock()
        mock_client.chat = mock_chat

        proxy = provider.wrap(mock_client)

        # The proxy should have the same shape
        assert hasattr(proxy, "chat")
        assert hasattr(proxy.chat, "completions")
        assert hasattr(proxy.chat.completions, "create")


class TestTokenExtraction:
    """Tests for extract_usage()."""

    def test_extract_usage_from_mock_response(self, provider: OpenAIProvider) -> None:
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 150
        mock_usage.completion_tokens = 50

        mock_response = MagicMock()
        mock_response.usage = mock_usage

        input_tokens, output_tokens = provider.extract_usage(mock_response)
        assert input_tokens == 150
        assert output_tokens == 50

    def test_extract_usage_no_usage_attr(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock(spec=[])  # no attributes at all
        input_tokens, output_tokens = provider.extract_usage(mock_response)
        assert input_tokens == 0
        assert output_tokens == 0
