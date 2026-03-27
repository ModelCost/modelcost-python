"""Tests for cache token extraction across all providers.

Uses SimpleNamespace objects that mirror real provider response structures
to verify extract_usage_detailed() handles cache tokens correctly.
"""

from __future__ import annotations

from types import SimpleNamespace

from modelcost.providers.anthropic import AnthropicProvider
from modelcost.providers.google import GoogleProvider
from modelcost.providers.openai import OpenAIProvider

# ===================== Anthropic Tests =====================


class TestAnthropicExtractUsageDetailed:

    def test_with_cache_tokens(self):
        """Anthropic returns cache tokens as separate fields; input_tokens already excludes cache."""
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=150,
                output_tokens=300,
                cache_creation_input_tokens=1000,
                cache_read_input_tokens=500,
            )
        )
        result = AnthropicProvider.extract_usage_detailed(response)
        assert result == (150, 300, 1000, 500)

    def test_without_cache_tokens(self):
        """Responses without caching should return 0 for cache fields."""
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=200,
                output_tokens=100,
            )
        )
        result = AnthropicProvider.extract_usage_detailed(response)
        assert result == (200, 100, 0, 0)

    def test_no_usage_object(self):
        """Missing usage returns all zeros."""
        response = SimpleNamespace()
        result = AnthropicProvider.extract_usage_detailed(response)
        assert result == (0, 0, 0, 0)

    def test_backward_compat_static(self):
        """extract_usage_static() still returns 2-tuple for backward compat."""
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=150,
                output_tokens=300,
                cache_creation_input_tokens=1000,
                cache_read_input_tokens=500,
            )
        )
        result = AnthropicProvider.extract_usage_static(response)
        assert result == (150, 300)


# ===================== OpenAI Tests =====================


class TestOpenAIExtractUsageDetailed:

    def test_with_cached_tokens(self):
        """OpenAI includes cached tokens in prompt_tokens — subtract to get regular input."""
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=1650,
                completion_tokens=300,
                total_tokens=1950,
                prompt_tokens_details=SimpleNamespace(cached_tokens=500),
            )
        )
        result = OpenAIProvider.extract_usage_detailed(response)
        # regular_input = 1650 - 500 = 1150
        assert result == (1150, 300, 0, 500)

    def test_without_prompt_tokens_details(self):
        """No details object means no cached tokens."""
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=200,
            )
        )
        result = OpenAIProvider.extract_usage_detailed(response)
        assert result == (1000, 200, 0, 0)

    def test_cached_exceeds_prompt(self):
        """Guard: cached_tokens > prompt_tokens should not produce negative input."""
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                prompt_tokens_details=SimpleNamespace(cached_tokens=200),
            )
        )
        result = OpenAIProvider.extract_usage_detailed(response)
        assert result == (0, 50, 0, 200)  # max(0, 100-200) = 0

    def test_no_usage_object(self):
        """Missing usage returns all zeros."""
        response = SimpleNamespace()
        result = OpenAIProvider.extract_usage_detailed(response)
        assert result == (0, 0, 0, 0)

    def test_cached_tokens_none(self):
        """prompt_tokens_details exists but cached_tokens is None."""
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=500,
                completion_tokens=100,
                prompt_tokens_details=SimpleNamespace(cached_tokens=None),
            )
        )
        result = OpenAIProvider.extract_usage_detailed(response)
        assert result == (500, 100, 0, 0)


# ===================== Google Tests =====================


class TestGoogleExtractUsageDetailed:

    def test_with_cached_content(self):
        """Google includes cached in prompt_token_count — subtract to get regular input."""
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=2000,
                candidates_token_count=400,
                cached_content_token_count=800,
            )
        )
        result = GoogleProvider.extract_usage_detailed(response)
        # regular_input = 2000 - 800 = 1200
        assert result == (1200, 400, 0, 800)

    def test_without_cached_content(self):
        """No cached content means all input is regular-rate."""
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=1000,
                candidates_token_count=300,
            )
        )
        result = GoogleProvider.extract_usage_detailed(response)
        assert result == (1000, 300, 0, 0)

    def test_cached_exceeds_prompt(self):
        """Guard: cached > prompt should not produce negative input."""
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=50,
                candidates_token_count=100,
                cached_content_token_count=200,
            )
        )
        result = GoogleProvider.extract_usage_detailed(response)
        assert result == (0, 100, 0, 200)  # max(0, 50-200) = 0

    def test_no_usage_metadata(self):
        """Missing usage_metadata returns all zeros."""
        response = SimpleNamespace()
        result = GoogleProvider.extract_usage_detailed(response)
        assert result == (0, 0, 0, 0)
