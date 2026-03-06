"""Tests for modelcost.rate_limiter (TokenBucketRateLimiter)."""

from __future__ import annotations

import time

import pytest

from modelcost.exceptions import RateLimitedError
from modelcost.rate_limiter import TokenBucketRateLimiter


class TestTokenBucket:
    """Tests for the token-bucket rate limiter."""

    def test_allows_within_burst(self) -> None:
        limiter = TokenBucketRateLimiter(rate=10.0, burst=5)
        for _ in range(5):
            assert limiter.allow() is True

    def test_rejects_when_exhausted(self) -> None:
        limiter = TokenBucketRateLimiter(rate=10.0, burst=2)
        assert limiter.allow() is True
        assert limiter.allow() is True
        assert limiter.allow() is False

    def test_strict_mode_raises(self) -> None:
        limiter = TokenBucketRateLimiter(rate=10.0, burst=1)
        assert limiter.allow(strict=True) is True
        with pytest.raises(RateLimitedError) as exc_info:
            limiter.allow(strict=True)
        assert exc_info.value.retry_after_seconds > 0
        assert exc_info.value.limit_dimension == "token_bucket"

    def test_tokens_refill_over_time(self) -> None:
        limiter = TokenBucketRateLimiter(rate=100.0, burst=1)
        # Exhaust the single token
        assert limiter.allow() is True
        assert limiter.allow() is False

        # Wait long enough for at least one token to refill (100/s = 10ms per token)
        time.sleep(0.05)

        assert limiter.allow() is True

    def test_wait_blocks_until_available(self) -> None:
        limiter = TokenBucketRateLimiter(rate=100.0, burst=1)
        assert limiter.allow() is True  # exhaust

        start = time.monotonic()
        limiter.wait()  # should block briefly then succeed
        elapsed = time.monotonic() - start

        # Should have waited roughly 10ms (1 token / 100 per second)
        assert elapsed < 0.5  # generous upper bound
