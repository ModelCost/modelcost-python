"""Token-bucket rate limiter."""

from __future__ import annotations

import threading
import time

from modelcost.exceptions import RateLimitedError


class TokenBucketRateLimiter:
    """A thread-safe token-bucket rate limiter.

    Parameters
    ----------
    rate:
        Tokens added per second.
    burst:
        Maximum number of tokens the bucket can hold.
    """

    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate
        self._burst = burst
        self._tokens: float = float(burst)
        self._last_refill: float = time.monotonic()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Add tokens based on elapsed time. Must be called with lock held."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allow(self, *, strict: bool = False) -> bool:
        """Consume one token and return ``True``, or ``False`` if empty.

        When *strict* is ``True``, a :class:`RateLimitedError` is raised
        instead of returning ``False``.
        """
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True

        if strict:
            with self._lock:
                # Calculate how long until next token
                deficit = 1.0 - self._tokens
                wait_seconds = deficit / self._rate if self._rate > 0 else 0.0
            raise RateLimitedError(
                message="Rate limit exceeded",
                retry_after_seconds=wait_seconds,
                limit_dimension="token_bucket",
            )
        return False

    def wait(self) -> None:
        """Block until a token is available, then consume it."""
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Calculate sleep time
                deficit = 1.0 - self._tokens
                sleep_time = deficit / self._rate if self._rate > 0 else 0.01

            time.sleep(sleep_time)
