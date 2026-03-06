"""Agent session governance with local-first enforcement."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("modelcost")


@dataclass
class SessionCallRecord:
    """Record of a single call within a session."""

    call_sequence: int
    call_type: str  # "llm" | "tool"
    tool_name: Optional[str]
    input_tokens: int
    output_tokens: int
    cumulative_input_tokens: int
    cost_usd: float
    cumulative_cost_usd: float
    pii_detected: bool
    created_at: datetime


class SessionContext:
    """Thread-safe in-memory session state with local-first budget enforcement.

    All limit checks are performed against local counters (sub-microsecond).
    Server sync is fire-and-forget for durability.
    """

    def __init__(
        self,
        *,
        session_id: str,
        server_session_id: Optional[str],
        feature: Optional[str],
        user_id: Optional[str],
        max_spend_usd: Optional[float],
        max_iterations: Optional[int],
    ) -> None:
        self.session_id = session_id
        self.server_session_id = server_session_id
        self.feature = feature
        self.user_id = user_id
        self.max_spend_usd = max_spend_usd
        self.max_iterations = max_iterations

        # Mutable state — guarded by _lock
        self._lock = threading.Lock()
        self._current_spend_usd: float = 0.0
        self._iteration_count: int = 0
        self._cumulative_input_tokens: int = 0
        self._status: str = "active"
        self._termination_reason: Optional[str] = None
        self._calls: list[SessionCallRecord] = []
        self._started_at: datetime = datetime.now(timezone.utc)

    # ---- Read-only properties ----

    @property
    def current_spend_usd(self) -> float:
        with self._lock:
            return self._current_spend_usd

    @property
    def iteration_count(self) -> int:
        with self._lock:
            return self._iteration_count

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def remaining_budget(self) -> Optional[float]:
        if self.max_spend_usd is None:
            return None
        with self._lock:
            return self.max_spend_usd - self._current_spend_usd

    @property
    def remaining_iterations(self) -> Optional[int]:
        if self.max_iterations is None:
            return None
        with self._lock:
            return self.max_iterations - self._iteration_count

    # ---- Core enforcement (called BEFORE each API call) ----

    def pre_call_check(self, estimated_cost: float = 0.0) -> None:
        """Check session limits before an API call.

        Raises SessionBudgetExceeded or SessionIterationLimitExceeded.
        Pure local check — no network calls, sub-microsecond.
        """
        from modelcost.exceptions import (
            SessionBudgetExceeded,
            SessionIterationLimitExceeded,
        )

        with self._lock:
            if self._status != "active":
                raise SessionBudgetExceeded(
                    message=f"Session '{self.session_id}' is {self._status}",
                    session_id=self.session_id,
                    current_spend=self._current_spend_usd,
                    max_spend=self.max_spend_usd,
                )

            # Iteration limit check
            if self.max_iterations is not None:
                if self._iteration_count >= self.max_iterations:
                    self._status = "terminated"
                    self._termination_reason = "iteration_limit"
                    raise SessionIterationLimitExceeded(
                        message=(
                            f"Session '{self.session_id}' reached iteration limit "
                            f"({self._iteration_count}/{self.max_iterations})"
                        ),
                        session_id=self.session_id,
                        current_iterations=self._iteration_count,
                        max_iterations=self.max_iterations,
                    )

            # Budget limit check
            if self.max_spend_usd is not None:
                projected = self._current_spend_usd + estimated_cost
                if projected > self.max_spend_usd:
                    self._status = "terminated"
                    self._termination_reason = "budget"
                    raise SessionBudgetExceeded(
                        message=(
                            f"Session '{self.session_id}' would exceed budget "
                            f"(${projected:.4f} > ${self.max_spend_usd:.2f})"
                        ),
                        session_id=self.session_id,
                        current_spend=self._current_spend_usd,
                        max_spend=self.max_spend_usd,
                    )

    # ---- Post-call recording (called AFTER each successful API call) ----

    def record_call(
        self,
        *,
        call_type: str = "llm",
        tool_name: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        pii_detected: bool = False,
    ) -> SessionCallRecord:
        """Record a completed call. Updates local counters atomically."""
        with self._lock:
            self._iteration_count += 1
            self._current_spend_usd += cost_usd
            self._cumulative_input_tokens += input_tokens

            record = SessionCallRecord(
                call_sequence=self._iteration_count,
                call_type=call_type,
                tool_name=tool_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cumulative_input_tokens=self._cumulative_input_tokens,
                cost_usd=cost_usd,
                cumulative_cost_usd=self._current_spend_usd,
                pii_detected=pii_detected,
                created_at=datetime.now(timezone.utc),
            )
            self._calls.append(record)
            return record

    # ---- Lifecycle ----

    def close(self, reason: str = "completed") -> None:
        """Mark session as closed."""
        with self._lock:
            if self._status == "active":
                self._status = "completed" if reason == "completed" else "terminated"
                self._termination_reason = reason if reason != "completed" else None
