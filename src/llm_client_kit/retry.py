"""Retry policy: which failures are worth retrying, and how long to wait.

Two decisions here are the ones that matter in production.

**Full jitter, not equal jitter.** When N clients retry a shared dependency
after an outage they synchronise, and the retry storm is often what keeps the
dependency down. Full jitter -- sleeping uniformly in [0, backoff] rather than
backoff/2 + random -- spreads the herd widest. It is the AWS architecture-blog
recommendation and it costs nothing.

**Deadline propagation.** A caller with a 10s budget and 3 retries at 4s each
would otherwise wait 12s. The deadline is checked before each sleep AND
truncates the sleep, so the budget is honoured rather than merely intended.
This is the difference between a timeout and a suggestion.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

# 408 request timeout, 409 conflict, 429 rate limited, 5xx server-side.
# NOT retried: 400 (malformed -- will fail identically), 401/403 (bad key),
# 404 (wrong model name), 422 (schema violation). Retrying a deterministic
# client error just burns the budget and delays the real error.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class RetryBudgetExceeded(RuntimeError):
    """Raised when the deadline would be exceeded by another attempt."""


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    # Full jitter by default. Set False only to make tests deterministic.
    jitter: bool = True

    def should_retry(self, attempt: int, status: int | None, exc: Exception | None) -> bool:
        if attempt >= self.max_retries:
            return False
        if status is not None:
            return status in RETRYABLE_STATUS
        # Transport-level failures (connection reset, DNS, read timeout) are
        # retryable; programming errors are not.
        return isinstance(exc, (TimeoutError, ConnectionError, OSError))

    def delay_for(self, attempt: int, retry_after: float | None = None,
                  rng: random.Random | None = None) -> float:
        """Backoff for the given attempt (0-indexed).

        A server-provided Retry-After always wins: the server knows its own
        capacity better than our exponential guess does. Ignoring it is how
        you get rate-limited twice.
        """
        if retry_after is not None:
            return min(retry_after, self.max_delay_s)
        exp = min(self.base_delay_s * (2 ** attempt), self.max_delay_s)
        if not self.jitter:
            return exp
        r = rng or random
        return r.uniform(0.0, exp)


class Deadline:
    """A wall-clock budget for one logical operation, shared across retries."""

    def __init__(self, total_s: float | None, clock=time.monotonic) -> None:
        self._clock = clock
        self._total = total_s
        self._start = clock()

    @property
    def remaining(self) -> float:
        if self._total is None:
            return float("inf")
        return max(0.0, self._total - (self._clock() - self._start))

    @property
    def expired(self) -> bool:
        return self.remaining <= 0.0

    def clamp(self, seconds: float) -> float:
        """Truncate a sleep so it cannot overrun the budget."""
        return min(seconds, self.remaining)

    def check(self) -> None:
        if self.expired:
            raise RetryBudgetExceeded(f"deadline of {self._total}s exhausted")
