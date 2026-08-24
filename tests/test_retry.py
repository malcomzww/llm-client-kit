"""Retry policy tests.

These encode decisions, not implementation details: which statuses are worth
retrying, that Retry-After wins over our own backoff, and that a deadline is
actually enforced rather than merely declared.
"""

from __future__ import annotations

import random

import pytest

from llm_client_kit.retry import (
    Deadline,
    RetryBudgetExceeded,
    RetryPolicy,
)


class FakeClock:
    """Controllable monotonic clock so deadline tests need no sleeping."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# --- which failures are worth retrying ---------------------------------


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504, 529])
def test_transient_statuses_are_retried(status):
    assert RetryPolicy().should_retry(0, status, None)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_deterministic_client_errors_are_not_retried(status):
    """Retrying a malformed request or a bad key burns budget and delays the
    real error. The failure is deterministic; the retry cannot help."""
    assert not RetryPolicy().should_retry(0, status, None)


def test_transport_errors_are_retried_but_bugs_are_not():
    p = RetryPolicy()
    assert p.should_retry(0, None, ConnectionError("reset"))
    assert p.should_retry(0, None, TimeoutError("read timeout"))
    assert not p.should_retry(0, None, ValueError("bad argument"))


def test_retries_stop_at_the_limit():
    p = RetryPolicy(max_retries=2)
    assert p.should_retry(0, 503, None)
    assert p.should_retry(1, 503, None)
    assert not p.should_retry(2, 503, None)


# --- backoff shape -----------------------------------------------------


def test_backoff_is_exponential_and_capped():
    p = RetryPolicy(base_delay_s=1.0, max_delay_s=8.0, jitter=False)
    assert [p.delay_for(i) for i in range(6)] == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_full_jitter_spans_the_whole_window():
    """Full jitter samples [0, backoff), not backoff/2 + noise. The wider
    spread is what actually disperses a retry storm."""
    p = RetryPolicy(base_delay_s=1.0, jitter=True)
    rng = random.Random(0)
    samples = [p.delay_for(3, rng=rng) for _ in range(400)]
    assert min(samples) < 1.0, "never sampled the low end -- not full jitter"
    assert max(samples) > 7.0, "never sampled the high end"
    assert all(0.0 <= s <= 8.0 for s in samples)


def test_server_retry_after_overrides_our_guess():
    """The server knows its own capacity. Ignoring Retry-After is how you get
    rate-limited a second time."""
    p = RetryPolicy(base_delay_s=1.0, max_delay_s=30.0)
    assert p.delay_for(0, retry_after=12.0) == 12.0


def test_retry_after_is_still_capped():
    p = RetryPolicy(max_delay_s=10.0)
    assert p.delay_for(0, retry_after=600.0) == 10.0


# --- deadline propagation ---------------------------------------------


def test_deadline_counts_down():
    clk = FakeClock()
    d = Deadline(10.0, clock=clk)
    assert d.remaining == 10.0
    clk.advance(4.0)
    assert d.remaining == 6.0
    assert not d.expired


def test_deadline_truncates_a_sleep_that_would_overrun():
    """The reason this class exists: 3 retries x 4s against a 10s budget
    would wait 12s. Clamping makes the budget real."""
    clk = FakeClock()
    d = Deadline(10.0, clock=clk)
    clk.advance(8.0)
    assert d.clamp(4.0) == pytest.approx(2.0)


def test_expired_deadline_raises_rather_than_sleeping():
    clk = FakeClock()
    d = Deadline(1.0, clock=clk)
    clk.advance(1.5)
    assert d.expired
    with pytest.raises(RetryBudgetExceeded):
        d.check()


def test_no_deadline_means_unbounded():
    d = Deadline(None)
    assert d.remaining == float("inf")
    assert not d.expired
    d.check()
