"""Bounded-concurrency tests.

The claims under test are the ones the README makes: the limit is actually
enforced, results stay in input order, and queue time is separated from
service time so a latency number cannot hide which one dominates.
"""

from __future__ import annotations

import asyncio

import pytest

from llm_client_kit.concurrency import RunStats, Timing, bounded_map


@pytest.mark.asyncio
async def test_concurrency_limit_is_actually_enforced():
    peak = 0
    live = 0

    async def work(i: int) -> int:
        nonlocal peak, live
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return i

    await bounded_map(work, range(40), limit=5)
    assert peak <= 5, f"limit breached: {peak} concurrent"
    assert peak == 5, "limit never reached -- the test is not exercising it"


@pytest.mark.asyncio
async def test_results_keep_input_order_despite_varying_duration():
    """Deliberately make later items finish first. Silently reordered
    results are a genuinely nasty class of bug."""

    async def work(i: int) -> int:
        await asyncio.sleep(0.02 - i * 0.002)
        return i * 10

    out, _ = await bounded_map(work, range(10), limit=10)
    assert out == [i * 10 for i in range(10)]


@pytest.mark.asyncio
async def test_one_failure_does_not_sink_the_batch():
    async def work(i: int) -> int:
        if i == 3:
            raise ValueError("boom")
        return i

    out, stats = await bounded_map(work, range(6), limit=3)
    assert out[3] is None
    assert stats.errors == 1
    assert [o for o in out if o is not None] == [0, 1, 2, 4, 5]


@pytest.mark.asyncio
async def test_queue_time_dominates_when_limit_is_the_bottleneck():
    """The measurement that justifies the whole module: at a tight limit the
    wait is admission, not compute. A single total-latency number would hide
    that -- and the two are fixed by opposite actions."""

    async def work(_: int) -> int:
        await asyncio.sleep(0.02)
        return 1

    _, tight = await bounded_map(work, range(32), limit=2)
    _, loose = await bounded_map(work, range(32), limit=32)

    t, lo = tight.summary(), loose.summary()
    assert t["p95_queued_s"] > t["p95_service_s"], "expected admission-bound"
    assert lo["p95_queued_s"] < t["p95_queued_s"], "raising the limit must cut queueing"
    # Service time is a property of the work, not of the limit.
    assert t["p95_service_s"] == pytest.approx(lo["p95_service_s"], abs=0.02)


@pytest.mark.asyncio
async def test_empty_input_is_not_a_special_case():
    out, stats = await bounded_map(lambda x: asyncio.sleep(0, x), [], limit=4)
    assert out == []
    assert stats.summary()["n"] == 0


def test_percentile_is_nearest_rank_and_documented():
    """Percentile definitions differ; a silent choice makes numbers
    unreproducible. Pin the one we use."""
    s = RunStats(timings=[Timing(service_s=float(i)) for i in range(1, 101)])
    out = s.summary()
    assert out["p50_total_s"] == pytest.approx(50.0)
    assert out["p95_total_s"] == pytest.approx(95.0)
    assert out["p99_total_s"] == pytest.approx(99.0)


def test_timing_total_is_queue_plus_service():
    assert Timing(queued_s=1.5, service_s=0.5).total_s == 2.0
