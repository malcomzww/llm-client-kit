"""Bounded concurrency, and why unbounded gather is the wrong default.

`asyncio.gather` over 500 requests does not issue 500 concurrent requests --
it issues as many as the transport will accept and queues the rest inside
the event loop, where they are invisible. Three consequences:

1. **p95 latency becomes meaningless.** Queue time inside the loop is not
   measured by a per-request timer that starts when the coroutine finally
   gets a connection. The number you report is compute time; the number the
   user feels includes the queue.
2. **Timeouts fire on queued work.** A 30s timeout on a request that spent
   28s waiting for a connection fails despite the server being healthy.
3. **Nothing sheds load.** With no admission limit the only backpressure is
   memory, so the failure mode is an OOM rather than a clean 429.

A semaphore makes the limit explicit and moves the wait somewhere you can
measure it. This module keeps queue time and service time as separate fields
for exactly that reason.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class Timing:
    """Where a request's wall-clock actually went.

    Reporting only `total` is the mistake this class exists to prevent: it
    hides whether you are compute-bound or admission-bound, which are fixed
    by opposite actions (bigger machine vs higher limit).
    """

    queued_s: float = 0.0
    service_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.queued_s + self.service_s


@dataclass
class RunStats:
    """Aggregate outcome of a bounded run."""

    timings: list[Timing] = field(default_factory=list)
    errors: int = 0

    def _pct(self, values: Sequence[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        # Nearest-rank percentile: the smallest value at or above rank
        # ceil(p/100 * N). Stated explicitly because percentile definitions
        # differ by up to one rank and a silent choice makes a reported p95
        # impossible to reproduce.
        rank = math.ceil((p / 100.0) * len(s))
        k = max(0, min(len(s) - 1, rank - 1))
        return s[k]

    def summary(self) -> dict[str, float | int]:
        tot = [t.total_s for t in self.timings]
        svc = [t.service_s for t in self.timings]
        que = [t.queued_s for t in self.timings]
        return {
            "n": len(self.timings),
            "errors": self.errors,
            "p50_total_s": round(self._pct(tot, 50), 4),
            "p95_total_s": round(self._pct(tot, 95), 4),
            "p99_total_s": round(self._pct(tot, 99), 4),
            "p95_service_s": round(self._pct(svc, 95), 4),
            "p95_queued_s": round(self._pct(que, 95), 4),
            "mean_queued_s": round(sum(que) / len(que), 4) if que else 0.0,
        }


async def bounded_map(
    fn: Callable[[T], Awaitable[R]],
    items: Iterable[T],
    *,
    limit: int = 8,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[list[R | None], RunStats]:
    """Apply `fn` across `items` with at most `limit` in flight.

    Returns results in input order (None where the call raised) plus stats
    that separate queue time from service time.

    Order preservation matters more than it looks: `asyncio.as_completed`
    is faster to first result but loses the mapping back to inputs, and
    silently reordered results are a genuinely nasty class of bug.
    """
    items = list(items)
    sem = asyncio.Semaphore(limit)
    stats = RunStats()
    results: list[R | None] = [None] * len(items)
    timings: list[Timing | None] = [None] * len(items)

    async def one(idx: int, item: T) -> None:
        t = Timing()
        submitted = clock()
        async with sem:
            t.queued_s = clock() - submitted
            started = clock()
            try:
                results[idx] = await fn(item)
            except Exception:
                stats.errors += 1
            finally:
                t.service_s = clock() - started
        timings[idx] = t

    await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
    stats.timings = [t for t in timings if t is not None]
    return results, stats
