# llm-client-kit

A provider-agnostic async client for OpenAI-shape endpoints, built around one
question:

> **Where does the wall-clock actually go when you fan out LLM calls, and what
> does bounded concurrency buy you at p95?**

Everything here serves answering that. It is a small library on purpose.

```
$ python -m pytest -q
31 passed in 0.82s

$ python scripts/generate_results.py
wrote results/concurrency.md
```

## The measurement

128 tasks, 20 ms simulated service time each, varying the concurrency limit:

| limit | p50 total | p95 total | p95 queued | p95 service | bound by |
|---|---|---|---|---|---|
| 1 | 1985 ms | 3787 ms | 3756 ms | 32 ms | admission |
| 4 | 495 ms | 961 ms | 930 ms | 33 ms | admission |
| 16 | 122 ms | 247 ms | 216 ms | 32 ms | admission |
| 64 | 31 ms | 62 ms | 31 ms | 31 ms | admission |
| 128 | 29 ms | 29 ms | 0 ms | 29 ms | service |

**Service time never changed.** It sat at ~31 ms across every limit, because
it is a property of the work rather than of the client. Everything else was
admission delay.

That is the whole point: a single reported latency number cannot tell you
which of the two you are looking at, and **the two are fixed by opposite
actions** — admission-bound means raise the limit or shed load, service-bound
means a faster backend. Acting on the wrong one wastes the effort.

Full table and limitations: [`results/concurrency.md`](results/concurrency.md).

## Why this exists

`asyncio.gather` over 500 requests does not issue 500 concurrent requests. It
issues as many as the transport accepts and queues the rest inside the event
loop, where they are invisible. Three consequences:

1. **p95 becomes meaningless.** A per-request timer that starts when the
   coroutine finally gets a connection measures compute. The user feels
   compute *plus* queue.
2. **Timeouts fire on healthy servers.** A 30 s timeout on a request that
   spent 28 s waiting for a connection fails while the backend is fine.
3. **Nothing sheds load.** With no admission limit the only backpressure is
   memory, so the failure mode is an OOM rather than a clean 429.

A semaphore makes the limit explicit and moves the wait somewhere you can
measure it. This library keeps `queued_s` and `service_s` as separate fields
for exactly that reason.

## Quickstart

```bash
uv sync --extra dev
uv run pytest -q
uv run python scripts/generate_results.py
```

```python
from llm_client_kit.concurrency import bounded_map

results, stats = await bounded_map(call_model, prompts, limit=8)
print(stats.summary())
# {'n': 128, 'p95_total_s': 0.49, 'p95_queued_s': 0.46, 'p95_service_s': 0.031, ...}
```

## Design decisions

**Full jitter, not equal jitter.** When N clients retry a shared dependency
after an outage they synchronise, and the retry storm is often what keeps the
dependency down. Sleeping uniformly in `[0, backoff]` spreads the herd widest.
It costs nothing.

**A deadline that clamps sleeps, not just checks them.** Three retries at 4 s
against a 10 s budget would wait 12 s. `Deadline.clamp()` truncates each sleep
so the budget is honoured rather than merely intended — the difference between
a timeout and a suggestion.

**400/401/403/404/422 are never retried.** Those failures are deterministic.
Retrying burns the budget and delays surfacing the real error. A
server-supplied `Retry-After` always beats our own exponential guess, because
the server knows its capacity and we are guessing.

**Results keep input order.** `as_completed` reaches the first result sooner
but loses the mapping back to inputs, and silently reordered results are a
genuinely nasty class of bug. First-result latency is not what a batch caller
cares about.

**Nearest-rank percentiles, named in the docstring.** Definitions differ by up
to one rank. An unreproducible p95 is worse than no p95.

**Two environment variables only** — `OPENAI_BASE_URL` and `OPENAI_API_KEY`,
never per-provider names. Swapping to a local vLLM server, Ollama, OpenRouter
or Groq is then a `base_url` change with no code change.

## Limitations

- **The measurement uses a simulated backend, not a live endpoint.** The claim
  under test is about this client's admission control; real API variance would
  swamp the effect. It does **not** predict latency against any real provider.
- `asyncio.sleep` is a lower bound on real I/O — no connection-pool
  contention, no TLS, no DNS, no head-of-line blocking from HTTP/2
  multiplexing.
- Single machine, single event loop. Nothing here is tested across processes.
- The HTTP transport and cassette record/replay layer are **not yet
  implemented**; `config.py`, `retry.py` and `concurrency.py` are.

## Status

| module | state |
|---|---|
| `config.py` — endpoint resolution, key redaction | done |
| `retry.py` — policy, full jitter, deadline propagation | done |
| `concurrency.py` — bounded map, queue/service split | done |
| `transport.py` — httpx client, streaming | not started |
| `cassettes.py` — record/replay for free deterministic CI | not started |
| `cost.py` — token and cost ledger | not started |

## Concepts covered

Anchored to a personal concept inventory (bucket 5, software engineering):

- async/await, the event loop, the GIL
- asyncio for high-throughput LLM calls, semaphores, bounded concurrency
- connection pooling
- retries with backoff and jitter, circuit breakers, deadline propagation
- mocking non-deterministic LLM calls
- prompt/prefix caching, cache-aware ordering, hit-rate measurement

## License

MIT
