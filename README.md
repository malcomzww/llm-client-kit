# llm-client-kit

A provider-agnostic async client for OpenAI-shape endpoints, built around one
question:

> **Where does the wall-clock actually go when you fan out LLM calls, and what
> does bounded concurrency buy you at p95?**

Everything here serves answering that. It is a small library on purpose.

```
$ python -m pytest -q
104 passed

$ python scripts/generate_results.py
wrote concurrency.md (committed) and concurrency-raw.md (gitignored)
```

**No test makes a live API call.** Every test drives the real client — real
retry loop, real deadlines, real connection pooling — over an injected
`httpx.MockTransport` or a recorded cassette, so only the socket is replaced.
A committed script asserts it: see
[`results/test-suite.md`](results/test-suite.md).

## The measurement

128 tasks, 20 ms simulated service time each, sweeping the concurrency limit:

| limit | queued vs service | bound by |
|---|---|---|
| 1 | >50x | admission |
| 4 | 10-50x | admission |
| 16 | 2-10x | admission |
| 64 | ~1x | **crossover** (flips between runs) |
| 128 | <1x | service |

**Service time never changed.** It stays within 25% of its maximum across a
128x range of limits, because it is a property of the work rather than of the
client. Everything else was admission delay.

That is the whole point: a single reported latency number cannot tell you
which of the two you are looking at, and **the two are fixed by opposite
actions** — admission-bound means raise the limit or shed load, service-bound
means a faster backend. Acting on the wrong one wastes the effort.

Note what the table does *not* claim. At `limit=64` queue and service time are
within 2x of each other and the classification flips between runs, so it is
labelled a crossover rather than given a verdict it cannot support.

Absolute millisecond timings are deliberately **not** committed — see
[`results/concurrency.md`](results/concurrency.md) for why, and for the full
set of asserted claims.

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
from llm_client_kit.config import ClientConfig
from llm_client_kit.cost import CostLedger, ModelPrice
from llm_client_kit.transport import ChatMessage, LLMClient

# A budget that raises, so an unattended loop stops itself.
ledger = CostLedger({"gpt-4o-mini": ModelPrice(0.15, 0.60)}, budget_usd=1.00)

async with LLMClient(ClientConfig.from_env(), ledger=ledger) as client:
    reply = await client.chat([ChatMessage.user("hello")])
    print(reply.text, reply.usage.cache_hit_rate)

    # Many calls under one admission limit, results in input order.
    results, stats = await client.chat_many(batches, limit=8)
    print(stats.summary())
```

Record once, replay free forever:

```python
from llm_client_kit.cassette import Cassette

with Cassette("tests/cassettes/chat.json", mode="auto") as tape:
    async with LLMClient(cfg, transport=tape.transport()) as client:
        await client.chat([ChatMessage.user("hello")])
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

**A budget that raises, not one that logs.** The failure mode of a runaway
agent loop is not a crash — it is a correct-looking run and an invoice. A
warning in a log nobody reads during an unattended run is indistinguishable
from no budget at all, so `check_budget()` raises. Recording and enforcing are
separate: the call that broke the budget is still recorded, because the tokens
were spent either way and that is the entry you most want to see.

**`Authorization` is scrubbed when a cassette is recorded, never when it is
loaded.** A cassette that ever contained a key is a leaked key — the file is
committed, pushed and mirrored long before anyone reviews it, and scrubbing on
load is too late because the bytes already reached disk. A test asserts on the
raw file contents, so a key smuggled through an unlisted header or a URL
parameter fails it too.

**A cassette miss is fatal, never a fallback to the network.** Falling back is
how a suite quietly starts billing you, and it makes a green CI run meaningless
because you cannot tell whether it replayed or dialled out.

**Cassettes match on method, URL and a canonical body hash — not headers.**
Header matching is how cassette suites rot: a client version bump changes
`user-agent` and every cassette misses. JSON keys are sorted before hashing,
because dict ordering is not part of a request's meaning.

**Money is `Decimal`, not `float`.** `0.1 + 0.2 != 0.3` in binary floating
point, and a ledger that drifts a fraction of a cent per call is one nobody
trusts after a million calls.

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
- **The client has never been run against a live provider.** No API key was
  configured while it was built, so every claim about it rests on mock
  transports and cassettes. Those prove the client parses what a server *did*
  send; they cannot prove a provider still sends it, or that a real endpoint
  will not fail in a way nothing here anticipates.
- **The cost ledger is arithmetic, not a bill.** It costs the tokens a provider
  reported. Reported token counts are the provider's own accounting, and
  nothing here reconciles them against an invoice.
- Streaming and a circuit breaker are not implemented — see the status table.

## Status

| module | state |
|---|---|
| `config.py` — endpoint resolution, key redaction | done |
| `retry.py` — policy, full jitter, deadline propagation | done |
| `concurrency.py` — bounded map, queue/service split | done |
| `transport.py` — httpx client, `chat`/`chat_many`, pooling | done |
| `cassette.py` — record/replay for free deterministic CI | done |
| `cost.py` — token and cost ledger, hard budget | done |
| streaming (`stream=true`, SSE) | not started |
| circuit breaker | not started |

Streaming and a circuit breaker are listed in the concept inventory and are
**not** implemented. They are named here rather than omitted, because a status
table that only lists what exists is a marketing page.

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
