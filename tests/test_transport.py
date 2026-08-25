"""Transport tests.

No live call is made anywhere in this file, or anywhere in this suite. Every
test drives the real `LLMClient` -- its real retry loop, real deadline
handling, real pooling -- over an injected `httpx.MockTransport` or a cassette.
Only the socket is replaced, so what is under test is the client rather than a
stand-in for it.

The tests pin claims: that retries follow the policy rather than a second copy
of it, that a deadline truncates rather than merely warns, that a batch keeps
input order, and that one bad prompt does not lose the rest of the batch.
"""

from __future__ import annotations

import httpx
import pytest

from llm_client_kit.cassette import Cassette
from llm_client_kit.config import ClientConfig
from llm_client_kit.cost import CostLedger, ModelPrice
from llm_client_kit.retry import Deadline, RetryBudgetExceeded, RetryPolicy
from llm_client_kit.transport import (
    APIStatusError,
    ChatMessage,
    Completion,
    LLMClient,
    LLMError,
    ResponseShapeError,
    Usage,
)

CFG = ClientConfig(base_url="https://api.example.com/v1", api_key="sk-test", model="m")
HELLO = [ChatMessage.user("hello")]


def completion_payload(text: str = "hi", model: str = "m", **usage: int) -> dict:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 10),
            "completion_tokens": usage.get("completion_tokens", 5),
            "prompt_tokens_details": {"cached_tokens": usage.get("cached_tokens", 0)},
        },
    }


def scripted(*responses: httpx.Response | int) -> tuple[httpx.MockTransport, list]:
    """A transport replaying a fixed script, recording the requests it saw."""
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, int):
            return httpx.Response(item, json={"error": f"status {item}"})
        return item

    return httpx.MockTransport(handler), seen


def client(transport: httpx.MockTransport, **kw) -> LLMClient:
    """A client whose sleeps are instant, so retry tests take no wall-clock."""

    async def no_sleep(_seconds: float) -> None:
        return None

    kw.setdefault("sleep", no_sleep)
    kw.setdefault("retry", RetryPolicy(max_retries=3, base_delay_s=0.01, jitter=False))
    return LLMClient(kw.pop("config", CFG), transport=transport, **kw)


# --- the happy path -----------------------------------------------------


async def test_chat_returns_the_parsed_completion():
    transport, seen = scripted(httpx.Response(200, json=completion_payload("world")))
    async with client(transport) as c:
        result = await c.chat(HELLO)

    assert isinstance(result, Completion)
    assert result.text == "world"
    assert result.finish_reason == "stop"
    assert result.attempts == 1
    assert len(seen) == 1


async def test_the_request_is_openai_shaped():
    """Coding against the shape rather than the vendor is the whole premise."""
    transport, seen = scripted(httpx.Response(200, json=completion_payload()))
    async with client(transport) as c:
        await c.chat(
            [ChatMessage.system("be terse"), ChatMessage.user("hi")],
            model="other-model",
            temperature=0.0,
        )

    request = seen[0]
    assert request.url.path.endswith("/chat/completions")
    body = request.read().decode()
    assert '"model":"other-model"' in body.replace(" ", "")
    assert '"temperature":0.0' in body.replace(" ", "")
    assert request.headers["authorization"] == "Bearer sk-test"


async def test_usage_surfaces_cached_tokens_from_the_nested_payload():
    """Prefix-cache hit rate is one of the things this repo measures; a number
    buried two dicts down does not get measured."""
    transport, _ = scripted(
        httpx.Response(200, json=completion_payload(prompt_tokens=100, cached_tokens=80))
    )
    async with client(transport) as c:
        usage = (await c.chat(HELLO)).usage

    assert usage.cached_tokens == 80
    assert usage.cache_hit_rate == pytest.approx(0.8)
    assert usage.total_tokens == 105


def test_usage_defaults_to_zero_on_servers_that_omit_it():
    assert Usage.from_payload(None) == Usage()
    assert Usage.from_payload({"prompt_tokens": 3}).cached_tokens == 0
    assert Usage().cache_hit_rate == 0.0


# --- retries: the policy decides, this module only drives ---------------


async def test_a_retryable_status_is_retried_and_then_succeeds():
    transport, seen = scripted(
        503, 503, httpx.Response(200, json=completion_payload("recovered"))
    )
    async with client(transport) as c:
        result = await c.chat(HELLO)

    assert result.text == "recovered"
    assert result.attempts == 3
    assert len(seen) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_deterministic_client_errors_are_not_retried(status):
    """Retrying a request that will fail identically burns the budget and
    delays the real error."""
    transport, seen = scripted(status)
    async with client(transport) as c:
        with pytest.raises(APIStatusError) as exc:
            await c.chat(HELLO)

    assert exc.value.status_code == status
    assert len(seen) == 1, "a deterministic failure must be attempted once"


async def test_retries_stop_at_the_policy_limit():
    transport, seen = scripted(503)
    async with client(transport, retry=RetryPolicy(max_retries=2, base_delay_s=0.0,
                                                   jitter=False)) as c:
        with pytest.raises(APIStatusError):
            await c.chat(HELLO)

    assert len(seen) == 3, "max_retries=2 means one attempt plus two retries"


async def test_retry_after_beats_our_own_backoff():
    """The server knows its capacity; our exponential guess does not."""
    slept: list[float] = []

    async def record_sleep(seconds: float) -> None:
        slept.append(seconds)

    transport, _ = scripted(
        httpx.Response(429, headers={"retry-after": "7"}, json={}),
        httpx.Response(200, json=completion_payload()),
    )
    async with client(
        transport,
        sleep=record_sleep,
        retry=RetryPolicy(max_retries=3, base_delay_s=0.5, jitter=False),
    ) as c:
        await c.chat(HELLO)

    assert slept == [7.0]


@pytest.mark.parametrize(
    "exc_type",
    [
        httpx.ConnectError,
        httpx.ReadError,
        httpx.ReadTimeout,
        httpx.ConnectTimeout,
        httpx.PoolTimeout,
        httpx.RemoteProtocolError,
    ],
)
async def test_transport_failures_are_retried(exc_type):
    """No httpx exception subclasses OSError or TimeoutError.

    RetryPolicy classifies on those stdlib types and imports no HTTP library.
    Without a translation at this boundary every real connection reset and read
    timeout is treated as non-retryable and surfaces on the first blip -- which
    is precisely what retries exist to absorb.
    """
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise exc_type("transient network failure")
        return httpx.Response(200, json=completion_payload("after reconnect"))

    async with client(httpx.MockTransport(handler)) as c:
        assert (await c.chat(HELLO)).text == "after reconnect"
    assert attempts == 2


async def test_a_programming_error_is_not_retried():
    """The translation must not turn every exception into a retryable one."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise ValueError("a bug in our own code")

    async with client(httpx.MockTransport(handler)) as c:
        with pytest.raises(ValueError):
            await c.chat(HELLO)
    assert attempts == 1, "a programming error must not be retried"


async def test_the_original_exception_reaches_the_caller():
    """Classification is internal; the caller must still see the real error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns went away")

    async with client(
        httpx.MockTransport(handler), retry=RetryPolicy(max_retries=1, base_delay_s=0.0)
    ) as c:
        with pytest.raises(httpx.ConnectError):
            await c.chat(HELLO)


# --- deadlines ----------------------------------------------------------


async def test_a_deadline_truncates_the_sleep_rather_than_merely_checking_it():
    """Three retries at 4s against a 10s budget would otherwise wait 12s."""
    slept: list[float] = []

    async def record_sleep(seconds: float) -> None:
        slept.append(seconds)

    transport, _ = scripted(503, httpx.Response(200, json=completion_payload()))
    async with client(
        transport,
        sleep=record_sleep,
        retry=RetryPolicy(max_retries=3, base_delay_s=100.0, jitter=False),
    ) as c:
        await c.chat(HELLO, deadline=Deadline(1.0))

    assert slept, "expected a backoff sleep"
    assert slept[0] <= 1.0, f"sleep of {slept[0]}s overran a 1s budget"


async def test_an_exhausted_deadline_stops_the_retry_loop():
    class Exhausted(Deadline):
        @property
        def remaining(self) -> float:
            return 0.0

    transport, seen = scripted(503)
    async with client(transport) as c:
        with pytest.raises(RetryBudgetExceeded):
            await c.chat(HELLO, deadline=Exhausted(1.0))

    assert seen == [], "an expired deadline must not issue a request"


async def test_the_remaining_budget_becomes_the_per_attempt_timeout():
    """Otherwise the deadline is honoured only at retry boundaries."""
    timeouts: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions.get("timeout", {}).get("read"))
        return httpx.Response(200, json=completion_payload())

    cfg = ClientConfig(base_url=CFG.base_url, api_key="k", model="m", timeout_s=60.0)
    async with client(httpx.MockTransport(handler), config=cfg) as c:
        await c.chat(HELLO, deadline=Deadline(2.0))

    assert timeouts[0] is not None
    assert timeouts[0] <= 2.0, "a 2s budget must not issue a 60s request"


# --- malformed successes ------------------------------------------------


async def test_a_200_without_choices_is_its_own_error():
    """A 500 means retry; a 200 with no choices means you are pointed at the
    wrong path, and retrying will never fix it."""
    transport, seen = scripted(httpx.Response(200, json={"object": "list", "data": []}))
    async with client(transport) as c:
        with pytest.raises(ResponseShapeError):
            await c.chat(HELLO)
    assert len(seen) == 1


async def test_a_200_that_is_not_json_is_a_shape_error():
    transport, _ = scripted(httpx.Response(200, text="<html>proxy error</html>"))
    async with client(transport) as c:
        with pytest.raises(ResponseShapeError):
            await c.chat(HELLO)


async def test_empty_messages_are_rejected_before_a_request_is_made():
    transport, seen = scripted(httpx.Response(200, json=completion_payload()))
    async with client(transport) as c:
        with pytest.raises(ValueError):
            await c.chat([])
    assert seen == []


# --- lifecycle ----------------------------------------------------------


async def test_the_context_manager_closes_the_pool():
    transport, _ = scripted(httpx.Response(200, json=completion_payload()))
    c = client(transport)
    async with c:
        await c.chat(HELLO)

    with pytest.raises(LLMError, match="closed"):
        await c.chat(HELLO)


async def test_aclose_is_idempotent():
    transport, _ = scripted(httpx.Response(200, json=completion_payload()))
    c = client(transport)
    await c.aclose()
    await c.aclose()


async def test_an_injected_client_is_not_closed_by_us():
    """Closing someone else's pool out from under them is a nasty bug."""
    transport, _ = scripted(httpx.Response(200, json=completion_payload()))
    shared = httpx.AsyncClient(base_url=CFG.base_url, transport=transport)
    async with LLMClient(CFG, client=shared):
        pass
    assert not shared.is_closed
    await shared.aclose()


# --- chat_many ----------------------------------------------------------


async def test_chat_many_preserves_input_order():
    """Silently reordered results are a genuinely nasty class of bug."""

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = request.read().decode()
        marker = prompt.split('"content":"')[1].split('"')[0]
        return httpx.Response(200, json=completion_payload(text=marker))

    prompts = [[ChatMessage.user(f"p{i}")] for i in range(20)]
    async with client(httpx.MockTransport(handler)) as c:
        results, stats = await c.chat_many(prompts, limit=4)

    assert [r.text for r in results] == [f"p{i}" for i in range(20)]
    assert stats.summary()["n"] == 20
    assert stats.errors == 0


async def test_chat_many_respects_the_admission_limit():
    """The limit is the point of the module; an unbounded gather would show
    every request in flight at once."""
    import asyncio

    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, json=completion_payload())

    prompts = [[ChatMessage.user(f"p{i}")] for i in range(24)]
    async with client(httpx.MockTransport(handler)) as c:
        await c.chat_many(prompts, limit=3)

    assert peak <= 3, f"admission limit of 3 exceeded: {peak} in flight"


async def test_one_failing_prompt_does_not_lose_the_rest_of_the_batch():
    """One bad prompt in a batch of 500 should not cost you the other 499."""

    def handler(request: httpx.Request) -> httpx.Response:
        if b'"poison"' in request.read():
            return httpx.Response(400, json={"error": "bad prompt"})
        return httpx.Response(200, json=completion_payload("ok"))

    prompts = [
        [ChatMessage.user("fine")],
        [ChatMessage.user("poison")],
        [ChatMessage.user("fine")],
    ]
    async with client(httpx.MockTransport(handler)) as c:
        results, stats = await c.chat_many(prompts, limit=2)

    assert results[0] is not None and results[2] is not None
    assert results[1] is None, "a failed call must be None, not a fabricated result"
    assert stats.errors == 1


async def test_chat_many_separates_queue_time_from_service_time():
    """The queue/service split is inherited from bounded_map, not re-derived."""
    transport, _ = scripted(httpx.Response(200, json=completion_payload()))
    prompts = [[ChatMessage.user(f"p{i}")] for i in range(8)]
    async with client(transport) as c:
        _, stats = await c.chat_many(prompts, limit=2)

    summary = stats.summary()
    assert "p95_queued_s" in summary
    assert "p95_service_s" in summary


# --- cost integration ---------------------------------------------------


async def test_a_budget_stops_the_client_mid_batch():
    """The end-to-end claim: an over-budget run raises instead of billing on."""
    from llm_client_kit.cost import BudgetExceeded

    transport, seen = scripted(
        httpx.Response(200, json=completion_payload(prompt_tokens=1_000_000,
                                                    completion_tokens=0))
    )
    ledger = CostLedger({"m": ModelPrice(1.0, 1.0)}, budget_usd=1.5)
    async with client(transport, ledger=ledger) as c:
        await c.chat(HELLO)          # $1.00, under budget
        with pytest.raises(BudgetExceeded):
            await c.chat(HELLO)      # $2.00 total, over

    assert len(seen) == 2
    assert ledger.total_usd == pytest.approx(2.0)


async def test_the_over_budget_call_is_still_recorded():
    from llm_client_kit.cost import BudgetExceeded

    transport, _ = scripted(
        httpx.Response(200, json=completion_payload(prompt_tokens=1_000_000,
                                                    completion_tokens=0))
    )
    ledger = CostLedger({"m": ModelPrice(1.0, 1.0)}, budget_usd=0.5)
    async with client(transport, ledger=ledger) as c:
        with pytest.raises(BudgetExceeded):
            await c.chat(HELLO)

    assert len(ledger.entries) == 1


# --- the client under a cassette ----------------------------------------


async def test_the_real_client_replays_from_a_cassette(tmp_path):
    """The integration that makes CI free and deterministic: the full client,
    retry loop included, driven entirely off recorded bytes."""
    path = tmp_path / "chat.json"
    upstream, _ = scripted(
        503, httpx.Response(200, json=completion_payload("from the cassette"))
    )

    with Cassette(path, mode="record") as rec:
        async with client(rec.transport(upstream)) as c:
            first = await c.chat(HELLO)
    assert first.attempts == 2, "the recording should capture the 503 and the retry"

    replay = Cassette(path, mode="replay")
    async with client(replay.transport()) as c:
        result = await c.chat(HELLO)

    assert result.text == "from the cassette"
    assert result.attempts == 2, "the retry path replayed, not just the success"
    assert replay.hits == 2


async def test_a_cassette_recorded_through_the_client_holds_no_key(tmp_path):
    """The scrubbing claim, asserted through the real client rather than a
    hand-built request -- this is the path that would actually leak."""
    path = tmp_path / "client.json"
    secret = "sk-live-CLIENT-PATH-9999"
    cfg = ClientConfig(base_url=CFG.base_url, api_key=secret, model="m")
    upstream, _ = scripted(httpx.Response(200, json=completion_payload()))

    with Cassette(path, mode="record") as rec:
        async with client(rec.transport(upstream), config=cfg) as c:
            await c.chat(HELLO)

    assert secret not in path.read_text(encoding="utf-8")
