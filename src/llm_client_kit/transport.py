"""The async client itself: one HTTP call, and many of them under a limit.

This module deliberately contains *no* retry maths and *no* concurrency
primitives. `RetryPolicy` and `Deadline` already decide what is worth retrying
and how long a caller's budget has left; `bounded_map` already runs work under
an admission limit while separating queue time from service time. Reimplementing
either here would give two places to fix a bug and two answers to a question.

What this module does own:

**One `AsyncClient` for the client's lifetime.** A new client per request means
a new TCP connection and a new TLS handshake per request -- typically 1-2 extra
round trips before a single byte of prompt moves. Pooling is the single largest
latency win available to an HTTP client and it costs one shared object.

**Deadline before attempt, not just between them.** The remaining budget is
passed down as the per-attempt timeout, so a 10 s budget with 8 s already spent
issues a request with a 2 s timeout rather than a fresh 60 s one. Without that,
the deadline is honoured only at retry boundaries -- which is to say, not when
it matters.

**`chat_many` shares the client and the limit, but not the deadline.** Each
call gets its own budget because a deadline is a per-request promise; one slow
call should not consume the budget of the 99 behind it.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from .concurrency import RunStats, bounded_map
from .config import ClientConfig
from .cost import CostLedger
from .retry import Deadline, RetryPolicy

Role = str


class LLMError(RuntimeError):
    """Base for client failures."""


class APIStatusError(LLMError):
    """A non-retryable, or finally-exhausted, HTTP error response."""

    def __init__(self, status_code: int, body: str, url: str = "") -> None:
        self.status_code = status_code
        self.body = body
        self.url = url
        super().__init__(f"HTTP {status_code} from {url}: {body[:400]}")


class ResponseShapeError(LLMError):
    """The endpoint answered 200 with something that is not a completion.

    Its own error type because the fix differs: a 500 means retry, whereas a
    200 with no `choices` means you are pointed at the wrong path or a proxy
    is intercepting, and no amount of retrying will change it.
    """


@dataclass(frozen=True)
class ChatMessage:
    """One turn. `role` is a plain string so a provider that invents a new one
    does not require a library release to use."""

    role: Role
    content: str
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        return cls("assistant", content)


@dataclass(frozen=True)
class Usage:
    """Token counts for one call.

    `cached_tokens` is pulled out of the nested `prompt_tokens_details` that
    OpenAI-shape servers use, and defaults to 0 on servers that omit it. It is
    a top-level field here because prompt-cache hit rate is one of the things
    this repo exists to measure, and a number buried two dicts down does not
    get measured.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompt tokens served from the provider's prefix cache."""
        if self.prompt_tokens <= 0:
            return 0.0
        return self.cached_tokens / self.prompt_tokens

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any] | None) -> Usage:
        if not raw:
            return cls()
        details = raw.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0) if isinstance(details, Mapping) else 0
        return cls(
            prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
            completion_tokens=int(raw.get("completion_tokens", 0) or 0),
            cached_tokens=int(cached or 0),
        )


@dataclass(frozen=True)
class Completion:
    """A single completion plus the accounting a caller needs afterwards.

    `attempts` and `latency_s` are on the result rather than in a log line
    because "how often did we retry, and did it cost latency" is a question
    asked after a batch finishes, when the logs have scrolled away.
    """

    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    attempts: int = 1
    latency_s: float = 0.0
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        attempts: int = 1,
        latency_s: float = 0.0,
    ) -> Completion:
        choices = payload.get("choices")
        if not isinstance(choices, Sequence) or not choices:
            raise ResponseShapeError(
                f"response has no choices; keys were {sorted(payload)}"
            )
        first = choices[0]
        message = first.get("message") or {}
        content = message.get("content")
        return cls(
            text="" if content is None else str(content),
            model=str(payload.get("model", "")),
            usage=Usage.from_payload(payload.get("usage")),
            finish_reason=first.get("finish_reason"),
            attempts=attempts,
            latency_s=latency_s,
            raw=payload,
        )


def as_retryable_os_error(exc: Exception) -> Exception:
    """Translate an httpx transport failure into the stdlib type the policy knows.

    `RetryPolicy` classifies transport failures as `TimeoutError`,
    `ConnectionError` or `OSError`, and deliberately imports no HTTP library --
    that independence is what lets it be tested and reused without httpx.

    But no httpx exception subclasses any of those: `httpx.ConnectError` derives
    from `httpx.TransportError`, not `OSError`. Without this translation every
    real connection reset, DNS failure and read timeout is classified
    non-retryable and surfaces on the first blip -- the exact failures retries
    exist for. Caught by a test that raises `httpx.ConnectError`.

    Translating here rather than teaching `retry.py` about httpx keeps the
    library dependency at the one boundary that already has it.
    """
    if isinstance(exc, httpx.TimeoutException):
        return TimeoutError(str(exc) or exc.__class__.__name__)
    if isinstance(exc, httpx.TransportError):
        # NetworkError, ProtocolError, PoolTimeout and friends: all transient
        # at the socket level and worth another attempt.
        return ConnectionError(str(exc) or exc.__class__.__name__)
    return exc


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse Retry-After, seconds form only.

    The HTTP-date form is accepted by the spec and essentially never sent by
    JSON APIs. Returning None on anything unparseable falls back to our own
    backoff, which is the safe direction: a mis-parsed date could produce a
    multi-hour sleep.
    """
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


class LLMClient:
    """Async client for an OpenAI-shape `/chat/completions` endpoint.

    Use as an async context manager so the connection pool is closed::

        async with LLMClient(ClientConfig.from_env()) as client:
            completion = await client.chat([ChatMessage.user("hi")])

    A `transport` can be injected -- that is how cassettes and fakes replace
    the socket while leaving the retry loop, timeouts and pooling under test.
    """

    def __init__(
        self,
        config: ClientConfig | None = None,
        *,
        retry: RetryPolicy | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
        ledger: CostLedger | None = None,
        rng: random.Random | None = None,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self.config = config or ClientConfig()
        self.retry = retry or RetryPolicy(max_retries=self.config.max_retries)
        self.ledger = ledger
        self._rng = rng or random.Random()
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=self.config.headers(),
            timeout=self.config.timeout_s,
            transport=transport,
            # Pool sized to the admission limit: more connections than the
            # semaphore will ever use is memory spent on sockets that idle.
            limits=httpx.Limits(
                max_connections=max(self.config.max_concurrency, 1),
                max_keepalive_connections=max(self.config.max_concurrency, 1),
            ),
        )
        self._closed = False

    # --- lifecycle ------------------------------------------------------

    async def aclose(self) -> None:
        """Close the connection pool. Idempotent.

        Only closes a client this object created. A caller who injected their
        own client owns its lifetime -- closing someone else's pool out from
        under them is a genuinely confusing bug to chase.
        """
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # --- one call -------------------------------------------------------

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        model: str | None,
        extra: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": [m.to_dict() for m in messages],
        }
        if extra:
            body.update(extra)
        return body

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        deadline: Deadline | None = None,
        **extra: Any,
    ) -> Completion:
        """One completion, retried per the policy and inside the deadline.

        The retry loop is a thin driver over `RetryPolicy`: this method decides
        *when* to ask, never *whether* a status is retryable or *how long* to
        wait. Those live in `retry.py` and have their own tests.
        """
        if self._closed:
            raise LLMError("client is closed")
        if not messages:
            raise ValueError("messages must not be empty")

        budget = deadline or Deadline(self.config.deadline_s)
        body = self._payload(messages, model, extra)
        loop = asyncio.get_running_loop()
        started = loop.time()
        last_error: Exception | None = None
        # What the policy classifies, which may be a stdlib translation of an
        # httpx error. The original is what the caller finally sees.
        classify_as: Exception | None = None
        attempt = 0

        while True:
            budget.check()
            try:
                # Per-attempt timeout is the smaller of the configured timeout
                # and what remains of the caller's budget.
                timeout = min(self.config.timeout_s, budget.remaining)
                response = await self._client.post(
                    "/chat/completions", json=body, timeout=timeout
                )
            except Exception as exc:  # noqa: BLE001 - classified by the policy
                last_error, status, retry_after = exc, None, None
                classify_as = as_retryable_os_error(exc)
            else:
                if response.status_code < 400:
                    return self._finish(response, attempt + 1, loop.time() - started)
                status = response.status_code
                retry_after = _retry_after_seconds(response)
                last_error = APIStatusError(
                    status, response.text, str(response.request.url)
                )
                classify_as = last_error

            assert last_error is not None
            if not self.retry.should_retry(attempt, status, classify_as):
                raise last_error
            delay = budget.clamp(
                self.retry.delay_for(attempt, retry_after, rng=self._rng)
            )
            if budget.remaining <= 0:
                budget.check()
            await self._sleep(max(delay, 0.0))
            attempt += 1

    def _finish(
        self, response: httpx.Response, attempts: int, latency_s: float
    ) -> Completion:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ResponseShapeError(
                f"200 response was not JSON: {response.text[:200]}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ResponseShapeError(f"expected a JSON object, got {type(payload).__name__}")
        completion = Completion.from_payload(
            payload, attempts=attempts, latency_s=latency_s
        )
        if self.ledger is not None:
            # Record before the budget check so the call that broke the budget
            # still appears in the ledger -- the tokens were spent regardless.
            self.ledger.record_usage(completion.model or self.config.model, completion.usage)
            self.ledger.check_budget()
        return completion

    # --- many calls -----------------------------------------------------

    async def chat_many(
        self,
        batches: Iterable[Sequence[ChatMessage]],
        *,
        model: str | None = None,
        limit: int | None = None,
        deadline_s: float | None = None,
        **extra: Any,
    ) -> tuple[list[Completion | None], RunStats]:
        """Run many chats under an admission limit.

        Delegates to `bounded_map`, so the queue-time/service-time split and
        the percentile definitions are the same ones the benchmark measures.
        Results keep input order, with `None` where a call finally failed --
        one bad prompt in a batch of 500 should not lose the other 499.

        Each call receives its own `Deadline`: a budget is a per-request
        promise, and sharing one would let the first slow call consume the
        allowance of everything queued behind it.
        """

        async def one(messages: Sequence[ChatMessage]) -> Completion:
            return await self.chat(
                messages,
                model=model,
                deadline=Deadline(deadline_s if deadline_s is not None
                                  else self.config.deadline_s),
                **extra,
            )

        return await bounded_map(
            one, batches, limit=limit or self.config.max_concurrency
        )
