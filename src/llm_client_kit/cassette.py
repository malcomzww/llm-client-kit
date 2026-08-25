"""Record/replay of HTTP exchanges, so CI never makes a live call.

The brief's constraint is that CI must never touch a real provider. That is
not only about money -- a test suite whose outcome depends on a third party's
uptime and on a non-deterministic sampler is not a test suite, it is a status
page. Cassettes make the LLM call deterministic without mocking the client
under test, which is the point: a hand-written fake tests the fake, whereas a
cassette replays bytes a real server actually sent.

**The security decision.** `Authorization` is scrubbed at *record* time, never
at load time. A cassette that ever contains a key is a leaked key -- the file
gets committed, pushed, and mirrored before anyone thinks about it, and
scrubbing on load is too late because the bytes are already on disk. There is
a test asserting the recorded file does not contain the key.

**Matching is on method, URL and a canonical body hash**, not on header set.
Header matching is how cassette suites become brittle: a client-version bump
changes `user-agent` and every cassette misses. The body is canonicalised
(sorted JSON keys) so an equivalent request in a different key order still
hits, because dict ordering is not part of the request's meaning.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

Mode = Literal["record", "replay", "auto"]

# Header names, lowercased, whose values never reach disk. Anything that can
# authenticate a request belongs here. The list is deliberately broad: the
# cost of scrubbing a harmless header is nil, the cost of missing a secret is
# a rotated key and an incident.
SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "api-key",
        "x-api-key",
        "openai-api-key",
        "anthropic-api-key",
        "x-auth-token",
        "cookie",
        "set-cookie",
    }
)

SCRUBBED = "<scrubbed>"

CASSETTE_VERSION = 1


class CassetteError(RuntimeError):
    """Base for cassette failures."""


class CassetteMiss(CassetteError):
    """No recorded interaction matches this request.

    In replay mode this is fatal on purpose. Falling back to a live call on a
    miss is the behaviour that lets a cassette suite quietly start billing
    you, and it makes a green CI run meaningless because you cannot tell
    whether it replayed or dialled out.
    """


def scrub_headers(headers: Any) -> dict[str, str]:
    """Return headers with every sensitive value replaced.

    Accepts anything dict-like or httpx.Headers. Returns plain lowercase-keyed
    strings so a cassette's on-disk shape does not depend on the HTTP library
    version that produced it.
    """
    items = headers.items() if hasattr(headers, "items") else headers
    out: dict[str, str] = {}
    for k, v in items:
        key = str(k).lower()
        out[key] = SCRUBBED if key in SENSITIVE_HEADERS else str(v)
    return out


def _canonical_body(body: bytes) -> str:
    """Stable fingerprint of a request body.

    JSON is re-serialised with sorted keys so two semantically identical
    requests match regardless of dict ordering. Non-JSON bodies fall back to
    raw bytes, which is correct rather than clever.
    """
    if not body:
        return ""
    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return hashlib.sha256(body).hexdigest()
    return hashlib.sha256(
        json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def request_key(method: str, url: str, body: bytes) -> str:
    """The identity of a request for matching purposes."""
    return f"{method.upper()} {url} {_canonical_body(body)}"


@dataclass
class Interaction:
    """One recorded request/response pair."""

    method: str
    url: str
    request_headers: dict[str, str]
    request_body: str
    status: int
    response_headers: dict[str, str]
    response_body: str
    # Present only when the body was not valid UTF-8; base64 then.
    body_encoding: str = "utf-8"

    @property
    def key(self) -> str:
        return request_key(self.method, self.url, self.request_body.encode("utf-8"))

    def response_bytes(self) -> bytes:
        if self.body_encoding == "base64":
            return base64.b64decode(self.response_body)
        return self.response_body.encode("utf-8")

    def to_json(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "url": self.url,
            "request_headers": self.request_headers,
            "request_body": self.request_body,
            "status": self.status,
            "response_headers": self.response_headers,
            "response_body": self.response_body,
            "body_encoding": self.body_encoding,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Interaction:
        return cls(
            method=d["method"],
            url=d["url"],
            request_headers=d.get("request_headers", {}),
            request_body=d.get("request_body", ""),
            status=d["status"],
            response_headers=d.get("response_headers", {}),
            response_body=d.get("response_body", ""),
            body_encoding=d.get("body_encoding", "utf-8"),
        )


class Cassette:
    """A file of recorded HTTP exchanges.

    Modes:
        ``record``  always call through, save every exchange (overwrites).
        ``replay``  never call through; a miss raises `CassetteMiss`.
        ``auto``    replay if the file exists, otherwise record.

    ``auto`` is the mode a developer wants and ``replay`` is the mode CI wants.
    CI pins it explicitly so a deleted cassette fails the build instead of
    quietly falling back to a live, billed call.
    """

    def __init__(self, path: str | os.PathLike[str], mode: Mode = "auto") -> None:
        if mode not in ("record", "replay", "auto"):
            raise ValueError(f"unknown cassette mode {mode!r}")
        self.path = Path(path)
        self.interactions: list[Interaction] = []
        self._played: set[int] = set()
        self.hits = 0
        self.misses = 0

        if mode == "auto":
            mode = "replay" if self.path.is_file() else "record"
        self.mode: Literal["record", "replay"] = mode

        if self.mode == "replay":
            self._load()

    # --- persistence ---------------------------------------------------

    def _load(self) -> None:
        if not self.path.is_file():
            raise CassetteMiss(
                f"cassette {self.path} does not exist and mode is 'replay'. "
                "Re-record it, or use mode='auto'."
            )
        data = json.loads(self.path.read_text(encoding="utf-8"))
        version = data.get("version")
        if version != CASSETTE_VERSION:
            raise CassetteError(
                f"cassette {self.path} is version {version}, expected "
                f"{CASSETTE_VERSION}; re-record it"
            )
        self.interactions = [Interaction.from_json(d) for d in data["interactions"]]

    def save(self) -> None:
        """Write the cassette. Only meaningful after recording."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CASSETTE_VERSION,
            # Stated in the file itself so anyone reading a cassette knows the
            # absence of a key is by construction, not by luck.
            "note": "Sensitive headers are scrubbed at record time.",
            "interactions": [i.to_json() for i in self.interactions],
        }
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )

    # --- recording -----------------------------------------------------

    def record(self, request: httpx.Request, response: httpx.Response) -> Interaction:
        """Store one exchange, with sensitive request headers scrubbed."""
        body = response.content
        try:
            text, encoding = body.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            text, encoding = base64.b64encode(body).decode("ascii"), "base64"

        interaction = Interaction(
            method=request.method,
            url=str(request.url),
            request_headers=scrub_headers(request.headers),
            request_body=request.content.decode("utf-8", errors="replace"),
            status=response.status_code,
            response_headers=scrub_headers(response.headers),
            response_body=text,
            body_encoding=encoding,
        )
        self.interactions.append(interaction)
        return interaction

    # --- replaying -----------------------------------------------------

    def find(self, request: httpx.Request) -> Interaction | None:
        """First unplayed interaction matching this request.

        Unplayed-first means a cassette recording the same request twice with
        different responses -- a retry that failed then succeeded -- replays
        that sequence rather than returning the first response forever. That
        sequence is exactly what a retry test needs to exercise.
        """
        want = request_key(request.method, str(request.url), request.content)
        for idx, interaction in enumerate(self.interactions):
            if idx in self._played:
                continue
            if interaction.key == want:
                self._played.add(idx)
                return interaction
        # Fall back to a played one: repeated identical calls in a run should
        # not need N copies on disk.
        for interaction in self.interactions:
            if interaction.key == want:
                return interaction
        return None

    def play(self, request: httpx.Request) -> httpx.Response:
        interaction = self.find(request)
        if interaction is None:
            self.misses += 1
            raise CassetteMiss(
                f"no recorded interaction for {request.method} {request.url} "
                f"in {self.path} ({len(self.interactions)} recorded). "
                "Re-record the cassette if the request shape changed."
            )
        self.hits += 1
        return httpx.Response(
            status_code=interaction.status,
            headers=interaction.response_headers,
            content=interaction.response_bytes(),
            request=request,
        )

    # --- httpx integration ---------------------------------------------

    def transport(self, inner: httpx.AsyncBaseTransport | None = None) -> CassetteTransport:
        """An httpx transport wired to this cassette."""
        return CassetteTransport(self, inner)

    def __enter__(self) -> Cassette:
        return self

    def __exit__(self, *exc: object) -> None:
        if self.mode == "record":
            self.save()

    def __len__(self) -> int:
        return len(self.interactions)


class CassetteTransport(httpx.AsyncBaseTransport):
    """httpx transport that records to, or replays from, a cassette.

    Sitting at the transport layer rather than patching the client means the
    code under test is the real `LLMClient` with its real retry loop, real
    timeouts and real connection handling. Only the socket is replaced. Tests
    that patch higher up end up asserting on the mock.
    """

    def __init__(
        self,
        cassette: Cassette,
        inner: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.cassette = cassette
        if cassette.mode == "record" and inner is None:
            inner = httpx.AsyncHTTPTransport()
        self.inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.cassette.mode == "replay":
            return self.cassette.play(request)

        assert self.inner is not None
        # Materialise the request body before it is consumed by the wire, so
        # the recorded key matches what a replay will later compute.
        await request.aread()
        response = await self.inner.handle_async_request(request)
        await response.aread()
        self.cassette.record(request, response)
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=response.content,
            request=request,
        )

    async def aclose(self) -> None:
        if self.inner is not None:
            await self.inner.aclose()
