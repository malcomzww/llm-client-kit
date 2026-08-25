"""Cassette tests.

The first test in this file is the one that matters: a cassette that ever
contains an API key is a leaked key. It asserts on the *bytes on disk*, not on
an in-memory dict, because the disk is what gets committed and pushed.
"""

from __future__ import annotations

import json

import httpx
import pytest

from llm_client_kit.cassette import (
    SCRUBBED,
    Cassette,
    CassetteMiss,
    scrub_headers,
)

SECRET = "sk-live-DO-NOT-LEAK-abcdef0123456789"


def ok_json(payload: dict) -> httpx.MockTransport:
    """A transport that answers everything with one JSON body."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


# --- the security claim -------------------------------------------------


async def test_recorded_cassette_file_does_not_contain_the_api_key(tmp_path):
    """The load-bearing security test.

    Asserts on the raw file contents rather than on parsed fields, so a key
    smuggled into any part of the JSON -- a header we forgot to list, a URL
    query parameter, a nested dict -- still fails this test.
    """
    path = tmp_path / "secrets.json"
    cassette = Cassette(path, mode="record")
    inner = ok_json({"choices": [{"message": {"content": "hi"}}]})

    async with httpx.AsyncClient(
        transport=cassette.transport(inner),
        headers={"Authorization": f"Bearer {SECRET}"},
    ) as client:
        await client.post("https://api.example.com/v1/chat/completions", json={"a": 1})
    cassette.save()

    on_disk = path.read_text(encoding="utf-8")
    assert SECRET not in on_disk, "the API key reached the cassette file"
    assert "sk-live" not in on_disk
    assert SCRUBBED in on_disk

    parsed = json.loads(on_disk)
    headers = parsed["interactions"][0]["request_headers"]
    assert headers["authorization"] == SCRUBBED


@pytest.mark.parametrize(
    "header",
    ["Authorization", "X-API-Key", "api-key", "Cookie", "Proxy-Authorization"],
)
def test_every_credential_header_is_scrubbed(header):
    scrubbed = scrub_headers({header: SECRET, "User-Agent": "kit/1.0"})
    assert scrubbed[header.lower()] == SCRUBBED
    assert scrubbed["user-agent"] == "kit/1.0", "harmless headers must survive"


def test_scrubbing_is_case_insensitive():
    """Header names are case-insensitive in HTTP; a case-sensitive scrubber
    leaks the moment a client capitalises differently."""
    assert scrub_headers({"AUTHORIZATION": SECRET})["authorization"] == SCRUBBED
    assert scrub_headers({"authorization": SECRET})["authorization"] == SCRUBBED


async def test_scrubbing_survives_a_record_replay_round_trip(tmp_path):
    """Scrubbing must not be undone by loading -- and replay must still work."""
    path = tmp_path / "round-trip.json"
    with Cassette(path, mode="record") as rec:
        async with httpx.AsyncClient(
            transport=rec.transport(ok_json({"ok": True})),
            headers={"Authorization": f"Bearer {SECRET}"},
        ) as client:
            await client.post("https://api.example.com/v1/x", json={"q": 1})

    replay = Cassette(path, mode="replay")
    assert SECRET not in path.read_text(encoding="utf-8")
    assert replay.interactions[0].request_headers["authorization"] == SCRUBBED

    async with httpx.AsyncClient(transport=replay.transport()) as client:
        response = await client.post("https://api.example.com/v1/x", json={"q": 1})
    assert response.json() == {"ok": True}


# --- record / replay behaviour -----------------------------------------


async def test_replay_returns_the_recorded_response_without_calling_out(tmp_path):
    path = tmp_path / "c.json"
    with Cassette(path, mode="record") as rec:
        async with httpx.AsyncClient(transport=rec.transport(ok_json({"n": 42}))) as c:
            await c.post("https://api.example.com/v1/x", json={"p": "hello"})

    calls = 0

    def exploding(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("replay must never reach the network")

    replay = Cassette(path, mode="replay")
    async with httpx.AsyncClient(
        transport=replay.transport(httpx.MockTransport(exploding))
    ) as c:
        response = await c.post("https://api.example.com/v1/x", json={"p": "hello"})

    assert response.json() == {"n": 42}
    assert calls == 0
    assert replay.hits == 1


async def test_replay_miss_is_fatal_rather_than_falling_back_to_the_network(tmp_path):
    """A silent fallback is how a cassette suite quietly starts billing you,
    and it makes a green CI run meaningless."""
    path = tmp_path / "c.json"
    with Cassette(path, mode="record") as rec:
        async with httpx.AsyncClient(transport=rec.transport(ok_json({"n": 1}))) as c:
            await c.post("https://api.example.com/v1/x", json={"p": "recorded"})

    replay = Cassette(path, mode="replay")
    async with httpx.AsyncClient(transport=replay.transport()) as c:
        with pytest.raises(CassetteMiss):
            await c.post("https://api.example.com/v1/x", json={"p": "never seen"})


def test_replay_mode_on_a_missing_file_fails_loudly(tmp_path):
    with pytest.raises(CassetteMiss):
        Cassette(tmp_path / "nope.json", mode="replay")


async def test_auto_records_then_replays(tmp_path):
    """The mode a developer wants: first run records, later runs replay."""
    path = tmp_path / "auto.json"
    first = Cassette(path, mode="auto")
    assert first.mode == "record"
    with first:
        async with httpx.AsyncClient(transport=first.transport(ok_json({"v": 1}))) as c:
            await c.post("https://api.example.com/v1/x", json={})

    second = Cassette(path, mode="auto")
    assert second.mode == "replay"
    async with httpx.AsyncClient(transport=second.transport()) as c:
        assert (await c.post("https://api.example.com/v1/x", json={})).json() == {"v": 1}


def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        Cassette(tmp_path / "c.json", mode="playback")  # type: ignore[arg-type]


# --- matching -----------------------------------------------------------


async def test_json_key_order_does_not_change_the_match(tmp_path):
    """Dict ordering is not part of a request's meaning; matching on raw bytes
    would make cassettes miss for a reason nobody can see."""
    path = tmp_path / "order.json"
    with Cassette(path, mode="record") as rec:
        async with httpx.AsyncClient(transport=rec.transport(ok_json({"v": 1}))) as c:
            await c.post("https://api.example.com/v1/x", json={"a": 1, "b": 2})

    replay = Cassette(path, mode="replay")
    async with httpx.AsyncClient(transport=replay.transport()) as c:
        response = await c.post("https://api.example.com/v1/x", json={"b": 2, "a": 1})
    assert response.json() == {"v": 1}


async def test_a_changed_body_is_a_different_request(tmp_path):
    path = tmp_path / "body.json"
    with Cassette(path, mode="record") as rec:
        async with httpx.AsyncClient(transport=rec.transport(ok_json({"v": 1}))) as c:
            await c.post("https://api.example.com/v1/x", json={"prompt": "one"})

    replay = Cassette(path, mode="replay")
    async with httpx.AsyncClient(transport=replay.transport()) as c:
        with pytest.raises(CassetteMiss):
            await c.post("https://api.example.com/v1/x", json={"prompt": "two"})


async def test_a_changed_user_agent_does_not_break_a_cassette(tmp_path):
    """Matching on the header set is how cassette suites rot on a version bump."""
    path = tmp_path / "ua.json"
    with Cassette(path, mode="record") as rec:
        async with httpx.AsyncClient(
            transport=rec.transport(ok_json({"v": 1})), headers={"user-agent": "old/1.0"}
        ) as c:
            await c.post("https://api.example.com/v1/x", json={"p": 1})

    replay = Cassette(path, mode="replay")
    async with httpx.AsyncClient(
        transport=replay.transport(), headers={"user-agent": "new/9.9"}
    ) as c:
        assert (await c.post("https://api.example.com/v1/x", json={"p": 1})).json() == {
            "v": 1
        }


async def test_repeated_identical_requests_replay_their_recorded_sequence(tmp_path):
    """A retry that failed then succeeded must replay as failed-then-succeeded,
    or the cassette cannot exercise a retry path at all."""
    path = tmp_path / "seq.json"
    statuses = iter([503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses), json={"n": 1})

    with Cassette(path, mode="record") as rec:
        async with httpx.AsyncClient(
            transport=rec.transport(httpx.MockTransport(handler))
        ) as c:
            await c.post("https://api.example.com/v1/x", json={"p": 1})
            await c.post("https://api.example.com/v1/x", json={"p": 1})

    replay = Cassette(path, mode="replay")
    async with httpx.AsyncClient(transport=replay.transport()) as c:
        first = await c.post("https://api.example.com/v1/x", json={"p": 1})
        second = await c.post("https://api.example.com/v1/x", json={"p": 1})
    assert (first.status_code, second.status_code) == (503, 200)


async def test_error_responses_are_recorded_not_dropped(tmp_path):
    """A cassette that only stores 200s cannot test the error paths, which are
    the ones worth testing."""
    path = tmp_path / "err.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "slow down"})

    with Cassette(path, mode="record") as rec:
        async with httpx.AsyncClient(
            transport=rec.transport(httpx.MockTransport(handler))
        ) as c:
            await c.post("https://api.example.com/v1/x", json={"p": 1})

    replay = Cassette(path, mode="replay")
    async with httpx.AsyncClient(transport=replay.transport()) as c:
        assert (await c.post("https://api.example.com/v1/x", json={"p": 1})).status_code == 429
