"""Tests for the Telnyx PSTN sink's wire contract and webhook handling.

The two bugs these pin first are the ones that made the sink a no-op against
the live API: ``POST /v2/calls`` requires ``connection_id`` alongside ``to``
and ``from``, and the playback action's field is ``audio_url`` (there is no
``media_url`` on ``playback_start``). Both failed with a 422 that the old
code surfaced as an opaque 500, so the request bodies are asserted directly.

The rest cover the webhook gate: a per-call token compared in constant time,
and single-fire playback under Telnyx's at-least-once webhook delivery.
"""

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services import sinks  # noqa: E402


class _Row:
    """Minimal stand-in for the TelnyxSettings/TelnyxCall ORM rows."""

    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)


class _FakeDB:
    """Records commits so claim-ordering can be asserted without SQLite."""

    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1


def _client_with(capture, *, status=200, json_body=None):
    """Build a TelnyxClient whose transport records the outgoing request."""

    def handler(request: httpx.Request) -> httpx.Response:
        capture["url"] = str(request.url)
        capture["body"] = request.read().decode() or ""
        return httpx.Response(status, json=json_body if json_body is not None else {})

    client = sinks.TelnyxClient("test-key")
    client._client = httpx.AsyncClient(
        base_url=sinks.TELNYX_API_BASE,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test-key"},
    )
    return client


# ─── Wire contract ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_call_sends_connection_id():
    """connection_id is required by the Dial API; omitting it 422s every call."""
    capture = {}
    body = {"data": {"call_control_id": "ccid-1"}}
    async with _client_with(capture, json_body=body) as client:
        data = await client.create_call(
            to="+15551234567",
            from_="+15557654321",
            connection_id="conn-abc",
            webhook_url="https://example.test/sinks/telnyx/webhook?token=t",
        )

    import json

    sent = json.loads(capture["body"])
    assert sent["connection_id"] == "conn-abc"
    assert sent["to"] == "+15551234567"
    assert sent["from"] == "+15557654321"
    assert data["call_control_id"] == "ccid-1"


@pytest.mark.asyncio
async def test_playback_uses_audio_url_not_media_url():
    """The playback_start action takes `audio_url`; `media_url` is not a field."""
    capture = {}
    async with _client_with(capture) as client:
        await client.playback_start("ccid-1", "https://example.test/audio/gen-1")

    import json

    sent = json.loads(capture["body"])
    assert sent["audio_url"] == "https://example.test/audio/gen-1"
    assert "media_url" not in sent


@pytest.mark.asyncio
async def test_telnyx_error_detail_is_surfaced():
    """A 422 should carry Telnyx's explanation, not collapse into a bare 500."""
    capture = {}
    errors = {"errors": [{"title": "Invalid connection", "detail": "not found"}]}
    async with _client_with(capture, status=422, json_body=errors) as client:
        with pytest.raises(ValueError) as exc:
            await client.playback_start("ccid-1", "https://example.test/audio/g")

    assert "Invalid connection" in str(exc.value)
    assert "not found" in str(exc.value)


@pytest.mark.asyncio
async def test_hangup_tolerates_already_ended_call():
    """404/422 on hangup means the call is already gone — not an error."""
    capture = {}
    async with _client_with(capture, status=404) as client:
        await client.hangup("ccid-1")  # must not raise


# ─── Configuration gate ────────────────────────────────────────────────────


def test_missing_settings_names_each_gap():
    row = _Row(
        enabled=True,
        api_key="k",
        connection_id=None,
        from_number=None,
        public_base_url="https://x.test",
    )
    assert sinks.missing_settings(row) == ["connection_id", "from_number"]


def test_missing_settings_reports_disabled_sink():
    """Fully credentialed but toggled off still can't dial — say so."""
    row = _Row(
        enabled=False,
        api_key="k",
        connection_id="c",
        from_number="+15551234567",
        public_base_url="https://x.test",
    )
    assert sinks.missing_settings(row) == ["enabled"]


def test_fully_configured_sink_has_no_gaps():
    row = _Row(
        enabled=True,
        api_key="k",
        connection_id="c",
        from_number="+15551234567",
        public_base_url="https://x.test",
    )
    assert sinks.missing_settings(row) == []


def test_mask_key_keeps_only_last_four():
    assert sinks.mask_key("abcdefghij") == "******ghij"
    assert sinks.mask_key(None) == ""


# ─── Webhook gate ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_webhook_rejects_wrong_token():
    row = _Row(call_control_id="ccid-1", webhook_secret="right", status="initiating")
    db = _FakeDB()
    db.query = lambda *a, **k: _Query(row)

    result = await sinks.handle_telnyx_webhook(
        body={"data": {"event_type": "call.answered",
                       "payload": {"call_control_id": "ccid-1"}}},
        token="wrong",
        db=db,
    )
    assert result == {"ok": False, "reason": "bad_token"}
    assert row.status == "initiating"  # never claimed


@pytest.mark.asyncio
async def test_webhook_fails_closed_when_row_has_no_secret():
    """A row without a stored secret must not be treated as trusted."""
    row = _Row(call_control_id="ccid-1", webhook_secret=None, status="initiating")
    db = _FakeDB()
    db.query = lambda *a, **k: _Query(row)

    result = await sinks.handle_telnyx_webhook(
        body={"data": {"event_type": "call.answered",
                       "payload": {"call_control_id": "ccid-1"}}},
        token=None,
        db=db,
    )
    assert result == {"ok": False, "reason": "bad_token"}


@pytest.mark.asyncio
async def test_unknown_call_is_acked_without_work():
    db = _FakeDB()
    db.query = lambda *a, **k: _Query(None)

    result = await sinks.handle_telnyx_webhook(
        body={"data": {"event_type": "call.answered",
                       "payload": {"call_control_id": "nope"}}},
        token="whatever",
        db=db,
    )
    assert result == {"ok": True, "reason": "unknown_call"}


def test_playback_claim_fires_once():
    """Telnyx retries webhooks; only the first delivery may start playback."""
    row = _Row(status="initiating")
    db = _FakeDB()

    assert sinks._claim_for_playback(row, db) is True
    assert row.status == "preparing"
    # A retry arriving while the first is still preparing must not double-fire.
    assert sinks._claim_for_playback(row, db) is False


@pytest.mark.parametrize("status", ["preparing", "playing", "playback_ended", "hangup", "failed"])
def test_playback_claim_refused_once_advanced(status):
    assert sinks._claim_for_playback(_Row(status=status), _FakeDB()) is False


class _Query:
    """Chainable query stub returning a fixed row."""

    def __init__(self, row):
        self._row = row

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._row
