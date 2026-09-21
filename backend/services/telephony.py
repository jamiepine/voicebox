"""Telephony providers for the voice agent.

The agent core only needs three things from the phone network: place a
call, hang it up, and tell whether the provider does its own audio
transport. Two implementations ship:

- :class:`LocalProvider` — no phone line. Agent speech plays through the
  machine's speakers (the same path ``voicebox.speak`` uses, pill and
  all) and the customer's side arrives through ``POST /calls/{id}/turn``
  (text or audio) or the MCP tools. Good for testing scripts, demos, and
  attended calling where a person holds the handset.
- :class:`TwilioProvider` — real PSTN calls over Twilio's REST API and
  TwiML webhooks. Implemented with ``httpx`` against the public API so
  there is no SDK to bundle. Needs ``TWILIO_ACCOUNT_SID``,
  ``TWILIO_AUTH_TOKEN`` and a publicly reachable ``VOICEBOX_PUBLIC_URL``
  for the webhooks (ngrok / a tunnel is fine for development).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
from dataclasses import dataclass
from typing import Protocol
from xml.sax.saxutils import escape as xml_escape

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised when a provider can't do what was asked (missing config,
    API failure). The agent core turns this into a failed call."""


@dataclass
class DialResult:
    provider_call_id: str | None
    # True when the provider streams the agent's audio itself (webhooks);
    # False when Voicebox should play it locally.
    remote_audio: bool


class CallProvider(Protocol):
    name: str

    async def dial(
        self, *, call_id: str, to_number: str, from_number: str | None, machine_detection: str = "Enable"
    ) -> DialResult: ...

    async def hangup(self, provider_call_id: str | None) -> None: ...

    async def send_sms(self, *, to_number: str, from_number: str | None, body: str) -> str | None: ...


class LocalProvider:
    """Speakers + API. ``dial`` is a no-op; there is nothing to connect."""

    name = "local"

    async def dial(
        self, *, call_id: str, to_number: str, from_number: str | None, machine_detection: str = "Enable"
    ) -> DialResult:
        return DialResult(provider_call_id=None, remote_audio=False)

    async def hangup(self, provider_call_id: str | None) -> None:
        return None

    async def send_sms(self, *, to_number: str, from_number: str | None, body: str) -> str | None:
        raise ProviderError("The local provider cannot send text messages; use the Twilio provider.")


class TwilioProvider:
    """Outbound + inbound calls via Twilio Programmable Voice.

    The call flow is webhook-driven: Twilio fetches TwiML from
    ``/webhooks/twilio/{call_id}`` which ``<Play>``s the agent's latest
    turn and ``<Record>``s the customer's answer; the recording callback
    transcribes it, runs a turn, and returns the next TwiML.
    """

    name = "twilio"
    api_base = "https://api.twilio.com/2010-04-01"

    def __init__(
        self,
        account_sid: str | None = None,
        auth_token: str | None = None,
        public_url: str | None = None,
    ) -> None:
        self.account_sid = account_sid or os.environ.get("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.environ.get("TWILIO_AUTH_TOKEN")
        self.public_url = (public_url or os.environ.get("VOICEBOX_PUBLIC_URL") or "").rstrip("/")

    def _require_config(self) -> None:
        missing = [
            name
            for name, value in (
                ("TWILIO_ACCOUNT_SID", self.account_sid),
                ("TWILIO_AUTH_TOKEN", self.auth_token),
                ("VOICEBOX_PUBLIC_URL", self.public_url),
            )
            if not value
        ]
        if missing:
            raise ProviderError(
                "Twilio provider is not configured. Set the environment variables: " + ", ".join(missing)
            )

    def webhook_url(self, call_id: str, leg: str = "answer") -> str:
        return f"{self.public_url}/webhooks/twilio/{call_id}/{leg}"

    def turn_audio_url(self, call_id: str, turn_id: str) -> str:
        return f"{self.public_url}/calls/{call_id}/turns/{turn_id}/audio"

    async def dial(
        self, *, call_id: str, to_number: str, from_number: str | None, machine_detection: str = "Enable"
    ) -> DialResult:
        """``machine_detection`` is "Enable" (answer webhook fires as soon as a
        machine is suspected) or "DetectMessageEnd" (fires after the
        voicemail greeting, so a message can be left)."""
        self._require_config()
        if not from_number:
            raise ProviderError("Twilio provider needs a `from_number` on the agent (a Twilio-owned number).")
        import httpx  # lazy: keep import cost off the hot path for local users

        url = f"{self.api_base}/Accounts/{self.account_sid}/Calls.json"
        data = {
            "To": to_number,
            "From": from_number,
            "Url": self.webhook_url(call_id, "answer"),
            "StatusCallback": self.webhook_url(call_id, "status"),
            "StatusCallbackEvent": "completed",
            "MachineDetection": machine_detection,
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, data=data, auth=(self.account_sid, self.auth_token))
        if resp.status_code >= 300:
            raise ProviderError(f"Twilio dial failed ({resp.status_code}): {resp.text[:300]}")
        sid = resp.json().get("sid")
        logger.info("Twilio call %s created for voice-agent call %s", sid, call_id)
        return DialResult(provider_call_id=sid, remote_audio=True)

    async def hangup(self, provider_call_id: str | None) -> None:
        if not provider_call_id:
            return
        self._require_config()
        import httpx  # lazy

        url = f"{self.api_base}/Accounts/{self.account_sid}/Calls/{provider_call_id}.json"
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, data={"Status": "completed"}, auth=(self.account_sid, self.auth_token))
        if resp.status_code >= 300:
            logger.warning("Twilio hangup for %s returned %s", provider_call_id, resp.status_code)

    async def send_sms(self, *, to_number: str, from_number: str | None, body: str) -> str | None:
        self._require_config()
        if not from_number:
            raise ProviderError("Twilio provider needs a `from_number` on the agent to send SMS.")
        import httpx  # lazy

        url = f"{self.api_base}/Accounts/{self.account_sid}/Messages.json"
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                url, data={"To": to_number, "From": from_number, "Body": body}, auth=(self.account_sid, self.auth_token)
            )
        if resp.status_code >= 300:
            raise ProviderError(f"Twilio SMS failed ({resp.status_code}): {resp.text[:300]}")
        return resp.json().get("sid")

    async def fetch_recording(self, recording_url: str) -> bytes:
        """Download a recording (Twilio serves WAV when ``.wav`` is appended)."""
        self._require_config()
        import httpx  # lazy

        url = recording_url if recording_url.endswith(".wav") else recording_url + ".wav"
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(url, auth=(self.account_sid, self.auth_token))
        if resp.status_code >= 300:
            raise ProviderError(f"Could not fetch Twilio recording ({resp.status_code})")
        return resp.content

    # ── TwiML ─────────────────────────────────────────────────────────

    def twiml_play_and_record(self, call_id: str, turn_id: str, *, max_seconds: int = 30) -> str:
        """Speak the agent's turn, then record the customer's reply."""
        play = xml_escape(self.turn_audio_url(call_id, turn_id))
        action = xml_escape(self.webhook_url(call_id, "recording"))
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f"<Play>{play}</Play>"
            f'<Record action="{action}" method="POST" maxLength="{max_seconds}" timeout="3" '
            'playBeep="false" trim="trim-silence" />'
            "</Response>"
        )

    def twiml_play_and_hangup(self, call_id: str, turn_id: str) -> str:
        play = xml_escape(self.turn_audio_url(call_id, turn_id))
        return f'<?xml version="1.0" encoding="UTF-8"?><Response><Play>{play}</Play><Hangup/></Response>'

    def twiml_play_and_dial(self, call_id: str, turn_id: str, number: str, *, caller_id: str | None = None) -> str:
        """Warm transfer: play the hand-off line, then bridge to a person."""
        play = xml_escape(self.turn_audio_url(call_id, turn_id))
        cid = f' callerId="{xml_escape(caller_id)}"' if caller_id else ""
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<Response><Play>{play}</Play><Dial{cid} timeout="30">{xml_escape(number)}</Dial></Response>'
        )

    def twiml_pause_and_retry(self, call_id: str, *, seconds: int = 2) -> str:
        """TTS isn't ready yet — hold the line briefly and re-fetch."""
        redirect = xml_escape(self.webhook_url(call_id, "answer"))
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<Response><Pause length="{seconds}"/><Redirect method="POST">{redirect}</Redirect></Response>'
        )

    @staticmethod
    def twiml_hangup() -> str:
        return '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'

    # ── Signature validation ──────────────────────────────────────────

    def validate_signature(self, url: str, params: dict[str, str], signature: str | None) -> bool:
        """Twilio signs webhooks with HMAC-SHA1 over URL + sorted form params.

        https://www.twilio.com/docs/usage/webhooks/webhooks-security
        """
        if not self.auth_token or not signature:
            return False
        payload = url + "".join(k + params[k] for k in sorted(params))
        digest = hmac.new(self.auth_token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha1).digest()
        expected = base64.b64encode(digest).decode("ascii")
        return hmac.compare_digest(expected, signature)


_providers: dict[str, CallProvider] = {}


def get_provider(name: str) -> CallProvider:
    """Provider registry. Instances are cached; Twilio reads its env on
    first use so a user can set the variables and restart."""
    key = (name or "local").lower()
    if key in _providers:
        return _providers[key]
    if key == "local":
        provider: CallProvider = LocalProvider()
    elif key == "twilio":
        provider = TwilioProvider()
    else:
        raise ProviderError(f"Unknown telephony provider '{name}'. Use 'local' or 'twilio'.")
    _providers[key] = provider
    return provider
