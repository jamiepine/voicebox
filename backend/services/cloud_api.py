"""
HTTP client for the Voicebox Cloud API (api.voicebox.sh).

A thin async wrapper over the bearer-key endpoints the sync client needs:
device/key distribution, the encrypted object store, and the sync feed. Every
payload sent through here is ciphertext or metadata about ciphertext — the
encryption itself happens in ``cloud_crypto`` before bytes reach this module.

Blob bytes don't flow through the API at all: pushes receive presigned PUT
URLs and pulls receive presigned GET URLs, and the client transfers ciphertext
directly with the storage host. Those transfers use a separate unauthenticated
HTTP client so the bearer key is never sent to the storage host.
"""

import contextlib
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0
_BLOB_TIMEOUT = 120.0  # audio assets can be tens of MB


class CloudApiError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class CloudApiClient:
    """One authenticated session against the cloud API. Use as an async context
    manager so both underlying connection pools are closed."""

    def __init__(self, api_url: str, api_key: str, *, transport: httpx.AsyncBaseTransport | None = None):
        self._api = httpx.AsyncClient(
            base_url=api_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_TIMEOUT,
            transport=transport,
        )
        # Presigned-URL transfers: no Authorization header, longer timeout.
        self._blobs = httpx.AsyncClient(timeout=_BLOB_TIMEOUT, transport=transport)

    async def __aenter__(self) -> "CloudApiClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._api.aclose()
        await self._blobs.aclose()

    async def _call(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
    ) -> Any:
        try:
            resp = await self._api.request(method, path, json=json, params=params)
        except httpx.HTTPError as err:
            raise CloudApiError(f"could not reach Voicebox Cloud: {err}") from err
        if resp.status_code >= 400:
            message = f"{method} {path} failed ({resp.status_code})"
            with contextlib.suppress(ValueError):
                message = resp.json().get("error", {}).get("message", message)
            raise CloudApiError(message, status=resp.status_code)
        payload = resp.json()
        if not payload.get("ok"):
            raise CloudApiError(f"{method} {path} returned ok=false", status=resp.status_code)
        return payload.get("data")

    # -- account ------------------------------------------------------------

    async def me(self) -> dict:
        return await self._call("GET", "/v1/account/me")

    # -- devices & key distribution ------------------------------------------

    async def register_device(self, name: str, public_key_b64: str) -> dict:
        """Returns {deviceId, accountHasKey}."""
        return await self._call("POST", "/v1/devices", json={"name": name, "publicKey": public_key_b64})

    async def list_devices(self) -> list[dict]:
        return await self._call("GET", "/v1/devices")

    async def put_wrapped_key(self, device_id: str, wrapped_master_key_b64: str) -> None:
        await self._call(
            "POST",
            f"/v1/devices/{device_id}/wrapped-key",
            json={"wrappedMasterKey": wrapped_master_key_b64},
        )

    async def get_wrapped_key(self, device_id: str) -> str | None:
        data = await self._call("GET", f"/v1/devices/{device_id}/wrapped-key")
        return data.get("wrappedMasterKey") if data else None

    async def put_account_key(self, recovery_wrapped_key_b64: str, kdf_salt_b64: str, kdf_params: str) -> None:
        await self._call(
            "PUT",
            "/v1/devices/account-key",
            json={
                "recoveryWrappedKey": recovery_wrapped_key_b64,
                "kdfSalt": kdf_salt_b64,
                "kdfParams": kdf_params,
            },
        )

    async def get_account_key(self) -> dict | None:
        """Returns {recoveryWrappedKey, kdfSalt, kdfParams} or None if the
        account has no escrow yet."""
        return await self._call("GET", "/v1/devices/account-key")

    # -- encrypted object store ----------------------------------------------

    async def push_object(
        self,
        *,
        kind: str,
        client_id: str,
        version: int,
        record: dict | None,
        assets: list[dict] | None = None,
    ) -> dict:
        """Upsert object metadata. ``record`` is {hash, size}; each asset is
        {role, clientAssetId, hash, size}. Returns {objectId, seq, uploads},
        where uploads lists presigned PUT URLs for exactly the changed blobs."""
        return await self._call(
            "POST",
            "/v1/objects",
            json={
                "kind": kind,
                "clientId": client_id,
                "version": version,
                "record": record,
                "assets": assets or [],
            },
        )

    async def commit_object(self, object_id: str) -> None:
        """Ask the server to verify all claimed blobs actually landed in storage."""
        await self._call("POST", f"/v1/objects/{object_id}/commit")

    async def delete_object(self, object_id: str) -> None:
        await self._call("DELETE", f"/v1/objects/{object_id}")

    async def get_changes(self, since: int, limit: int = 200) -> dict:
        """Sync pull: {changes, cursor, hasMore} for everything newer than ``since``."""
        return await self._call("GET", "/v1/sync/changes", params={"since": since, "limit": limit})

    # -- blob transfer (presigned URLs, ciphertext only) ----------------------

    async def upload_blob(self, url: str, data: bytes) -> None:
        try:
            resp = await self._blobs.put(url, content=data, headers={"Content-Type": "application/octet-stream"})
        except httpx.HTTPError as err:
            raise CloudApiError(f"blob upload failed: {err}") from err
        if resp.status_code >= 400:
            raise CloudApiError(f"blob upload rejected ({resp.status_code})", status=resp.status_code)

    async def download_blob(self, url: str) -> bytes:
        try:
            resp = await self._blobs.get(url)
        except httpx.HTTPError as err:
            raise CloudApiError(f"blob download failed: {err}") from err
        if resp.status_code >= 400:
            raise CloudApiError(f"blob download rejected ({resp.status_code})", status=resp.status_code)
        return resp.content
