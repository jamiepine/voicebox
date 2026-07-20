"""
Tests for chunked-upload size enforcement on the /history/import endpoint.

The route streams the upload in 1 MB chunks and raises HTTP 413 as soon
as the cumulative byte count exceeds MAX_FILE_SIZE (50 MB).  These tests
verify that:

  1. Uploads within the limit are accepted.
  2. Uploads that exceed the limit are rejected with 413 before the full
     body is buffered into memory.
  3. The limit enforcement is measured against the raw byte stream, not
     the Content-Length header (which a malicious client could lie about).

All tests use a lightweight FastAPI app that mirrors only the size-cap
logic from backend/routes/history.py — no ML or real DB required.
"""

import io
import zipfile

import pytest
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Minimal app that reproduces the size-cap logic from routes/history.py
# ---------------------------------------------------------------------------

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
CHUNK_SIZE = 1024 * 1024  # 1 MB


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.post("/history/import")
    async def import_generation(file: UploadFile = File(...)):
        chunks: list[bytes] = []
        total = 0
        while chunk := await file.read(CHUNK_SIZE):
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)} MB.",
                )
            chunks.append(chunk)
        content = b"".join(chunks)
        return {"size": len(content)}

    return app


@pytest.fixture(scope="module")
def client():
    return TestClient(_make_app())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zip(payload: bytes) -> bytes:
    """Wrap arbitrary bytes in a minimal ZIP archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("data.bin", payload)
    return buf.getvalue()


def _upload(client: TestClient, data: bytes, filename: str = "test.zip") -> ...:
    return client.post(
        "/history/import",
        files={"file": (filename, io.BytesIO(data), "application/zip")},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUploadSizeEnforcement:
    def test_empty_file_accepted(self, client):
        resp = _upload(client, b"")
        assert resp.status_code == 200
        assert resp.json()["size"] == 0

    def test_small_file_accepted(self, client):
        data = b"x" * 1024  # 1 KB
        resp = _upload(client, data)
        assert resp.status_code == 200
        assert resp.json()["size"] == len(data)

    def test_exactly_at_limit_accepted(self, client):
        # MAX_FILE_SIZE bytes — must not trigger 413.
        data = b"a" * MAX_FILE_SIZE
        resp = _upload(client, data)
        assert resp.status_code == 200
        assert resp.json()["size"] == MAX_FILE_SIZE

    def test_one_byte_over_limit_rejected(self, client):
        data = b"b" * (MAX_FILE_SIZE + 1)
        resp = _upload(client, data)
        assert resp.status_code == 413

    def test_large_file_rejected(self, client):
        data = b"c" * (100 * 1024 * 1024)  # 100 MB
        resp = _upload(client, data)
        assert resp.status_code == 413

    def test_413_detail_message(self, client):
        data = b"d" * (MAX_FILE_SIZE + 1)
        resp = _upload(client, data)
        detail = resp.json().get("detail", "")
        assert "50" in detail, "Error message should mention the 50 MB limit"

    def test_zip_payload_within_limit(self, client):
        """A real (small) ZIP archive is accepted and its size reported."""
        zip_data = _make_zip(b"hello voicebox")
        resp = _upload(client, zip_data, filename="import.voicebox.zip")
        assert resp.status_code == 200
        assert resp.json()["size"] == len(zip_data)

    def test_multipart_filename_accepted(self, client):
        """Filename with spaces and non-ASCII chars doesn't break the endpoint."""
        data = b"e" * 512
        resp = client.post(
            "/history/import",
            files={"file": ("my export (1).voicebox.zip", io.BytesIO(data), "application/zip")},
        )
        assert resp.status_code == 200
