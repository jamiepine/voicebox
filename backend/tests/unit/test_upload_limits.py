"""Unit tests for the chunked-upload size-enforcement logic.

The upload helpers in routes/captures.py, routes/profiles.py, and
routes/history.py all share the same pattern:

    while chunk := await file.read(CHUNK_SIZE):
        total += len(chunk)
        if total > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, ...)
        chunks.append(chunk)

These tests verify the boundary conditions (exactly at the limit, one byte
over, well within) using a pure-Python helper that mirrors the logic without
any FastAPI or IO imports.
"""


# ---------------------------------------------------------------------------
# Mirror the accumulation logic as a pure function under test
# ---------------------------------------------------------------------------


def _simulate_chunked_read(
    chunks: list[bytes],
    max_bytes: int,
) -> tuple[bool, int]:
    """Simulate the chunked-read loop.

    Returns:
        (rejected, total_bytes_read)
        rejected=True  when the limit was exceeded (413 would be raised)
        rejected=False when the upload was accepted
    """
    total = 0
    for chunk in chunks:
        total += len(chunk)
        if total > max_bytes:
            return True, total
    return False, total


# ---------------------------------------------------------------------------
# Fixtures — common limits mirrored from the route files
# ---------------------------------------------------------------------------

CAPTURE_MAX = 500 * 1024 * 1024  # 500 MB  (captures.py)
PROFILE_MAX = 100 * 1024 * 1024  # 100 MB  (profiles.py)
HISTORY_MAX = 50 * 1024 * 1024  #  50 MB  (history.py import)
CHUNK_SIZE = 1 * 1024 * 1024  #   1 MB


def _make_chunks(total_bytes: int, chunk_size: int = CHUNK_SIZE) -> list[bytes]:
    """Return a list of byte-chunks that sum to *total_bytes*."""
    chunks = []
    remaining = total_bytes
    while remaining > 0:
        size = min(chunk_size, remaining)
        chunks.append(b"\x00" * size)
        remaining -= size
    return chunks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCaptureUploadLimit:
    MAX = CAPTURE_MAX

    def test_exactly_at_limit_is_accepted(self):
        rejected, total = _simulate_chunked_read(_make_chunks(self.MAX), self.MAX)
        assert not rejected
        assert total == self.MAX

    def test_one_byte_over_is_rejected(self):
        rejected, _ = _simulate_chunked_read(_make_chunks(self.MAX + 1), self.MAX)
        assert rejected

    def test_well_within_limit_is_accepted(self):
        rejected, total = _simulate_chunked_read(_make_chunks(1024), self.MAX)
        assert not rejected
        assert total == 1024

    def test_empty_upload_is_accepted_by_loop(self):
        """An empty file produces zero chunks — the loop body never executes."""
        rejected, total = _simulate_chunked_read([], self.MAX)
        assert not rejected
        assert total == 0


class TestProfileImportLimit:
    MAX = PROFILE_MAX

    def test_exactly_at_limit_is_accepted(self):
        rejected, _ = _simulate_chunked_read(_make_chunks(self.MAX), self.MAX)
        assert not rejected

    def test_one_byte_over_is_rejected(self):
        rejected, _ = _simulate_chunked_read(_make_chunks(self.MAX + 1), self.MAX)
        assert rejected

    def test_capture_sized_file_is_rejected(self):
        """A file legal for captures (500 MB) is too big for profile import (100 MB)."""
        rejected, _ = _simulate_chunked_read(_make_chunks(CAPTURE_MAX), self.MAX)
        assert rejected


class TestHistoryImportLimit:
    MAX = HISTORY_MAX

    def test_exactly_at_limit_is_accepted(self):
        rejected, _ = _simulate_chunked_read(_make_chunks(self.MAX), self.MAX)
        assert not rejected

    def test_one_byte_over_is_rejected(self):
        rejected, _ = _simulate_chunked_read(_make_chunks(self.MAX + 1), self.MAX)
        assert rejected

    def test_profile_sized_file_is_rejected(self):
        """A file legal for profile import (100 MB) is too big for history import (50 MB)."""
        rejected, _ = _simulate_chunked_read(_make_chunks(PROFILE_MAX), self.MAX)
        assert rejected


class TestChunkBoundaryEdgeCases:
    """Verify the loop correctly catches overflows mid-stream."""

    def test_overflow_detected_mid_stream(self):
        """Overflow that only becomes visible after the second chunk is caught."""
        # Limit is 3 bytes; first chunk 2 bytes (ok), second chunk 2 bytes (over)
        chunks = [b"\x00" * 2, b"\x00" * 2]
        rejected, total = _simulate_chunked_read(chunks, max_bytes=3)
        assert rejected
        assert total == 4

    def test_exact_multi_chunk_is_accepted(self):
        """Three chunks of 1 byte each exactly at a 3-byte limit."""
        chunks = [b"\x00", b"\x00", b"\x00"]
        rejected, total = _simulate_chunked_read(chunks, max_bytes=3)
        assert not rejected
        assert total == 3
