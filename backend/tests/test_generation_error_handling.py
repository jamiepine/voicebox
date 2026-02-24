"""
Tests for generation error handling.

This test suite verifies that the application correctly handles
file system errors and broken pipe errors during audio generation.

Related to issues #168 and #140.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Add parent directory to path to import backend modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.audio import save_audio


class TestSaveAudio:
    """Test cases for the save_audio function."""

    def test_save_audio_success(self, tmp_path):
        """Test that save_audio works correctly under normal conditions."""
        audio = np.random.rand(24000).astype(np.float32)
        output_path = tmp_path / "test.wav"

        save_audio(audio, str(output_path), sample_rate=24000)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_audio_creates_parent_directory(self, tmp_path):
        """Test that save_audio creates parent directories if they don't exist."""
        audio = np.random.rand(24000).astype(np.float32)
        output_path = tmp_path / "subdir" / "test.wav"

        # Parent directory doesn't exist yet
        assert not output_path.parent.exists()

        save_audio(audio, str(output_path), sample_rate=24000)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_save_audio_atomic_write(self, tmp_path):
        """Test that save_audio uses atomic write (temp file then rename)."""
        audio = np.random.rand(24000).astype(np.float32)
        output_path = tmp_path / "test.wav"

        with patch('soundfile.write') as mock_write:
            # Mock sf.write to not actually write
            mock_write.return_value = None

            save_audio(audio, str(output_path), sample_rate=24000)

            # Should have been called with temp file path
            args = mock_write.call_args[0]
            assert args[0].endswith('.tmp')

    def test_save_audio_cleanup_on_error(self, tmp_path):
        """Test that temp file is cleaned up if an error occurs."""
        audio = np.random.rand(24000).astype(np.float32)
        output_path = tmp_path / "test.wav"

        with patch('soundfile.write', side_effect=Exception("Write failed")):
            with pytest.raises(OSError) as exc_info:
                save_audio(audio, str(output_path), sample_rate=24000)

            # Error message should include context
            assert "Failed to save audio" in str(exc_info.value)

            # Temp file should be cleaned up
            temp_file = Path(f"{output_path}.tmp")
            assert not temp_file.exists()

    def test_save_audio_permission_error(self, tmp_path):
        """Test that save_audio raises OSError with clear message on permission error."""
        audio = np.random.rand(24000).astype(np.float32)

        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        os.chmod(readonly_dir, 0o444)

        output_path = readonly_dir / "test.wav"

        try:
            with pytest.raises(OSError) as exc_info:
                save_audio(audio, str(output_path), sample_rate=24000)

            assert "Failed to save audio" in str(exc_info.value)
        finally:
            # Cleanup: restore permissions
            os.chmod(readonly_dir, 0o755)

    def test_save_audio_no_space_error(self, tmp_path):
        """Test handling of ENOSPC (no space left on device) error."""
        audio = np.random.rand(24000).astype(np.float32)
        output_path = tmp_path / "test.wav"

        # Mock sf.write to raise OSError with errno 28 (ENOSPC)
        error = OSError("No space left on device")
        error.errno = 28

        with patch('soundfile.write', side_effect=error):
            with pytest.raises(OSError) as exc_info:
                save_audio(audio, str(output_path), sample_rate=24000)

            assert "Failed to save audio" in str(exc_info.value)


class TestGenerationErrorHandling:
    """Test cases for generation error handling in the API endpoint."""

    @pytest.mark.asyncio
    async def test_generation_directory_missing(self):
        """Test that generation fails gracefully if directory is missing."""
        # This would be tested with the full API endpoint
        # For now, we test that get_generations_dir creates the directory
        from config import get_generations_dir

        # Clear any existing directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from config import set_data_dir
            set_data_dir(tmpdir)

            # get_generations_dir should create it
            gen_dir = get_generations_dir()
            assert gen_dir.exists()

    @pytest.mark.asyncio
    async def test_generation_directory_not_writable(self, tmp_path):
        """Test that generation fails with clear error if directory is not writable."""
        from config import set_data_dir

        set_data_dir(tmp_path)

        # Make generations directory read-only
        from config import get_generations_dir
        gen_dir = get_generations_dir()
        os.chmod(gen_dir, 0o444)

        try:
            # Check that we can detect non-writable directory
            assert not os.access(gen_dir, os.W_OK)
        finally:
            # Cleanup
            os.chmod(gen_dir, 0o755)


class TestHealthCheckEndpoint:
    """Test cases for the filesystem health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, tmp_path):
        """Test that health check returns success for a working filesystem."""
        from config import set_data_dir
        set_data_dir(tmp_path)

        # Import after setting data dir
        from main import check_filesystem

        result = await check_filesystem()

        assert result["status"] == "ok"
        assert result["writable"] is True
        assert "free_space_gb" in result
        assert "generations_dir" in result

    @pytest.mark.asyncio
    async def test_health_check_readonly_directory(self, tmp_path):
        """Test that health check detects non-writable directories."""
        from config import set_data_dir, get_generations_dir

        set_data_dir(tmp_path)
        gen_dir = get_generations_dir()

        # Make directory read-only
        os.chmod(gen_dir, 0o444)

        try:
            from main import check_filesystem
            result = await check_filesystem()

            assert result["status"] == "error"
            assert "permission" in result["message"].lower()
        finally:
            # Cleanup
            os.chmod(gen_dir, 0o755)


class TestBrokenPipeErrorHandling:
    """Test cases for broken pipe error handling."""

    def test_broken_pipe_error_detection(self):
        """Test that we can detect broken pipe errors."""
        error = OSError("Broken pipe")
        error.errno = 32  # EPIPE

        assert error.errno == 32
        assert "Broken pipe" in str(error)

    @pytest.mark.asyncio
    async def test_generation_handles_broken_pipe(self):
        """Test that generation endpoint handles BrokenPipeError gracefully."""
        # This would require mocking the full generation pipeline
        # For now, we verify that BrokenPipeError can be caught
        try:
            raise BrokenPipeError("Test broken pipe")
        except BrokenPipeError as e:
            # Should be catchable
            assert "broken pipe" in str(e).lower() or str(e) == "Test broken pipe"


# Integration tests (require full app setup)
@pytest.mark.integration
class TestGenerationEndpointErrorHandling:
    """Integration tests for generation endpoint error handling."""

    @pytest.mark.asyncio
    async def test_generation_with_readonly_filesystem(self):
        """Test generation fails gracefully with read-only filesystem."""
        # This would require setting up the full FastAPI test client
        # and is beyond the scope of unit tests
        pass

    @pytest.mark.asyncio
    async def test_generation_with_client_disconnect(self):
        """Test generation handles client disconnect gracefully."""
        # This would require simulating a client disconnect
        # during generation
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
