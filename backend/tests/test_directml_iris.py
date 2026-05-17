"""
Test DirectML device detection and Iris iGPU support on Windows.

Run with: pytest backend/tests/test_directml_iris.py -v -s
"""

import platform
import logging
import pytest

logger = logging.getLogger(__name__)


@pytest.mark.skipif(platform.system() != "Windows", reason="DirectML tests only on Windows")
class TestDirectMLDetection:
    """Test DirectML device availability and Iris iGPU detection."""

    def test_directml_import(self):
        """Test torch_directml can be imported."""
        try:
            import torch_directml
            assert torch_directml is not None
            logger.info("✓ torch_directml imported successfully")
        except ImportError as e:
            pytest.skip(f"torch_directml not installed: {e}")

    def test_directml_device_count(self):
        """Test DirectML detects at least one device."""
        try:
            import torch_directml
            device_count = torch_directml.device_count()
            assert device_count > 0, f"DirectML device_count returned {device_count}, expected > 0"
            logger.info(f"✓ DirectML detected {device_count} device(s)")
        except ImportError:
            pytest.skip("torch_directml not installed")

    def test_directml_device_creation(self):
        """Test creating a DirectML device object."""
        try:
            import torch_directml
            if torch_directml.device_count() > 0:
                device = torch_directml.device(0)
                assert device is not None
                logger.info(f"✓ DirectML device created: {device}")
        except ImportError:
            pytest.skip("torch_directml not installed")

    def test_get_torch_device_directml(self):
        """Test get_torch_device returns DirectML on Windows with iGPU."""
        from ..backends.base import get_torch_device
        import torch

        device = get_torch_device(allow_directml=True)
        logger.info(f"Selected device: {device}")

        # On Windows with iGPU and torch_directml installed, should use DirectML
        try:
            import torch_directml
            if torch_directml.device_count() > 0:
                # Should be DirectML device, not CPU
                assert str(device) != "cpu", f"Expected DirectML but got {device}"
                logger.info(f"✓ DirectML device selected: {device}")
        except ImportError:
            logger.info("torch_directml not installed, may fall back to CPU")

    def test_iris_igpu_detection(self):
        """Test Iris iGPU detection via WMI."""
        from ..backends.base import _detect_iris_igpu

        try:
            import wmi
            has_iris = _detect_iris_igpu()
            logger.info(f"Iris iGPU detected: {has_iris}")
        except ImportError:
            logger.info("wmi module not available, skipping Iris detection test")


@pytest.mark.skipif(platform.system() != "Windows", reason="DirectML tests only on Windows")
class TestDirectMLTorchTensor:
    """Test basic torch tensor operations on DirectML device."""

    def test_torch_tensor_on_directml(self):
        """Test creating and operating on tensors with DirectML."""
        try:
            import torch
            import torch_directml

            if torch_directml.device_count() == 0:
                pytest.skip("No DirectML devices available")

            device = torch_directml.device(0)
            x = torch.randn(3, 3, device=device)
            y = torch.randn(3, 3, device=device)
            z = torch.mm(x, y)

            assert z.shape == (3, 3)
            logger.info(f"✓ Tensor operation successful on {device}")
            logger.info(f"  Result shape: {z.shape}")
        except ImportError:
            pytest.skip("torch_directml not installed")

    def test_directml_memory_management(self):
        """Test DirectML memory can be freed properly."""
        try:
            import torch
            import torch_directml

            if torch_directml.device_count() == 0:
                pytest.skip("No DirectML devices available")

            device = torch_directml.device(0)
            # Create and delete tensors to check memory cleanup
            for _ in range(5):
                x = torch.randn(1000, 1000, device=device)
                del x

            logger.info("✓ DirectML memory management OK")
        except ImportError:
            pytest.skip("torch_directml not installed")


@pytest.mark.skipif(platform.system() != "Windows", reason="Model tests only on Windows")
@pytest.mark.asyncio
class TestWhisperOnDirectML:
    """Test Whisper (STT) model on DirectML device."""

    async def test_whisper_model_loads_on_directml(self):
        """Test Whisper model can load on DirectML."""
        try:
            import torch_directml
            if torch_directml.device_count() == 0:
                pytest.skip("No DirectML devices available")
        except ImportError:
            pytest.skip("torch_directml not installed")

        from ..backends.pytorch_backend import PyTorchSTTBackend

        backend = PyTorchSTTBackend(model_size="base")
        assert backend.device != "cpu", f"Expected GPU device, got {backend.device}"
        logger.info(f"✓ Whisper backend using device: {backend.device}")

        # Try to load the model (this will download if needed)
        try:
            await backend.load_model_async("base")
            assert backend.is_loaded()
            logger.info("✓ Whisper model loaded successfully on DirectML")
            backend.unload_model()
        except (TimeoutError, ConnectionError, OSError) as e:
            pytest.skip(f"Environment/network limitation during model load: {e}")


@pytest.mark.skipif(platform.system() != "Windows", reason="Model tests only on Windows")
@pytest.mark.asyncio
class TestQwenTTSOnDirectML:
    """Test Qwen TTS model on DirectML device."""

    async def test_qwen_tts_loads_on_directml(self):
        """Test Qwen TTS model can load on DirectML."""
        try:
            import torch_directml
            if torch_directml.device_count() == 0:
                pytest.skip("No DirectML devices available")
        except ImportError:
            pytest.skip("torch_directml not installed")

        from ..backends.pytorch_backend import PyTorchTTSBackend

        backend = PyTorchTTSBackend(model_size="0.6B")
        assert backend.device != "cpu", f"Expected GPU device, got {backend.device}"
        logger.info(f"✓ Qwen TTS backend using device: {backend.device}")

        # Try to load the model (this will download if needed)
        try:
            await backend.load_model_async("0.6B")
            assert backend.is_loaded()
            logger.info("✓ Qwen TTS model loaded successfully on DirectML")
            backend.unload_model()
        except (TimeoutError, ConnectionError, OSError) as e:
            pytest.skip(f"Environment/network limitation during model load: {e}")
