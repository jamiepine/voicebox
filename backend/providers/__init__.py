"""
Provider management system for TTS providers.
"""

from typing import Optional
import asyncio
import platform
from pathlib import Path

from .base import TTSProvider
from .types import ProviderType
from .bundled import BundledProvider
from .local import LocalProvider
from .installer import get_provider_binary_path, _get_providers_dir
from ..config import get_data_dir
import subprocess
import socket


class ProviderManager:
    """Manages TTS provider lifecycle."""
    
    def __init__(self):
        self.active_provider: Optional[TTSProvider] = None
        self._default_provider: Optional[TTSProvider] = None
        self._provider_process: Optional[subprocess.Popen] = None
        self._provider_port: Optional[int] = None
    
    def _get_default_provider(self) -> TTSProvider:
        """Get the default bundled provider."""
        if self._default_provider is None:
            self._default_provider = BundledProvider()
        return self._default_provider
    
    async def get_active_provider(self) -> TTSProvider:
        """
        Get the currently active provider.
        
        Returns:
            Active TTS provider instance
        """
        if self.active_provider is None:
            # Default to bundled provider
            self.active_provider = self._get_default_provider()
        return self.active_provider
    
    async def start_provider(self, provider_type: str) -> None:
        """
        Start a TTS provider.
        
        Args:
            provider_type: Type of provider to start
        """
        if provider_type == "apple-mlx":
            # Use bundled MLX provider
            self.active_provider = self._get_default_provider()
        elif provider_type in ["pytorch-cpu", "pytorch-cuda"]:
            # Try to start external provider subprocess if binary exists
            provider_path = get_provider_binary_path(provider_type)
            if provider_path and provider_path.exists():
                # External downloaded provider exists, start it
                # Find a free port
                port = self._get_free_port()

                # Start provider subprocess with stdout/stderr capture
                from ..config import get_data_dir
                import logging
                logger = logging.getLogger(__name__)

                logger.info(f"Starting provider {provider_type} on port {port}")
                logger.info(f"Provider binary: {provider_path}")
                logger.info(f"Data directory: {get_data_dir()}")

                # Create log files for provider output (easier debugging on Windows)
                logs_dir = get_data_dir() / "logs"
                logs_dir.mkdir(exist_ok=True)
                stdout_log = logs_dir / f"{provider_type}-stdout.log"
                stderr_log = logs_dir / f"{provider_type}-stderr.log"

                logger.info(f"Provider logs will be written to: {logs_dir}")

                process = subprocess.Popen(
                    [
                        str(provider_path),
                        "--port", str(port),
                        "--data-dir", str(get_data_dir()),
                    ],
                    stdout=open(stdout_log, 'w'),
                    stderr=open(stderr_log, 'w'),
                    text=True,
                    bufsize=1,
                )

                # Wait for provider to be ready
                base_url = f"http://127.0.0.1:{port}"
                try:
                    await self._wait_for_provider_health(base_url, timeout=30)
                except TimeoutError as e:
                    # Read log files for debugging (works on all platforms unlike select)
                    stdout_content = ""
                    stderr_content = ""

                    # Try to read available output (works on Windows and Unix)
                    try:
                        # Use non-blocking read with timeout
                        import threading
                        import queue

                        def enqueue_output(stream, queue):
                            try:
                                for line in iter(stream.readline, ''):
                                    queue.put(line)
                            except:
                                pass

                        stdout_queue = queue.Queue()
                        stderr_queue = queue.Queue()

                        if process.stdout:
                            t = threading.Thread(target=enqueue_output, args=(process.stdout, stdout_queue))
                            t.daemon = True
                            t.start()

                        if process.stderr:
                            t2 = threading.Thread(target=enqueue_output, args=(process.stderr, stderr_queue))
                            t2.daemon = True
                            t2.start()

                        # Give threads a moment to read
                        import time
                        time.sleep(0.5)

                        # Collect output
                        while not stdout_queue.empty():
                            stdout_lines.append(stdout_queue.get_nowait())
                        while not stderr_queue.empty():
                            stderr_lines.append(stderr_queue.get_nowait())
                    except Exception as ex:
                        logger.warning(f"Could not capture subprocess output: {ex}")

                    logger.error(f"Provider failed to start within 30 seconds")
                    logger.error(f"Check logs at: {logs_dir}")
                    if stdout_content:
                        logger.error(f"Stdout: {stdout_content[-2000:]}")  # Last 2000 chars
                    if stderr_content:
                        logger.error(f"Stderr: {stderr_content[-2000:]}")  # Last 2000 chars

                    # Terminate the process
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()

                    # Raise with log file location for user
                    raise TimeoutError(
                        f"Provider {provider_type} failed to start. Check logs at: {logs_dir}"
                    )

                # Create LocalProvider instance
                self.active_provider = LocalProvider(base_url)
                self._provider_process = process
                self._provider_port = port

                # Logs are written directly to files (stdout_log, stderr_log)
                # No need for background task - users can check {logs_dir} for debugging
            else:
                # No external binary, use bundled provider (if available)
                if provider_type == "pytorch-cpu":
                    # PyTorch CPU can use bundled backend
                    self.active_provider = self._get_default_provider()
                else:
                    raise ValueError(f"Provider {provider_type} is not installed. Please download it first.")
        elif provider_type == "remote":
            # Remote provider - will be implemented in Phase 5
            raise NotImplementedError("Remote provider not yet implemented")
        elif provider_type == "openai":
            # OpenAI provider - will be implemented in Phase 5
            raise NotImplementedError("OpenAI provider not yet implemented")
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    async def stop_provider(self) -> None:
        """Stop the active provider."""
        if self.active_provider:
            # Only stop if it's not the default bundled provider
            if self.active_provider is not self._default_provider:
                if hasattr(self.active_provider, 'stop'):
                    await self.active_provider.stop()
                self.active_provider = None
            
            # Stop subprocess if running
            if self._provider_process:
                self._provider_process.terminate()
                try:
                    self._provider_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._provider_process.kill()
                self._provider_process = None
                self._provider_port = None
    
    async def list_installed(self) -> list[str]:
        """
        List installed provider types.
        
        Returns:
            List of installed provider type strings
        """
        installed = []
        
        # Bundled providers are always available
        system = platform.system()
        machine = platform.machine()

        if system == "Darwin" and machine == "arm64":
            # Apple Silicon gets MLX bundled
            installed.append("apple-mlx")
        elif system == "Windows" or (system == "Darwin" and machine != "arm64"):
            # Windows and Intel macOS get PyTorch CPU bundled
            installed.append("pytorch-cpu")
        # Linux: no bundled provider - users must download
        
        # Check for downloaded providers by checking if binary path exists
        for provider_type in ["pytorch-cpu", "pytorch-cuda"]:
            binary_path = get_provider_binary_path(provider_type)
            if binary_path and binary_path.exists() and provider_type not in installed:
                installed.append(provider_type)
        
        return installed
    
    async def get_provider_info(self, provider_type: str) -> dict:
        """
        Get information about a provider.
        
        Args:
            provider_type: Type of provider
            
        Returns:
            Provider information dictionary
        """
        if provider_type in ["apple-mlx", "bundled-pytorch"]:
            return {
                "type": provider_type,
                "name": "Bundled Provider",
                "installed": True,
                "size_mb": None,  # Bundled, no separate size
            }
        elif provider_type == "pytorch-cpu":
            return {
                "type": provider_type,
                "name": "PyTorch CPU",
                "installed": provider_type in await self.list_installed(),
                "size_mb": 300,
            }
        elif provider_type == "pytorch-cuda":
            return {
                "type": provider_type,
                "name": "PyTorch CUDA",
                "installed": provider_type in await self.list_installed(),
                "size_mb": 2400,
            }
        else:
            return {
                "type": provider_type,
                "name": provider_type,
                "installed": False,
                "size_mb": None,
            }


    def _get_free_port(self) -> int:
        """Get a free port for the provider server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    async def _wait_for_provider_health(self, base_url: str, timeout: int = 30) -> None:
        """Wait for provider to become healthy."""
        import httpx
        import asyncio

        start_time = asyncio.get_event_loop().time()
        while True:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(f"{base_url}/tts/health")
                    if response.status_code == 200:
                        return
            except Exception:
                pass

            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Provider did not become healthy within {timeout} seconds")

            await asyncio.sleep(0.5)

    async def _log_subprocess_output(self, process: subprocess.Popen) -> None:
        """Log subprocess stdout and stderr."""
        import logging
        logger = logging.getLogger(__name__)

        async def read_stream(stream, prefix):
            if stream:
                loop = asyncio.get_event_loop()
                while True:
                    line = await loop.run_in_executor(None, stream.readline)
                    if not line:
                        break
                    logger.info(f"{prefix}: {line.rstrip()}")

        await asyncio.gather(
            read_stream(process.stdout, "Provider stdout"),
            read_stream(process.stderr, "Provider stderr"),
            return_exceptions=True,
        )


# Global provider manager instance
_provider_manager: Optional[ProviderManager] = None


def get_provider_manager() -> ProviderManager:
    """Get the global provider manager instance."""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = ProviderManager()
    return _provider_manager
