"""
Provider management system for TTS providers.
"""

from typing import Optional
import platform
from pathlib import Path

from .base import TTSProvider
from .types import ProviderType
from .bundled import BundledProvider
from .local import LocalProvider
from .installer import get_provider_binary_path
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
        if provider_type in ["bundled-mlx", "bundled-pytorch"]:
            # Use bundled provider
            self.active_provider = self._get_default_provider()
        elif provider_type in ["pytorch-cpu", "pytorch-cuda"]:
            # Start local provider subprocess
            provider_path = get_provider_binary_path(provider_type)
            if not provider_path or not provider_path.exists():
                raise ValueError(f"Provider {provider_type} is not installed. Please download it first.")
            
            # Find a free port
            port = self._get_free_port()
            
            # Start provider subprocess
            from ..config import get_data_dir
            process = subprocess.Popen(
                [
                    str(provider_path),
                    "--port", str(port),
                    "--data-dir", str(get_data_dir()),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            # Wait for provider to be ready
            base_url = f"http://127.0.0.1:{port}"
            await self._wait_for_provider_health(base_url, timeout=30)
            
            # Create LocalProvider instance
            self.active_provider = LocalProvider(base_url)
            self._provider_process = process
            self._provider_port = port
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
            installed.append("bundled-mlx")
        else:
            installed.append("bundled-pytorch")
        
        # Check for downloaded providers (Phase 2)
        providers_dir = _get_providers_dir()
        if providers_dir.exists():
            for provider_file in providers_dir.glob("tts-provider-*"):
                if provider_file.is_file() and provider_file.stat().st_size > 0:
                    name = provider_file.name
                    if "pytorch-cpu" in name:
                        installed.append("pytorch-cpu")
                    elif "pytorch-cuda" in name:
                        installed.append("pytorch-cuda")
        
        return installed
    
    async def get_provider_info(self, provider_type: str) -> dict:
        """
        Get information about a provider.
        
        Args:
            provider_type: Type of provider
            
        Returns:
            Provider information dictionary
        """
        if provider_type in ["bundled-mlx", "bundled-pytorch"]:
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


# Global provider manager instance
_provider_manager: Optional[ProviderManager] = None


def get_provider_manager() -> ProviderManager:
    """Get the global provider manager instance."""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = ProviderManager()
    return _provider_manager
