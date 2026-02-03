"""
Configuration module for voicebox backend.

Handles data directory configuration for production bundling.
"""

from pathlib import Path
import os

# Default TTS model type

def get_tts_backend_type() -> str:
    """Get TTS backend type from database settings."""
    try:
        # Try to import here to avoid circular dependency
        from .database import SessionLocal, Settings
        
        if SessionLocal is not None:
            db = SessionLocal()
            try:
                setting = db.query(Settings).filter(Settings.key == "tts_backend").first()
                if setting:
                    return setting.value
            finally:
                db.close()
    except Exception:
        pass  # Database not initialized yet, use default
    
    # Default if database not ready
    return "chatterbox_turbo"


# Keep for backward compatibility during initialization
TTS_MODEL_TYPE = get_tts_backend_type()


# Default data directory (used in development)
_data_dir = Path("data")

def set_data_dir(path: str | Path):
    """
    Set the data directory path.

    Args:
        path: Path to the data directory
    """
    global _data_dir
    _data_dir = Path(path)
    _data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory set to: {_data_dir.absolute()}")

def get_data_dir() -> Path:
    """
    Get the data directory path.

    Returns:
        Path to the data directory
    """
    return _data_dir

def get_db_path() -> Path:
    """Get database file path."""
    return _data_dir / "voicebox.db"

def get_profiles_dir() -> Path:
    """Get profiles directory path."""
    path = _data_dir / "profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_generations_dir() -> Path:
    """Get generations directory path."""
    path = _data_dir / "generations"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_cache_dir() -> Path:
    """Get cache directory path."""
    path = _data_dir / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_models_dir() -> Path:
    """Get models directory path."""
    path = _data_dir / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path
