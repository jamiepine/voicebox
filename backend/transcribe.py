"""
Whisper ASR module for transcription.
"""

from typing import Optional, List, Dict
import asyncio
import torch
import numpy as np
from pathlib import Path
from .utils.progress import get_progress_manager
from .utils.hf_progress import HFProgressTracker, create_hf_progress_callback
from .utils.tasks import get_task_manager


class WhisperModel:
    """Manages Whisper model loading and transcription."""
    
    def __init__(self, model_size: str = "base"):
        self.model = None
        self.processor = None
        self.model_size = model_size
        self.device = self._get_device()
    
    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS support for Whisper
            return "cpu"  # Use CPU for stability
        return "cpu"
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def load_model(self, model_size: Optional[str] = None):
        """
        Lazy load the Whisper model.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
        """
        if model_size is None:
            model_size = self.model_size
        
        if self.model is not None and self.model_size == model_size:
            return
        
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            model_name = f"openai/whisper-{model_size}"
            
            # Set up progress tracking
            progress_manager = get_progress_manager()
            progress_model_name = f"whisper-{model_size}"
            
            # Start tracking download task
            task_manager = get_task_manager()
            task_manager.start_download(progress_model_name)
            
            print(f"Loading Whisper model {model_size} on {self.device}...")
            
            # Initialize progress state to show download has started
            progress_manager.update_progress(
                model_name=progress_model_name,
                current=0,
                total=1,  # Set to 1 initially, will be updated by callback
                filename="",
                status="downloading",
            )
            
            # Set up progress callback
            progress_callback = create_hf_progress_callback(progress_model_name, progress_manager)
            tracker = HFProgressTracker(progress_callback)
            
            # Use progress tracker during download
            with tracker.patch_download():
                self.processor = WhisperProcessor.from_pretrained(model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            
            self.model.to(self.device)
            self.model_size = model_size
            
            # Mark as complete
            progress_manager.mark_complete(progress_model_name)
            task_manager.complete_download(progress_model_name)
            
            print(f"Whisper model {model_size} loaded successfully")
            
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            progress_manager = get_progress_manager()
            task_manager = get_task_manager()
            progress_model_name = f"whisper-{model_size}"
            progress_manager.mark_error(progress_model_name, str(e))
            task_manager.error_download(progress_model_name, str(e))
            raise
    
    async def load_model_async(self, model_size: Optional[str] = None):
        """
        Async version of load_model that runs in thread pool.
        
        This prevents blocking the event loop during model loading.
        """
        if model_size is None:
            model_size = self.model_size
            
        # If already loaded with correct size, return immediately
        if self.model is not None and self.model_size == model_size:
            return
        
        # Run the blocking load operation in a thread pool
        await asyncio.to_thread(self.load_model, model_size)
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("Whisper model unloaded")
    
    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            language: Optional language hint (en or zh)
            
        Returns:
            Transcribed text
        """
        await self.load_model_async()
        
        from .utils.audio import load_audio
        
        def _transcribe_sync():
            """Run synchronous transcription in thread pool."""
            # Load audio
            audio, sr = load_audio(audio_path, sample_rate=16000)
            
            # Process audio
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Set language if provided
            forced_decoder_ids = None
            if language:
                # Support all languages from frontend: en, zh, ja, ko, de, fr, ru, pt, es, it
                # Whisper supports these and many more
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language,
                    task="transcribe",
                )
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs["input_features"],
                    forced_decoder_ids=forced_decoder_ids,
                )
            
            # Decode
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )[0]
            
            return transcription.strip()
        
        # Run blocking transcription in thread pool
        return await asyncio.to_thread(_transcribe_sync)
    
    async def transcribe_with_timestamps(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> List[Dict[str, any]]:
        """
        Transcribe audio with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Optional language hint
            
        Returns:
            List of word segments with timestamps
        """
        await self.load_model_async()
        
        from .utils.audio import load_audio
        
        def _transcribe_timestamps_sync():
            """Run synchronous transcription with timestamps in thread pool."""
            # Load audio
            audio, sr = load_audio(audio_path, sample_rate=16000)
            
            # Process audio
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Set language if provided
            forced_decoder_ids = None
            if language:
                # Support all languages from frontend: en, zh, ja, ko, de, fr, ru, pt, es, it
                # Whisper supports these and many more
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language,
                    task="transcribe",
                )
            
            # Generate with timestamps
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs["input_features"],
                    forced_decoder_ids=forced_decoder_ids,
                    return_timestamps=True,
                )
            
            # Parse timestamps (simplified - would need more robust parsing)
            # For now, return basic transcription
            # TODO: Implement proper timestamp parsing
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )[0]
            
            return [
                {
                    "text": transcription,
                    "start": 0.0,
                    "end": len(audio) / sr,
                }
            ]
        
        # Run blocking transcription in thread pool
        return await asyncio.to_thread(_transcribe_timestamps_sync)


# Global model instance
_whisper_model: Optional[WhisperModel] = None


def get_whisper_model() -> WhisperModel:
    """Get or create Whisper model instance."""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel()
    return _whisper_model


def unload_whisper_model():
    """Unload Whisper model to free memory."""
    global _whisper_model
    if _whisper_model is not None:
        _whisper_model.unload_model()
