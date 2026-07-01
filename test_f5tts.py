import asyncio
import os
import sys

# Setup path so we can import backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from backend.backends.f5tts_backend import F5TTSBackend

async def test_base():
    print("Testing F5-TTS Base...")
    backend = F5TTSBackend()
    
    # Try generating
    # For zero-shot, we need a ref_audio. We'll try just generating text without it or check if it throws
    voice_prompt = {}
    print("Loading model...")
    await backend.load_model("f5-tts")
    
    print("Generating audio...")
    audio, sr = await backend.generate("Hello world, this is F5-TTS.", voice_prompt)
    print(f"Generated {len(audio)} samples at {sr} Hz")
    
    backend.unload_model()
    
async def test_ro():
    print("Testing F5-TTS Romanian...")
    backend = F5TTSBackend()
    
    voice_prompt = {}
    print("Loading model...")
    await backend.load_model("f5-tts-ro")
    
    print("Generating audio...")
    audio, sr = await backend.generate("Salut, acesta este un test.", voice_prompt)
    print(f"Generated {len(audio)} samples at {sr} Hz")
    
    backend.unload_model()

async def main():
    await test_base()
    await test_ro()

if __name__ == "__main__":
    asyncio.run(main())
