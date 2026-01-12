"""
Quick test for Practice Mode dependencies and basic functionality.
"""

import os

import pytest

# This file is a manual/integration smoke test. It depends on optional local
# speech packages (e.g., faster-whisper) and hardware/runtime configuration.
# Skip during normal test runs unless explicitly enabled.
if os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip("practice mode integration test (set RUN_INTEGRATION_TESTS=1 to run)", allow_module_level=True)

import asyncio
from app.config_practice_mode import get_practice_config
from app.services.local_stt_service import LocalSTTService
from app.services.local_tts_service import LocalTTSService

async def test_practice_mode():
    """Test Practice Mode basic functionality."""
    
    print("=" * 60)
    print("PRACTICE MODE DEPENDENCY TEST")
    print("=" * 60)
    
    # Get configuration
    config = get_practice_config()
    print(f"\n✓ Configuration loaded")
    print(f"  - TTS Engine: {config.tts.engine}")
    print(f"  - STT Model Size: {config.stt.stt_model_size}")
    print(f"  - STT Device: {config.stt.device}")
    
    # Test TTS Service
    print(f"\n[TTS Test]")
    tts_service = LocalTTSService(config.tts, output_dir=config.audio_storage_path)
    tts_service.warmup()
    print(f"✓ TTS service initialized ({config.tts.engine})")
    
    # Generate a test audio
    test_text = "Hello! Welcome to the interview practice mode."
    try:
        audio_path = await tts_service.synthesize(test_text)
        print(f"✓ TTS synthesis successful: {audio_path}")
    except Exception as e:
        print(f"✗ TTS synthesis failed: {e}")
    
    # Test STT Service
    print(f"\n[STT Test]")
    stt_service = LocalSTTService(config.stt)
    stt_service.warmup()
    print(f"✓ STT service initialized (model: {config.stt.stt_model_size})")
    
    print("\n" + "=" * 60)
    print("PRACTICE MODE TEST COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start server: uvicorn app.main:app --reload")
    print("2. Check status: GET http://localhost:8000/api/practice/status")
    print("3. Start interview: POST http://localhost:8000/api/practice/interview/start")

if __name__ == "__main__":
    asyncio.run(test_practice_mode())
