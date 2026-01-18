"""
Local STT Service using faster-whisper.
Production-grade speech-to-text with performance optimizations.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import time

from faster_whisper import WhisperModel

from app.schemas import STTConfig

logger = logging.getLogger(__name__)


class LocalSTTService:
    """
    Local Speech-to-Text service using faster-whisper.
    Target: Transcribe 1 min audio in <3 seconds.
    """
    
    def __init__(self, config: STTConfig):
        """
        Initialize the STT service.
        
        Args:
            config: STT configuration
        """
        self.config = config
        self.model: Optional[WhisperModel] = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize faster-whisper model."""
        try:
            logger.info(
                f"Loading faster-whisper model: {self.config.stt_model_size} "
                f"on {self.config.device} with {self.config.compute_type}"
            )
            
            start_time = time.time()
            
            self.model = WhisperModel(
                model_size_or_path=self.config.stt_model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
                download_root=None,  # Use default cache
                local_files_only=False
            )
            
            load_time = time.time() - start_time
            logger.info(f"Faster-whisper model loaded in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize faster-whisper: {e}", exc_info=True)
            raise
    
    def transcribe(self, audio_path: str, language: str = "en") -> Tuple[str, dict]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: "en")
            
        Returns:
            Tuple of (transcript, metadata)
        """
        if not self.model:
            raise RuntimeError("STT model not initialized")
        
        try:
            start_time = time.time()
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Transcribe with faster-whisper
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=True,
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=100
                )
            )
            
            # Combine segments
            transcript_parts = []
            segment_durations = 0.0
            for segment in segments:
                transcript_parts.append(segment.text.strip())
                segment_durations += (segment.end - segment.start)
            
            transcript = " ".join(transcript_parts)
            
            transcription_time = time.time() - start_time
            
            # Calculate VAD removed duration (approximate)
            # Total audio duration - actual speech segments duration
            vad_removed = max(0, info.duration - segment_durations)
            
            # Metadata
            metadata = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "transcription_time": transcription_time,
                "model_size": self.config.stt_model_size,
                "vad_removed_duration": round(vad_removed, 2)  # NEW: Silence removed by VAD
            }
            
            logger.info(
                f"Transcription complete in {transcription_time:.2f}s "
                f"({info.duration:.1f}s audio)"
            )
            
            # Performance check
            if transcription_time > self.config.max_transcription_time:
                logger.warning(
                    f"Transcription took {transcription_time:.2f}s, "
                    f"exceeds target {self.config.max_transcription_time}s"
                )
            
            return transcript, metadata
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            raise
    
    async def transcribe_async(
        self, 
        audio_path: str, 
        language: str = "en"
    ) -> Tuple[str, dict]:
        """
        Async wrapper for transcription (runs in thread pool).
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Returns:
            Tuple of (transcript, metadata)
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.transcribe, 
            audio_path, 
            language
        )
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_size": self.config.stt_model_size,
            "device": self.config.device,
            "compute_type": self.config.compute_type,
            "initialized": self.model is not None
        }
    
    def warmup(self):
        """Warmup model with dummy audio."""
        try:
            logger.info("Warming up STT model...")
            import numpy as np
            import soundfile as sf
            import tempfile
            
            # Generate 1 second of silence
            dummy_audio = np.zeros(16000, dtype=np.float32)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, dummy_audio, 16000)
                temp_path = f.name
            
            # Transcribe dummy audio
            self.transcribe(temp_path)
            
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
            
            logger.info("STT model warmed up")
            
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")


class TranscriptionCache:
    """Simple cache for transcriptions to avoid re-processing."""
    
    def __init__(self, max_size: int = 100):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum cache entries
        """
        self.cache: dict = {}
        self.max_size = max_size
    
    def get(self, audio_path: str) -> Optional[Tuple[str, dict]]:
        """Get cached transcription."""
        import hashlib
        
        # Create hash of file
        try:
            with open(audio_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            return self.cache.get(file_hash)
        except Exception:
            return None
    
    def set(self, audio_path: str, transcript: str, metadata: dict):
        """Cache transcription."""
        import hashlib
        
        try:
            with open(audio_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            # Evict oldest if full
            if len(self.cache) >= self.max_size:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
            
            self.cache[file_hash] = (transcript, metadata)
        except Exception:
            pass
