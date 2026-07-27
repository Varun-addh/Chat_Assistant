"""
Speech Analytics Agent - Production-grade implementation.
Extracts advanced speech delivery metrics using librosa and faster-whisper.
"""

import librosa
import numpy as np
import re
from typing import List, Tuple, Dict
import logging
from pathlib import Path
import soundfile as sf

from app.schemas import SpeechMetrics, SpeechAnalyticsConfig

logger = logging.getLogger(__name__)


class SpeechAnalyticsAgent:
    """
    Agent 2: Speech Analytics Agent
    Handles local audio transcription and metric extraction.
    """
    
    def __init__(self, config: SpeechAnalyticsConfig):
        """
        Initialize the speech analytics agent.
        
        Args:
            config: Speech analytics configuration
        """
        self.config = config
        # Dynamic filler detection - no hardcoded lists
        self.common_fillers = self._get_reference_fillers()  # Only for reference
        logger.info("Speech Analytics Agent initialized with dynamic filler detection")
    
    def _get_reference_fillers(self) -> List[str]:
        """Get reference filler words for baseline comparison only."""
        return ["um", "uh", "like", "you know", "sort of", "kind of", "actually", "basically"]
    
    def analyze_audio(
        self, 
        audio_path: str, 
        transcript: str,
        time_limit: int = 90,
        stt_metadata: dict = None
    ) -> SpeechMetrics:
        """
        Complete speech analysis pipeline.
        
        Args:
            audio_path: Path to audio file
            transcript: Transcribed text
            time_limit: Question time limit in seconds
            stt_metadata: Optional STT metadata (includes VAD info)
            
        Returns:
            SpeechMetrics object with all extracted metrics
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            
            # Extract all metrics
            filler_count, filler_words = self._detect_fillers(transcript)
            wpm = self._calculate_wpm(transcript, duration)
            longest_silence, pause_count = self._detect_silences(audio, sr)
            confidence_score, pitch_variance = self._calculate_confidence(audio, sr, transcript)
            overtalked = self._detect_overtalking(duration, time_limit)
            
            # Get VAD info from STT if available
            vad_removed = None
            if stt_metadata and "vad_removed_duration" in stt_metadata:
                vad_removed = stt_metadata["vad_removed_duration"]
            
            metrics = SpeechMetrics(
                filler_count=filler_count,
                wpm=round(wpm, 1),
                longest_silence=round(longest_silence, 2),
                # 2 decimals: confidence is a 0-1 value whose most populated
                # band spans only 0.9-1.0, so rounding to 1 decimal collapsed
                # every natural speaker into {0.9, 1.0}.
                confidence_score=round(confidence_score, 2),
                overtalked=overtalked,
                duration=round(duration, 2),
                filler_words=filler_words,
                pause_count=pause_count,
                pitch_variance=round(pitch_variance, 4),
                silence_removed=vad_removed  # NEW: VAD removed silence
            )
            
            logger.info(f"Speech analysis complete: {metrics.dict()}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}", exc_info=True)
            # Return default metrics on error
            return self._get_default_metrics(audio_path)
    
    def _detect_fillers(self, transcript: str) -> Tuple[int, List[str]]:
        """
        Dynamically detect filler words and speech disfluencies using AI techniques.
        Detects:
        - Repeated words ("the the", "I I")
        - Word fragments/stuttering ("w- what", "I I I mean")
        - False starts ("I think- I mean")
        - Unnatural hesitations (detected via pattern analysis)
        - Crutch phrases (overused transitional phrases)
        
        Args:
            transcript: Transcribed text
            
        Returns:
            Tuple of (filler_count, list of detected fillers)
        """
        if not transcript or len(transcript.strip()) == 0:
            return 0, []
        
        fillers_detected = []
        
        # 1. Detect word repetitions ("the the", "I I I think")
        fillers_detected.extend(self._detect_repetitions(transcript))
        
        # 2. Detect word fragments and stuttering ("w- what", "I I I")
        fillers_detected.extend(self._detect_fragments(transcript))
        
        # 3. Detect false starts ("I was- I mean")
        fillers_detected.extend(self._detect_false_starts(transcript))
        
        # 4. Detect unnatural hesitation patterns
        fillers_detected.extend(self._detect_hesitation_patterns(transcript))
        
        # 5. Detect overused transitional phrases
        fillers_detected.extend(self._detect_crutch_phrases(transcript))
        
        return len(fillers_detected), fillers_detected
    
    def _detect_repetitions(self, transcript: str) -> List[str]:
        """
        Detect immediate word repetitions ("the the", "I I I").
        
        Args:
            transcript: Transcribed text
            
        Returns:
            List of detected repetitions
        """
        repetitions = []
        words = transcript.lower().split()
        
        i = 0
        while i < len(words) - 1:
            current_word = words[i]
            repeat_count = 1
            
            # Count consecutive repetitions
            while i + repeat_count < len(words) and words[i + repeat_count] == current_word:
                repeat_count += 1
            
            # If word repeated 2+ times, it's a filler
            if repeat_count >= 2:
                repetitions.append(f"{current_word} (x{repeat_count})")
                i += repeat_count
            else:
                i += 1
        
        return repetitions
    
    def _detect_fragments(self, transcript: str) -> List[str]:
        """
        Detect word fragments and partial words ("w- what", "I- I mean").
        
        Args:
            transcript: Transcribed text
            
        Returns:
            List of detected fragments
        """
        # Match patterns like "w-", "I-", "th-"
        fragment_pattern = r'\b[a-zA-Z]{1,3}-\s'
        matches = re.finditer(fragment_pattern, transcript)
        fragments = [match.group().strip() for match in matches]
        return fragments
    
    def _detect_false_starts(self, transcript: str) -> List[str]:
        """
        Detect false starts where speaker changes direction mid-sentence.
        Pattern: "word- other words"
        
        Args:
            transcript: Transcribed text
            
        Returns:
            List of detected false starts
        """
        # Pattern: word(s) followed by dash, then different words
        false_start_pattern = r'\b[a-zA-Z]+\s*-\s*(?:I\s+)?(?:mean|think|believe|guess|say)\b'
        matches = re.finditer(false_start_pattern, transcript, re.IGNORECASE)
        false_starts = [match.group() for match in matches]
        return false_starts
    
    def _detect_hesitation_patterns(self, transcript: str) -> List[str]:
        """
        Detect hesitation patterns using linguistic analysis.
        Detects single-letter or very short hesitations that STT captures.
        
        Args:
            transcript: Transcribed text
            
        Returns:
            List of detected hesitations
        """
        hesitations = []
        
        # Pattern 1: Single letters standing alone (often transcribed hesitations)
        # Use negative lookbehind/lookahead for apostrophe to avoid false positives
        # from contractions like "I'm" (→ m) or "that's" (→ s)
        single_letter_pattern = r"(?<!')\b[a-zA-Z]\b(?!')"
        matches = re.finditer(single_letter_pattern, transcript)
        for match in matches:
            word = match.group()
            # Exclude valid single letters (I, a)
            if word.lower() not in ['i', 'a']:
                hesitations.append(word)
        
        # Pattern 2: Very short repeated sounds
        short_sound_pattern = r'\b(uh|um|er|ah|oh|eh|mm|hm)\b'
        matches = re.finditer(short_sound_pattern, transcript, re.IGNORECASE)
        hesitations.extend([match.group() for match in matches])
        
        return hesitations
    
    def _detect_crutch_phrases(self, transcript: str) -> List[str]:
        """
        Detect overused transitional/filler phrases dynamically.
        Uses frequency analysis to find phrases used more than expected.
        
        Args:
            transcript: Transcribed text
            
        Returns:
            List of detected crutch phrases
        """
        crutch_phrases = []
        
        # Common 2-3 word phrases to check
        potential_crutches = [
            r'\byou know\b',
            r'\bI mean\b',
            r'\bI think\b',
            r'\bkind of\b',
            r'\bsort of\b',
            r'\bright\?\b',
            r'\bbasically\b',
            r'\bactually\b',
            r'\bto be honest\b',
            r'\bat the end of the day\b'
        ]
        
        word_count = len(transcript.split())
        threshold = max(2, word_count // 50)  # Dynamic threshold: 1 per 50 words
        
        for pattern in potential_crutches:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            count = len(matches)
            
            # If phrase used more than threshold, it's a crutch
            if count >= threshold:
                phrase = matches[0] if matches else pattern.replace(r'\b', '').replace('\\', '')
                crutch_phrases.extend([f"{phrase} (x{count})"] * count)
        
        return crutch_phrases
    
    def _calculate_wpm(self, transcript: str, duration: float) -> float:
        """
        Calculate words per minute.
        
        Args:
            transcript: Transcribed text
            duration: Audio duration in seconds
            
        Returns:
            Words per minute (WPM)
        """
        if duration == 0:
            return 0.0
        
        # Count words (split by whitespace)
        word_count = len(transcript.split())
        wpm = (word_count / duration) * 60
        return wpm
    
    def _detect_silences(self, audio: np.ndarray, sr: int) -> Tuple[float, int]:
        """
        Detect silence periods in audio.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            Tuple of (longest_silence, pause_count)
        """
        try:
            # Split audio on silence
            intervals = librosa.effects.split(
                audio, 
                top_db=self.config.silence_top_db,
                frame_length=2048,
                hop_length=512
            )
            
            # Calculate gaps between speech segments
            silences = []
            for i in range(len(intervals) - 1):
                gap_start = intervals[i][1]
                gap_end = intervals[i + 1][0]
                gap_duration = (gap_end - gap_start) / sr
                silences.append(gap_duration)
            
            # Find longest silence and count significant pauses
            longest_silence = max(silences) if silences else 0.0
            pause_count = sum(
                1 for s in silences 
                if s >= self.config.significant_pause_threshold
            )
            
            return longest_silence, pause_count
            
        except Exception as e:
            logger.warning(f"Error detecting silences: {e}")
            return 0.0, 0
    
    @staticmethod
    def confidence_from_pitch_variance(pitch_variance: float) -> float:
        """Map raw pitch variance to a 0-1 confidence score.

        Most human speech pitch variance falls between 800 and 6000:
          - < 800        Monotone/robotic       -> 0.70 - 0.95
          - 800 - 3500   Natural, confident     -> 0.90 - 1.00  (sweet spot)
          - 3500 - 7000  Hesitant/wavering      -> 0.50 - 0.90
          - > 7000       Very shaky             -> 0.00 - 0.50

        Pure function of the variance, separated from audio handling so the
        curve can be tested directly rather than through a duplicated copy.

        The sweet spot interpolates across its ACTUAL width (2700). It was
        previously ``0.9 + min(0.1, (pitch_variance - 800) / 10000)``, where the
        min() saturated at any variance >= 1800 — 63% of the band, and where
        essentially all real speech lands. Every speaker scored exactly 1.0,
        making the metric non-discriminating and pinning the delivery dimension
        (25% of the overall practice score) to a constant 100.
        """
        if pitch_variance < 800:
            return 0.7 + (pitch_variance / 800) * 0.25
        if pitch_variance <= 3500:
            return 0.9 + ((pitch_variance - 800) / 2700) * 0.1
        if pitch_variance <= 7000:
            return 0.9 - ((pitch_variance - 3500) / 3500) * 0.4
        return max(0.0, 0.5 - ((pitch_variance - 7000) / 5000))

    def _calculate_confidence(self, audio: np.ndarray, sr: int, transcript: str = "") -> Tuple[float, float]:
        """
        Calculate confidence score based on pitch variance.
        Lower variance = more stable = more confident.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            transcript: Transcribed text (used to detect empty/silent submissions)
            
        Returns:
            Tuple of (confidence_score 0-1, raw pitch_variance)
        """
        try:
            # GUARD: If the user said nothing (or nearly nothing), confidence is 0.
            # Pitch analysis on silence/noise produces misleadingly stable (high)
            # variance values because background hum is monotone.
            word_count = len(transcript.strip().split()) if transcript and transcript.strip() else 0
            if word_count < 3:
                logger.info(f"Empty/near-empty transcript ({word_count} words) — confidence=0.0")
                return 0.0, 0.0

            # GUARD: If the audio has negligible speech energy, confidence is 0.
            # RMS below ~0.005 means essentially silence / very faint noise.
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 0.005:
                logger.info(f"Audio RMS too low ({rms:.4f}) — confidence=0.0")
                return 0.0, 0.0

            # Extract pitch using piptrack
            pitches, magnitudes = librosa.piptrack(
                y=audio, 
                sr=sr,
                fmin=75,  # Typical human voice min
                fmax=300  # Typical human voice max
            )
            
            # Get pitch values where magnitude is significant
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:  # Only non-zero pitches
                    pitch_values.append(pitch)
            
            if len(pitch_values) < 10:  # Not enough data
                return 0.5, 0.0  # Return middle confidence
            
            # Calculate variance
            pitch_variance = float(np.var(pitch_values))
            
            # --- WORLD-CLASS CONFIDENCE SCORING MODEL ---
            # Most human speech pitch variance falls between 800 and 6000.
            # - Too low (< 500): Monotone/Robotic (Slight penalty)
            # - Sweet Spot (800-3500): Confident, natural professional tone (Peak score)
            # - High (3500-6000): Some hesitation or wavering (Moderate score)
            # - Extreme (> 7000): Very shaky or uncertain (Low score)
            
            confidence_score = self.confidence_from_pitch_variance(pitch_variance)

            return float(confidence_score), pitch_variance
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.5, 0.0  # Return middle confidence on error
    
    def _detect_overtalking(self, duration: float, time_limit: int) -> bool:
        """
        Detect if answer exceeded time limit by >10%.
        
        Args:
            duration: Actual duration in seconds
            time_limit: Time limit in seconds
            
        Returns:
            True if overtalked
        """
        threshold = time_limit * self.config.overtalk_threshold
        return duration > threshold
    
    def _get_default_metrics(self, audio_path: str) -> SpeechMetrics:
        """Return default metrics when analysis fails."""
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
        except Exception:
            duration = 0.0
        
        return SpeechMetrics(
            filler_count=0,
            wpm=0.0,
            longest_silence=0.0,
            confidence_score=0.5,  # Middle confidence (0-1 scale)
            overtalked=False,
            duration=duration,
            filler_words=[],
            pause_count=0,
            pitch_variance=0.0
        )
    
    def get_detailed_analysis(self, metrics: SpeechMetrics) -> Dict[str, str]:
        """
        Get detailed analysis interpretation of metrics.
        
        Args:
            metrics: Speech metrics
            
        Returns:
            Dictionary with human-readable interpretations
        """
        analysis = {}
        
        # WPM analysis
        if metrics.wpm < self.config.ideal_wpm_min:
            analysis['pace'] = "Speaking too slowly"
        elif metrics.wpm > self.config.ideal_wpm_max:
            analysis['pace'] = "Speaking too fast"
        else:
            analysis['pace'] = "Good speaking pace"
        
        # Filler words
        if metrics.filler_count == 0:
            analysis['fillers'] = "Excellent - no filler words"
        elif metrics.filler_count <= 3:
            analysis['fillers'] = "Good - minimal filler words"
        elif metrics.filler_count <= 7:
            analysis['fillers'] = "Moderate filler word usage"
        else:
            analysis['fillers'] = "High filler word usage"
        
        # Confidence (0-1 scale)
        if metrics.confidence_score >= 0.75:
            analysis['confidence'] = "High confidence"
        elif metrics.confidence_score >= 0.5:
            analysis['confidence'] = "Moderate confidence"
        else:
            analysis['confidence'] = "Low confidence - voice instability"
        
        # Pauses
        if metrics.longest_silence > 5:
            analysis['pauses'] = "Very long pauses detected"
        elif metrics.longest_silence > 3:
            analysis['pauses'] = "Some long pauses"
        else:
            analysis['pauses'] = "Good flow"
        
        return analysis


class FillerWordDetector:
    """Advanced filler word detection with context awareness."""
    
    def __init__(self, filler_words: List[str]):
        """Initialize with filler word list."""
        self.filler_words = filler_words
        self.pattern = self._compile_pattern()
    
    def _compile_pattern(self) -> re.Pattern:
        """Compile regex with word boundaries."""
        words = [re.escape(word) for word in self.filler_words]
        pattern = r'\b(' + '|'.join(words) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def detect(self, text: str) -> Tuple[int, List[str], List[int]]:
        """
        Detect filler words with positions.
        
        Returns:
            Tuple of (count, words, positions)
        """
        matches = []
        positions = []
        
        for match in self.pattern.finditer(text):
            matches.append(match.group().lower())
            positions.append(match.start())
        
        return len(matches), matches, positions
    
    def get_density(self, text: str) -> float:
        """
        Calculate filler word density (fillers per 100 words).
        
        Returns:
            Filler density percentage
        """
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        filler_count = len(self.pattern.findall(text))
        density = (filler_count / word_count) * 100
        return round(density, 2)
