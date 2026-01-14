import os
import logging
import re
import asyncio
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import hashlib

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logging.warning("pyttsx3 not installed. Install with: pip install pyttsx3")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logging.warning("gtts not installed. Install with: pip install gtts")

from app.schemas import TTSConfig

logger = logging.getLogger(__name__)


class LocalTTSService:
    """
    Local Text-to-Speech service using pyttsx3 (offline) with gTTS fallback.
    
    Features:
    - Offline synthesis with pyttsx3 (Windows SAPI, macOS NSSpeechSynthesizer, Linux espeak)
    - gTTS fallback for reliability
    - Async synthesis
    - Configurable voice parameters
    """
    
    def __init__(self, config: TTSConfig, output_dir: str = "data/practice_audio"):
        self.config = config
        self.output_dir = output_dir
        self.engine = None
        self._initialized = False
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"LocalTTSService initialized - Engine: {config.engine}")
    
    def _init_pyttsx3(self) -> bool:
        """Initialize pyttsx3 engine with conversational settings."""
        if not PYTTSX3_AVAILABLE:
            return False
        
        try:
            self.engine = pyttsx3.init()
            
            # Configure voice parameters for natural, conversational tone
            self.engine.setProperty('rate', 190)  # Faster for high-stakes professional interview (180-200 WPM)
            self.engine.setProperty('volume', 0.95)  # Clear but not overwhelming
            
            # Try to set a specific voice if available
            voices = self.engine.getProperty('voices')
            if voices:
                # 👔 WORLD-CLASS MALE VOICE SELECTION 👔
                # Priority: Male voices (Microsoft David, Alex, Mark)
                male_keywords = ['david', 'mark', 'alex', 'james', 'george', 'male', 'man']
                female_keywords = ['zira', 'hazel', 'female', 'woman', 'samantha', 'victoria']
                
                selected_voice = None
                
                # Step 1: Search for high-quality male voices first
                for pref in male_keywords:
                    for voice in voices:
                        if pref in voice.name.lower():
                            selected_voice = voice
                            self.engine.setProperty('voice', voice.id)
                            logger.info(f"Selected priority male voice: {voice.name}")
                            break
                    if selected_voice: break
                
                # Step 2: Fallback to any voice that doesn't explicitly contain female keywords
                if not selected_voice:
                    for voice in voices:
                        voice_name_lower = voice.name.lower()
                        if not any(f_key in voice_name_lower for f_key in female_keywords):
                            selected_voice = voice
                            self.engine.setProperty('voice', voice.id)
                            logger.info(f"Found alternative male/neutral voice: {voice.name}")
                            break
                
                # Step 3: Absolute final fallback (any voice, even female)
                if not selected_voice:
                    selected_voice = voices[0]
                    self.engine.setProperty('voice', selected_voice.id)
                    logger.info(f"Using final fallback voice: {selected_voice.name}")
            
            self._initialized = True
            logger.info("pyttsx3 engine initialized successfully with conversational settings")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to initialize pyttsx3: {e}")
            logger.info("TTS will use fallback mode (gTTS or mock) - This is expected in Docker without eSpeak/SAPI")
            return False
    
    def cleanup(self):
        """Cleanup TTS resources and stop any running threads."""
        try:
            if self.engine and self._initialized:
                # Stop pyttsx3 engine and cleanup threads
                try:
                    self.engine.stop()
                except:
                    pass
                self.engine = None
                self._initialized = False
                logger.info("TTS engine cleaned up")
        except Exception as e:
            logger.error(f"Error during TTS cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
    
    def _normalize_text_for_speech(self, text: str) -> str:
        """
        Intelligently normalize text for natural speech synthesis.
        Expands abbreviations, formats numbers, handles technical terms.
        
        This is future-proof and extensible - add new patterns as needed.
        
        Args:
            text: Raw text
            
        Returns:
            Normalized text optimized for TTS
        """
        # SECTION 1: Common Abbreviations & Latin Phrases
        abbreviations = {
            # Latin phrases (most common in technical writing)
            r'\be\.g\.': 'for example',
            r'\bi\.e\.': 'that is',
            r'\betc\.': 'et cetera',
            r'\bvs\.': 'versus',
            r'\bcf\.': 'compare',
            r'\bet al\.': 'and others',
            r'\bviz\.': 'namely',
            
            # Titles
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Missus',
            r'\bMs\.': 'Miss',
            r'\bProf\.': 'Professor',
            r'\bSr\.': 'Senior',
            r'\bJr\.': 'Junior',
            
            # Business/Organizational
            r'\bInc\.': 'Incorporated',
            r'\bCorp\.': 'Corporation',
            r'\bLtd\.': 'Limited',
            r'\bCo\.': 'Company',
            r'\bDept\.': 'Department',
            
            # Time/Date
            r'\bJan\.': 'January',
            r'\bFeb\.': 'February',
            r'\bMar\.': 'March',
            r'\bApr\.': 'April',
            r'\bAug\.': 'August',
            r'\bSept\.': 'September',
            r'\bOct\.': 'October',
            r'\bNov\.': 'November',
            r'\bDec\.': 'December',
            r'\bMon\.': 'Monday',
            r'\bTue\.': 'Tuesday',
            r'\bWed\.': 'Wednesday',
            r'\bThu\.': 'Thursday',
            r'\bFri\.': 'Friday',
            r'\bSat\.': 'Saturday',
            r'\bSun\.': 'Sunday',
        }
        
        # SECTION 2: Technical Abbreviations (Programming/IT)
        tech_abbreviations = {
            r'\bAPI\b': 'A P I',  # Spell out acronyms for clarity
            r'\bAPIs\b': 'A P I s',
            r'\bREST\b': 'REST',  # Keep as-is, commonly pronounced
            r'\bHTTP\b': 'H T T P',
            r'\bHTTPS\b': 'H T T P S',
            r'\bURL\b': 'U R L',
            r'\bURLs\b': 'U R L s',
            r'\bSQL\b': 'S Q L',  # or 'sequel' depending on preference
            r'\bHTML\b': 'H T M L',
            r'\bCSS\b': 'C S S',
            r'\bJSON\b': 'J SON',
            r'\bXML\b': 'X M L',
            r'\bAWS\b': 'A W S',
            r'\bGCP\b': 'G C P',
            r'\bCI/CD\b': 'C I C D',
            r'\bOOP\b': 'O O P',
            r'\bUI\b': 'U I',
            r'\bUX\b': 'U X',
            r'\bDB\b': 'database',
            r'\bDBs\b': 'databases',
            r'\bOS\b': 'operating system',
            r'\bCPU\b': 'C P U',
            r'\bRAM\b': 'R A M',
            r'\bGPU\b': 'G P U',
            r'\bSSD\b': 'S S D',
            r'\bIoT\b': 'I o T',
            r'\bML\b': 'machine learning',
            r'\bAI\b': 'A I',
            r'\bNLP\b': 'N L P',
            r'\bCRUD\b': 'C R U D',
            r'\bSSH\b': 'S S H',
            r'\bFTP\b': 'F T P',
            r'\bDNS\b': 'D N S',
            r'\bVPN\b': 'V P N',
        }
        
        # SECTION 3: Units & Measurements
        units = {
            r'\bkm\b': 'kilometers',
            r'\bkg\b': 'kilograms',
            r'\bmg\b': 'milligrams',
            r'\bms\b': 'milliseconds',
            r'\bGB\b': 'gigabytes',
            r'\bMB\b': 'megabytes',
            r'\bKB\b': 'kilobytes',
            r'\bTB\b': 'terabytes',
            r'\bGHz\b': 'gigahertz',
            r'\bMHz\b': 'megahertz',
        }
        
        # Apply all normalizations
        normalized = text
        
        # Apply common abbreviations
        for pattern, replacement in abbreviations.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Apply tech abbreviations (case-sensitive for acronyms)
        for pattern, replacement in tech_abbreviations.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        # Apply units
        for pattern, replacement in units.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        # SECTION 4: Number Formatting
        # Format numbers with commas (e.g., "1000" -> "1 thousand")
        def format_large_number(match):
            num = int(match.group())
            if num >= 1000000:
                return f"{num/1000000:.1f} million".replace('.0', '')
            elif num >= 1000:
                return f"{num/1000:.1f} thousand".replace('.0', '')
            return str(num)
        
        normalized = re.sub(r'\b\d{4,}\b', format_large_number, normalized)
        
        # SECTION 5: Code-like patterns
        # Replace underscores with spaces in variable names
        # e.g., "user_profile" -> "user profile"
        normalized = re.sub(r'\b(\w+)_(\w+)\b', r'\1 \2', normalized)
        
        # SECTION 6: Special characters for better speech
        replacements = {
            '&': 'and',
            '@': 'at',
            '#': 'hashtag',
            '%': 'percent',
            '+': 'plus',
            '=': 'equals',
            '<': 'less than',
            '>': 'greater than',
            '/': 'slash',
            '\\': 'backslash',
        }
        
        for char, word in replacements.items():
            normalized = normalized.replace(char, f" {word} ")
        
        # Clean up multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    async def synthesize(
        self,
        text: str,
        filename: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Convert text to speech and save as audio file.
        
        Args:
            text: Text to synthesize
            filename: Optional output filename (generated if not provided)
            use_cache: Whether to use cached audio if available
            
        Returns:
            Path to generated audio file
        """
        # STEP 1: Normalize text for natural speech
        normalized_text = self._normalize_text_for_speech(text)
        
        # Generate filename if not provided
        if not filename:
            text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
            filename = f"tts_{text_hash}.wav"
        
        output_path = Path(self.output_dir) / filename
        
        # Check cache
        if use_cache and output_path.exists():
            logger.debug(f"Using cached TTS: {output_path}")
            return str(output_path)
        
        # Try pyttsx3 first (offline)
        if self.config.engine in ["pyttsx3", "offline"]:
            try:
                audio_path = await self._synthesize_pyttsx3(normalized_text, output_path)
                if audio_path:
                    return audio_path
            except Exception as e:
                logger.warning(f"pyttsx3 synthesis failed: {e}")
        
        # Fallback to gTTS (requires internet)
        if GTTS_AVAILABLE:
            try:
                audio_path = await self._synthesize_gtts(normalized_text, output_path)
                if audio_path:
                    return audio_path
            except Exception as e:
                logger.error(f"gTTS synthesis failed: {e}")
        
        raise RuntimeError("All TTS engines failed. Install pyttsx3 or gtts.")
    
    async def synthesize_async(self, text: str, output_path: str) -> str:
        """
        Async wrapper for synthesize with specific output path.
        
        Args:
            text: Text to synthesize
            output_path: Full path where audio should be saved
            
        Returns:
            Path to generated audio file
        """
        # Extract filename from full path
        filename = Path(output_path).name
        
        # Call main synthesize method
        result = await self.synthesize(text, filename=filename, use_cache=True)
        
        return result
    
    async def _synthesize_pyttsx3(self, text: str, output_path: Path) -> Optional[str]:
        """Synthesize using pyttsx3 (offline)."""
        if not self._initialized:
            if not self._init_pyttsx3():
                return None
        
        def _generate():
            try:
                # pyttsx3 saves to file directly
                self.engine.save_to_file(text, str(output_path))
                self.engine.runAndWait()
                return str(output_path)
            except Exception as e:
                logger.error(f"pyttsx3 generation error: {e}")
                return None
        
        # Run in executor with shorter timeout to avoid blocking
        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _generate),
                timeout=5.0  # 5 second timeout (reduced from 10s)
            )
            
            if result and output_path.exists():
                logger.info(f"pyttsx3 synthesis complete: {output_path}")
                return str(output_path)
        except asyncio.TimeoutError:
            logger.warning(f"pyttsx3 synthesis timed out after 5s - falling back to gTTS")
            # Don't raise, just return None to trigger gTTS fallback
        except Exception as e:
            logger.warning(f"pyttsx3 synthesis failed: {e} - falling back to gTTS")
        
        return None
    
    async def _synthesize_gtts(self, text: str, output_path: Path) -> Optional[str]:
        """
        Synthesize using Google TTS with conversational tone (requires internet).
        gTTS provides more natural intonation than offline TTS.
        """
        if not GTTS_AVAILABLE:
            return None
        
        def _generate():
            try:
                # Change extension to mp3 for gTTS
                mp3_path = output_path.with_suffix('.mp3')
                # Use gTTS with US English for professional interview tone
                tts = gTTS(
                    text=text, 
                    lang='en', 
                    slow=False,  # Normal conversational pace
                    tld='com'    # US English accent
                )
                tts.save(str(mp3_path))
                return str(mp3_path)
            except Exception as e:
                logger.error(f"gTTS generation error: {e}")
                return None
        
        # Run in executor
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _generate)
        
        if result and Path(result).exists():
            logger.info(f"gTTS synthesis complete with conversational tone: {result}")
            return result
        
        return None
    
    def warmup(self):
        """Warm up the TTS engine with a test synthesis."""
        try:
            # Only initialize pyttsx3 if it's the selected engine
            if self.config.engine in ["pyttsx3", "offline"]:
                if not self._initialized and PYTTSX3_AVAILABLE:
                    success = self._init_pyttsx3()
                    if not success:
                        logger.info("TTS pyttsx3 initialization skipped - fallback mode active")
            
            logger.info("TTS warmup complete")
        except Exception as e:
            logger.warning(f"TTS warmup had issues (non-critical): {e}")
            logger.info("TTS will operate in fallback mode")
    
    def get_engine_info(self) -> dict:
        """Get information about the TTS engine."""
        info = {
            "engine": self.config.engine,
            "initialized": self._initialized,
            "available_engines": []
        }
        
        if PYTTSX3_AVAILABLE:
            info["available_engines"].append("pyttsx3")
        if GTTS_AVAILABLE:
            info["available_engines"].append("gtts")
        
        # Get voice info if pyttsx3 is initialized
        if self._initialized and self.engine:
            try:
                voices = self.engine.getProperty('voices')
                if voices:
                    info["voices_available"] = len(voices)
                    info["current_voice"] = self.engine.getProperty('voice')
                    info["rate"] = self.engine.getProperty('rate')
                    info["volume"] = self.engine.getProperty('volume')
            except:
                pass
        
        return info
    
    def cleanup(self):
        """Clean up TTS resources."""
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
        
        logger.info("TTS cleanup complete")
