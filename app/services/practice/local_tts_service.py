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

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logging.info("edge-tts not installed. Install with: pip install edge-tts")

from app.schemas import TTSConfig
from app.config import settings

logger = logging.getLogger(__name__)


def _voice_keyword_match(text: str, keyword: str) -> bool:
    """Case-insensitive match for voice selection keywords.

    Important: gender terms must be whole-word matches to avoid false positives
    such as "Manipuri" matching the keyword "man".
    """
    text_l = (text or "").lower()
    key = (keyword or "").strip().lower()
    if not key:
        return False
    if key in {"male", "female", "man", "woman"}:
        return re.search(rf"\b{re.escape(key)}\b", text_l) is not None
    return key in text_l


def _voice_prefers_english(name: str = "", voice_id: str = "", languages: object = None) -> bool:
    """Best-effort check for a US/UK English voice.

    We intentionally restrict this to American/British English variants to avoid
    surprising selections like "English (Caribbean)" when the user wants a
    standard interview voice.
    """
    parts: list[str] = []
    if name:
        parts.append(str(name))
    if voice_id:
        parts.append(str(voice_id))

    if isinstance(languages, (list, tuple)):
        for item in languages:
            try:
                if isinstance(item, (bytes, bytearray)):
                    parts.append(item.decode(errors="ignore"))
                else:
                    parts.append(str(item))
            except Exception:
                continue

    combined = " ".join(parts).lower()
    # Explicit locale markers (preferred): en-US / en-GB
    if re.search(r"\ben[_-]us\b", combined) is not None:
        return True
    if re.search(r"\ben[_-]gb\b", combined) is not None:
        return True

    # Common name patterns (Windows SAPI + some engines)
    if re.search(r"\benglish\s*\(\s*united\s+states\s*\)", combined) is not None:
        return True
    if re.search(r"\benglish\s*\(\s*united\s+kingdom\s*\)", combined) is not None:
        return True

    # Espeak-style variants
    if "english-us" in combined or "english (us" in combined:
        return True
    if "english-uk" in combined or "english (uk" in combined:
        return True

    return False


def _voice_is_windows_david(name: str = "", voice_id: str = "") -> bool:
    combined = f"{name or ''} {voice_id or ''}".lower()
    return "david" in combined


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
        self._pyttsx3_lock = asyncio.Lock()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"LocalTTSService initialized - Engine: {config.engine}")
    
    def _init_pyttsx3(self) -> bool:
        """Initialize pyttsx3 engine with conversational settings."""
        if not PYTTSX3_AVAILABLE:
            return False
        
        try:
            # Force Windows to use SAPI5 explicitly for consistent system voices.
            if os.name == "nt":
                self.engine = pyttsx3.init(driverName="sapi5")
            else:
                self.engine = pyttsx3.init()
            
            # Configure voice parameters for natural, conversational tone
            self.engine.setProperty('rate', 190)  # Faster for high-stakes professional interview (180-200 WPM)
            self.engine.setProperty('volume', 0.95)  # Clear but not overwhelming
            
            # Try to set a specific voice if available
            voices = self.engine.getProperty('voices')
            if voices:
                # Optional hard override: pick an exact voice id if provided.
                if getattr(settings, "practice_tts_voice_id", None):
                    voice_id = settings.practice_tts_voice_id
                    try:
                        self.engine.setProperty("voice", voice_id)
                        logger.info(f"Selected configured voice id: {voice_id}")
                    except Exception as e:
                        logger.warning(f"Failed to set configured voice id '{voice_id}': {e}")
                else:
                    # Optional soft override: pick first voice whose name contains this substring.
                    name_contains = (getattr(settings, "practice_tts_voice_name_contains", None) or "").strip().lower()
                    if name_contains:
                        for voice in voices:
                            if name_contains in (getattr(voice, "name", "") or "").lower():
                                try:
                                    self.engine.setProperty("voice", voice.id)
                                    logger.info(f"Selected configured voice by name match: {voice.name}")
                                    break
                                except Exception as e:
                                    logger.warning(f"Failed to set voice '{voice.name}': {e}")

                # On Windows, strongly prefer Microsoft David if present.
                # This is the "standard" interview voice most users expect.
                if os.name == "nt" and not getattr(settings, "practice_tts_voice_name_contains", None):
                    for voice in voices:
                        if _voice_is_windows_david(getattr(voice, "name", "") or "", getattr(voice, "id", "") or ""):
                            try:
                                self.engine.setProperty("voice", voice.id)
                                logger.info(f"Selected Windows preferred voice: {voice.name}")
                                break
                            except Exception as e:
                                logger.warning(f"Failed to set Windows preferred voice '{voice.name}': {e}")
			
                # 👔 WORLD-CLASS MALE VOICE SELECTION 👔
                # Priority: Male voices (Microsoft David, Alex, Mark)
                male_keywords = list(getattr(settings, "tts_male_voice_keywords", None) or [])
                female_keywords = list(getattr(settings, "tts_female_voice_keywords", None) or [])
                
                selected_voice = None
                
                # Step 1: Search for high-quality male voices first
                for pref in male_keywords:
                    for voice in voices:
                        if _voice_keyword_match(getattr(voice, "name", "") or "", pref):
                            selected_voice = voice
                            try:
                                self.engine.setProperty('voice', voice.id)
                                logger.info(f"Selected priority voice (keyword='{pref}'): {voice.name}")
                            except Exception as e:
                                logger.warning(f"Failed to set male voice '{voice.name}': {e}")
                            break
                    if selected_voice: break
                
                # Step 2: Fallback to any voice that doesn't explicitly contain female keywords
                if not selected_voice:
                    non_female = []
                    for voice in voices:
                        voice_name_lower = (getattr(voice, "name", "") or "").lower()
                        if any(_voice_keyword_match(voice_name_lower, f_key) for f_key in female_keywords):
                            continue
                        non_female.append(voice)

                    # Prefer US/UK English voices first.
                    for voice in non_female:
                        if _voice_prefers_english(
                            name=getattr(voice, "name", "") or "",
                            voice_id=getattr(voice, "id", "") or "",
                            languages=getattr(voice, "languages", None),
                        ):
                            selected_voice = voice
                            try:
                                self.engine.setProperty('voice', voice.id)
                                logger.info(f"Found alternative voice (preferred language): {voice.name}")
                            except Exception as e:
                                logger.warning(f"Failed to set alternative voice '{voice.name}': {e}")
                            break

                    # If no preferred US/UK English voice is available, pick the first non-female voice,
                    # but be explicit in logs so it's easy to diagnose missing voice packs.
                    if not selected_voice and non_female:
                        voice = non_female[0]
                        selected_voice = voice
                        try:
                            self.engine.setProperty('voice', voice.id)
                            logger.warning(
                                "No preferred US/UK English voice found; using fallback voice: %s",
                                getattr(voice, "name", ""),
                            )
                        except Exception as e:
                            logger.warning(f"Failed to set alternative voice '{voice.name}': {e}")
                
                # Step 3: Absolute final fallback (any voice, even female)
                if not selected_voice:
                    selected_voice = voices[0]
                    try:
                        self.engine.setProperty('voice', selected_voice.id)
                        logger.info(f"Using final fallback voice: {selected_voice.name}")
                    except Exception as e:
                        logger.warning(f"Failed to set final fallback voice '{selected_voice.name}': {e}")
            
            self._initialized = True
            logger.info("pyttsx3 engine initialized successfully with conversational settings")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to initialize pyttsx3: {e}")
            logger.info("TTS will use fallback mode (gTTS or mock) - This is expected in Docker without eSpeak/SAPI")
            return False

    def _configure_pyttsx3_engine(self, engine) -> None:
        """Apply rate/volume/voice selection to a pyttsx3 engine instance."""
        # Configure voice parameters for natural, conversational tone
        engine.setProperty('rate', 190)  # Faster for high-stakes professional interview (180-200 WPM)
        engine.setProperty('volume', 0.95)  # Clear but not overwhelming

        # Try to set a specific voice if available
        voices = engine.getProperty('voices')
        if not voices:
            return

        # Optional hard override: pick an exact voice id if provided.
        if getattr(settings, "practice_tts_voice_id", None):
            voice_id = settings.practice_tts_voice_id
            engine.setProperty("voice", voice_id)
            logger.info(f"Selected configured voice id: {voice_id}")
            return

        # Optional soft override: pick first voice whose name contains this substring.
        name_contains = (getattr(settings, "practice_tts_voice_name_contains", None) or "").strip().lower()
        if name_contains:
            for voice in voices:
                if name_contains in (getattr(voice, "name", "") or "").lower():
                    engine.setProperty("voice", voice.id)
                    logger.info(f"Selected configured voice by name match: {voice.name}")
                    return

        # On Windows, strongly prefer Microsoft David if present.
        if os.name == "nt":
            for voice in voices:
                if _voice_is_windows_david(getattr(voice, "name", "") or "", getattr(voice, "id", "") or ""):
                    engine.setProperty("voice", voice.id)
                    logger.info(f"Selected Windows preferred voice: {voice.name}")
                    return

        # 👔 WORLD-CLASS MALE VOICE SELECTION 👔
        male_keywords = list(getattr(settings, "tts_male_voice_keywords", None) or [])
        female_keywords = list(getattr(settings, "tts_female_voice_keywords", None) or [])

        # Step 1: Search for high-quality male voices first
        for pref in male_keywords:
            for voice in voices:
                if _voice_keyword_match(getattr(voice, "name", "") or "", pref):
                    engine.setProperty('voice', voice.id)
                    logger.info(f"Selected priority voice (keyword='{pref}'): {voice.name}")
                    return

        # Step 2: Fallback to any voice that doesn't explicitly contain female keywords
        non_female = []
        for voice in voices:
            voice_name_lower = (getattr(voice, "name", "") or "").lower()
            if any(_voice_keyword_match(voice_name_lower, f_key) for f_key in female_keywords):
                continue
            non_female.append(voice)

        for voice in non_female:
            if _voice_prefers_english(
                name=getattr(voice, "name", "") or "",
                voice_id=getattr(voice, "id", "") or "",
                languages=getattr(voice, "languages", None),
            ):
                engine.setProperty('voice', voice.id)
                logger.info(f"Found alternative voice (preferred language): {voice.name}")
                return

        if non_female:
            engine.setProperty('voice', non_female[0].id)
            logger.warning(f"No preferred US/UK English voice found; using fallback voice: {non_female[0].name}")
            return

        # Step 3: Absolute final fallback (any voice)
        engine.setProperty('voice', voices[0].id)
        logger.info(f"Using final fallback voice: {voices[0].name}")

    def _pick_windows_sapi_voice_token(self, sp_voice):
        """Pick a Windows SAPI voice token matching configured preferences (best-effort)."""
        try:
            voices = list(sp_voice.GetVoices())
        except Exception:
            voices = []

        if not voices:
            return None

        # Optional hard override: voice id token (rarely used); we treat it as substring match.
        name_contains = (getattr(settings, "practice_tts_voice_name_contains", None) or "").strip().lower()
        if name_contains:
            for token in voices:
                try:
                    name = (token.GetDescription() or "").lower()
                except Exception:
                    name = ""
                if name_contains in name:
                    return token

        # Strong preference: Microsoft David (Windows SAPI). If the host does not have it,
        # we'll fall back to the keyword heuristics below.
        for token in voices:
            try:
                name = (token.GetDescription() or "")
            except Exception:
                name = ""
            if "david" in name.lower():
                return token

        male_keywords = list(getattr(settings, "tts_male_voice_keywords", None) or [])
        female_keywords = list(getattr(settings, "tts_female_voice_keywords", None) or [])

        # Prefer male keywords
        for pref in male_keywords:
            for token in voices:
                try:
                    name = (token.GetDescription() or "").lower()
                except Exception:
                    name = ""
                if _voice_keyword_match(name, pref):
                    return token

        # Avoid explicitly female keywords
        non_female_tokens = []
        for token in voices:
            try:
                name = (token.GetDescription() or "")
            except Exception:
                name = ""
            name_l = name.lower()
            if any(_voice_keyword_match(name_l, f) for f in female_keywords):
                continue
            non_female_tokens.append((token, name))

        for token, name in non_female_tokens:
            if _voice_prefers_english(name=name):
                return token

        if non_female_tokens:
            return non_female_tokens[0][0]

        return voices[0]

    async def _synthesize_windows_sapi(self, text: str, output_path: Path) -> Optional[str]:
        """Synthesize speech to WAV using Windows SAPI directly (more reliable than pyttsx3)."""
        if os.name != "nt":
            return None

        # Import inside method so non-Windows environments don’t require pywin32.
        try:
            import win32com.client  # type: ignore
        except Exception:
            return None

        timeout_seconds = float(getattr(self.config, "max_generation_time", None) or 30.0)
        wav_path = output_path.with_suffix(".wav")

        def _generate() -> Optional[str]:
            try:
                voice = win32com.client.Dispatch("SAPI.SpVoice")
                token = self._pick_windows_sapi_voice_token(voice)
                if token is not None:
                    try:
                        voice.Voice = token
                        logger.info(f"Selected Windows SAPI voice: {token.GetDescription()}")
                    except Exception:
                        pass

                stream = win32com.client.Dispatch("SAPI.SpFileStream")
                # 3 == SSFMCreateForWrite
                stream.Open(str(wav_path), 3)
                voice.AudioOutputStream = stream
                voice.Speak(text)
                stream.Close()
                return str(wav_path) if wav_path.exists() else None
            except Exception as e:
                logger.error(f"Windows SAPI generation error: {e}")
                try:
                    stream.Close()
                except Exception:
                    pass
                return None

        async with self._pyttsx3_lock:
            try:
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(loop.run_in_executor(None, _generate), timeout=timeout_seconds)
                if result and Path(result).exists():
                    return result
            except asyncio.TimeoutError:
                logger.warning(f"Windows SAPI synthesis timed out after {timeout_seconds:.1f}s")
            except Exception as e:
                logger.warning(f"Windows SAPI synthesis failed: {e}")

        return None
    
    def cleanup(self):
        """Cleanup TTS resources and stop any running threads."""
        try:
            if self.engine and self._initialized:
                # Stop pyttsx3 engine and cleanup threads
                try:
                    self.engine.stop()
                except Exception:
                    pass
                self.engine = None
                self._initialized = False
                logger.info("TTS engine cleaned up")
        except Exception as e:
            logger.error(f"Error during TTS cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

    # ------------------------------------------------------------------
    # Edge-TTS (Microsoft Neural Voices) — Primary high-quality engine
    # ------------------------------------------------------------------

    # Curated interviewer voices — professional, clear, natural
    _EDGE_INTERVIEWER_VOICES = [
        "en-US-GuyNeural",          # Deep, confident male — excellent for interviewer
        "en-US-ChristopherNeural",  # Warm, professional male
        "en-US-EricNeural",         # Clear, authoritative male
        "en-US-SteffanNeural",      # Calm, measured male
    ]

    def _build_interviewer_ssml(self, text: str, voice: str) -> str:
        """
        Convert plain interviewer text into SSML with natural prosody.

        Adds:
        - Strategic <break/> pauses at ellipsis and sentence boundaries
        - Emphasis on question words for rising intonation

        NOTE: This SSML is used for engines that accept raw SSML.
        For edge-tts we use the simpler Communicate API (rate/pitch params).
        """
        import re as _re

        # Insert <break/> at ellipsis markers the format_tts_text engine leaves behind
        processed = _re.sub(r'\.{2,3}', ' <break time="400ms"/> ', text)

        # Add a short breath pause after intro sentences (first sentence ending with . or ,)
        first_sentence_end = _re.search(r'(?<=[.!])\s', processed)
        if first_sentence_end:
            pos = first_sentence_end.end()
            processed = processed[:pos] + '<break time="350ms"/> ' + processed[pos:]

        # Light emphasis on question words at start of clauses
        processed = _re.sub(
            r'\b(What|How|Why|When|Where|Who|Which|Can you|Could you|Tell me|Describe|Explain)\b',
            r'<emphasis level="moderate">\1</emphasis>',
            processed,
            count=2,  # Only first 2 occurrences to avoid over-emphasis
        )

        ssml = (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">'
            f'<voice name="{voice}">'
            '<prosody rate="-5%" pitch="-3%">'
            f'{processed}'
            '</prosody>'
            '</voice>'
            '</speak>'
        )
        return ssml

    async def _synthesize_edge_tts(self, text: str, output_path: Path) -> Optional[str]:
        """
        Synthesize speech using Microsoft Edge Neural TTS.

        Produces natural, human-sounding interviewer audio using Azure Neural voices
        — the same voices that power Edge browser's Read Aloud feature.
        Free, no API key required.  Needs internet connectivity.

        The neural voices inherently handle:
        - Question intonation (rising pitch)
        - Natural pauses at punctuation
        - Emphasis based on sentence context
        - Human-like warmth and authority

        Combined with format_tts_text() which adds interviewer intros/transitions,
        this produces realistic interviewer audio.
        """
        if not EDGE_TTS_AVAILABLE:
            return None

        mp3_path = output_path.with_suffix(".mp3")

        # Pick voice: use configured setting or default to GuyNeural
        voice = (
            getattr(settings, "practice_edge_tts_voice", None)
            or self._EDGE_INTERVIEWER_VOICES[0]
        )

        try:
            # Edge-tts Communicate handles SSML internally.
            # We pass plain text + prosody params for the best result.
            # The neural voice does the heavy lifting — natural intonation,
            # question inflection, pauses — far beyond what SAPI/espeak can do.
            communicate = edge_tts.Communicate(
                text,
                voice,
                rate="-5%",   # Slightly slower than default for interview gravitas
                pitch="-3Hz", # Slightly deeper pitch for authoritative interviewer
            )
            await communicate.save(str(mp3_path))

            if mp3_path.exists() and mp3_path.stat().st_size > 0:
                logger.info(f"Edge-TTS synthesis complete ({voice}): {mp3_path}")
                return str(mp3_path)
            else:
                logger.warning("Edge-TTS produced empty file")
                return None

        except Exception as e:
            logger.warning(f"Edge-TTS synthesis failed: {e}")
            return None

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
        
        # Check cache — also check .mp3 variant since edge-tts/gTTS produce mp3
        if use_cache:
            if output_path.exists():
                logger.debug(f"Using cached TTS: {output_path}")
                return str(output_path)
            mp3_variant = output_path.with_suffix(".mp3")
            if mp3_variant.exists():
                logger.debug(f"Using cached TTS (mp3): {mp3_variant}")
                return str(mp3_variant)

        # =================================================================
        # ENGINE CASCADE — best quality first, offline fallbacks last
        # =================================================================

        # 1. Edge-TTS (Neural voices — sounds like a real interviewer)
        if EDGE_TTS_AVAILABLE and self.config.engine in ["edge_tts", "edge-tts", "neural", "auto", "pyttsx3", "offline"]:
            try:
                audio_path = await self._synthesize_edge_tts(normalized_text, output_path)
                if audio_path:
                    return audio_path
            except Exception as e:
                logger.warning(f"Edge-TTS synthesis failed, trying fallbacks: {e}")

        # 2. pyttsx3 / Windows SAPI (offline fallback)
        if self.config.engine in ["pyttsx3", "offline", "auto"]:
            try:
                audio_path = await self._synthesize_pyttsx3(normalized_text, output_path)
                if audio_path:
                    return audio_path
            except Exception as e:
                logger.warning(f"pyttsx3 synthesis failed: {e}")

        # If you want a consistent device voice, you can hard-disable gTTS fallback.
        if getattr(settings, "practice_tts_disable_gtts_fallback", False):
            raise RuntimeError(
                "All primary TTS engines failed and gTTS fallback is disabled "
                "(practice_tts_disable_gtts_fallback=true)."
            )

        # 3. gTTS (Google online fallback — flat but reliable)
        if GTTS_AVAILABLE:
            try:
                audio_path = await self._synthesize_gtts(normalized_text, output_path)
                if audio_path:
                    return audio_path
            except Exception as e:
                logger.error(f"gTTS synthesis failed: {e}")

        raise RuntimeError(
            "All TTS engines failed. Install edge-tts (recommended), pyttsx3, or gtts."
        )
    
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
        # On Windows, prefer direct SAPI output to avoid pyttsx3 second-call hangs.
        if os.name == "nt":
            try:
                sapi_path = await self._synthesize_windows_sapi(text, output_path)
                if sapi_path:
                    logger.info(f"Windows SAPI synthesis complete: {sapi_path}")
                    return sapi_path
            except Exception:
                # Fall back to pyttsx3 below.
                pass

        if not PYTTSX3_AVAILABLE:
            return None

        timeout_seconds = float(getattr(self.config, "max_generation_time", None) or 15.0)

        def _generate() -> Optional[str]:
            # Create a fresh engine per synthesis.
            # This avoids pyttsx3 engine state issues that commonly hang on the 2nd+ run.
            try:
                if os.name == "nt":
                    engine = pyttsx3.init(driverName="sapi5")
                else:
                    engine = pyttsx3.init()

                try:
                    self._configure_pyttsx3_engine(engine)
                except Exception as e:
                    logger.warning(f"Failed to configure pyttsx3 voice settings: {e}")

                engine.save_to_file(text, str(output_path))
                engine.runAndWait()
                try:
                    engine.stop()
                except Exception:
                    pass

                return str(output_path) if output_path.exists() else None
            except Exception as e:
                # IMPORTANT: this can fail for many reasons (missing espeak on Linux,
                # missing voices/COM issues on Windows, engine state issues, etc.).
                hint = ""
                if os.name == "nt":
                    hint = " (Windows: ensure SAPI voices are installed and accessible)"
                else:
                    hint = " (Linux: ensure 'espeak-ng' is installed in the image)"
                logger.error(f"pyttsx3 generation error{hint}: {type(e).__name__}: {e}")
                return None
        
        # pyttsx3 isn't thread-safe; serialize calls to avoid deadlocks/timeouts.
        async with self._pyttsx3_lock:
            # Run in executor with timeout to avoid blocking the event loop.
            try:
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, _generate),
                    timeout=timeout_seconds,
                )

                if result and output_path.exists():
                    logger.info(f"pyttsx3 synthesis complete: {output_path}")
                    return str(output_path)
            except asyncio.TimeoutError:
                logger.warning(
                    f"pyttsx3 synthesis timed out after {timeout_seconds:.1f}s"
                )
            except Exception as e:
                logger.warning(f"pyttsx3 synthesis failed: {e}")
        
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

        if EDGE_TTS_AVAILABLE:
            info["available_engines"].append("edge_tts")
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
            except Exception:
                pass
        
        return info
    
    def cleanup(self):
        """Clean up TTS resources."""
        if self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass
        
        logger.info("TTS cleanup complete")
