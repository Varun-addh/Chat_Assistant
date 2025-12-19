"""
Evaluation Agent - Production-grade implementation.
Uses Gemini Pro API for final interview coaching report.
"""

import logging
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import asyncio

import google.generativeai as genai

from app.schemas import (
    EvaluationReport,
    EvaluationStrengths,
    EvaluationImprovements,
    MetricsSummary,
    ActionPlan,
    AnswerSubmission,
    SpeechMetrics
)

logger = logging.getLogger(__name__)


class EvaluationAgent:
    """
    Agent 3: Evaluation Agent
    Uses Gemini Pro for comprehensive interview evaluation.
    Called ONCE at the end of the interview.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        """
        Initialize the evaluation agent.
        
        Args:
            api_key: Gemini API key
            model_name: Model name (default: gemini-1.5-pro)
        """
        genai.configure(api_key=api_key)
        
        # Try to list available models to validate API key
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            logger.info(f"Available Gemini models: {available_models}")
            
            # Use first available model if default doesn't work
            if available_models:
                # Prefer gemini-1.5-pro or gemini-1.5-flash
                for preferred in ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro']:
                    matching = [m for m in available_models if preferred in m]
                    if matching:
                        model_name = matching[0].replace('models/', '')
                        logger.info(f"Using model: {model_name}")
                        break
                else:
                    # Use first available
                    model_name = available_models[0].replace('models/', '')
                    logger.warning(f"No preferred model found, using: {model_name}")
        except Exception as e:
            logger.warning(f"Could not list models (will try default): {e}")
        
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        logger.info(f"Evaluation Agent initialized with {model_name}")
    
    async def evaluate_interview(
        self, 
        answers: List[AnswerSubmission],
        session_id: str,
        api_key: Optional[str] = None
    ) -> EvaluationReport:
        """
        Generate comprehensive evaluation report using Gemini Pro.
        
        Args:
            answers: List of all submitted answers
            session_id: Session identifier for logging
            
        Returns:
            Complete EvaluationReport
        """
        try:
            logger.info(f"Starting evaluation for session {session_id} with {len(answers)} answers")
            
            # Validate we have answers
            if not answers:
                logger.error("No answers provided for evaluation")
                return self._get_fallback_evaluation([])
            
            # Calculate aggregated metrics
            metrics_summary = self._calculate_metrics_summary(answers)
            logger.info(f"Metrics: {metrics_summary.total_fillers} fillers, "
                       f"{metrics_summary.avg_wpm} WPM, "
                       f"{metrics_summary.avg_confidence}/10 confidence")
            
            # Prepare data for Gemini
            prompt = self._build_evaluation_prompt(answers, metrics_summary)
            
            # Call Gemini Pro
            logger.info("Calling Gemini Pro API for evaluation...")
            response = await self._call_gemini(prompt, api_key)
            
            # Log raw response for debugging
            logger.info(f"Raw Gemini response length: {len(response)} chars")
            logger.info(f"First 500 chars: {response[:500]}")
            logger.info(f"Last 500 chars: {response[-500:]}")
            
            # Parse response
            evaluation_data = self._parse_gemini_response(response)
            
            # Build evaluation report
            report = EvaluationReport(
                strengths=EvaluationStrengths(items=evaluation_data["strengths"]),
                improvements=EvaluationImprovements(items=evaluation_data["improvements"]),
                metrics_summary=metrics_summary,
                action_plan=ActionPlan(steps=evaluation_data["action_plan"]),
                practice_recommendation=evaluation_data["practice_recommendation"]
            )
            
            logger.info(f"Evaluation complete for session {session_id}")
            return report
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            # Return fallback evaluation
            return self._get_fallback_evaluation(answers)
    
    def _calculate_metrics_summary(self, answers: List[AnswerSubmission]) -> MetricsSummary:
        """
        Calculate aggregated metrics across all answers.
        
        Args:
            answers: List of answer submissions
            
        Returns:
            MetricsSummary object
        """
        total_fillers = sum(a.metrics.filler_count for a in answers)
        avg_wpm = sum(a.metrics.wpm for a in answers) / len(answers) if answers else 0
        longest_pause = max((a.metrics.longest_silence for a in answers), default=0)
        avg_confidence = sum(a.metrics.confidence_score for a in answers) / len(answers) if answers else 0
        total_duration = sum(a.metrics.duration for a in answers)
        overtalked_count = sum(1 for a in answers if a.metrics.overtalked)
        
        return MetricsSummary(
            total_fillers=total_fillers,
            avg_wpm=round(avg_wpm, 1),
            longest_pause=round(longest_pause, 2),
            avg_confidence=round(avg_confidence, 1),
            total_duration=round(total_duration, 2),
            overtalked_count=overtalked_count
        )
    
    def _build_evaluation_prompt(
        self, 
        answers: List[AnswerSubmission], 
        metrics: MetricsSummary
    ) -> str:
        """
        Build comprehensive prompt for Gemini Pro.
        
        Args:
            answers: List of answer submissions
            metrics: Aggregated metrics
            
        Returns:
            Formatted prompt string
        """
        # Prepare transcript data
        transcripts = []
        for i, answer in enumerate(answers, 1):
            transcripts.append({
                "question_num": i,
                "transcript": answer.transcript,
                "filler_count": answer.metrics.filler_count,
                "wpm": answer.metrics.wpm,
                "confidence": answer.metrics.confidence_score
            })
        
        transcripts_json = json.dumps(transcripts, indent=2)
        
        prompt = f"""You are an expert interview coach with 15 years of experience. Analyze this practice interview performance.

Interview Data:
- Total filler words: {metrics.total_fillers}
- Average speaking pace: {metrics.avg_wpm} WPM (ideal 140-160)
- Longest pause: {metrics.longest_pause} seconds
- Average confidence score: {metrics.avg_confidence}/10 (pitch stability analysis)
- Total duration: {metrics.total_duration:.0f} seconds
- Questions where overtalked: {metrics.overtalked_count}/5

Transcripts by Question:
{transcripts_json}

Generate a professional coaching report with:

1. STRENGTHS (2-3 specific items)
   - Must be evidence-based from the data
   - Reference specific metrics or transcript examples
   - Be specific and actionable

2. AREAS TO IMPROVE (2-3 specific items)
   - Must be evidence-based with examples
   - Reference specific metrics or patterns in transcripts
   - Prioritize by impact

3. ACTION PLAN (2-3 concrete steps)
   - Must be specific and actionable
   - Focus on delivery improvement
   - Realistic and achievable

4. PRACTICE RECOMMENDATION
   - Estimate sessions needed (e.g., "2-3 more sessions recommended")
   - OR "Ready for real interviews with minor polish"
   - Be realistic based on the data

CRITICAL RULES:
- NO scores or percentages in the report
- NO generic statements like "good job" without evidence
- Be specific with examples from transcripts or metrics
- Focus on delivery (pace, fillers, pauses, confidence)
- Keep each item under 30 words
- Use professional coaching tone

Return ONLY a valid JSON object with this exact structure. DO NOT include markdown code blocks or explanations.
Ensure all strings are properly escaped and closed with quotes:
{{
  "strengths": ["strength 1 - properly escaped string", "strength 2", "strength 3"],
  "improvements": ["improvement 1 - properly escaped string", "improvement 2", "improvement 3"],
  "action_plan": ["step 1 - properly escaped string", "step 2", "step 3"],
  "practice_recommendation": "recommendation text - properly escaped"
}}

IMPORTANT: Return ONLY the JSON object. No markdown, no code blocks, no explanations before or after. Ensure ALL strings are properly terminated."""
        
        return prompt
    
    async def _call_gemini(self, prompt: str, api_key: Optional[str] = None) -> str:
        """
        Call Gemini Pro API with safety filter handling.
        
        Args:
            prompt: Evaluation prompt
            
        Returns:
            API response text
        """
        try:
            # Configure with relaxed safety settings
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            if api_key:
                genai.configure(api_key=api_key)

            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=self.generation_config,
                safety_settings=safety_settings
            )
            
            # Check if response was blocked
            if not response.text:
                if hasattr(response, 'prompt_feedback'):
                    logger.warning(f"Gemini blocked prompt: {response.prompt_feedback}")
                if hasattr(response, 'candidates') and response.candidates:
                    finish_reason = response.candidates[0].finish_reason
                    logger.warning(f"Gemini finish_reason: {finish_reason}")
                    if finish_reason == 2:  # SAFETY
                        logger.warning("Content blocked by safety filters, using fallback")
                raise ValueError("Gemini response blocked by safety filters")
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            raise
    
    def _parse_gemini_response(self, response: str) -> Dict[str, Any]:
        """
        Parse Gemini response into structured data.
        
        Args:
            response: Raw Gemini response
            
        Returns:
            Parsed evaluation data
        """
        try:
            # Try to extract JSON from response
            # Sometimes Gemini wraps JSON in markdown code blocks
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            
            if response.endswith("```"):
                response = response[:-3]
            
            response = response.strip()
            
            # Repair common JSON issues before parsing
            response = self._repair_json(response)
            
            # Parse JSON
            try:
                data = json.loads(response)
            except json.JSONDecodeError as first_error:
                # Try more aggressive repair
                logger.warning(f"Initial JSON parse failed, trying aggressive repair: {first_error}")
                response = self._aggressive_json_repair(response)
                try:
                    data = json.loads(response)
                except json.JSONDecodeError as second_error:
                    # Last resort: manual extraction
                    logger.error(f"Aggressive repair failed: {second_error}")
                    data = self._manual_json_extract(response)
                    if not data:
                        raise
            
            # Validate required fields
            required = ["strengths", "improvements", "action_plan", "practice_recommendation"]
            for field in required:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure lists have correct length
            data["strengths"] = data["strengths"][:3]  # Max 3
            data["improvements"] = data["improvements"][:3]  # Max 3
            data["action_plan"] = data["action_plan"][:3]  # Max 3
            
            # Ensure minimum length
            if len(data["strengths"]) < 2:
                data["strengths"].append("Completed the practice interview")
            if len(data["improvements"]) < 2:
                data["improvements"].append("Continue practicing for consistency")
            if len(data["action_plan"]) < 2:
                data["action_plan"].append("Practice regularly with varied questions")
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {e}")
            logger.debug(f"Problematic response (first 1000 chars): {response[:1000]}")
            raise ValueError("Invalid JSON response from Gemini")
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            raise
    
    def _repair_json(self, text: str) -> str:
        """Repair common JSON formatting issues from LLM responses."""
        import re
        
        # Step 1: Fix unterminated strings more aggressively
        # Look for patterns like: "some text without closing quote,
        # or "some text without closing quote}
        # or "some text without closing quote]
        text = re.sub(r'([^"])(\n\s*[,\]\}])', r'\1"\2', text)
        
        # Step 2: Fix strings that span multiple lines incorrectly
        # Replace actual newlines within strings with space
        lines = text.split('\n')
        repaired_lines = []
        in_string = False
        
        for line in lines:
            # Track if we're inside a string
            escaped = False
            new_line = ""
            for char in line:
                if char == '\\' and not escaped:
                    escaped = True
                    new_line += char
                elif char == '"' and not escaped:
                    in_string = not in_string
                    new_line += char
                    escaped = False
                else:
                    new_line += char
                    escaped = False
            
            # If line ends while in_string and doesn't end with quote, add quote
            if in_string and new_line.rstrip() and not new_line.rstrip().endswith('"'):
                # Check if next character would be , } or ]
                stripped = new_line.rstrip()
                if stripped.endswith(',') or stripped.endswith('}') or stripped.endswith(']'):
                    new_line = stripped[:-1] + '"' + stripped[-1]
                else:
                    new_line = new_line.rstrip() + '"'
                in_string = False
            
            repaired_lines.append(new_line)
        
        text = '\n'.join(repaired_lines)
        
        # Step 3: Remove trailing commas before closing brackets/braces
        text = re.sub(r',\s*([\]}])', r'\1', text)
        
        # Step 4: Fix common escape issues
        text = text.replace('\\n', ' ').replace('\\t', ' ')
        
        # Step 5: Ensure proper quote balance in array items
        text = re.sub(r'\[\s*"([^"]*?)\s*,\s*"([^"]*?)\s*,\s*"([^"]*?)"\s*\]', 
                      r'["\1", "\2", "\3"]', text)
        
        return text
    
    def _aggressive_json_repair(self, text: str) -> str:
        """More aggressive JSON repair as fallback."""
        import re
        
        # Remove all actual newlines from within string values
        # This is aggressive but necessary for badly formatted responses
        text = re.sub(r'"\s*\n\s*([^"{}[\],:]+)\s*\n\s*"', r'"\1"', text)
        
        # Fix strings that got broken across lines
        text = re.sub(r':\s*"([^"]*?)\n([^"]*?)"', r': "\1 \2"', text)
        
        # Ensure all strings in arrays are quoted
        text = re.sub(r'\[\s*([^"\[\]]+?)\s*\]', lambda m: f'["{m.group(1)}"]' if ',' not in m.group(1) else m.group(0), text)
        
        return text
    
    def _manual_json_extract(self, text: str) -> dict:
        """Manual extraction as last resort when JSON parsing completely fails."""
        import re
        
        logger.warning("Attempting manual JSON extraction")
        
        result = {
            "strengths": [],
            "improvements": [],
            "action_plan": [],
            "practice_recommendation": ""
        }
        
        # Extract arrays by key
        for key in ["strengths", "improvements", "action_plan"]:
            # Look for "key": ["item1", "item2", "item3"]
            pattern = rf'"{key}"\s*:\s*\[(.*?)\]'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                items_text = match.group(1)
                # Extract quoted strings
                items = re.findall(r'"([^"]*)"', items_text)
                result[key] = items[:3]  # Max 3
        
        # Extract practice recommendation
        pattern = r'"practice_recommendation"\s*:\s*"([^"]*)"'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result["practice_recommendation"] = match.group(1)
        
        # Validate we got something
        if not result["strengths"] and not result["improvements"]:
            logger.error("Manual extraction failed - no data found")
            return None
        
        logger.info(f"Manual extraction successful: {len(result['strengths'])} strengths, {len(result['improvements'])} improvements")
        return result
    
    def _get_fallback_evaluation(self, answers: List[AnswerSubmission]) -> EvaluationReport:
        """
        Generate fallback evaluation when Gemini fails.
        
        Args:
            answers: List of answer submissions
            
        Returns:
            Basic evaluation report
        """
        logger.warning("Using fallback evaluation")
        
        metrics_summary = self._calculate_metrics_summary(answers)
        
        # Generate basic strengths
        strengths = ["Completed all 5 practice questions"]
        if metrics_summary.avg_wpm >= 140 and metrics_summary.avg_wpm <= 160:
            strengths.append("Maintained good speaking pace throughout")
        if metrics_summary.total_fillers < 15:
            strengths.append("Kept filler word usage relatively low")
        
        # Generate basic improvements
        improvements = []
        if metrics_summary.total_fillers > 10:
            improvements.append(f"Reduce filler words (used {metrics_summary.total_fillers} total)")
        if metrics_summary.avg_wpm > 180:
            improvements.append("Slow down speaking pace for clarity")
        elif metrics_summary.avg_wpm < 120:
            improvements.append("Increase speaking pace for better engagement")
        if metrics_summary.longest_pause > 4:
            improvements.append("Work on reducing long pauses")
        if metrics_summary.avg_confidence < 0.6:
            improvements.append("Practice to improve voice stability and confidence")
        
        # Ensure minimum 2 improvements
        if len(improvements) < 2:
            improvements.append("Continue practicing for consistency")
        if len(improvements) < 2:
            improvements.append("Focus on clarity and conciseness in answers")
        
        # Basic action plan
        action_plan = [
            "Practice answering common questions daily",
            "Record yourself and review filler word usage"
        ]
        
        if metrics_summary.avg_wpm > 160:
            action_plan.append("Practice speaking more slowly with pauses")
        
        # Practice recommendation
        if metrics_summary.total_fillers < 10 and 140 <= metrics_summary.avg_wpm <= 160:
            practice_rec = "Ready for real interviews with minor polish"
        else:
            practice_rec = "2-3 more practice sessions recommended"
        
        return EvaluationReport(
            strengths=EvaluationStrengths(items=strengths[:3]),
            improvements=EvaluationImprovements(items=improvements[:3]),
            metrics_summary=metrics_summary,
            action_plan=ActionPlan(steps=action_plan[:3]),
            practice_recommendation=practice_rec
        )


class GeminiResponseValidator:
    """Validates and sanitizes Gemini API responses."""
    
    @staticmethod
    def validate_structure(data: Dict[str, Any]) -> bool:
        """Validate response structure."""
        required_fields = ["strengths", "improvements", "action_plan", "practice_recommendation"]
        
        for field in required_fields:
            if field not in data:
                return False
            
            if field != "practice_recommendation":
                if not isinstance(data[field], list):
                    return False
                if len(data[field]) < 2:
                    return False
        
        return True
    
    @staticmethod
    def sanitize_items(items: List[str], max_length: int = 3) -> List[str]:
        """Sanitize list items."""
        # Remove empty strings
        items = [item.strip() for item in items if item.strip()]
        
        # Limit length
        items = items[:max_length]
        
        # Ensure minimum length
        while len(items) < 2:
            items.append("Continue practicing regularly")
        
        return items
