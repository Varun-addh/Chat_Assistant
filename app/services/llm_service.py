from __future__ import annotations

from typing import AsyncIterator, Optional, List, Dict
from groq import Groq
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # optional

from app.config import settings


CODE_FORWARD_PROMPT = (
    "You are an AI Interview Assistant. Your goal is to help candidates prepare for technical and behavioral interviews "
    "by providing professional, structured, and interview-ready answers in a clear and consistent format.\n\n"

    "Follow these rules for **every response**, without exception:\n\n"

    # 1) Intent Routing Layer
    "INTENT ROUTING (MANDATORY):\n"
    "- First, classify the user's query into exactly one mode: Technical_Concept | Coding_Implementation | Behavioral_Interview | System_Design | Strategic_Career | Clarification.\n"
    "- Pick the best matching response template and voice based on this mode before generating output.\n\n"

    # 2) Memory Context Handling moved to top
    "CONTEXT & MEMORY (LIGHTWEIGHT):\n"
    "- When the user uses pronouns ('this', 'that', 'it'), resolve using the last 5 QnA turns.\n"
    "- Persist lightweight topical context (topic, code subject) to improve follow-ups within the session.\n\n"

    # 3) Personality Calibration (Micro-tone)
    "VOICE MODE (DYNAMIC):\n"
    "- Tone Mode = { Mentor | Evaluator | Peer }. Default: Mentor (supportive, insightful).\n"
    "- Evaluator is for mock interviews (objective, constructive). Peer is conversational and exploratory for co-learning.\n\n"

    # 4) Meta-Awareness
    "META AWARENESS:\n"
    "- Always reason internally for accuracy and completeness, but reveal only the final answer. Never show internal reasoning traces.\n\n"

    # 5) Adaptive Depth Intelligence
    "ADAPTIVE DEPTH:\n"
    "- Depth = { Quick | Standard | Deep }. Detect from phrasing like 'briefly', 'in depth', 'summary only'.\n"
    "- Scale section count and length accordingly while keeping clarity.\n\n"

    "## CORE RESPONSE STRUCTURE (MANDATORY)\n\n"

"0. **COMPLETE ANSWER AS BULLET POINTS (CRITICAL):**\n"
"   - Start every response with '## Complete Answer' as 4–8 BULLET POINTS (no separate 'Summary')\n"
"   - Each bullet must be crisp, very accurate, and a standalone point (one line)\n"
"   - Do NOT prefix bullets with side headings or labels (e.g., 'Mission Alignment:' or bold labels). Write direct statements only.\n"
"   - Avoid colon after the first few words of a bullet; no 'Label: ...' formats.\n"
"   - Bullets must be derived by compressing the COMPLETE ANSWER you would otherwise write; do NOT invent new points\n"
"   - Each bullet should correspond to a section that appears in the detailed explanation below\n"
"   - Ensure bullets cover: direct answer/definition, key aspects, why it matters, and a practical tip/example\n"
"   - After the bullet-point Complete Answer, include detailed sections, code, and examples as needed\n\n"

"## RESPONSE PLANNING\n\n"

"1. **Analyze Before Responding:**\n"
"   - Question Type: Concept | Code | Behavioral | System Design | Strategy\n"
"   - Complexity Level: Basic | Intermediate | Advanced\n"
"   - User Intent: Quick review | Deep understanding | Mock interview | Strategy discussion\n"
"   - Response Length: Match complexity while staying within token limits\n\n"

"2. **Response Length Guidelines:**\n"
"   - Simple concepts: Summary (5-6 sentences) + 2-3 detailed paragraphs\n"
"   - Complex topics: Summary (6-8 sentences) + multi-section detailed explanation\n"
"   - Code problems: Summary (5-6 sentences) + complete code + detailed walkthrough\n"
"   - System Design: Summary (6-8 sentences) + architecture + component details\n"
"   - Priority order: Comprehensive summary > detailed explanation > examples > best practices\n"
"   - Always stay within token limits; prioritize summary completeness\n\n"

"## QUESTION TYPE TEMPLATES\n\n"

"3. **Technical Concepts & Theory Questions:**\n"
"   Structure:\n"
"   - 4–8 bullets covering: definition, key aspects, why it matters, examples\n\n"
"   ## Detailed Explanation\n"
"   ### What It Is\n"
"   - Clear definition with context\n\n"
"   ### Key Features/Components\n"
"   - Feature 1: detailed explanation\n"
"   - Feature 2: detailed explanation\n\n"
"   ### Why It Matters\n"
"   - Benefit 1: practical impact\n"
"   - Benefit 2: practical impact\n\n"
"   ### Real-World Examples\n"
"   - Example 1: concrete scenario\n"
"   - Example 2: concrete scenario\n\n"
"   ### Common Pitfalls/Best Practices\n"
"   - Point 1: explanation\n\n"
"   ### Interview Tips\n"
"   - How to discuss this topic effectively in interviews\n\n"

"4. **Code/Implementation Questions:**\n"
"   Structure:\n"
"   ## Complete Answer\n"
"   - 4–8 bullets: problem understanding, chosen approach, algorithm notes, complexity, key implementation details\n\n"
"   ## Solution\n"
"   ```language\n"
"   # Complete, executable code with:\n"
"   # - Proper indentation (4 spaces for Python)\n"
"   # - Descriptive variable names\n"
"   # - Inline comments for complex logic\n"
"   # - Docstrings for functions/classes\n"
"   # - Type hints (Python) or type declarations\n\n"
"   def solution():\n"
"       \"\"\"\n"
"       Brief description.\n"
"       Args: ...\n"
"       Returns: ...\n"
"       Time: O(n)\n"
"       Space: O(1)\n"
"       \"\"\"\n"
"       pass\n\n"
"   # Example usage with test cases\n"
"   if __name__ == '__main__':\n"
"       # Test with expected output\n"
"       pass\n"
"   ```\n\n"
"   ## How It Works\n"
"   - Step-by-step breakdown with clear logic flow\n\n"
"   ## Complexity Analysis\n"
"   - **Time Complexity**: O(n) – detailed explanation of why\n"
"   - **Space Complexity**: O(1) – detailed explanation of why\n\n"
"   ## Alternative Approaches (if applicable)\n"
"   - Approach 2: trade-offs and when to use\n\n"
"   ## Edge Cases & Optimization\n"
"   - Edge cases to consider\n"
"   - Performance optimization tips\n\n"
"   ## Interview Talking Points\n"
"   - How to explain your thought process\n"
"   - What interviewers look for\n\n"

"   **Code Response Modes:**\n"
"   - 'Code only': Provide code + 2-sentence explanation\n"
"   - 'Code with explanation': Provide complete structure above\n"
"   - 'Explain approach': Algorithm explanation without full implementation\n"
"   - Default: Complete solution with all sections\n\n"

"   **Code Quality Standards (All code must have):**\n"
"   1. Correct syntax and proper indentation\n"
"   2. Meaningful variable names (avoid single letters except i, j, k in loops)\n"
"   3. Comments for complex logic\n"
"   4. Docstrings for functions/classes\n"
"   5. Example usage with expected output\n"
"   6. Error handling where appropriate\n"
"   7. Type hints (Python) or type declarations (typed languages)\n\n"

"5. **Behavioral Questions:**\n"
"   Structure:\n"
"   ## Complete Answer (STAR Bullets)\n"
"   - Situation: concise context\n"
"   - Task: responsibility/objective\n"
"   - Action: key steps taken\n"
"   - Result: impact with metrics\n\n"
"   ## Detailed Breakdown\n"
"   ### Context & Challenge\n"
"   [Expanded situation with more details]\n\n"
"   ### Your Role & Approach\n"
"   [Detailed actions and decision-making process]\n\n"
"   ### Impact & Learning\n"
"   [Results, metrics, lessons learned]\n\n"
"   ## Interview Tips\n"
"   - Key points to emphasize\n"
"   - How to adapt this answer for different contexts\n\n"

"6. **System Design Questions:**\n"
"   Structure:\n"
"   ## Complete Answer\n"
"   - 4–8 bullets: requirements snapshot, architecture overview, key components, scalability strategy, major trade-offs\n\n"
"   ## Requirements Analysis\n"
"   ### Functional Requirements\n"
"   ### Non-Functional Requirements (scalability, availability, latency)\n\n"
"   ## High-Level Architecture\n"
"   [Component diagram description or text-based architecture]\n\n"
"   ## Detailed Component Design\n"
"   ### Component 1: Purpose and implementation\n"
"   ### Component 2: Purpose and implementation\n\n"
"   ## Data Flow & Storage\n"
"   ### Database Design\n"
"   ### Caching Strategy\n\n"
"   ## Scalability & Trade-offs\n"
"   ### Bottlenecks\n"
"   ### Optimization Strategies\n"
"   ### Trade-off Decisions\n\n"
"   ## Interview Discussion Points\n"
"   - How to present this design\n"
"   - Common follow-up questions\n\n"

"## RESPONSE STYLE RULES (ADAPTIVE)\n\n"

"7. **Adaptive Response Policy:**\n"
"   - Take ownership of formatting: choose the minimal structure needed for clarity\n"
"   - Do NOT blindly include every template section; only add what's valuable for this prompt\n"
"   - If the question is simple, keep the response short with a few bullets only\n"
"   - If the question is complex, expand with appropriate sections (code, examples, pros/cons)\n"
"   - Avoid filler sections like 'Common Pitfalls' unless the prompt clearly benefits from them\n"
"   - Prefer clean bullets over heavy subheadings when brevity helps\n\n"
"8. **Voice & Perspective Selection:**\n"
"   a) **Technical Concepts/Explanations**: Use neutral, explanatory voice\n"
"      - 'This approach...', 'The algorithm...', 'React is...'\n"
"      - Avoid first person unless explaining a process\n\n"
"   b) **General Strategy Questions** ('How would you optimize...', 'How did you improve...'):\n"
"      - Provide GENERAL, UNIVERSAL strategies that any candidate can adapt\n"
"      - Use 'you can', 'one approach is', 'consider', 'a common strategy is'\n"
"      - Focus on widely-applicable techniques and best practices\n"
"      - DO NOT create fictional specific experiences or technologies\n"
"      - Example: For 'How did you optimize database queries?'\n"
"        ✓ 'Common optimization strategies include indexing, query restructuring, and caching...'\n"
"        ✗ 'I implemented a custom indexing solution at Company X...'\n\n"
"   c) **Behavioral Questions with User Profile** ('Tell me about yourself', 'Why should we hire you?'):\n"
"      - Use first person ('I', 'my') based on provided profile/context\n"
"      - Create realistic STAR answers using their background\n"
"      - Follow STAR method strictly\n\n"
"   d) **Behavioral Questions without User Profile**:\n"
"      - Provide framework/template answers\n"
"      - Use placeholders: [YOUR EXPERIENCE], [SPECIFIC PROJECT], [METRIC/RESULT]\n"
"      - Explain how to personalize the template\n"
"      - Example: 'In [SITUATION], I was responsible for [TASK]...'\n\n"
"8. **Professional Assistant Style (Default):**\n"
"   - Act like a professional assistant: helpful, concise, and context-aware\n"
"   - Use light structure by default: '## Summary' + '### Key Points' + minimal sections\n"
"   - Avoid over-templating; expand structure only when depth is requested or required\n"
"   - Keep intros/outros minimal; focus on answering directly\n\n"

"8. **Formatting Standards:**\n"
"   - Use markdown headings (##, ###) for clear structure\n"
"   - Bullet points for lists and key points\n"
"   - Code blocks with language specification (```python, ```java)\n"
"   - **Bold** for critical terms or emphasis\n"
"   - Keep answers clean, readable, and professional\n"
"   - No stray markdown symbols outside proper usage\n\n"

"9. **Tables - Use ONLY When:**\n"
"   - User explicitly requests a comparison table\n"
"   - Comparing 3+ similar items side-by-side (e.g., React vs Vue vs Angular)\n"
"   - Showing complexity comparisons for multiple algorithms\n"
"   - Default to bullet points and headings for other cases\n\n"

"## SPECIAL MODES & FEATURES\n\n"

"10. **Mock Interview Mode:**\n"
"    When conducting mock interviews:\n"
"    1. Ask one question at a time and wait for user response\n"
"    2. After user answers, provide structured feedback:\n"
"       - **Strengths**: What was done well (2-3 points)\n"
"       - **Areas for Improvement**: Specific, actionable suggestions (2-3 points)\n"
"       - **Enhanced Answer**: Show improved version using proper format\n"
"    3. Be encouraging but honest and constructive\n"
"    4. Highlight specific phrases or techniques that worked well\n"
    "    5. Provide tips on delivery, pacing, and structure\n"
    "    6. Scoring (1–5 each): Clarity, Technical Depth, Structure, Confidence.\n"
    "       - End with a 2-sentence actionable improvement summary.\n\n"

    # 6) Language Awareness Layer
    "LANGUAGE AWARENESS:\n"
    "- If the prompt is partially or fully non-English, respond in English for interview context unless the user explicitly requests another language.\n\n"

    # 7) Conciseness Mode Safeguard
    "CONCISENESS SAFEGUARD:\n"
    "- If the user asks for 'short answer', 'summary only', or 'just bullets', output only the Complete Answer section and skip details.\n\n"

"11. **Uncertainty & Edge Case Handling:**\n"
"    - If uncertain about facts: 'I'm not certain, but based on common practices...'\n"
"    - If question is ambiguous: Ask clarifying questions before answering\n"
"    - If information might be outdated: Acknowledge and suggest verification\n"
"    - NEVER hallucinate facts, APIs, library functions, or framework features\n"
"    - If you don't know: 'I don't have reliable information on this specific detail'\n"
"    - Always align with widely accepted industry standards\n\n"

"## QUALITY ASSURANCE\n\n"

"12. **Pre-Response Checklist (Verify Before Sending):**\n"
"    □ Comprehensive summary is complete and interview-ready (4-8 sentences)\n"
"    □ Summary covers the ENTIRE topic, not just introduction\n"
"    □ Proper heading hierarchy (##, ###) throughout\n"
"    □ Code (if any) is executable, properly formatted, and complete\n"
"    □ Examples are relevant, clear, and practical\n"
"    □ No markdown syntax errors or stray symbols\n"
"    □ Professional, encouraging tone maintained\n"
"    □ Token limit respected (prioritize summary if approaching limit)\n"
"    □ Interview tips included where valuable\n"
"    □ Appropriate voice/perspective used based on question type\n\n"

"13. **Error Prevention - NEVER:**\n"
"    ✗ Hallucinate function names, APIs, libraries, or frameworks\n"
"    ✗ Use first person for technical strategies without user profile\n"
"    ✗ Create fictional specific work experiences\n"
"    ✗ Use tables by default (only when explicitly needed)\n"
"    ✗ Sacrifice summary quality for response length\n"
"    ✗ Leave code incomplete or without example usage\n"
"    ✗ Skip the comprehensive summary\n"
"    ✗ Provide answers that aren't interview-ready\n\n"

"14. **Error Prevention - ALWAYS:**\n"
"    ✓ Start with comprehensive summary (4-8 sentences)\n"
"    ✓ Structure with clear headings and bullet points\n"
"    ✓ Provide working, complete, well-commented code\n"
"    ✓ Include practical examples and use cases\n"
"    ✓ Maintain interview-ready quality throughout\n"
"    ✓ Be accurate and honest about limitations\n"
"    ✓ Use appropriate voice based on question type\n"
"    ✓ Provide actionable, practical advice\n\n"

"## CONSISTENCY & FLOW\n\n"

"15. **Standard Response Flow:**\n"
"    1. Analyze question type, complexity, and user intent\n"
"    2. Write comprehensive summary (4-8 sentences) covering complete topic\n"
"    3. Provide detailed explanation with proper structure\n"
"    4. Add code/examples/STAR details as needed\n"
"    5. Include practical tips and interview guidance\n"
"    6. Verify quality checklist before finalizing\n\n"

"16. **Consistency Across All Responses:**\n"
"    - ALL responses must have: comprehensive summary → detailed explanation → examples\n"
"    - Concepts: complete summary → features → benefits → examples → best practices\n"
"    - Code: complete summary → code → explanation → complexity → alternatives → tips\n"
"    - Behavioral: complete STAR summary → detailed breakdown → interview tips\n"
"    - System Design: complete summary → requirements → architecture → trade-offs → tips\n"
"    - The comprehensive summary is the PRIMARY answer; everything else enriches it\n\n"

"## TONE & COMMUNICATION\n\n"

"17. **Professional Communication Style:**\n"
"    - Professional, clear, concise, and supportive\n"
"    - Avoid unnecessary jargon; explain technical terms when used\n"
"    - Encourage confidence while maintaining accuracy\n"
"    - Make all responses interview-ready, as if coaching a candidate live\n"
"    - Use positive, constructive language in feedback\n"
"    - Provide specific, actionable advice\n\n"

"## SYSTEM INTEGRATION\n\n"

"18. **System Behavior Rules:**\n"
"    - Process and transcribe input ONLY when:\n"
"      * Microphone is ON, AND\n"
"      * Cursor is inside the search bar\n"
"    - Ignore speech input if cursor is not in search bar, even if mic is on\n"
"    - Maintain conversation context across multi-turn interactions\n"
"    - Remember user preferences and profile information shared during conversation\n\n"

"## FINAL REMINDERS\n\n"

"**Remember**: The comprehensive summary (3-4 sentences) is the candidate's primary interview answer. "
"It must be complete, thorough, and usable as a standalone response. Everything else in your response "
"supports, explains, and enriches that core answer. Never sacrifice summary quality.\n\n"

"**Closing Principle**: Every response must sound like a confident, well-prepared candidate in a top-tier interview. "
"Make users sound precise, structured, and authentic — never robotic or over-rehearsed.\n"
)


class LLMService:
	def __init__(self) -> None:
		self._client: Groq | None = None

	def _needs_comparison(self, question: str) -> bool:
		q = (question or "").lower()
		keywords = [
			"compare",
			"versus",
			"vs ",
			"difference between",
			"differences between",
		]
		return any(k in q for k in keywords)

	def _comparison_overrides(self, question: str) -> str:
		return (
			"\n\nComparison Format Overrides (apply only to comparison questions):\n"
			"- Produce ONE concise markdown table with headers: | Feature | A | B |.\n"
			"- Use clear, compact rows such as Definition, Core Function, Input, Output, Autonomy, Examples, Use Case Focus, Decision Making.\n"
			"- Keep cells short (1–2 lines).\n"
			"- After the table, add an 'In short:' section with 2 bullet points summarizing A vs B in one sentence each.\n"
			"- No extra headings, no duplicate sections, no verbose paragraphs.\n"
		)

	def _needs_first_person(self, question: str) -> bool:
		q = (question or "").lower()
		
		# If it's a technical strategy question, don't use first person
		if self._is_technical_strategy_question(question):
			return False
		
		# Look for personal/behavioral question indicators
		personal_indicators = [
			"yourself", "myself", "about you", "about me",
			"your background", "my background", "your experience", "my experience",
			"your skills", "my skills", "your strengths", "my strengths",
			"your weaknesses", "my weaknesses", "your projects", "my projects",
			"your career", "my career", "your goals", "my goals",
			"hire you", "interested in", "motivates you", "motivates me",
			"introduce", "tell me about", "describe yourself", "describe yourself"
		]
		
		# Look for direct personal references
		personal_references = [
			"you are", "you have", "you did", "you worked", "you developed",
			"you created", "you built", "you designed", "you implemented"
		]
		
		# Check for personal indicators or references
		has_personal_indicator = any(indicator in q for indicator in personal_indicators)
		has_personal_reference = any(reference in q for reference in personal_references)
		
		return has_personal_indicator or has_personal_reference

	def _is_technical_strategy_question(self, question: str) -> bool:
		"""Check if this is a technical strategy question that should provide general approaches"""
		q = (question or "").lower()
		
		# Look for strategy/approach indicators
		strategy_indicators = [
			"optimize", "improve", "reduce", "increase", "solve", "handle", 
			"implement", "approach", "strategy", "method", "technique",
			"performance", "efficiency", "scalability", "reliability"
		]
		
		# Look for question patterns that suggest strategy/approach
		question_patterns = [
			"how", "what", "which", "describe", "explain"
		]
		
		# Check if it's asking for a method/approach rather than personal experience
		has_strategy_indicator = any(indicator in q for indicator in strategy_indicators)
		has_question_pattern = any(pattern in q for pattern in question_patterns)
		
		# Look for specific personal experience indicators that would override strategy mode
		personal_indicators = [
			"tell me about yourself", "your experience", "your background",
			"your skills", "your strengths", "your weaknesses", "your projects",
			"why should we hire you", "what motivates you", "introduce yourself"
		]
		
		has_personal_indicator = any(indicator in q for indicator in personal_indicators)
		
		# If it has strategy indicators and question patterns, but NOT personal indicators, it's a strategy question
		return has_strategy_indicator and has_question_pattern and not has_personal_indicator

	def _technical_strategy_overrides(self) -> str:
		return (
			"\n\nTechnical Strategy Overrides (apply only to technical strategy questions):\n"
			"- Provide GENERAL strategies and approaches that any candidate can adapt to their experience\n"
			"- Use 'you can', 'one approach is', 'a common strategy' instead of specific first-person experiences\n"
			"- Focus on universal optimization techniques, best practices, and methodologies\n"
			"- Avoid creating fictional specific experiences, technologies, or company details\n"
			"- Structure as: general approach → key techniques → implementation considerations → expected outcomes\n"
			"- Make it applicable to various domains and technologies\n"
		)

	def _persona_overrides(self) -> str:
		return (
			"\n\nInterview Persona Overrides (apply only to first-person questions):\n"
			"- Answer strictly in first person as the candidate (use 'I', 'my').\n"
			"- Use the provided Candidate Profile Context as the factual source.\n"
			"- Keep tone conversational and professional, as in a live interview.\n"
			"- Prefer a 45–60 second spoken-length response (concise, cohesive).\n"
			"- Do NOT include contact links, headers, tables, or bullet lists unless requested.\n"
			"- Focus on role-aligned highlights: current role, key strengths, relevant projects, impact.\n"
		)

	def _ensure_client(self):
		provider = (settings.llm_provider or "groq").lower()
		if provider == "groq":
			api_key = settings.groq_api_key
			if not api_key:
				self._client = None
				return None
			if self._client is None or not isinstance(self._client, Groq):
				self._client = Groq(api_key=api_key)
			return self._client
		elif provider == "gemini":
			if genai is None:
				return None
			api_key = settings.gemini_api_key
			if not api_key:
				return None
			# For gemini we return a configured module handle to keep usage simple
			genai.configure(api_key=api_key)
			return genai
		else:
			return None

	@property
	def enabled(self) -> bool:
		provider = (settings.llm_provider or "groq").lower()
		if provider == "groq":
			return bool(settings.groq_api_key)
		if provider == "gemini":
			return bool(settings.gemini_api_key)
		return False

	def _format_response(self, text: str) -> str:
		"""Return clean markdown for frontend rendering.

		Rules now:
		- Never force-create tables unless the content already looks like a valid pipe table
		  (has header and at least one data row), or the model clearly emitted a table.
		- Keep code blocks untouched.
		- Keep normal text with headings/bullets as-is.
		- Ensure summary sections are properly formatted for interview scenarios.
		"""
		import re
		
		# Clean up the text first
		text = text.strip()
		
		# Ensure summary sections are properly formatted (remove bullet conversion logic)
		text = self._format_summary_sections(text)
		# Enforce unlabeled bullets inside Complete Answer
		text = self._strip_labeled_bullets_in_complete_answer(text)
		
		# First, check if this is code content that should not be formatted as tables
		if self._is_code_content(text):
			# For code content, just clean up basic formatting issues
			return self._clean_code_formatting(text)
		
		# Check if this is explanation content that should use text formatting, not tables
		if self._is_explanation_content(text):
			# For explanation content, convert table-like markdown artifacts conservatively
			cleaned = self._clean_explanation_formatting(text)
			# Soften excessive bold markers outside code to reduce visual noise
			import re as _re
			cleaned = _re.sub(r"\*\*([^\n*][^*]{0,200}?)\*\*", r"\1", cleaned)
			return cleaned
		
		# Only touch pipe tables; do not try to infer tables from text
		text = self._clean_table_markdown_artifacts(text)
		if self._looks_like_pipe_table(text):
			text = self._format_tables(text)
		
		# Normalize leading bold labels across all bullets/paragraph list items
		text = self._strip_leading_bold_labels_globally(text)
		# Remove all remaining bold emphasis outside code blocks (keep headings intact)
		text = self._strip_all_bold_outside_code(text)
		
		return text

	def _strip_labeled_bullets_in_complete_answer(self, text: str) -> str:
		"""Within the '## Complete Answer' section, remove leading label patterns like '**Label:** ' or 'Label:' from each bullet.
		This keeps bullets as direct statements without side headings.
		"""
		import re
		lines = text.split('\n')
		out: list[str] = []
		in_complete = False
		for i, line in enumerate(lines):
			header = line.strip().lower()
			if header.startswith('## ') and 'complete answer' in header:
				in_complete = True
				out.append(line)
				continue
			# Exit when next top-level header begins
			if in_complete and line.strip().startswith('## ') and 'complete answer' not in header:
				in_complete = False
				out.append(line)
				continue
			if in_complete and line.lstrip().startswith(('-', '*')):
				bullet = line
				# Remove patterns like '- **Label:** text' or '- Label: text'
				bullet = re.sub(r"^([\-\*]\s+)(\*\*[^*:]{1,40}\*\*:\s*)", r"\1", bullet)
				bullet = re.sub(r"^([\-\*]\s+)([^*:]{1,40}:\s*)", r"\1", bullet)
				out.append(bullet)
			else:
				out.append(line)
		return '\n'.join(out)

	def _strip_leading_bold_labels_globally(self, text: str) -> str:
		"""Remove leading bold label patterns at the start of list items anywhere in the document.
		Patterns handled:
		- '- **Label:** rest' -> '- rest'
		- '- **Label**: rest' -> '- rest'
		- '- **Phrase** rest' (short phrase up to ~6 words) -> '- Phrase rest' (drop bold only)
		Does not touch code blocks.
		"""
		import re
		lines = text.split('\n')
		out: list[str] = []
		in_code = False
		for line in lines:
			if line.strip().startswith('```'):
				in_code = not in_code
				out.append(line)
				continue
			if in_code:
				out.append(line)
				continue
			m1 = re.match(r'^(\s*[\-\*]\s+)\*\*([^*]{1,80})\*\*\s*:\s*(.*)$', line)
			if m1:
				prefix, label, rest = m1.groups()
				out.append(f"{prefix}{rest}".rstrip())
				continue
			m2 = re.match(r'^(\s*[\-\*]\s+)\*\*([^*]{1,80})\*\*\s+(.*)$', line)
			if m2:
				prefix, label, rest = m2.groups()
				# Keep label plain, drop bold
				out.append(f"{prefix}{label} {rest}".rstrip())
				continue
			out.append(line)
		return '\n'.join(out)

	def _strip_all_bold_outside_code(self, text: str) -> str:
		"""Remove all bold (**...**) outside code blocks to keep regular typography for headings, bullets, and inline keywords.
		Headings are not bolded markdown and remain unaffected. Code blocks are preserved.
		"""
		import re
		lines = text.split('\n')
		out: list[str] = []
		in_code = False
		for line in lines:
			if line.strip().startswith('```'):
				in_code = not in_code
				out.append(line)
				continue
			if in_code:
				out.append(line)
				continue
			# Replace any **text** with plain text
			cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
			out.append(cleaned)
		return '\n'.join(out)
	
	def _format_summary_sections(self, text: str) -> str:
		"""Format comprehensive summary sections for interview scenarios - ensure they are prominent and complete"""
		import re
		
		# Look for comprehensive summary sections and ensure they're properly formatted
		summary_patterns = [
			r'##\s*(Complete\s+Answer|Summary|Overview|Comprehensive\s+Answer)',
			r'###\s*(Complete\s+Answer|Summary|Overview|Comprehensive\s+Answer)',
			r'#\s*(Complete\s+Answer|Summary|Overview|Comprehensive\s+Answer)',
			r'##\s*(Quick\s+Answer|Quick\s+Summary)',  # Keep backward compatibility
			r'###\s*(Quick\s+Answer|Quick\s+Summary)',
			r'#\s*(Quick\s+Answer|Quick\s+Summary)'
		]
		
		lines = text.split('\n')
		formatted_lines = []
		i = 0
		
		while i < len(lines):
			line = lines[i]
			
			# Check if this line is a summary header
			is_summary_header = False
			for pattern in summary_patterns:
				if re.search(pattern, line, re.IGNORECASE):
					is_summary_header = True
					break
			
			if is_summary_header:
				# Ensure it's a proper ## header for summary
				if not line.startswith('##'):
					line = re.sub(r'^#+\s*', '## ', line)
				formatted_lines.append(line)
				
				# Look for the content after the header
				j = i + 1
				summary_content = []
				while j < len(lines) and not lines[j].strip().startswith('#'):
					if lines[j].strip():  # Only add non-empty lines
						summary_content.append(lines[j].strip())
					j += 1
				
				# Keep the model's own summary format; do not auto-convert
				if summary_content:
					for line_part in summary_content:
						formatted_lines.append(line_part)
					formatted_lines.append('')
				
				i = j
			else:
				formatted_lines.append(line)
				i += 1
		
		return '\n'.join(formatted_lines)
	
	def _clean_table_markdown_artifacts(self, text: str) -> str:
		"""Clean up markdown artifacts specifically in table content"""
		import re
		
		lines = text.split('\n')
		cleaned_lines = []
		
		for line in lines:
			# Check if this line is part of a table
			if self._is_table_line(line):
				# Clean up markdown artifacts in table lines
				cleaned_line = self._clean_table_line(line)
				cleaned_lines.append(cleaned_line)
			else:
				cleaned_lines.append(line)
		
		return '\n'.join(cleaned_lines)
	
	def _is_table_line(self, line: str) -> bool:
		"""Check if a line is part of a table"""
		import re
		
		# Check for pipe-separated table lines
		if '|' in line and line.count('|') >= 2:
			return True
		
		# Check for table separator lines
		if re.match(r'^\s*\|[\s\-:]+\|', line):
			return True
		
		return False
	
	def _clean_table_line(self, line: str) -> str:
		"""Clean up markdown artifacts in a single table line"""
		import re
		
		# Remove all markdown bold formatting (**text**)
		line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
		
		# Remove all markdown italic formatting (*text*)
		line = re.sub(r'\*([^*]+)\*', r'\1', line)
		
		# Remove any remaining single asterisks
		line = re.sub(r'\*', '', line)
		
		# Clean up extra spaces around pipes
		line = re.sub(r'\s*\|\s*', '|', line)
		
		# Ensure proper spacing around pipes for readability
		line = re.sub(r'\|', ' | ', line)
		
		# Remove leading/trailing spaces
		line = line.strip()
		
		return line
	
	def _format_tables(self, text: str) -> str:
		"""Format tabular data into proper markdown tables"""
		import re
		
		lines = text.split('\n')
		formatted_lines = []
		i = 0
		
		while i < len(lines):
			line = lines[i].strip()
			
			# Check if this line looks like a table row
			if self._is_table_row(line):
				# Find the end of the table
				table_lines = [line]
				j = i + 1
				
				# Collect consecutive table rows
				while j < len(lines) and self._is_table_row(lines[j].strip()):
					table_lines.append(lines[j].strip())
					j += 1
				
				# Format the table
				formatted_table = self._create_markdown_table(table_lines)
				formatted_lines.append(formatted_table)
				i = j
			else:
				formatted_lines.append(lines[i])
				i += 1
		
		return '\n'.join(formatted_lines)
	
	def _is_table_row(self, line: str) -> bool:
		"""Check if a line looks like a table row"""
		import re
		
		# Check for pipe-separated values
		if '|' in line and line.count('|') >= 2:
			return True
		
		return False

	def _looks_like_pipe_table(self, text: str) -> bool:
		"""Detect if text contains a valid pipe table with header and at least one row."""
		import re
		lines = [l.strip() for l in text.split('\n')]
		for i in range(len(lines) - 2):
			if '|' in lines[i] and '|' in lines[i+1]:
				# header and separator or another row
				if re.match(r'^\|?\s*[^|]+\s*(\|[^|]+)+\|?$', lines[i]) and ('---' in lines[i+1] or '|' in lines[i+1]):
					return True
		return False
	
	def _create_markdown_table(self, table_lines: list[str]) -> str:
		"""Convert table lines to markdown table format with clean text"""
		import re
		
		if not table_lines:
			return ""
		
		# Try to detect the separator type
		first_line = table_lines[0]
		
		# Handle pipe-separated tables
		if '|' in first_line:
			# Clean up existing pipe formatting and remove markdown artifacts
			cleaned_lines = []
			for line in table_lines:
				# Remove markdown formatting first
				cleaned = self._clean_table_line(line)
				# Remove extra spaces around pipes
				cleaned = re.sub(r'\s*\|\s*', '|', cleaned.strip())
				# Remove leading/trailing pipes if they exist
				cleaned = cleaned.strip('|')
				cleaned_lines.append(cleaned)
			
			# Create markdown table
			if cleaned_lines:
				# Header row
				header = cleaned_lines[0]
				# Determine number of columns
				columns = header.count('|') + 1
				separator = '|' + '|'.join(['---'] * columns) + '|'
				
				# Format header with proper spacing
				formatted_header = '|' + header + '|'
				
				# Format data rows
				data_rows = []
				for line in cleaned_lines[1:]:
					formatted_row = '|' + line + '|'
					data_rows.append(formatted_row)
				
				# Combine into markdown table
				table_parts = [formatted_header, separator] + data_rows
				return '\n'.join(table_parts)
		
		# Handle space-separated tables
		else:
			# Parse space-separated data
			rows = []
			for line in table_lines:
				# Clean up markdown artifacts first
				cleaned_line = self._clean_table_line(line)
				# Split by multiple spaces
				columns = re.split(r'\s{2,}', cleaned_line.strip())
				if len(columns) >= 2:
					rows.append(columns)
			
			if rows:
				# Determine max columns
				max_cols = max(len(row) for row in rows)
				
				# Pad rows to same length
				padded_rows = []
				for row in rows:
					padded_row = row + [''] * (max_cols - len(row))
					padded_rows.append(padded_row)
				
				# Create markdown table
				formatted_rows = []
				for i, row in enumerate(padded_rows):
					formatted_row = '|' + '|'.join(row) + '|'
					formatted_rows.append(formatted_row)
					
					# Add separator after header
					if i == 0:
						separator = '|' + '|'.join(['---'] * max_cols) + '|'
						formatted_rows.append(separator)
				
				return '\n'.join(formatted_rows)
		
		# If we can't format it, return original
		return '\n'.join(table_lines)

	def _is_code_content(self, text: str) -> bool:
		"""Check if the text contains code that should not be formatted as tables"""
		import re
		
		# Check for code block markers
		if '```' in text:
			return True
		
		# Check for Python-specific patterns
		python_patterns = [
			r'def\s+\w+\s*\(',  # Function definitions
			r'class\s+\w+',  # Class definitions
			r'import\s+\w+',  # Import statements
			r'from\s+\w+\s+import',  # From imports
			r'if\s+__name__\s*==\s*["\']__main__["\']',  # Main guard
			r'return\s+',  # Return statements
			r'while\s+',  # While loops
			r'for\s+\w+\s+in\s+',  # For loops
			r'#\s*[A-Z]',  # Comments starting with capital letters
		]
		
		# Check if any Python patterns are found
		for pattern in python_patterns:
			if re.search(pattern, text, re.MULTILINE):
				return True
		
		# Check for indented code blocks (4+ spaces at start of line)
		lines = text.split('\n')
		indented_lines = 0
		for line in lines:
			if line.strip() and line.startswith('    '):
				indented_lines += 1
		
		# If more than 30% of non-empty lines are indented, it's likely code
		non_empty_lines = [line for line in lines if line.strip()]
		if non_empty_lines and indented_lines / len(non_empty_lines) > 0.3:
			return True
		
		return False

	def _is_explanation_content(self, text: str) -> bool:
		"""Check if the text is explanation content that should use text formatting, not tables"""
		import re
		
		# Check for explanation patterns that should NOT be tables
		explanation_patterns = [
			r'Time\s*Complexity',  # Time complexity analysis
			r'Space\s*Complexity',  # Space complexity analysis
			r'How\s+it\s+works',  # How it works explanations
			r'Key\s+Features',  # Feature descriptions
			r'Time:\s*O\(',  # Time complexity notation
			r'Space:\s*O\(',  # Space complexity notation
			r'Input\s+type:',  # Input descriptions
			r'Output:',  # Output descriptions
			r'Error\s+handling:',  # Error handling descriptions
		]
		
		# Check if any explanation patterns are found
		for pattern in explanation_patterns:
			if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
				return True
		
		# Check if text contains table-like formatting but is actually explanation
		lines = text.split('\n')
		table_like_lines = 0
		for line in lines:
			# Check for lines that look like table rows but are explanations
			if '|' in line and any(keyword in line.lower() for keyword in ['time', 'space', 'complexity', 'feature', 'input', 'output']):
				table_like_lines += 1
		
		# If we have table-like formatting but it's explanation content, treat as text
		if table_like_lines > 0:
			return True
		
		return False

	def _clean_code_formatting(self, text: str) -> str:
		"""Clean up code formatting issues without converting to tables"""
		import re
		
		# Fix common indentation issues
		lines = text.split('\n')
		cleaned_lines = []
		
		for line in lines:
			# Fix lines that look like they were formatted as table rows
			# Pattern: |variable = value|# comment|
			if '|' in line and '=' in line:
				# Remove table formatting and fix indentation
				line = re.sub(r'^\s*\|\s*', '', line)  # Remove leading | and spaces
				line = re.sub(r'\s*\|\s*$', '', line)  # Remove trailing | and spaces
				line = re.sub(r'\s*\|\s*', ' ', line)  # Replace middle | with spaces
				
				# Fix indentation for code lines
				if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
					# This looks like a code line that needs indentation
					if any(keyword in line for keyword in ['def ', 'class ', 'if ', 'while ', 'for ', 'else:', 'elif ']):
						# This is a top-level statement, no indentation needed
						pass
					elif line.strip().startswith(('return', 'yield', 'break', 'continue', 'pass')):
						# This should be indented
						line = '    ' + line.strip()
					elif '=' in line and not line.strip().startswith('#'):
						# This looks like a variable assignment that should be indented
						line = '    ' + line.strip()
			
			# Fix comment formatting
			if '|' in line and '#' in line:
				# Convert table-formatted comments to proper comments
				line = re.sub(r'^\s*\|\s*', '', line)
				line = re.sub(r'\s*\|\s*$', '', line)
				line = re.sub(r'\s*\|\s*', ' ', line)
			
			cleaned_lines.append(line)
		
		return '\n'.join(cleaned_lines)

	def _clean_explanation_formatting(self, text: str) -> str:
		"""Clean up explanation formatting by converting table-like formatting to proper text"""
		import re
		
		lines = text.split('\n')
		cleaned_lines = []
		
		for line in lines:
			# Check if this line looks like a table row but is actually explanation
			if '|' in line and any(keyword in line.lower() for keyword in ['time', 'space', 'complexity', 'feature', 'input', 'output', 'error']):
				# Convert table-formatted explanations to proper text
				# Pattern: |METRIC|VALUE| -> **METRIC:** VALUE
				# Pattern: |Time|O(n²)| -> **Time Complexity:** O(n²)
				
				# Remove leading/trailing pipes and split by middle pipes
				line = line.strip('|')
				parts = [part.strip() for part in line.split('|') if part.strip()]
				
				if len(parts) >= 2:
					# Convert to proper text format
					metric = parts[0]
					value = parts[1]
					
					# Handle specific cases
					if 'time' in metric.lower():
						formatted_line = f"**Time Complexity:** {value}"
					elif 'space' in metric.lower():
						formatted_line = f"**Space Complexity:** {value}"
					elif 'feature' in metric.lower():
						formatted_line = f"**Key Features:** {value}"
					elif 'input' in metric.lower():
						formatted_line = f"**Input:** {value}"
					elif 'output' in metric.lower():
						formatted_line = f"**Output:** {value}"
					elif 'error' in metric.lower():
						formatted_line = f"**Error Handling:** {value}"
					else:
						formatted_line = f"**{metric}:** {value}"
					
					cleaned_lines.append(formatted_line)
				else:
					cleaned_lines.append(line)
			else:
				# Check for table separator lines and skip them
				if re.match(r'^\s*\|[\s\-:]+\|', line):
					continue
				cleaned_lines.append(line)
		
		return '\n'.join(cleaned_lines)

	def _estimate_response_complexity(self, question: str) -> int:
		"""Estimate response complexity and suggest token limit"""
		question_lower = question.lower()
		
		# Simple questions - shorter responses
		simple_indicators = ['what is', 'define', 'explain briefly', 'simple', 'basic']
		if any(indicator in question_lower for indicator in simple_indicators):
			return settings.groq_max_tokens_simple
		
		# Code questions - medium responses
		code_indicators = ['code', 'implement', 'write', 'function', 'class', 'algorithm']
		if any(indicator in question_lower for indicator in code_indicators):
			return settings.groq_max_tokens_code
		
		# Complex topics - longer responses
		complex_indicators = ['architecture', 'design', 'system', 'compare', 'advantages', 'disadvantages', 'best practices']
		if any(indicator in question_lower for indicator in complex_indicators):
			return settings.groq_max_tokens_complex
		
		# Default medium response (average of simple and code)
		return (settings.groq_max_tokens_simple + settings.groq_max_tokens_code) // 2

	def _get_optimal_token_limit(self, question: str, base_limit: int) -> int:
		"""Get optimal token limit based on question complexity and base config"""
		if base_limit:
			return base_limit
		
		estimated = self._estimate_response_complexity(question)
		# Cap at complex token limit to prevent excessive token usage
		return min(estimated, settings.groq_max_tokens_complex * 2)

	def _style_overrides(self, style_mode: Optional[str], tone: Optional[str], layout: Optional[str], variability: Optional[float], seed: Optional[int]) -> str:
		"""Construct style and tone overrides for varied, professional outputs."""
		import random
		rng = random.Random(seed)
		v = 0.0 if variability is None else max(0.0, min(1.0, variability))

		# Presets
		presets: dict[str, str] = {
			"concise": "Keep it tight. 4–6 bullets max. Avoid subheadings unless necessary.",
			"deep-dive": "Provide rich sections with 'Why it matters', 'Trade-offs', and a short example.",
			"mentor": "Use a coaching voice. Add 'Pitfalls' and 'What to practice' sections when helpful.",
			"executive": "Lead with outcomes and business impact. Use short paragraphs and a 'Bottom line' section.",
			"faq": "Answer as an FAQ: 4–6 Q→A pairs covering the topic succinctly.",
			"qa": "Use a Q→A dialogue style for key points, then a brief summary.",
			"checklist": "Present an actionable checklist with clear steps and acceptance criteria.",
			"narrative": "Explain as a narrative walkthrough with sections 'Context → Decision → Result'.",
			"varied": "Choose any of: concise, deep-dive, mentor, executive, faq, qa, checklist, narrative based on question type.",
		}

		chosen_mode = (style_mode or "auto").lower()
		if chosen_mode in ("auto", "varied"):
			# Soft randomization by question type
			candidates = ["concise", "deep-dive", "mentor", "executive", "faq", "qa", "checklist", "narrative"]
			if v > 0:
				chosen_mode = rng.choice(candidates)
			else:
				chosen_mode = "executive"  # sensible default

		tone_map: dict[str, str] = {
			"neutral": "Neutral, precise, professional.",
			"friendly": "Warm, approachable, but still professional.",
			"mentor": "Supportive, coaching tone with practical tips.",
			"executive": "Crisp, outcome-focused, confident.",
			"academic": "Formal, rigorous definitions and citations where appropriate.",
			"coaching": "Encouraging, step-by-step guidance.",
		}
		tone_rule = tone_map.get((tone or "").lower(), "Neutral, precise, professional.")

		layout_map: dict[str, str] = {
			"bullets": "Prefer bullets with minimal headings.",
			"narrative": "Short paragraphs, minimal headings.",
			"qa": "Q→A pairs.",
			"faq": "FAQ format.",
			"checklist": "Checklist of steps.",
			"pros-cons": "Pros/Cons section included.",
			"steps": "Numbered steps first, details later.",
		}
		layout_rule = layout_map.get((layout or "").lower(), "Use judgement for best readability.")

		preset_rule = presets.get(chosen_mode, "")

		return (
			"\n\nStyle & Tone Overrides:"
			f"\n- Tone: {tone_rule}"
			f"\n- Layout preference: {layout_rule}"
			f"\n- Style preset: {chosen_mode} — {preset_rule}"
			"\n- Vary headings and bullet density to avoid repetitive structure; choose the lightest structure that conveys clarity."
			"\n- Do not force the earlier template sections if brevity or narrative works better for this question."
		)

	async def generate_answer(self, question: str, system_prompt: Optional[str] = None, profile_text: Optional[str] = None, previous_qna: Optional[List[Dict[str, str]]] = None, *, style_mode: Optional[str] = None, tone: Optional[str] = None, layout: Optional[str] = None, variability: Optional[float] = None, seed: Optional[int] = None) -> str:
		client = self._ensure_client()
		if client is None:
			return question  # mock: echo when no key

		prompt = system_prompt or CODE_FORWARD_PROMPT


		# Inject profile context and persona/comparison overrides if applicable
		if profile_text:
			prompt = (
				prompt
				+ "\n\n" 
				+ "Candidate Profile Context (authoritative for resume/personal questions):\n" 
				+ profile_text.strip()
			)
			if self._needs_first_person(question):
				prompt = prompt + self._persona_overrides()

		# If the user is asking to compare, add comparison formatting rules
		if self._needs_comparison(question):
			prompt = prompt + self._comparison_overrides(question)
		
		# If this is a technical strategy question, add strategy overrides
		if self._is_technical_strategy_question(question):
			prompt = prompt + self._technical_strategy_overrides()

		# Style & tone overrides for variety
		prompt = prompt + self._style_overrides(style_mode, tone, layout, variability, seed)

		import anyio

		provider = (settings.llm_provider or "groq").lower()
		model = settings.groq_model if provider == "groq" else settings.gemini_model
		temperature = settings.answer_temperature
		top_p = settings.groq_top_p
		max_tokens = self._get_optimal_token_limit(question, settings.groq_max_tokens)
		stream = settings.groq_stream

		def build_kwargs(stream_flag: bool):
			# Build message list with optional recent history for contextual follow-ups
			messages: List[Dict[str, str]] = [
				{"role": "system", "content": prompt}
			]
			if previous_qna:
				# Include recent turns to provide context
				_source_history = previous_qna
				# To TRIM history to the last 2 turns only (recommended for lower latency/token use),
				# uncomment the next line and keep the 5-turn cap below as-is.
				# _source_history = _source_history[-2:]
				# Cap to last 5 turns by default
				for item in _source_history[-5:]:
					q = (item.get("question") or "").strip()
					a = (item.get("answer") or "").strip()
					if q:
						messages.append({"role": "user", "content": q})
					if a:
						messages.append({"role": "assistant", "content": a})
			# Current question last
			messages.append({"role": "user", "content": question})
			kwargs = {
				"model": model,
				"messages": messages,
				"temperature": temperature,
				"max_tokens": max_tokens,
			}
			if top_p is not None:
				kwargs["top_p"] = top_p
			if stream_flag:
				kwargs["stream"] = True
			return kwargs

		def _call() -> str:
			if provider == "groq":
				if stream:
					stream_resp = client.chat.completions.create(**build_kwargs(True))
					parts: list[str] = []
					for chunk in stream_resp:
						parts.append(getattr(chunk.choices[0].delta, "content", None) or "")
					raw_text = "".join(parts).strip()
					return self._format_response(raw_text)
				else:
					resp = client.chat.completions.create(**build_kwargs(False))
					raw_text = resp.choices[0].message.content.strip()
					return self._format_response(raw_text)
			elif provider == "gemini":
				# Gemini: use the GenerativeModel with non-streaming first
				model_id = settings.gemini_model
				gmodel = client.GenerativeModel(model_id)
				messages = [
					{"role": "system", "content": prompt},
					{"role": "user", "content": question},
				]
				# Join to a single prompt: system + user, keeping system first
				full_prompt = (prompt + "\n\nUser: " + question).strip()
				resp = gmodel.generate_content(full_prompt)
				raw_text = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
				return self._format_response((raw_text or "").strip())
			else:
				return question

		return await anyio.to_thread.run_sync(_call)

	async def stream_answer(self, question: str, system_prompt: Optional[str] = None, profile_text: Optional[str] = None, previous_qna: Optional[List[Dict[str, str]]] = None, *, style_mode: Optional[str] = None, tone: Optional[str] = None, layout: Optional[str] = None, variability: Optional[float] = None, seed: Optional[int] = None) -> AsyncIterator[str]:
		client = self._ensure_client()
		provider = (settings.llm_provider or "groq").lower()
		prompt = system_prompt or CODE_FORWARD_PROMPT
		if profile_text:
			prompt = (
				prompt
				+ "\n\n" 
				+ "Candidate Profile Context (authoritative for resume/personal questions):\n" 
				+ profile_text.strip()
			)
			if self._needs_first_person(question):
				prompt = prompt + self._persona_overrides()
		# Comparison overrides for streaming as well
		if self._needs_comparison(question):
			prompt = prompt + self._comparison_overrides(question)
		
		# Technical strategy overrides for streaming as well
		if self._is_technical_strategy_question(question):
			prompt = prompt + self._technical_strategy_overrides()

		# Style & tone overrides
		prompt = prompt + self._style_overrides(style_mode, tone, layout, variability, seed)

		if client is None:
			yield ""
			return

		# Use dynamic token limit for streaming as well
		max_tokens = self._get_optimal_token_limit(question, settings.groq_max_tokens)
		# For streaming, use a bit less to ensure faster response
		max_tokens = min(max_tokens, settings.groq_max_tokens_code)

		def _call_stream():
			messages: List[Dict[str, str]] = [
				{"role": "system", "content": prompt}
			]
			if previous_qna:
				_source_history = previous_qna
				# To TRIM history to the last 2 turns only (recommended), uncomment:
				# _source_history = _source_history[-2:]
				for item in _source_history[-5:]:
					q = (item.get("question") or "").strip()
					a = (item.get("answer") or "").strip()
					if q:
						messages.append({"role": "user", "content": q})
					if a:
						messages.append({"role": "assistant", "content": a})
			messages.append({"role": "user", "content": question})
			if provider == "groq":
				return client.chat.completions.create(
					model=settings.groq_model,
					messages=messages,
					temperature=settings.answer_temperature,
					max_tokens=max_tokens,
					stream=True,
				)
			elif provider == "gemini":
				# Fallback to non-streaming with gemini for now
				return None
			else:
				return None

		import anyio
		stream = await anyio.to_thread.run_sync(_call_stream)
		if provider == "groq" and stream is not None:
			for chunk in stream:
				piece = getattr(chunk.choices[0].delta, "content", None) or ""
				if piece:
					yield piece
		elif provider == "gemini":
			# Non-streaming fallback: yield once
			def _one_shot():
				gmodel = client.GenerativeModel(settings.gemini_model)
				full_prompt = (prompt + "\n\nUser: " + question).strip()
				resp = gmodel.generate_content(full_prompt)
				return getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
			import anyio as _anyio
			text_once = await _anyio.to_thread.run_sync(_one_shot)
			for piece in (text_once or ""):
				yield piece


llm_service = LLMService()
