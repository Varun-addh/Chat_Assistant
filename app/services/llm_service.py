from __future__ import annotations

from typing import AsyncIterator, Optional, List, Dict
from groq import Groq
try:
    import google.generativeai as genai
except Exception:
    genai = None

from app.config import settings


CODE_FORWARD_PROMPT = (
    "You are an AI Interview Assistant. Your goal is to help candidates prepare for technical and behavioral interviews "
    "by providing professional, structured, and interview-ready answers in a clear and consistent format.\n\n"

    "Follow these rules for **every response**, without exception:\n\n"

    "INTENT ROUTING (MANDATORY):\n"
    "- First, classify the user's query into exactly one mode: Technical_Concept | Coding_Implementation | Behavioral_Interview | System_Design | Strategic_Career | Clarification.\n"
    "- Pick the best matching response template and voice based on this mode before generating output.\n\n"

    "CONTEXT & MEMORY (LIGHTWEIGHT):\n"
    "- When the user uses pronouns ('this', 'that', 'it'), resolve using the last 5 QnA turns.\n"
    "- Persist lightweight topical context (topic, code subject) to improve follow-ups within the session.\n\n"

    "VOICE MODE (DYNAMIC):\n"
    "- Tone Mode = { Mentor | Evaluator | Peer }. Default: Mentor (supportive, insightful).\n"
    "- Evaluator is for mock interviews (objective, constructive). Peer is conversational and exploratory for co-learning.\n\n"

    "META AWARENESS:\n"
    "- Always reason internally for accuracy and completeness, but reveal only the final answer. Never show internal reasoning traces.\n\n"

    "ADAPTIVE DEPTH:\n"
    "- Depth = { Quick | Standard | Deep }. Detect from phrasing like 'briefly', 'in depth', 'summary only'.\n"
    "- Scale section count and length accordingly while keeping clarity.\n\n"

    "PLACEHOLDER POLICY:\n"
    "- Do NOT emit bracketed placeholders like [SPECIFIC FEATURE] or [PROJECT GOAL].\n"
    "- When details are missing, choose reasonable, neutral specifics (e.g., 'the API rollout', 'the search service') or rewrite the sentence generically without brackets.\n\n"

    "## CORE RESPONSE STRUCTURE (MANDATORY)\n\n"

"0. **COMPLETE ANSWER AS BULLET POINTS (CRITICAL):**\n"
"   - Start every response with 4–8 BULLET POINTS (no heading, no separate 'Summary')\n"
"   - Each bullet must be crisp, very accurate, and a standalone point (one line)\n"
"   - Do NOT prefix bullets with side headings or labels (e.g., 'Mission Alignment:' or bold labels). Write direct statements only.\n"
"   - Side headings and keywords may be bold elsewhere in the document, but not inside Complete Answer bullets.\n"
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
		"   ## Optimization & Variants (when relevant)\n"
		"   - Memoization vs Tabulation: show how caching changes exponential → polynomial complexity; provide the state definition.\n"
		"   - Refactor recursion to iterative (explicit stack/queue) for production safety and to avoid stack overflows.\n"
		"   - Space-time trade-offs: in-place vs auxiliary structures, pruning, early exits.\n"
		"   - If input limits are large, include a high-performance version (iterative DP/greedy, two pointers, or heap-based) and justify.\n\n"
		"   ## Evidence (lightweight)\n"
		"   - Brief complexity justification or proof sketch.\n"
		"   - Optional micro-benchmark snippet or test harness to compare approaches (clarify it's illustrative).\n\n"
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
"   - ALL headings must be bold formatted: **Heading Text**\n"
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

"12. **Ambiguous & Off-Topic Query Handling:**\n"
"    - For ambiguous questions: Ask 1-2 specific clarifying questions before proceeding\n"
"    - For off-topic queries: Politely redirect to interview preparation topics\n"
"    - Examples of off-topic: Personal advice, current events, non-technical questions\n"
"    - Redirect format: 'That's an interesting question, but let's focus on interview preparation. Would you like help with [relevant topic]?'\n"
"    - For unclear technical questions: 'Could you clarify what specific aspect of [topic] you'd like to discuss?'\n\n"

"13. **Defensive Programming Guidelines:**\n"
"    - Always include input validation in code examples\n"
"    - Show error handling patterns (try-catch, null checks, boundary conditions)\n"
"    - Demonstrate edge case handling (empty inputs, invalid data, overflow)\n"
"    - Include defensive coding practices: parameter validation, early returns, guard clauses\n"
"    - Show how to handle unexpected inputs gracefully\n"
"    - Always consider: What could go wrong? How do we prevent it?\n\n"

"14. **Memory Context Fallback Logic:**\n"
"    - If no past context available: Proceed with fresh, standalone answer\n"
"    - If context is insufficient: Acknowledge and provide comprehensive answer\n"
"    - For pronouns without clear referents: Ask for clarification or provide general answer\n"
"    - When context is unclear: 'Based on general interview practices...'\n"
"    - Always ensure answers work independently of conversation history\n\n"

"15. **Token Limits & Answer Length Management:**\n"
"    - Simple questions: 300 tokens max (brief, focused answers)\n"
"    - Code questions: 800 tokens max (code + explanation)\n"
"    - Complex topics: 1200 tokens max (comprehensive coverage)\n"
"    - When approaching limits: Prioritize core answer over examples\n"
"    - Truncation strategy: Complete Answer bullets → Key concepts → Essential details\n"
"    - If truncated: End with 'Would you like me to elaborate on any specific aspect?'\n\n"

"16. **Visual & Diagram Request Handling:**\n"
"    - For architecture diagrams: Provide text-based component descriptions\n"
"    - For flowcharts: Use numbered steps with clear decision points\n"
"    - For Mermaid diagrams: Use proper syntax with one statement per line, proper indentation (2 spaces for subgraphs, 4 for nodes), and no semicolons as separators\n"
"    - Mermaid format: Start with 'flowchart LR' or 'flowchart TD', use proper node syntax [Label], and separate each connection on its own line\n"
"    - For system designs: Describe components, relationships, and data flow in text\n"
"    - Format: 'Here's how I would structure this visually:' followed by detailed text description\n"
"    - Include: Component names, connections, data flow direction, key interfaces\n"
"    - Note: 'While I can't generate actual diagrams, here's the textual representation'\n\n"

"17. **External Sources & Citation Guidelines:**\n"
"    - When citing standards: 'According to [standard name]...'\n"
"    - For official documentation: 'As documented in [framework] official docs...'\n"
"    - For best practices: 'Industry best practice suggests...'\n"
"    - When uncertain: 'Common approaches include...' or 'Typical implementations...'\n"
"    - Always disclaim: 'Specific implementations may vary by organization'\n"
"    - For recent changes: 'Please verify current documentation as APIs may have changed'\n"
"    - Never claim: 'This is the only way' or 'This is always true'\n\n"

"## QUALITY ASSURANCE\n\n"

"18. **Pre-Response Checklist (Verify Before Sending):**\n"
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

"20. **Error Prevention - NEVER:**\n"
"    ✗ Hallucinate function names, APIs, libraries, or frameworks\n"
"    ✗ Use first person for technical strategies without user profile\n"
"    ✗ Create fictional specific work experiences\n"
"    ✗ Use tables by default (only when explicitly needed)\n"
"    ✗ Sacrifice summary quality for response length\n"
"    ✗ Leave code incomplete or without example usage\n"
"    ✗ Skip the comprehensive summary\n"
"    ✗ Provide answers that aren't interview-ready\n\n"

"21. **Error Prevention - ALWAYS:**\n"
"    ✓ Start with comprehensive summary (4-8 sentences)\n"
"    ✓ Structure with clear headings and bullet points\n"
"    ✓ Provide working, complete, well-commented code\n"
"    ✓ Include practical examples and use cases\n"
"    ✓ Maintain interview-ready quality throughout\n"
"    ✓ Be accurate and honest about limitations\n"
"    ✓ Use appropriate voice based on question type\n"
"    ✓ Provide actionable, practical advice\n\n"

"## CONSISTENCY & FLOW\n\n"

"22. **Standard Response Flow:**\n"
"    1. Analyze question type, complexity, and user intent\n"
"    2. Write comprehensive summary (4-8 sentences) covering complete topic\n"
"    3. Provide detailed explanation with proper structure\n"
"    4. Add code/examples/STAR details as needed\n"
"    5. Include practical tips and interview guidance\n"
"    6. Verify quality checklist before finalizing\n\n"

"23. **Consistency Across All Responses:**\n"
"    - ALL responses must have: comprehensive summary → detailed explanation → examples\n"
"    - Concepts: complete summary → features → benefits → examples → best practices\n"
"    - Code: complete summary → code → explanation → complexity → alternatives → tips\n"
"    - Behavioral: complete STAR summary → detailed breakdown → interview tips\n"
"    - System Design: complete summary → requirements → architecture → trade-offs → tips\n"
"    - The comprehensive summary is the PRIMARY answer; everything else enriches it\n\n"

"## TONE & COMMUNICATION\n\n"

"24. **Professional Communication Style:**\n"
"    - Professional, clear, concise, and supportive\n"
"    - Avoid unnecessary jargon; explain technical terms when used\n"
"    - Encourage confidence while maintaining accuracy\n"
"    - Make all responses interview-ready, as if coaching a candidate live\n"
"    - Use positive, constructive language in feedback\n"
"    - Provide specific, actionable advice\n\n"

"## SYSTEM INTEGRATION\n\n"

"25. **System Behavior Rules:**\n"
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

	async def evaluate_code_with_critique(self, problem: str, code: str, language: str) -> str:
		"""Ask the model to produce a structured evaluation and approach explanation.

		Returns markdown text with sections: Summary, Strengths, Weaknesses, Scores JSON, Recommendations.
		"""
		client = self._ensure_client()
		if client is None:
			# fallback mock
			return (
				"Summary: Offline mode. Cannot evaluate without LLM.\n\n"
				"Strengths:\n- Runs locally\n\nWeaknesses:\n- No LLM available\n\n"
				"Scores: {\"correctness\":0.0,\"optimization\":0.0,\"approach_explanation\":0.0,\"complexity_discussion\":0.0,\"edge_cases_testing\":0.0,\"total\":0.0}\n\n"
				"Recommendations:\n- Configure LLM provider"
			)

		prompt = (
			"You are a senior coding interview evaluator. Given a coding problem (if provided), a candidate's source code, and the language, you must produce a concise, world-class critique.\n\n"
			"Output strictly in this format (exact headings):\n"
			"Summary:\n<3-6 sentence overview of approach and correctness>\n\n"
			"Strengths:\n- <bullet 1>\n- <bullet 2>\n\n"
			"Weaknesses:\n- <bullet 1>\n- <bullet 2>\n\n"
			"Scores: {\"correctness\":<0..1>,\"optimization\":<0..1>,\"approach_explanation\":<0..1>,\"complexity_discussion\":<0..1>,\"edge_cases_testing\":<0..1>,\"total\":<0..1>}\n\n"
			"Recommendations:\n- <actionable bullet 1>\n- <actionable bullet 2>\n\n"
			"Guidance: Be concrete. Do not use placeholders. If problem is missing, infer likely intent from code."
		)

		messages: List[Dict[str, str]] = [
			{"role": "system", "content": prompt},
			{"role": "user", "content": f"Problem: {problem or 'N/A'}\nLanguage: {language}\n\nCode:\n```{language}\n{code}\n```"},
		]
		provider = settings.llm_provider
		max_tokens = min(settings.groq_max_tokens, 2048)
		def _call():
			if provider == "groq":
				return client.chat.completions.create(
					model=settings.groq_model,
					messages=messages,
					temperature=0.2,
					max_tokens=max_tokens,
				)
			elif provider == "gemini":
				gmodel = client.GenerativeModel(settings.gemini_model)
				full_prompt = (prompt + "\n\nUser:\n" + messages[-1]["content"]).strip()
				resp = gmodel.generate_content(full_prompt)
				return getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
			else:
				return None

		import anyio
		result = await anyio.to_thread.run_sync(_call)
		if result is None:
			return "Summary: Provider not available.\n\nStrengths:\n- N/A\n\nWeaknesses:\n- N/A\n\nScores: {\"correctness\":0,\"optimization\":0,\"approach_explanation\":0,\"complexity_discussion\":0,\"edge_cases_testing\":0,\"total\":0}\n\nRecommendations:\n- Configure provider"
		if isinstance(result, str):
			return result
		return result.choices[0].message.content or ""

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

	def _is_greeting(self, question: str) -> bool:
		q = (question or "").strip().lower()
		if not q:
			return False
		# Normalize common punctuation
		for ch in ["!", ".", ","]:
			q = q.replace(ch, "")
		# Single/two-word salutations and courtesies
		greetings = {
			"hi", "hello", "hey", "yo", "hiya", "heya",
			"good morning", "good afternoon", "good evening", "gm", "gn",
			"thank you", "thanks", "thx", "ty",
			"bye", "goodbye", "see you", "see ya", "cya", "take care",
		}
		# Quick exact match
		if q in greetings:
			return True
		# Startswith match for polite variants
		prefixes = [
			"hi ", "hello ", "hey ",
			"thank you", "thanks", "thx", "ty ",
			"good morning", "good afternoon", "good evening",
			"bye", "goodbye", "see you", "see ya",
		]
		return any(q.startswith(p) for p in prefixes)

	def _is_off_topic(self, question: str) -> bool:
		"""Detect if question is off-topic for interview preparation"""
		q = (question or "").lower()
		if not q.strip():
			return False
		
		# Off-topic indicators
		off_topic_keywords = [
			"weather", "news", "politics", "sports", "entertainment",
			"personal advice", "relationship", "health", "medical",
			"cooking", "travel", "shopping", "finance", "investment",
			"current events", "celebrity", "movie", "music", "book",
			"game", "gaming", "social media", "dating", "family",
		]
		
		# Check for off-topic keywords
		if any(keyword in q for keyword in off_topic_keywords):
			return True
		
		# Check for non-interview question patterns
		off_topic_patterns = [
			"what's happening", "what's new", "how's your day",
			"tell me about yourself personally", "what do you think about",
			"do you know about", "have you heard about", "what's your opinion",
		]
		
		return any(pattern in q for pattern in off_topic_patterns)

	def _is_ambiguous(self, question: str) -> bool:
		"""Detect if question is ambiguous and needs clarification"""
		q = (question or "").strip()
		if len(q) < 10:  # Very short questions
			return True
		
		# Ambiguous patterns
		ambiguous_patterns = [
			"how do you", "what about", "tell me about", "explain",
			"what is", "how does", "why", "when", "where",
		]
		
		# Check if question is too vague
		if any(pattern in q.lower() for pattern in ambiguous_patterns):
			# If it's a very general question without specific context
			if len(q.split()) < 5:  # Very short
				return True
			# If it lacks specific technical terms
			technical_terms = [
				"algorithm", "data structure", "database", "api", "framework",
				"language", "coding", "programming", "system", "design",
				"interview", "technical", "behavioral", "experience"
			]
			if not any(term in q.lower() for term in technical_terms):
				return True
		
		return False

	def _has_sufficient_context(self, question: str, previous_qna: Optional[List[Dict[str, str]]]) -> bool:
		"""Check if there's sufficient context for the question"""
		if not previous_qna or len(previous_qna) == 0:
			return False
		
		q = (question or "").lower()
		
		# Check for pronouns that need context
		context_pronouns = ["this", "that", "it", "these", "those", "them"]
		if any(pronoun in q for pronoun in context_pronouns):
			return True
		
		# Check for references to previous topics
		context_references = ["previous", "earlier", "above", "before", "last"]
		if any(ref in q for ref in context_references):
			return True
		
		# Check for follow-up questions
		follow_up_patterns = ["also", "additionally", "furthermore", "more", "another"]
		if any(pattern in q for pattern in follow_up_patterns):
			return True
		
		return False

	def _greeting_overrides(self) -> str:
		return (
			"\n\nGreeting Overrides (apply only to salutations/thanks/parting):\n"
			"- Do NOT start with any 'Complete Answer' bullets or a Summary.\n"
			"- No headings. Respond briefly (one or two sentences) in a friendly tone.\n"
			"- Acknowledge the greeting/thanks and offer help if appropriate.\n"
		)

	def _off_topic_overrides(self) -> str:
		return (
			"\n\nOff-Topic Query Overrides (apply only to non-interview questions):\n"
			"- Politely redirect to interview preparation topics.\n"
			"- Format: 'That's an interesting question, but let's focus on interview preparation. Would you like help with [relevant topic]?'\n"
			"- Suggest relevant interview topics: technical concepts, coding problems, system design, behavioral questions.\n"
			"- Keep response brief and professional.\n"
		)

	def _ambiguous_query_overrides(self) -> str:
		return (
			"\n\nAmbiguous Query Overrides (apply only to unclear questions):\n"
			"- Ask 1-2 specific clarifying questions before proceeding.\n"
			"- Format: 'Could you clarify what specific aspect of [topic] you'd like to discuss?'\n"
			"- Provide examples of what you could help with.\n"
			"- Keep response brief and helpful.\n"
		)

	def _context_fallback_overrides(self) -> str:
		return (
			"\n\nContext Fallback Overrides (apply when context is insufficient):\n"
			"- If no past context available: Proceed with fresh, standalone answer.\n"
			"- If context is insufficient: Acknowledge and provide comprehensive answer.\n"
			"- For pronouns without clear referents: Ask for clarification or provide general answer.\n"
			"- When context is unclear: 'Based on general interview practices...'\n"
			"- Always ensure answers work independently of conversation history.\n"
		)

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

	def _is_system_design_question(self, question: str) -> bool:
		"""Detect explicit System Design / Architecture questions"""
		q = (question or "").lower()
		
		# Exclude questions that should generate other types of diagrams
		exclude_keywords = [
			"front page", "user interface", "ui design", "mobile app interface",
			"database schema", "er diagram", "entity relationship",
			"algorithm", "data structure", "sorting", "searching",
			"frontend", "ui/ux", "user experience", "wireframe",
			"mockup", "prototype", "visual design", "layout design"
		]
		
		# If it contains exclude keywords, it's not a system design question
		if any(k in q for k in exclude_keywords):
			return False
		
		# System design and architecture specific keywords - EXPANDED for better coverage
		keywords = [
			# Explicit system design terms
			"system design", "how would you design", "architecture", "architect",
			"high-level design", "hld", "low-level design", "scale to", 
			"million users", "billions", "throughput", "latency",
			"load balancer", "cache", "queue", "kafka", "replication",
			"microservices", "distributed system", "scalable", "scalability",
			"api design", "service design", "component design",
			
			# Specific system types
			"url shortener", "chat system", "social media", "e-commerce",
			"video streaming", "file storage", "search engine", "recommendation system",
			"notification system", "payment system", "booking system", "messaging system",
			"build a system", "create a system", "implement a system", "develop a system",
			
			# Architecture-related terms (EXPANDED)
			"how to build", "how to create", "how to implement", "how to develop",
			"how would you build", "how would you create", "how would you implement",
			"design a", "design an", "build a", "create a", "implement a", "develop a",
			"construct a", "setup a", "setup an", "configure a", "configure an",
			
			# Infrastructure and deployment terms
			"infrastructure", "deployment", "deploy", "hosting", "cloud architecture",
			"aws architecture", "azure architecture", "gcp architecture", "cloud design",
			"container", "docker", "kubernetes", "orchestration", "devops",
			
			# Performance and scaling terms
			"performance", "optimization", "optimize", "scaling", "scale",
			"high availability", "fault tolerance", "redundancy", "backup",
			"disaster recovery", "monitoring", "logging", "metrics",
			"load balancing", "load balancer", "auto-scaling", "auto scaling",
			
			# Data and storage architecture
			"data architecture", "data pipeline", "etl", "elt", "data warehouse",
			"data lake", "big data", "analytics", "reporting", "business intelligence",
			"real-time processing", "batch processing", "stream processing",
			
			# Security and networking
			"security architecture", "network design", "firewall", "vpn",
			"authentication", "authorization", "encryption", "ssl", "tls",
			
			# Integration and API terms
			"integration", "api integration", "third-party integration",
			"webhook", "rest api", "graphql", "soap", "rpc",
			
			# Application architecture patterns
			"mvc", "mvp", "mvvm", "microservices", "monolith", "serverless",
			"event-driven", "cqs", "cqrs", "event sourcing", "saga pattern",
			
			# Technology-specific architecture
			"react architecture", "angular architecture", "vue architecture",
			"node.js architecture", "python architecture", "java architecture",
			"spring architecture", "django architecture", "flask architecture",
			
			# Business and domain terms
			"business architecture", "domain architecture", "enterprise architecture",
			"solution architecture", "technical architecture", "application architecture"
		]
		return any(k in q for k in keywords)

	def _is_database_schema_question(self, question: str) -> bool:
		"""Detect database schema / ER diagram questions"""
		q = (question or "").lower()
		keywords = [
			"database schema", "er diagram", "entity relationship", "database design",
			"show the database", "database structure", "table design", "schema design",
			"relational model", "database model", "data model"
		]
		return any(k in q for k in keywords)

	def _is_ui_design_question(self, question: str) -> bool:
		"""Detect UI/UX design questions"""
		q = (question or "").lower()
		keywords = [
			"front page", "user interface", "ui design", "mobile app interface",
			"frontend design", "ui/ux", "user experience", "wireframe",
			"mockup", "prototype", "visual design", "layout design",
			"design the front", "design the interface", "design the page"
		]
		return any(k in q for k in keywords)

	def _is_algorithm_question(self, question: str) -> bool:
		"""Detect algorithm and data structure questions"""
		q = (question or "").lower()
		keywords = [
			"algorithm", "data structure", "sorting", "searching", "recommendation algorithm",
			"build a recommendation", "implement authentication", "authentication algorithm",
			"search algorithm", "matching algorithm", "optimization algorithm"
		]
		return any(k in q for k in keywords)

	def _database_schema_overrides(self) -> str:
		"""Overrides for database schema questions"""
		return (
			"\n\nDatabase Schema Overrides (apply only to database schema questions):\n"
			"- Include a 'Database Schema' section with an ER diagram using Mermaid.\n"
			"- Use erDiagram syntax with entities, relationships, and attributes.\n"
			"- Example format:\n"
			"  ```mermaid\n"
			"  erDiagram\n"
			"    USER ||--o{ ORDER : places\n"
			"    USER {\n"
			"      int id PK\n"
			"      string name\n"
			"      string email\n"
			"    }\n"
			"    ORDER {\n"
			"      int id PK\n"
			"      int user_id FK\n"
			"      decimal total\n"
			"    }\n"
			"  ```\n"
		)

	def _ui_design_overrides(self) -> str:
		"""Overrides for UI design questions"""
		return (
			"\n\nUI Design Overrides (apply only to UI/UX design questions):\n"
			"- Include a 'UI Design' section with a wireframe or layout diagram using Mermaid.\n"
			"- Use flowchart syntax to show component hierarchy and layout.\n"
			"- Example format:\n"
			"  ```mermaid\n"
			"  flowchart TD\n"
			"    A[Header] --> B[Navigation]\n"
			"    A --> C[Search Bar]\n"
			"    A --> D[User Menu]\n"
			"    E[Main Content] --> F[Article List]\n"
			"    E --> G[Sidebar]\n"
			"    H[Footer] --> I[Links]\n"
			"  ```\n"
		)

	def _algorithm_overrides(self) -> str:
		"""Overrides for algorithm questions"""
		return (
			"\n\nAlgorithm Overrides (apply only to algorithm questions):\n"
			"- Include a 'Algorithm Flow' section with a flowchart using Mermaid.\n"
			"- Use flowchart syntax to show the algorithm steps and decision points.\n"
			"- Example format:\n"
			"  ```mermaid\n"
			"  flowchart TD\n"
			"    A[Start] --> B{Input Valid?}\n"
			"    B -->|Yes| C[Process Data]\n"
			"    B -->|No| D[Return Error]\n"
			"    C --> E[Return Result]\n"
			"  ```\n"
		)

	def _system_design_overrides(self) -> str:
		"""Enforce the System Design response structure requested by the user."""
		return (
			"\n\nSystem Design Overrides (apply only to system/architecture questions):\n"
			"- Follow this exact markdown structure:\n"
			"\n### **Key Highlights**\n"
			"- 4–6 crisp bullets on core data structures, pipelines, algorithms, scalability ideas, trade-offs.\n"
			"\n### **Detailed Explanation**\n"
			"\n#### **1. Requirements Analysis**\n"
			"- **Functional Requirements:** Core outcomes.\n"
			"- **Non-Functional Requirements:** Latency/availability/scalability/freshness.\n"
			"\n#### **2. High-Level Architecture**\n"
			"- Provide a table with Component | Purpose | Technology/Layer.\n"
			"- Executive summary (copy-pasteable): Summarize the domain-specific strategy in 2–4 sentences. Example patterns to consider and adapt: streaming pipelines, event-driven fanout, CQRS, serverless ingestion, microservices vs monolith, or OLAP/OLTP separation. Choose stacks per domain and scale (e.g., messaging vs media vs ridesharing), and justify key trade-offs briefly.\n"
			"- **MANDATORY: Include a 'Visual Architecture Diagram' section with a Mermaid flowchart code block.**\n"
			"- **ALWAYS generate at least one domain-relevant Mermaid diagram (system, data, or cloud view depending on the question), not optional.**\n"
			"- **Generate diagrams for ALL architecture questions: system design, cloud architecture, data architecture, security architecture, etc.**\n"
			"- Use solid arrows (-->), subgraphs for layers (User, Backend, Services, Cache, Database), and colorful classDefs.\n"
			"- Choose appropriate flowchart direction: TD (top-down) for layered architectures, LR (left-right) for data flow.\n"
			"- Include all major components: clients, load balancers, API gateways, microservices, databases, caches, message queues.\n"
			"- Use descriptive node names and proper styling with classDef statements.\n"
			"- Adapt the diagram to the specific architecture type (system, cloud, data, security, etc.).\n"
			"- Diversify technology choices across answers: rotate clouds (AWS/Azure/GCP), data stores (Postgres/MySQL/MongoDB/Cassandra/DynamoDB), queues (Kafka/RabbitMQ/SQS/PubSub), caches (Redis/Memcached), and service languages (Go/Java/Node/Python) based on problem fit—avoid repeating the same stack each time and also use your own intellegence to pick the most appropriate stack.\n"
			"- When constraints are generic, pick a plausible stack and briefly justify choices (e.g., DynamoDB for write-heavy predictable access; Postgres for strong consistency and joins).\n"
			"- Example style guide to follow (adapt names to the problem). Treat nodes as placeholders; rename components to match the domain.\n"
			"  ```mermaid\nflowchart TD\n  subgraph Client[Client Layer]\n    Web[Web App]:::client\n    Mobile[Mobile App]:::client\n  end\n  subgraph CDN[Content Delivery Network]\n    CloudFlare[CloudFlare]:::cdn\n  end\n  subgraph Load_Balancer[Load Balancer Layer]\n    ALB[Application Load Balancer]:::lb\n  end\n  subgraph API_Gateway[API Gateway Layer]\n    Kong[Kong Gateway]:::gateway\n    Auth[Authentication]:::gateway\n  end\n  subgraph Microservices[Microservices Layer]\n    User_Service[User Service]:::service\n    Order_Service[Order Service]:::service\n  end\n  subgraph Database[Database Layer]\n    Postgres[(PostgreSQL)]:::db\n    Redis[(Redis Cache)]:::cache\n  end\n  Web --> CloudFlare\n  CloudFlare --> ALB\n  ALB --> Kong\n  Kong --> Auth\n  Auth --> User_Service\n  Auth --> Order_Service\n  User_Service --> Postgres\n  Order_Service --> Postgres\n  User_Service --> Redis\n  classDef client fill:#e1f5fe,stroke:#01579b,color:#000\n  classDef cdn fill:#f3e5f5,stroke:#4a148c,color:#000\n  classDef lb fill:#fff3e0,stroke:#e65100,color:#000\n  classDef gateway fill:#e8f5e8,stroke:#1b5e20,color:#000\n  classDef service fill:#fff8e1,stroke:#f57f17,color:#000\n  classDef db fill:#e3f2fd,stroke:#0d47a1,color:#000\n  classDef cache fill:#fff3e0,stroke:#f57c00,color:#000\n  ```\n"
			"\n#### **3. Component Design**\n"
			"- Cover ingestion, serving, ranking, caching with data structures, algorithms, storage, optimizations.\n"
			"- Domain-specific pattern example (social feed): for supernodes, apply backpressure: split fanout into segments, write top‑K followers synchronously and the remainder asynchronously; store supernode posts in a hot-post-store and rely on pull/merge at read time.\n"
			"\n#### **3.5. Capacity Planning & Calculations**\n"
			"- **ALWAYS include back-of-envelope math for scale questions.**\n"
			"- Calculate: Daily Active Users → QPS → Storage (per day/year) → Bandwidth → Cache size.\n"
			"- Example format:\n"
			"  * Assumptions: 100M DAU, 10 actions/user/day\n"
			"  * QPS = (100M × 10) / 86400 ≈ 11.6K requests/sec (peak 5x = 58K QPS)\n"
			"  * Storage: 1KB/action × 100M × 10 = 1TB/day → 365TB/year\n"
			"  * Bandwidth: 1TB/day ÷ 86400 = 11.6 MB/sec\n"
			"- Show realistic numbers and how they inform architecture decisions (sharding threshold, cache sizing).\n"
			"\n#### **4. Example Implementation**\n"
			"- Include at least one concise Python (or pseudocode) snippet showing a critical concept.\n"
			"\n#### **5. Scalability & Trade-offs**\n"
			"- Analyze memory vs latency, freshness vs stability, complexity vs maintainability, sharding and load balancing.\n"
			"\n#### **7. Reliability & Failure Handling**\n"
			"- **What breaks when:** Enumerate single points of failure and cascading failures.\n"
			"  * Database down → Read replicas/cache serve stale data, writes queue.\n"
			"  * Cache eviction → Database load spike → Circuit breaker → Degraded mode.\n"
			"  * Service crash → Load balancer health checks → Auto-scaling triggers.\n"
			"- **Recovery patterns:** Retry with exponential backoff, dead letter queues, chaos engineering.\n"
			"- **Disaster recovery:** RTO/RPO targets, multi-region failover, data replication strategies.\n"
			"\n#### **8. Security & Compliance**\n"
			"- **Authentication/Authorization:** OAuth 2.0/JWT, RBAC, API key rotation.\n"
			"- **Data protection:** Encryption at rest (AES-256), in transit (TLS 1.3), key management (KMS/Vault).\n"
			"- **Attack mitigation:** Rate limiting (token bucket), DDoS protection (CloudFlare/AWS Shield), input validation, SQL injection prevention.\n"
			"- **Compliance:** GDPR/CCPA considerations, data residency, audit logging.\n"
			"- **Zero trust:** Service mesh (Istio/Linkerd), mTLS between services, least privilege IAM.\n"
			"\n#### **9. Cost Analysis**\n"
			"- **Infrastructure costs:** EC2/compute ($X/month), storage ($Y/TB), data transfer ($Z/TB out).\n"
			"- **Trade-offs:** Reserved instances vs spot vs on-demand, S3 tiers (Standard/IA/Glacier).\n"
			"- **Optimization strategies:** Caching reduces DB reads by 80% (cost savings), compression, cold data archival.\n"
			"- **Example (illustrative only, scale accordingly):** '1M users → 10TB storage → $230/month S3, 50 c5.xlarge instances → $4K/month'.\n"
			"- Create a billing alert at 60% of monthly budget and an automated job to shut down non‑essential dev stacks.\n"
			"\n#### **10. Monitoring & Observability**\n"
			"- **Golden signals:** Latency (p50/p95/p99), Traffic (QPS), Errors (5xx rate), Saturation (CPU/memory).\n"
			"- **SLIs/SLOs:** Define: '99.9% of requests < 200ms', '99.95% uptime', error budget calculations.\n"
			"- **Tooling:** Metrics (Prometheus/Datadog), Logs (ELK/Splunk), Traces (Jaeger/Zipkin), Alerts (PagerDuty).\n"
			"- **Dashboards:** Show critical path metrics, dependency health, business KPIs.\n"
			"- **On-call playbooks:** Link alerts to runbooks, auto-remediation where possible.\n"
			"\n#### **11. Evolution Strategy**\n"
			"- **Phase 1 (MVP):** Monolith + single DB → Launch in 3 months, 10K users.\n"
			"- **Phase 2 (Scale):** Extract microservices, add caching, read replicas → 1M users.\n"
			"- **Phase 3 (Global):** Multi-region, CDN, eventual consistency → 100M users.\n"
			"- **Migration tactics:** Strangler pattern, feature flags, dark launches, canary deployments.\n"
			"- **Zero-downtime:** Blue-green deployments, rolling updates, database migrations (expand/contract).\n"
			"\n#### **12. Trade-offs Analysis**\n"
			"- Present decisions in table format:\n"
			"  | Decision | Option A | Option B | When to Choose |\n"
			"  |----------|----------|----------|----------------|\n"
			"  | Consistency | Strong (SQL) | Eventual (NoSQL) | Financial: A, Social feed: B |\n"
			"  | Caching | Write-through | Write-behind | Read-heavy: A, Write-heavy: B |\n"
			"- Explain CAP theorem implications for the specific use case.\n"
			"- Discuss latency vs consistency trade-offs with concrete numbers.\n"
			"\n#### **13. Interview Strategy**\n"
			"- **Clarifying questions to ask:** Scale (users/data), latency requirements, read/write ratio, consistency needs.\n"
			"- **Signals to demonstrate:**\n"
			"  * Junior: Functional design, basic scalability\n"
			"  * Mid: Trade-off analysis, caching strategies, basic sharding\n"
			"  * Senior: Cost awareness, failure handling, operational excellence, cross-regional complexity\n"
			"  * Staff+: Build vs buy decisions, org impact, multi-year evolution, team scalability\n"
			"- **Time management:** 5min requirements, 15min architecture, 15min deep-dive, 10min trade-offs.\n"
			"- **Red flags to avoid:** Over-engineering MVP, ignoring failure cases, no metrics/monitoring, unrealistic numbers.\n"
			"\n#### **Meta-Learning Guidance**\n"
			"- After each answer, include:\n"
			"  * **Follow-up questions an interviewer might ask:** 'How would you handle X?', 'What if Y increases 10x?'\n"
			"  * **Common mistakes candidates make:** List 2-3 pitfalls specific to this problem.\n"
			"  * **Leveling indicators:** What a L4 vs L5 vs L6 answer looks like for this question.\n"
			"  * **Related problems:** 3 similar systems to practice for pattern recognition.\n"
			"\n#### **Domain-Specific Optimizations**\n"
			"- Detect problem domain and add specific guidance:\n"
			"  * **Social media:** News feed ranking, viral content handling, graph databases\n"
			"  * **E-commerce:** Inventory consistency, payment idempotency, fraud detection\n"
			"  * **Streaming:** Adaptive bitrate, CDN strategy, live vs VOD\n"
			"  * **Fintech:** Double-entry ledger, audit trails, PCI compliance\n"
			"  * **ML systems:** Feature stores, model serving, A/B testing, drift detection\n"
			"  * **Real-time:** WebSocket/SSE, CRDT, operational transforms\n"
			"\n#### **Company Culture Signals**\n"
			"- Mention if specific companies are known for certain focuses:\n"
			"  * 'Google/Facebook often probe distributed consensus (Paxos/Raft)'\n"
			"  * 'Amazon emphasizes cost optimization and operational excellence'\n"
			"  * 'Netflix looks for chaos engineering mindset'\n"
			"  * 'Stripe focuses on API design and idempotency'\n"
			"\n#### **Adaptive Complexity**\n"
			"- Start with L4-L5 baseline, then:\n"
			"  * If user asks 'what about X edge case?' → Increase to L6-L7 depth\n"
			"  * If user says 'simpler please' → Focus on MVP, defer optimizations\n"
			"  * If user specifies 'Staff level' → Add org design, multi-year roadmap, build-vs-buy.\n"
			"\n#### **6. Interview Takeaways**\n"
			"- 3–5 bullets candidates should emphasize.\n"
			"\n#### **Advanced Enhancements (Include when relevant)**\n"
			"- Memory optimization: prefer Compressed Radix Tree/Patricia or Double-Array Trie for long single-child paths; immutable main index with batch rebuilds.\n"
			"- Hybrid indexing: immutable main index + real-time delta index from Kafka/Kinesis; merge results (delta → main).\n"
			"- Zero-downtime updates: atomic pointer swaps for index versions; blue/green deployment.\n"
			"- Neural re-ranking: apply lightweight encoder (e.g., DistilBERT) on top-K to boost relevance within latency budget.\n"
			"- Sharding: use consistent hashing on prefix/key ranges; auto-rebalance to avoid hot shards.\n"
			"- Caching: multi-level (L1 Redis/memcached, L2 in-process LFU), pre-warm from analytics; Bloom filters to skip cold misses.\n"
			"- Monitoring/feedback: track CTR/abandonment/dwell; A/B test and retrain weights periodically.\n"
			"- Memory layout: flat arrays/struct-of-arrays, contiguous allocations, mmap for fast startup (C++/Rust serving).\n"
			"- Privacy: isolate personalization vectors in a separate encrypted service; serve embeddings/session profiles only.\n"
			"- Ranking refinement: normalize features to [0,1], incorporate CTR, learn weights via logistic regression/GBDT.\n"
			"\n- Style: Senior, precise, 600–1200 words, no filler. Always include at least one code block.\n"
			"- Diagram rendering: Prefer Mermaid flowchart fenced as ```mermaid for UIs that support it.\n"
			"  If Mermaid is not supported, provide a Graphviz DOT fallback fenced as ```dot with solid edges and color attributes.\n"
		)

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
		
		# If content includes a Mermaid diagram, normalize and return as-is (don't treat as code)
		if self._contains_mermaid(text):
			return self._normalize_mermaid_blocks(text)
		
		# First, check if this is code content that should not be formatted as tables
		if self._is_code_content(text):
			# For code content, just clean up basic formatting issues
			text = self._clean_code_formatting(text)
			# Ensure headings are still bolded
			text = self._format_headings_bold(text)
			# Strip LaTeX markers in non-code segments
			text = self._strip_latex_math(text)
			# Normalize Mermaid blocks even inside mixed content
			text = self._normalize_mermaid_blocks(text)
			return text
		
		# Check if this is explanation content that should use text formatting, not tables
		if self._is_explanation_content(text):
			# For explanation content, convert table-like markdown artifacts conservatively
			text = self._clean_explanation_formatting(text)
			# Preserve bold emphasis for headings, side headings, and keywords
			# Ensure headings are still bolded
			text = self._format_headings_bold(text)
			# Remove LaTeX math markers for readability
			text = self._strip_latex_math(text)
			return text
		
		# Only touch pipe tables; do not try to infer tables from text
		text = self._clean_table_markdown_artifacts(text)
		if self._looks_like_pipe_table(text):
			text = self._format_tables(text)
		
		# Only enforce unlabeled bullets within the Complete Answer; elsewhere allow bold
		text = self._strip_labeled_bullets_in_complete_answer(text)
		# Remove bracketed placeholders by converting them to neutral phrasing
		text = self._deplaceholderize(text)
		
		# Ensure headings are properly bolded
		text = self._format_headings_bold(text)
		# Remove LaTeX math markers from non-code sections for readability
		text = self._strip_latex_math(text)
		# Normalize any Mermaid code blocks so each statement is on its own line
		text = self._normalize_mermaid_blocks(text)
		
		return text

	def _format_headings_bold(self, text: str) -> str:
		"""Ensure all headings are properly bolded, but never touch fenced code blocks."""
		import re
		
		lines = text.split('\n')
		formatted_lines = []
		in_code = False
		
		for line in lines:
			stripped = line.strip()
			# Toggle code fence regions
			if stripped.startswith('```'):
				in_code = not in_code
				formatted_lines.append(line)
				continue
			
			if not in_code and stripped.startswith(('##', '###', '####')):
				# Extract the heading text (remove the ##, ###, etc.)
				heading_match = re.match(r'^(#{2,4})\s*(.+)$', stripped)
				if heading_match:
					hashes, heading_text = heading_match.groups()
					# Check if already bolded
					if not heading_text.strip().startswith('**') or not heading_text.strip().endswith('**'):
						formatted_line = f"{hashes} **{heading_text.strip()}**"
						formatted_lines.append(formatted_line)
					else:
						formatted_lines.append(line)
				else:
					formatted_lines.append(line)
			else:
				formatted_lines.append(line)
		
		return '\n'.join(formatted_lines)

	def _strip_latex_math(self, text: str) -> str:
		"""Remove LaTeX math markers ($...$, \(...\), \[...\]) from non-code blocks while preserving inner text.
		Skips fenced code blocks entirely."""
		import re
		lines = text.split('\n')
		out: list[str] = []
		in_code = False
		for line in lines:
			stripped = line.strip()
			if stripped.startswith('```'):
				in_code = not in_code
				out.append(line)
				continue
			if in_code:
				out.append(line)
				continue
			# Replace inline math markers
			newline = re.sub(r'\$(.*?)\$', r'\1', line)
			newline = re.sub(r'\\\((.*?)\\\)', r'\1', newline)
			newline = re.sub(r'\\\[(.*?)\\\]', r'\1', newline, flags=re.DOTALL)
			out.append(newline)
		return '\n'.join(out)

	def _normalize_mermaid_blocks(self, text: str) -> str:
		"""Normalize Mermaid blocks without changing their content semantics.
		Rules (conservative):
		- Do NOT change letters, brackets, or punctuation inside labels.
		- Only insert newlines around structural tokens: subgraph, end, classDef, class, flowchart, and edge statements.
		- Ensure blocks are fenced with ```mermaid.
		"""
		import re
		
		def normalize_block(code: str) -> str:
			"""Bulletproof Mermaid normalizer that completely rebuilds valid syntax."""
			c = code.strip()
			
			# Remove stray backtick artifacts
			c = c.replace("`mermaid", "").replace("```", "").replace("`", "")
			
			# Clean up any leading/trailing whitespace and newlines
			c = c.strip()
			
			# Fix Mermaid syntax issues with special characters in labels
			# Remove parentheses from node labels (Mermaid doesn't handle them well)
			c = re.sub(r'\[([^\]]*?)\(([^)]*?)\)([^\]]*?)\]', r'[\1\2\3]', c)
			# Handle multiple parentheses in the same label
			c = re.sub(r'\[([^\]]*?)\(([^)]*?)\)([^\]]*?)\(([^)]*?)\)([^\]]*?)\]', r'[\1\2\3\4\5]', c)
			# Clean up any remaining parentheses in labels
			c = re.sub(r'\[([^\]]*?)\(([^)]*?)\)([^\]]*?)\]', r'[\1\2\3]', c)
			# Remove parentheses from subgraph names
			c = re.sub(r'subgraph\s+([^[]*?)\(([^)]*?)\)([^[]*?)\[', r'subgraph \1\2\3[', c)
			# Clean up any remaining parentheses in subgraph names
			c = re.sub(r'subgraph\s+([^[]*?)\(([^)]*?)\)([^[]*?)\[', r'subgraph \1\2\3[', c)
			
			# Extract flowchart type - preserve the original direction
			flowchart_match = re.match(r'^(flowchart\s+[A-Z]{2})', c)
			flowchart_type = flowchart_match.group(1) if flowchart_match else "flowchart LR"
			
			# Remove flowchart declaration
			remaining = re.sub(r'^flowchart\s+[A-Z]{2}\s*', '', c).strip()
			
			formatted_lines = [flowchart_type]
			
			# Extract classDef and class statements first to avoid duplication
			classdef_pattern = r'classDef\s+([^;]+)'
			classdef_matches = re.findall(classdef_pattern, c)
			classdef_statements = [f"classDef {classdef.strip()}" for classdef in classdef_matches]
			
			class_pattern = r'class\s+([^;]+)'
			class_matches = re.findall(class_pattern, c)
			class_statements = [f"class {class_stmt.strip()}" for class_stmt in class_matches]
			
			# Remove classDef and class statements from remaining content to avoid duplication
			remaining = re.sub(r'classDef\s+[^;]+;?', '', remaining)
			remaining = re.sub(r'class\s+[^;]+;?', '', remaining)
			
			# Process the content line by line to preserve structure
			lines = remaining.split('\n')
			in_subgraph = False
			subgraph_depth = 0
			
			for line in lines:
				line = line.strip()
				if not line:
					continue
					
				# Skip flowchart declaration as it's already added
				if re.match(r'^(flowchart\s+[A-Z]{2}|sequenceDiagram|classDiagram|erDiagram|stateDiagram|gantt|journey|pie|mindmap|timeline)\s*', line):
					continue
					
				# Check if this line starts a subgraph
				subgraph_match = re.match(r'subgraph\s+(.+)', line)
				if subgraph_match:
					subgraph_name = subgraph_match.group(1).strip()
					# Ensure subgraph name is properly formatted
					if not subgraph_name.endswith(']') and '[' in subgraph_name:
						# Add missing closing bracket if needed
						subgraph_name += ']'
					formatted_lines.append(f"subgraph {subgraph_name}")
					in_subgraph = True
					subgraph_depth += 1
					continue
				
				# Check if this line ends a subgraph
				if line == 'end' and in_subgraph:
					formatted_lines.append("end")
					subgraph_depth -= 1
					if subgraph_depth == 0:
						in_subgraph = False
					continue
				
				# Process regular statements
				if in_subgraph:
					# Indent content inside subgraphs
					formatted_lines.append(f"  {line}")
				else:
					# Regular content outside subgraphs
					formatted_lines.append(f"  {line}")
			
			# Add classDef and class statements at the end with proper formatting
			for classdef in classdef_statements:
				formatted_lines.append(classdef)
			for class_stmt in class_statements:
				formatted_lines.append(class_stmt)
			
			# Join lines and clean up
			result = '\n'.join(formatted_lines)
			
			# Final cleanup
			result = re.sub(r'\n\s*\n', '\n', result)
			result = re.sub(r'^\s*\n', '', result)
			result = result.strip()
			
			return result
		
		lines = text.split('\n')
		out: list[str] = []
		in_mermaid = False
		buffer: list[str] = []
		for line in lines:
			if line.strip().startswith("```mermaid"):
				in_mermaid = True
				buffer = []
				out.append(line)
				continue
			if in_mermaid and line.strip().startswith("```"):
				# close block
				normalized = normalize_block("\n".join(buffer))
				out.append(normalized)
				out.append(line)
				in_mermaid = False
				buffer = []
				continue
			if in_mermaid:
				buffer.append(line)
			else:
				out.append(line)
		
		# If there was orphan flowchart text without fences, try to wrap it
		joined = "\n".join(out)
		import re as _re
		if _re.search(r"^(flowchart|sequenceDiagram|classDiagram|erDiagram|stateDiagram|gantt|journey|pie|mindmap|timeline)\b", joined, _re.MULTILINE) and "```mermaid" not in joined:
			code = normalize_block(joined)
			return "```mermaid\n" + code + "```"
		return joined

	def _contains_mermaid(self, text: str) -> bool:
		import re
		if "```mermaid" in text:
			return True
		return bool(re.search(r"^(flowchart|sequenceDiagram|classDiagram|erDiagram|stateDiagram|gantt|journey|pie|mindmap|timeline)\b", text, re.MULTILINE))

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

	def _deplaceholderize(self, text: str) -> str:
		"""Convert bracketed placeholders like [SPECIFIC FEATURE/PROJECT TASK] into neutral, readable text.
		Rules:
		- Known mappings to concise phrases
		- Otherwise, drop brackets and lower-case the phrase in a generic way
		- Never introduce brackets [] in the output
		"""
		import re
		mappings = {
			"SPECIFIC FEATURE": "the feature",
			"SPECIFIC PRODUCT": "the product",
			"PROJECT GOAL": "the project goal",
			"SPECIFIC COMPROMISE DETAIL": "a balanced compromise",
			"FEATURE/PROJECT TASK": "the task",
			"SITUATION": "the situation",
			"TASK": "the task",
			"ACTION": "the action",
			"RESULT": "the result",
		}
		def repl(match: re.Match[str]) -> str:
			inside = match.group(1).strip()
			key = inside.upper()
			if key in mappings:
				return mappings[key]
			# Simplify multi-part tokens like 'SPECIFIC FEATURE/PROJECT' → 'the feature'
			parts = re.split(r"[\s/_-]+", inside)
			for part in parts:
				candidate = part.upper()
				if candidate in mappings:
					return mappings[candidate]
			# Fallback: plain, lower-cased phrase without brackets
			return inside.lower()
		# Replace all [ ... ] occurrences
		return re.sub(r"\[([^\]]{1,80})\]", repl, text)

	# Note: We intentionally removed global bold stripping to allow bold for headings, side headings, and keywords.
	
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
		
		# Check if this is a heading line - preserve bold formatting for headings
		if line.strip().startswith(('##', '###', '####')):
			# For headings, preserve bold formatting
			return line
		
		# Remove all markdown bold formatting (**text**) for non-heading lines
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

		# If this is a brief greeting/thanks/parting, suppress summary/bullets
		if self._is_greeting(question):
			prompt = prompt + self._greeting_overrides()
		
		# If this is an off-topic query, redirect to interview preparation
		if self._is_off_topic(question):
			prompt = prompt + self._off_topic_overrides()
		
		# If this is an ambiguous query, ask for clarification
		if self._is_ambiguous(question):
			prompt = prompt + self._ambiguous_query_overrides()
		
		# If context is insufficient, provide fallback handling
		if not self._has_sufficient_context(question, previous_qna):
			prompt = prompt + self._context_fallback_overrides()
		
		# If this is a system design question, enforce the SD structure
		if self._is_system_design_question(question):
			prompt = prompt + self._system_design_overrides()
		
		# If this is a database schema question, add ER diagram overrides
		if self._is_database_schema_question(question):
			prompt = prompt + self._database_schema_overrides()
		
		# If this is a UI design question, add UI design overrides
		if self._is_ui_design_question(question):
			prompt = prompt + self._ui_design_overrides()
		
		# If this is an algorithm question, add algorithm overrides
		if self._is_algorithm_question(question):
			prompt = prompt + self._algorithm_overrides()
		
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
