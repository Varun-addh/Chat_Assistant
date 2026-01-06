from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Header, Response
from typing import Optional, List
from fastapi.responses import StreamingResponse
from datetime import datetime

from app.schemas import (
	CreateSessionResponse, 
	QuestionIn, 
	AnswerOut, 
	SessionHistory, 
	QnA, 
	SessionList, 
	SessionSummary, 
	UpdateSessionTitleRequest
)
from app.services.session_manager import get_session_manager, SessionManager
from app.services.llm_service import get_llm_service
from app.services.architecture_generator import get_architecture_generator, ArchitectureViewType
from app.services.code_evaluation_service import evaluate_code
from app.utils.security import verify_api_key
from app.utils.audit import auditor
from app.config import settings
from app.middleware.auth import get_user_id_from_request
from fastapi import Request
import asyncio
import logging

from app.utils.story_contract import extract_mermaid_first, enforce_story_contract, sanitize_mermaid_subset

logger = logging.getLogger(__name__)

# Use intelligent provider selection with "copilot" feature flag for AI Assistant/Chat
llm_service = get_llm_service(feature="copilot")

router = APIRouter()


def _validate_specificity(text: str, system_description: str) -> tuple[bool, list[str]]:
	"""Check if explanation is specific enough for FAANG-level interviews.
	
	Returns (is_specific, list_of_issues).
	"""
	import re
	
	issues = []
	text_lower = text.lower()
	
	# Check for generic platitudes (indicates shallow thinking)
	generic_phrases = [
		"business logic", "process the request", "store the data",
		"handle the workflow", "manage the system", "provides scalability",
		"ensures reliability", "improves performance", "handle requests"
	]
	
	generic_count = sum(1 for phrase in generic_phrases if phrase in text_lower)
	if generic_count > 3:
		issues.append(f"Too many generic phrases ({generic_count}): use actual tech names and mechanisms")
	
	# Check for specific tech mentions (indicates real design thinking)
	tech_indicators = [
		r'\b(redis|postgres|postgresql|kafka|elasticsearch|cassandra|mongodb|dynamodb|s3|sqs|rabbitmq)\b',
		r'\b\d+\s*(ms|milliseconds?|sec|seconds?|minutes?|MB|GB|TB|PB|req/sec|QPS|TPS)\b',  # Numbers with units
		r'\b(sharding|partitioning|replication|caching|indexing|hashing)\b',
		r'\b(two-phase commit|optimistic locking|pessimistic locking|MVCC|CAP theorem)\b',
	]
	
	has_specifics = any(re.search(pattern, text, re.IGNORECASE) for pattern in tech_indicators)
	if not has_specifics:
		issues.append("Missing specific technology names, metrics, or architectural patterns")
	
	# Check for quantified reasoning (numbers, metrics, comparisons)
	has_numbers = bool(re.search(r'\b\d+\s*[a-z%]+\b', text))
	has_comparison = bool(re.search(r'\b(vs|versus|compared to|instead of|chose.*over)\b', text_lower))
	
	if not has_numbers and not has_comparison:
		issues.append("Missing quantified reasoning (numbers, metrics, or technology comparisons)")
	
	return len(issues) == 0, issues


def _sanitize_mermaid_from_llm(mermaid_code: str) -> str:
	"""Remove CSS/style/HTML artifacts that LLM might inject into Mermaid code.
	Also fix common Mermaid syntax errors.
	
	LLMs sometimes add @import, @keyframes, CSS selectors, or HTML despite
	being instructed not to. This breaks rendering.
	"""
	import re
	
	if not mermaid_code:
		return mermaid_code
	
	# Split into lines
	lines = mermaid_code.split('\n')
	cleaned_lines = []
	
	# Patterns to reject (CSS/HTML artifacts)
	reject_patterns = [
		'@import', '@keyframes', '@font-face', '@media',
		'#mermaid-svg', '#container', '.edge-', '.node-',
		'<style>', '</style>', '<script>', '</script>',
		'font-family:', 'font-size:', 'fill:', 'stroke:',
		'background-color:', 'color:', 'opacity:',
		'trebuchet', 'verdana', 'sans-serif',  # Font names indicate CSS leak
		'}}#', '{font-', ':root{'  # CSS selector patterns
	]
	
	for line in lines:
		stripped = line.strip()
		
		# Skip empty lines at the start
		if not stripped and not cleaned_lines:
			continue
		
		# Skip lines that look like CSS/HTML
		if any(pattern in line.lower() for pattern in reject_patterns):
			logger.warning(f"[Mermaid Sanitizer] Removing CSS/HTML line: {stripped[:80]}")
			continue
		
		# Skip lines that are pure CSS selectors (contain { } but not Mermaid syntax)
		if '{' in line and '}' in line and not any(x in line for x in ['[', ']', '(', ')']):
			logger.warning(f"[Mermaid Sanitizer] Removing CSS selector: {stripped[:80]}")
			continue
		
		# Skip init blocks
		if stripped.startswith('%%{init') or stripped.endswith('}%%'):
			logger.warning(f"[Mermaid Sanitizer] Removing init block: {stripped[:80]}")
			continue
		
		cleaned_lines.append(line)
	
	cleaned = '\n'.join(cleaned_lines)
	
	# Fix common arrow syntax errors: -> should be --> in Mermaid
	cleaned = re.sub(r'(\w+|\])\s*->\s*(\w+|\[)', r'\1 --> \2', cleaned)
	
	# Fix single dash arrows: A - B should be A --> B
	cleaned = re.sub(r'(\w+|\])\s+-\s+(\w+|\[)', r'\1 --> \2', cleaned)
	
	# Ensure first line is a valid diagram type
	first_lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
	if first_lines:
		first = first_lines[0].lower()
		if not any(first.startswith(t) for t in ['flowchart', 'graph', 'sequencediagram', 'classdiagram']):
			# Prepend flowchart LR if missing
			cleaned = 'flowchart LR\n' + cleaned
	
	# Log if we made changes
	if len(cleaned) < len(mermaid_code) * 0.9:
		logger.info(f"[Mermaid Sanitizer] Cleaned {len(mermaid_code) - len(cleaned)} chars of CSS/HTML")
	
	return cleaned.strip()


async def _auto_evaluate_if_code(session_manager: SessionManager, session_id: str, question: str, answer: str, api_key: Optional[str] = None) -> None:
	"""Auto-evaluate if the answer contains code blocks."""
	import re
	
	# Look for code blocks in the answer
	code_pattern = r'```(\w+)?\n(.*?)\n```'
	matches = re.findall(code_pattern, answer, re.DOTALL)
	
	if not matches:
		return
	
	# Get the most substantial code block
	best_code = ""
	best_lang = "python"  # Default to Python for coding questions
	for lang, code in matches:
		if len(code.strip()) > len(best_code.strip()):
			best_code = code.strip()
			best_lang = lang or "python"  # Default to Python if no language specified
	
	if not best_code:
		return
	
	# Auto-trigger evaluation in background
	try:
		# Get conversation context
		session_state = await session_manager.get_required(session_id)
		conversation_context = ""
		if session_state.qna:
			recent_qna = session_state.qna[-2:] if len(session_state.qna) >= 2 else session_state.qna
			for item in recent_qna:
				conversation_context += f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}\n"
		
		# Run evaluation
		await evaluate_code(question, best_code, best_lang, conversation_context, api_key=api_key)
		
		# Log auto-evaluation
		await auditor.log({
			"type": "auto_evaluation",
			"session_id": session_id,
			"question": question,
			"language": best_lang,
			"auto_triggered": True,
		})
	except Exception as e:
		# Don't fail the main request if evaluation fails
		await auditor.log({
			"type": "auto_evaluation_error",
			"session_id": session_id,
			"error": str(e),
		})


@router.post("/session", response_model=CreateSessionResponse)
async def create_session(request: Request):
	user_id = get_user_id_from_request(request) or "default"
	manager = get_session_manager(user_id)
	state = await manager.create_session()
	return CreateSessionResponse(session_id=state.session_id)


@router.post("/question")
async def submit_question(
	payload: QuestionIn,
	request: Request,
	response: Response,
	x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
	x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
	authorization: Optional[str] = Header(None, alias="Authorization")
):
	# 1. Extract and clean headers (handle empty strings from frontend)
	groq_key = (x_api_key.strip() if x_api_key else None) or None
	gemini_key = (x_gemini_key.strip() if x_gemini_key else None) or None
	
	if not groq_key and authorization:
		auth_val = authorization.strip()
		if auth_val.lower().startswith("bearer "):
			groq_key = auth_val.split(" ", 1)[1] if " " in auth_val else auth_val
		else:
			groq_key = auth_val

	# 2. Select the key to use. 
	# If the user has provided a Gemini key in Bridge Settings, we prefer it (as it's higher quality).
	# Otherwise, use the Groq key if provided.
	api_key = gemini_key or groq_key
	
	# 3. Fallback to server/env keys ONLY if user provided NO keys in Bridge Settings
	if not api_key:
		# Use preferred provider from settings if key is available
		if settings.llm_provider == "gemini" and settings.gemini_api_key:
			api_key = settings.gemini_api_key
			logger.info("🔧 Using environment GEMINI_API_KEY")
		elif settings.llm_provider == "groq" and settings.groq_api_key:
			api_key = settings.groq_api_key
			logger.info("🔧 Using environment GROQ_API_KEY")
		else:
			# Last resort: use whatever is available
			api_key = settings.gemini_api_key or settings.groq_api_key
			if api_key:
				logger.info(f"🔧 Using fallback environment key for {settings.llm_provider}")

	# 4. Final check: if still no key, raise error
	if not api_key:
		raise HTTPException(
			status_code=401,
			detail="No active API key. Please add your Groq or Gemini API key in Bridge Settings to continue."
		)
	
	user_id = get_user_id_from_request(request) or "default"
	manager = get_session_manager(user_id)
	
	try:
		state = await manager.get(payload.session_id)
		if state is None:
			# Auto-recovery: Create a new session if the requested one doesn't exist
			# This handles cases where backend was restarted or storage was cleared
			logger.info(f"Session {payload.session_id} not found. Creating new session for recovery.")
			state = await manager.create_session()
			# Update the response to include the new session_id so frontend can sync
			logger.info(f"Auto-created new session: {state.session_id}")
	except KeyError:
		# Fallback: create new session
		state = await manager.create_session()
		logger.info(f"Created new session due to KeyError: {state.session_id}")

	if not payload.question.strip():
		raise HTTPException(status_code=400, detail="Empty question")

	# Auto-detect architecture mode choice if user replies with "single" or "multi"
	# This handles the case where user is responding to the architecture choice prompt
	if not payload.architecture_mode:
		q_lower = payload.question.strip().lower()
		# Check if this is a simple mode selection response
		if q_lower in ["single", "single-view", "single view", "1"]:
			payload.architecture_mode = "single"
			logger.info(f"🎯 Auto-detected architecture mode choice: single")
			# Retrieve the original question from conversation history
			if state.qna and len(state.qna) > 0:
				last_qna = state.qna[-1]
				if isinstance(last_qna, dict) and "question" in last_qna:
					original_question = last_qna["question"]
					logger.info(f"📝 Retrieved original question from history: {original_question[:100]}")
					payload.question = original_question
				elif isinstance(last_qna, QnA):
					original_question = last_qna.question
					logger.info(f"📝 Retrieved original question from history: {original_question[:100]}")
					payload.question = original_question
		elif q_lower in ["multi", "multi-view", "multi view", "multiview", "2"]:
			payload.architecture_mode = "multi-view"
			logger.info(f"🎯 Auto-detected architecture mode choice: multi-view")
			# Retrieve the original question from conversation history
			if state.qna and len(state.qna) > 0:
				last_qna = state.qna[-1]
				if isinstance(last_qna, dict) and "question" in last_qna:
					original_question = last_qna["question"]
					logger.info(f"📝 Retrieved original question from history: {original_question[:100]}")
					payload.question = original_question
				elif isinstance(last_qna, QnA):
					original_question = last_qna.question
					logger.info(f"📝 Retrieved original question from history: {original_question[:100]}")
					payload.question = original_question

	# Deterministic identity/developer attribution answers (never call an LLM)
	# This is a belt-and-suspenders guard in addition to the LLMService short-circuit.
	if llm_service._is_identity_question(payload.question):
		identity_answer = llm_service._identity_response_text(payload.question)
		# Proof-of-handling: inspect this in DevTools -> Network -> Response Headers
		response.headers["X-Stratax-Guard"] = "identity"
		response.headers["X-Stratax-App"] = getattr(settings, "app_name", "Stratax AI")
		if payload.stream:
			async def _identity_event_gen():
				# SSE: prefix each line with 'data: '
				safe = identity_answer.replace("\n", "\ndata: ")
				yield f"data: {safe}\n\n"
				# Persist to history if requested
				if payload.save_to_history:
					await manager.append_qna(payload.session_id, payload.question, identity_answer)
				await auditor.log({
					"type": "qna",
					"session_id": payload.session_id,
					"question": payload.question,
					"answer": identity_answer,
					"saved_to_history": payload.save_to_history,
				})
				yield "event: end\n\n"
			# StreamingResponse doesn't automatically inherit headers from `response`
			return StreamingResponse(
				_identity_event_gen(),
				media_type="text/event-stream",
				headers={
					"X-Stratax-Guard": "identity",
					"X-Stratax-App": getattr(settings, "app_name", "Stratax AI"),
				},
			)

		if payload.save_to_history:
			await manager.append_qna(payload.session_id, payload.question, identity_answer)
		await auditor.log({
			"type": "qna",
			"session_id": payload.session_id,
			"question": payload.question,
			"answer": identity_answer,
			"saved_to_history": payload.save_to_history,
		})
		return AnswerOut(answer=identity_answer, created_at=datetime.utcnow(), truncated=False)

	# Read any stored profile for this session
	profile_text = state.profile_text

	# --- ARCHITECTURE DETECTION ---
	# Smart pattern-based detection (no LLM call needed - saves time and cost)
	q_lower = payload.question.lower()
	
	# Pattern 1: Explicit system design keywords
	has_explicit = any(kw in q_lower for kw in settings.architecture_detection_explicit_keywords)
	
	# Pattern 2: "Design/Build X" pattern (catches "design twitter", "build a cache", etc.)
	has_design_verb = any(kw in q_lower for kw in settings.architecture_detection_design_verbs)
	
	# Pattern 3: System components/concepts (indicates architecture discussion)
	has_system_concepts = (
		sum(
			[
				any(kw in q_lower for kw in settings.architecture_detection_system_concepts_scale),
				any(kw in q_lower for kw in settings.architecture_detection_system_concepts_data),
				any(kw in q_lower for kw in settings.architecture_detection_system_concepts_infra),
			]
		)
		>= 2
	)  # Need at least 2 system concepts
	
	# Pattern 4: Exclude coding problems
	is_code_problem = any(kw in q_lower for kw in settings.architecture_detection_code_problem_keywords)
	
	# Trigger if: explicit keywords OR design verb OR multiple system concepts, AND not a code problem
	is_architecture = (has_explicit or has_design_verb or has_system_concepts) and not is_code_problem

	# If system design detected AND user hasn't chosen a mode yet, ask for preference.
	arch_mode = payload.architecture_mode
	if is_architecture and not arch_mode:
		logger.info(f"🏗️ System design detected: {payload.question} - asking user for preference")
		
		# Return minimal trigger text that frontend will detect and replace with UI card
		# This message contains the keywords to trigger frontend detection but is meant to be hidden/replaced
		choice_message = (
			"Choose your preferred architecture format: "
			"Single Comprehensive or Multi-View. "
			"architecture_mode"
		)
		
		# Don't save this internal trigger message to history
		# The actual architecture response will be saved later
		
		return AnswerOut(
			answer=choice_message,
			created_at=datetime.utcnow(),
			truncated=False,
		)

	# Route based on user's choice
	if is_architecture and arch_mode == "multi-view":
		logger.info(f"🏗️ Detected system design question: {payload.question}")
		
		# View titles and descriptions for each architecture layer
		view_info = {
			"SYSTEM_OVERVIEW": ("🏗️ System Overview", "High-level components and how they connect"),
			"REQUEST_FLOW": ("🔄 Request Flow", "Step-by-step journey of a user request through the system"),
			"DATA_MODEL": ("🗄️ Data & Storage", "Databases, caches, and how data is persisted"),
			"DEPLOYMENT": ("📦 Deployment Architecture", "Infrastructure, scaling, and cloud services"),
			"OBSERVABILITY": ("📊 Observability & Monitoring", "Metrics, logs, alerts, and system health"),
		}
		
		# Define the generator for architecture responses
		async def architecture_stream_gen():
			try:
				# 1. Initialize services
				arch_gen = get_architecture_generator()

				def _sanitize_view_explanation(text: str) -> str:
					"""Make LLM output deterministic for the UI.

					We already emit an H2 title per view. This strips any accidental
					H1/H2 headings and removes stray 'Bottom line:' lines that often
					show up inside sections.
					"""
					import re
					if not text:
						return ""
					out_lines: list[str] = []
					for raw_line in text.splitlines():
						line = raw_line.rstrip()
						# Remove top-level headings (caller provides them)
						if re.match(r"^\s*#{1,2}\s+", line):
							continue
						# Remove stray 'Bottom line' inside layers (keep final summaries clean)
						if re.match(r"^\s*([\-*]\s*)?(\*\*?)?bottom line(\*\*?)?:", line, flags=re.IGNORECASE):
							continue
						out_lines.append(raw_line)
					return "\n".join(out_lines).strip()

				def _safe_ascii(text: str) -> str:
					# Keep UI stable and avoid non-ASCII issues.
					return (text or "").encode("ascii", errors="ignore").decode("ascii", errors="ignore")

				# 2. Select views: keep multi-view deterministic (the product promise is 5 views).
				views_to_gen = [
					ArchitectureViewType.SYSTEM_OVERVIEW,
					ArchitectureViewType.REQUEST_FLOW,
					ArchitectureViewType.DATA_MODEL,
					ArchitectureViewType.DEPLOYMENT,
					ArchitectureViewType.OBSERVABILITY,
				]
				
				# 3. Generate each view with diagram + story-driven explanation
				full_response = f"# System Design: {payload.question}\n\n"
				
				for idx, view_type in enumerate(views_to_gen):
					is_last_view = (idx == len(views_to_gen) - 1)
					
					# Get view title and description
					view_name = view_type.name if hasattr(view_type, 'name') else str(view_type)
					title, desc = view_info.get(view_name, (f"📐 {view_name}", "Architecture view"))
					
					# Send title
					yield f"data: ## {title}\n\n"
					
					prompt_data = arch_gen.get_view_prompt(view_type, payload.question)
					# prompt_data['system_prompt'] already contains a strict output contract.
					base_prompt = f"{prompt_data['user_prompt']}"
					combined_prompt = base_prompt

					# Generate view with up to 2 retries if diagram is too complex
					response_text = ""
					mermaid_code = ""
					explanation = ""
					last_issues: List[str] = []

					for attempt in range(3):
						response_text, _ = await llm_service.generate_answer(
							question=combined_prompt,
							system_prompt=prompt_data['system_prompt'],
							api_key=api_key,
							apply_auto_overrides=False,
						)

						# Parse response: Mermaid diagram first, then explanation
						mermaid_code, raw_explanation = extract_mermaid_first(response_text or "")
						if not mermaid_code:
							mermaid_code = "flowchart TD\n  Error[No diagram generated]"
						explanation = raw_explanation
						explanation = _sanitize_view_explanation(explanation)
						explanation = enforce_story_contract(view_name, payload.question, explanation)
						explanation = _safe_ascii(explanation)
						mermaid_code = sanitize_mermaid_subset(mermaid_code)

						# Validate specificity (skip for SYSTEM_OVERVIEW which is intentionally brief)
						if view_name != "SYSTEM_OVERVIEW":
							is_specific, specificity_issues = _validate_specificity(explanation, payload.question)
							if not is_specific and attempt < 2:
								logger.warning(f"[{view_name}] Content not specific enough (attempt {attempt+1}): {specificity_issues}")
								combined_prompt = (
									f"MAKE IT MORE SPECIFIC. Previous answer was too generic: {', '.join(specificity_issues)}.\n\n"
									f"You MUST include:\n"
									f"- Actual technology names (Redis, Postgres, Kafka, etc.)\n"
									f"- Actual numbers with units (50ms, 10K req/sec, 95% cache hit)\n"
									f"- Specific mechanisms/algorithms (GEORADIUS, two-phase commit, sharding by user_id)\n"
									f"- Technology comparisons (Chose X over Y because...)\n\n"
									f"BAD: 'caching layer stores data'\n"
									f"GOOD: 'Redis cluster: 3 primary + 3 replica, LRU eviction, 1-hour TTL, 95% hit rate'\n\n"
									+ base_prompt
								)
								continue  # Retry with specificity guidance

						# Validate complexity; if too complex, retry with a simplification instruction
						validation = arch_gen.validate_diagram_complexity(
							mermaid_code,
							view_type,
							max_nodes=prompt_data.get("max_nodes"),
							max_edges=prompt_data.get("max_edges"),
						)
						if validation.get("valid"):
							break

						last_issues = validation.get("issues") or []
						logger.warning(f"Complexity warning for {view_type}: {last_issues}")
						if attempt < 2:
							combined_prompt = (
								f"SIMPLIFY AND REGENERATE. Previous diagram violated constraints: {last_issues}.\n"
								f"You MUST reduce to <= {prompt_data.get('max_nodes')} nodes and <= {prompt_data.get('max_edges')} edges.\n"
								"Keep ONLY the critical path for this view. Remove optional components.\n"
								"Do NOT use subgraphs unless absolutely necessary.\n\n"
								"Also: Explanation must follow the OUTPUT CONTRACT exactly (no Key Highlights, no tables, no deep dives).\n\n"
								+ base_prompt
							)

					# If still invalid after retries, proceed (renderer will show placeholder if needed)
					
					# Validate it starts with a valid Mermaid diagram type
					first_line = (mermaid_code.split('\n')[0].strip().lower() if mermaid_code else "")
					valid_starts = ['flowchart', 'graph', 'sequencediagram', 'classdiagram', 'statediagram']
					if not any(first_line.startswith(vs) for vs in valid_starts):
						logger.error(f"Invalid Mermaid diagram - doesn't start with valid type. First line: {first_line}")
						mermaid_code = "flowchart TD\n  Error[Invalid diagram generated]"
					
					# CRITICAL: Strip any CSS/style content that LLM might have added (breaks rendering)
					mermaid_code = _sanitize_mermaid_from_llm(mermaid_code)
					
					# Log the mermaid code for debugging
					logger.info(f"Generated Mermaid code ({len(mermaid_code)} chars): {mermaid_code[:200]}...")
						
					# Validate (log only)
					validation = arch_gen.validate_diagram_complexity(mermaid_code, view_type)
					if not validation.get("valid"):
						logger.warning(f"Complexity warning for {view_type}: {validation.get('issues')}")

					# Send diagram (SSE-safe: prefix every line with 'data: ')
					diagram_block = f"```mermaid\n{mermaid_code}\n```"
					safe_diagram = diagram_block.replace("\n", "\ndata: ")
					yield f"data: {safe_diagram}\n\n"
					
					# Stream explanation (includes code/tips if last view)
					safe_explanation = explanation.replace(chr(10), chr(10) + 'data: ')
					yield f"data: {safe_explanation}\n\n"
					
					# Append to full response for history
					full_response += f"## {title}\n\n```mermaid\n{mermaid_code}\n```\n\n{explanation}\n\n"
				
				# Save to history (code + tips already included in last view)
				if payload.save_to_history:
					await manager.append_qna(state.session_id, payload.question, full_response)
					
				yield "event: end\n\n"
				
			except Exception as e:
				logger.error(f"Architecture generation failed: {e}")
				err_msg = str(e).replace(chr(10), " ")
				yield f"data: ⚠️ **Generation failed:** {err_msg}\n\n"
				yield "event: end\n\n"

		if payload.stream:
			return StreamingResponse(architecture_stream_gen(), media_type="text/event-stream")
		else:
			# Non-streaming fallback
			chunks = []
			async for chunk in architecture_stream_gen():
				if chunk.startswith("data: "):
					chunks.append(chunk[6:].replace("\ndata: ", "\n"))
			final_ans = "".join(chunks)
			if payload.save_to_history:
				await manager.append_qna(state.session_id, payload.question, final_ans)
			return AnswerOut(answer=final_ans, created_at=datetime.utcnow(), truncated=False)
	
	# Handle single comprehensive architecture (legacy single-diagram behavior)
	elif is_architecture and arch_mode == "single":
		logger.info(f"🏗️ Generating SINGLE comprehensive architecture for: {payload.question}")
		
		# IMPORTANT: "single" here means the legacy single-diagram system-design behavior
		# powered by LLMService's built-in system-design overrides (the "old architecture thing").
		# We do NOT run the newer multi-view architecture generator or SINGLE story-contract clamping.
		logger.info("📊 Using legacy SINGLE architecture path (LLMService auto system-design overrides)")

		if payload.stream:
			async def single_architecture_stream_gen():
				collected: list[str] = []
				previous_qna = state.qna
				# Quick feedback + consistent header for the UI
				yield "data: ## Single-View Architecture\n\n"
				yield "data: Generating architecture...\n\n"

				async for chunk in llm_service.stream_answer(
					payload.question,
					payload.system_prompt,
					profile_text=profile_text,
					previous_qna=previous_qna[-10:] if previous_qna else None,
					style_mode=payload.style_mode,
					tone=payload.tone,
					layout=payload.layout,
					variability=payload.variability,
					seed=payload.seed,
					api_key=api_key,
					# Legacy behavior: allow LLMService to apply its auto system-design overrides.
					apply_auto_overrides=True,
				):
					collected.append(chunk)
					safe_chunk = chunk.replace('\n', '\ndata: ')
					yield f"data: {safe_chunk}\n\n"

				if payload.save_to_history:
					await manager.append_qna(
						state.session_id,
						payload.question,
						"## Single-View Architecture\n\n" + "".join(collected),
					)
				yield "event: end\n\n"

			return StreamingResponse(single_architecture_stream_gen(), media_type="text/event-stream")
		else:
			previous_qna = state.qna
			answer, truncated = await llm_service.generate_answer(
				payload.question,
				payload.system_prompt,
				profile_text=profile_text,
				previous_qna=previous_qna[-10:] if previous_qna else None,
				style_mode=payload.style_mode,
				tone=payload.tone,
				layout=payload.layout,
				variability=payload.variability,
				seed=payload.seed,
				api_key=api_key,
				# Legacy behavior: allow LLMService to apply its auto system-design overrides.
				apply_auto_overrides=True,
			)
			final_ans = ("## Single-View Architecture\n\n" + (answer or "").strip()).strip()
			if payload.save_to_history:
				await manager.append_qna(state.session_id, payload.question, final_ans)
			return AnswerOut(answer=final_ans, created_at=datetime.utcnow(), truncated=truncated)
		

	
	# Regular non-architecture questions
	if payload.stream:
		async def event_gen():
			collected: list[str] = []
			# Provide recent QnA as context for follow-ups
			previous_qna = state.qna
			async for chunk in llm_service.stream_answer(
				payload.question,
				payload.system_prompt,
				profile_text=profile_text,
				previous_qna=previous_qna,
				style_mode=payload.style_mode,
				tone=payload.tone,
				layout=payload.layout,
				variability=payload.variability,
				seed=payload.seed,
				api_key=api_key,
			):
				collected.append(chunk)
				# Fix: Proper SSE encoding for multi-line chunks to prevent text loss/corruption
				# EventSource spec requires 'data: ' prefix for every line of the data payload
				safe_chunk = chunk.replace('\n', '\ndata: ')
				yield f"data: {safe_chunk}\n\n"
			# On stream end, persist the full answer
			full_answer = "".join(collected)
			if payload.save_to_history:
				await manager.append_qna(state.session_id, payload.question, full_answer)
			
			await auditor.log({
				"type": "qna",
				"session_id": state.session_id,
				"question": payload.question,
				"answer": full_answer,
				"saved_to_history": payload.save_to_history,
			})
			
			# Auto-evaluate if response contains code
			asyncio.create_task(_auto_evaluate_if_code(manager, state.session_id, payload.question, full_answer, api_key))
			
			yield "event: end\n\n"
		return StreamingResponse(event_gen(), media_type="text/event-stream")

	# Provide recent QnA as context for follow-ups
	previous_qna = state.qna
	answer, truncated = await llm_service.generate_answer(
		payload.question,
		payload.system_prompt,
		profile_text=profile_text,
		previous_qna=previous_qna,
		style_mode=payload.style_mode,
		tone=payload.tone,
		layout=payload.layout,
		variability=payload.variability,
		seed=payload.seed,
		api_key=api_key,
	)
	
	if payload.save_to_history:
		await manager.append_qna(state.session_id, payload.question, answer)
		
	await auditor.log({
		"type": "qna",
		"session_id": state.session_id,
		"question": payload.question,
		"answer": answer,
		"saved_to_history": payload.save_to_history,
	})
	
	# Auto-evaluate if response contains code
	asyncio.create_task(_auto_evaluate_if_code(manager, state.session_id, payload.question, answer, api_key))
	
	return AnswerOut(answer=answer, created_at=datetime.utcnow(), truncated=truncated)


@router.post("/upload_profile")
async def upload_profile(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    request: Request = None,
):
	# Get user-specific manager
	user_id = get_user_id_from_request(request) if request else "default"
	manager = get_session_manager(user_id)
	
	# Ensure session exists
	try:
		await manager.get_required(session_id)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found. Create one via POST /api/session and reuse its session_id.")

	# Determine how to read the file; support text and pdf minimally
	filename = (file.filename or "").lower()
	content_type = (file.content_type or "").lower()

	# Read bytes
	data = await file.read()

	text: str = ""
	try:
		if filename.endswith(".txt") or content_type.startswith("text/"):
			text = data.decode("utf-8", errors="ignore")
		elif filename.endswith(".md"):
			text = data.decode("utf-8", errors="ignore")
		elif filename.endswith(".docx") or content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
			# Handle Word documents (.docx)
			try:
				from docx import Document
				import io
				doc = Document(io.BytesIO(data))
				# Extract all paragraphs
				text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
				# Also extract text from tables
				for table in doc.tables:
					for row in table.rows:
						for cell in row.cells:
							if cell.text.strip():
								text += "\n" + cell.text
			except ImportError:
				raise HTTPException(status_code=415, detail="Word document support not available. Please install python-docx or upload a PDF/text file.")
			except Exception as e:
				raise HTTPException(status_code=415, detail=f"Unable to read Word document: {str(e)}. Please try converting to PDF or text format.")
		elif filename.endswith(".pdf") or content_type == "application/pdf":
			# Lazy import to avoid heavy dependency at startup
			try:
				from app.utils.text_extract import extract_text_from_pdf  # type: ignore
			except Exception:
				raise HTTPException(status_code=415, detail="PDF support not available. Please install pdfminer.six or upload a text/markdown file.")
			text = extract_text_from_pdf(data)
		else:
			# Fallback: try utf-8
			text = data.decode("utf-8", errors="ignore")
	except UnicodeDecodeError:
		raise HTTPException(status_code=415, detail="Unable to decode file. Please upload a UTF-8 text, markdown, PDF, or Word (.docx) file.")

	if not text.strip():
		raise HTTPException(status_code=400, detail="Uploaded file appears empty.")

	await manager.set_profile_text(session_id, text)
	await auditor.log({
		"type": "profile_upload",
		"session_id": session_id,
		"filename": file.filename,
		"bytes": len(data),
	})

	return {"status": "ok", "characters": len(text)}


@router.get("/sessions", response_model=SessionList)
async def list_sessions(request: Request):
	user_id = get_user_id_from_request(request) or "default"
	manager = get_session_manager(user_id)
	items_raw = await manager.list_sessions()
	items = [
		SessionSummary(
			session_id=i["session_id"],
			title=i["title"],
			last_update=datetime.fromisoformat(i["last_update"]),
			qna_count=i["qna_count"],
		)
		for i in items_raw
	]
	return SessionList(items=items)


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, request: Request):
	user_id = get_user_id_from_request(request) or "default"
	manager = get_session_manager(user_id)
	deleted = await manager.delete_session(session_id)
	if not deleted:
		raise HTTPException(status_code=404, detail="Session not found")
	return {"status": "ok", "deleted": True}


@router.delete("/history/{session_id}")
async def clear_history(session_id: str, request: Request):
	user_id = get_user_id_from_request(request) or "default"
	manager = get_session_manager(user_id)
	try:
		await manager.clear_history(session_id)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found")
	return {"status": "ok"}


@router.delete("/history/{session_id}/{index}")
async def delete_qna_item(session_id: str, index: int, request: Request):
	user_id = get_user_id_from_request(request) or "default"
	manager = get_session_manager(user_id)
	try:
		await manager.remove_qna(session_id, index)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found")
	except IndexError:
		raise HTTPException(status_code=400, detail="QnA index out of range")
	return {"status": "ok"}


@router.put("/session/{session_id}/title")
async def update_session_title(session_id: str, payload: UpdateSessionTitleRequest, request: Request):
	user_id = get_user_id_from_request(request) or "default"
	manager = get_session_manager(user_id)
	try:
		await manager.set_session_title(session_id, payload.title)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found")
	return {"status": "ok", "title": payload.title}


@router.get("/session/{session_id}/chat", response_model=SessionHistory)
async def get_session_chat(session_id: str, request: Request):
	"""Get chat history for a session (different from Search Intelligence history)"""
	user_id = get_user_id_from_request(request) or "default"
	manager = get_session_manager(user_id)
	try:
		state = await manager.get_required(session_id)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found")
	
	items = [
		QnA(
			question=i["question"],
			answer=i["answer"],
			created_at=datetime.fromisoformat(i["created_at"]),
		)
		for i in state.qna
	]
	return SessionHistory(session_id=session_id, items=items)
