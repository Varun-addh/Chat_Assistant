from __future__ import annotations

import re


_INTERNAL_LEAK_MARKERS = (
	"identity & attribution",
	"output formatting rule",
	"bullet format",
	"list format",
	"intent routing",
	"voice mode",
	"meta awareness",
	"adaptive depth",
	"context & memory",
	"placeholder policy",
	"core response structure",
	"response planning",
	"question type templates",
	"response style rules",
	"adaptive response policy",
	"code response modes",
	"code quality standards",
	"analyze before responding",
	"response length guidelines",
)


def _process_thinking_tags(self, text: str) -> str:
	"""Wrap <think> blocks in collapsible details tags"""
	if "<think>" in text:
		text = text.replace("<think>", "<details class='thinking-process'><summary>Thinking Process</summary>")
	if "</think>" in text:
		text = text.replace("</think>", "</details>")
	return text


def _wrap_loose_sql_blocks(self, text: str) -> str:
	"""Wrap standalone SQL snippets into fenced code blocks.

	This repairs a frequent provider formatting glitch where the model emits:
	  sql-- comment
	  INSERT INTO ...
	  VALUES (...);

	without surrounding ```sql fences, which makes the frontend render it as prose.

	Conservative rules:
	- Never touches existing fenced code blocks.
	- Only wraps runs that look strongly like SQL (keywords / semicolons / sql-- prefix).
	"""
	if not text:
		return text

	text = text.replace("\r\n", "\n").replace("\r", "\n")
	lines = text.split("\n")
	out: list[str] = []
	in_code = False

	sql_start_re = re.compile(r"^\s*(?:sql--\s*|--\s*|SELECT\b|WITH\b|INSERT\b|UPDATE\b|DELETE\b|CREATE\b|DROP\b|ALTER\b)", re.IGNORECASE)
	sql_kw_re = re.compile(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|FROM|WHERE|VALUES|JOIN|GROUP\s+BY|ORDER\s+BY)\b", re.IGNORECASE)

	def _looks_like_sql_run(run_lines: list[str]) -> bool:
		if not run_lines:
			return False
		joined = "\n".join(run_lines)
		score = 0
		# Strong signals
		if re.search(r"^\s*sql--\s*", run_lines[0], re.IGNORECASE):
			score += 3
		if re.search(r"^\s*(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b", run_lines[0], re.IGNORECASE):
			score += 3
		# General SQL-ish traits
		score += 2 * len(sql_kw_re.findall(joined))
		if ";" in joined:
			score += 2
		# Require at least 2 lines unless it's an extremely strong single-liner.
		if len(run_lines) >= 2 and score >= 5:
			return True
		if len(run_lines) == 1 and score >= 9:
			return True
		return False

	i = 0
	while i < len(lines):
		line = lines[i]
		stripped = line.strip()

		if stripped.startswith("```"):
			in_code = not in_code
			out.append(line)
			i += 1
			continue

		if in_code:
			out.append(line)
			i += 1
			continue

		# Candidate start of a loose SQL run.
		if stripped and sql_start_re.search(stripped):
			j = i
			run: list[str] = []
			while j < len(lines):
				s = lines[j].strip()
				if not s:
					break
				# Stop before headings/lists/new fences.
				if s.startswith(("```", "#", "- ", "* ", "+ ")):
					break
				run.append(lines[j])
				j += 1

			if _looks_like_sql_run(run):
				# Normalize the common 'sql--' prefix to a real SQL comment.
				norm_run = [re.sub(r"^\s*sql--\s*", "-- ", l, flags=re.IGNORECASE) for l in run]
				out.append("```sql")
				out.extend([l.rstrip() for l in norm_run])
				out.append("```")
				i = j
				continue

		out.append(line)
		i += 1

	return "\n".join(out)


def _drop_empty_example_code_blocks(self, text: str) -> str:
	"""Remove fenced code blocks that contain only 'Example:' (or are empty).

	Some provider outputs accidentally emit empty code fences like:
	```\nExample:\n```
	which the frontend renders as an empty CODE card.

	Never touches non-empty code blocks.
	"""
	if not text:
		return text

	text = text.replace("\r\n", "\n").replace("\r", "\n")
	lines = text.split("\n")
	out: list[str] = []
	i = 0
	while i < len(lines):
		line = lines[i]
		stripped = line.strip()
		if not stripped.startswith("```"):
			out.append(line)
			i += 1
			continue

		# Capture a fenced block
		fence_open = line
		block: list[str] = []
		i += 1
		found_close = False
		while i < len(lines):
			l = lines[i]
			if l.strip().startswith("```"):
				found_close = True
				break
			block.append(l)
			i += 1

		# If no closing fence, treat as plain text.
		if not found_close:
			out.append(fence_open)
			out.extend(block)
			break

		fence_close = lines[i]
		content = "\n".join(block).strip()
		content_lower = content.lower()
		if content_lower in ("", "example:", "example"):
			# Drop the whole fenced block.
			i += 1
			continue

		out.append(fence_open)
		out.extend(block)
		out.append(fence_close)
		i += 1

	return "\n".join(out)


def _fix_markdown_syntax(self, text: str) -> str:
	"""Fix common broken markdown syntax artifacts (code-fence-aware)."""
	import re

	if not text:
		return text

	lines = text.split("\n")
	out: list[str] = []
	in_code = False
	for line in lines:
		stripped = line.strip()
		if stripped.startswith("```"):
			in_code = not in_code
			out.append(line)
			continue
		if in_code:
			out.append(line)
			continue

		# Fix double colons :: -> : (safe for non-code text)
		line = re.sub(r"(\w)::\s", r"\1: ", line)
		line = re.sub(r"(\w)::$", r"\1:", line)

		# Fix unclosed bold tags near colons: "**Label: value" -> "**Label:** value"
		line = re.sub(r"(\*\*[^*\n]+:)(?!\*\*)", r"\1**", line)

		out.append(line)
	return "\n".join(out)


def _split_runon_plus_bullets(self, text: str) -> str:
	"""Split run-on '+ ' bullets that appear on a single line.

	Some models occasionally emit multiple bullets like:
	  + Item A + Item B + Item C
	which renders poorly in the UI. This normalizes them to:
	  - Item A
	  - Item B
	  - Item C

	We apply this only outside fenced code blocks and only when the line
	*starts* with '+ ' and contains at least two '+ ' bullet segments.
	"""
	import re

	lines = text.split("\n")
	out: list[str] = []
	in_code = False
	for line in lines:
		stripped = line.strip()
		if stripped.startswith("```"):
			in_code = not in_code
			out.append(line)
			continue
		if in_code:
			out.append(line)
			continue

		if stripped.startswith("+ ") and " + " in stripped:
			# Capture repeated '+ <text>' segments on the same line.
			segs = re.findall(r"(?:^|\s)(\+\s+[^+]+?)(?=(?:\s\+\s)|$)", stripped)
			if len(segs) >= 2:
				for seg in segs:
					item = seg.strip()[2:].strip()  # drop leading '+ '
					if item:
						out.append(f"- {item}")
				continue

		out.append(line)
	return "\n".join(out)


def _is_internal_prompt_leak_line(self, line: str) -> bool:
	"""Best-effort detection of system-prompt leakage.

	We only filter highly-specific internal headings/instructions that should
	never be shown to the user.
	"""
	s = (line or "").strip()
	if not s:
		return False

	# Normalize common markdown prefixes (headings, quotes, list markers, numbering, bold).
	# Keep this conservative to avoid removing real content.
	norm = re.sub(r"^\s*(?:[>#\-*]+\s*)?(?:\d+\.?\s*)?", "", s)
	norm = norm.strip()
	if norm.startswith("**"):
		norm = norm.strip("*")
	norm_lower = norm.lower().strip()

	if "internal only" in norm_lower:
		return True
	if "mandatory - internal only" in norm_lower:
		return True

	return any(marker in norm_lower for marker in _INTERNAL_LEAK_MARKERS)


def _strip_internal_prompt_leakage(self, text: str) -> str:
	"""Remove internal instruction headings that a model may accidentally echo.

	Skips fenced code blocks entirely.
	"""
	if not text:
		return ""
	lines = text.split("\n")
	out: list[str] = []
	in_code = False
	for line in lines:
		stripped = line.strip()
		if stripped.startswith("```"):
			in_code = not in_code
			out.append(line)
			continue
		if in_code:
			out.append(line)
			continue
		if self._is_internal_prompt_leak_line(line):
			continue
		out.append(line)
	return "\n".join(out).strip()


def _remove_forbidden_side_headings(self, text: str) -> str:
	"""Remove specific side-heading labels that must never appear to users.

	Targets labels like: Definition, Why it matters, Concrete example (and simple variants).
	Skips fenced code blocks and preserves the remainder of the line when heading and content
	appear on the same line; drops pure-heading lines.
	"""
	if not text:
		return ""
	import re

	lines = text.split("\n")
	out: list[str] = []
	in_code = False
	# Pattern: optional heading markers, optional list marker, optional bold, heading phrase, optional colon/dash, optional trailing content
	pat = re.compile(
		r"^\s*(?:#{1,6}\s*)?(?:[-\*\+]\s+)?(?:\*\*)?\s*(Definition|Why it matters|Concrete example)s?\s*[:\-–—]?\s*(.*)$",
		flags=re.IGNORECASE,
	)

	for line in lines:
		stripped = line.strip()
		if stripped.startswith("```"):
			in_code = not in_code
			out.append(line)
			continue
		if in_code:
			out.append(line)
			continue
		m = pat.match(line)
		if m:
			# If there's content after the heading on the same line, keep it (preserve indentation)
			rest = m.group(2) or ""
			if rest.strip():
				# Preserve original leading whitespace
				leading_ws = re.match(r"^(\s*)", line).group(1) or ""
				out.append(leading_ws + rest.strip())
			# If it was a pure heading line, drop it (do not append)
			continue
		out.append(line)
	return "\n".join(out)


def _break_wall_of_text_paragraphs(self, text: str) -> str:
	"""Break wall-of-text paragraphs into readable, structured content.

	Detection: any non-code, non-heading, non-bullet paragraph with more than
	~80 words is considered a "wall of text" and gets split at sentence
	boundaries into short paragraphs separated by blank lines.

	This acts as a backend safety net for when the LLM ignores the
	RESPONSE_CONTRACT anti-wall-of-text policy.
	"""
	import re

	if not text:
		return ""

	WORD_THRESHOLD = 80  # words before we consider it a wall of text

	lines = text.split("\n")
	out: list[str] = []
	in_code = False

	i = 0
	while i < len(lines):
		line = lines[i]
		stripped = line.strip()

		# Track fenced code blocks — never touch them.
		if stripped.startswith("```"):
			in_code = not in_code
			out.append(line)
			i += 1
			continue
		if in_code:
			out.append(line)
			i += 1
			continue

		# Skip headings, bullets, empty lines — they're fine as-is.
		if (
			not stripped
			or stripped.startswith("#")
			or stripped.startswith("- ")
			or stripped.startswith("* ")
			or stripped.startswith("+ ")
			or re.match(r"^\d+\.\s", stripped)
			or stripped.startswith("|")
		):
			out.append(line)
			i += 1
			continue

		# ── Candidate paragraph — check word count ──
		word_count = len(stripped.split())
		if word_count <= WORD_THRESHOLD:
			out.append(line)
			i += 1
			continue

		# ── Wall of text detected — split at sentence boundaries ──
		# Split on sentence-ending punctuation followed by a space.
		sentences = re.split(r'(?<=[.!?])\s+', stripped)

		if len(sentences) <= 2:
			# Only 1-2 very long sentences — not much we can do structurally.
			out.append(line)
			i += 1
			continue

		# Group sentences into short paragraphs (2-3 sentences each).
		chunk: list[str] = []
		chunk_words = 0
		for sentence in sentences:
			s_words = len(sentence.split())
			chunk.append(sentence)
			chunk_words += s_words
			if chunk_words >= 35 or len(chunk) >= 3:
				out.append(" ".join(chunk))
				out.append("")  # blank line between paragraphs
				chunk = []
				chunk_words = 0
		if chunk:
			out.append(" ".join(chunk))

		i += 1

	return "\n".join(out)


def _rewrite_onboarding_brochure_if_present(self, text: str) -> str:
	"""Rewrite common brochure-style onboarding into a natural chat reply.

	This targets the frequent pattern of:
	- 'Introduction'
	- 'How I Can Assist You'
	- 'Getting Started'
	- 'Example Questions'

	We only rewrite when multiple of these markers appear, to avoid harming
	legitimate content about introductions of a technical topic.
	"""
	if not text:
		return ""

	# Work only outside fenced code blocks.
	lines = text.split("\n")
	out: list[str] = []
	in_code = False
	for line in lines:
		stripped = line.strip()
		if stripped.startswith("```"):
			in_code = not in_code
			out.append(line)
			continue
		if in_code:
			out.append(line)
			continue
		out.append(line)
	text_no_code = "\n".join(out)

	markers = [
		"introduction",
		"how i can assist you",
		"getting started",
		"example questions",
		"some example questions",
	]
	hits = sum(1 for m in markers if m in text_no_code.lower())
	if hits < 2:
		return text

	# Debrochureize: remove the generic section headings, keep the lead paragraph.
	heading_re = re.compile(
		r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*(introduction|how i can assist you|getting started|example questions|some example questions)\s*(?:\*\*)?\s*$",
		flags=re.IGNORECASE,
	)
	clean_lines: list[str] = []
	in_code = False
	for line in text.split("\n"):
		stripped = line.strip()
		if stripped.startswith("```"):
			in_code = not in_code
			clean_lines.append(line)
			continue
		if in_code:
			clean_lines.append(line)
			continue
		# Drop brochure headings (keep everything else)
		if heading_re.match(stripped):
			continue
		clean_lines.append(line)

	# Extract lead paragraph (first non-empty block)
	lead_lines: list[str] = []
	for line in clean_lines:
		if not line.strip():
			if lead_lines:
				break
			continue
		lead_lines.append(line.strip())
	lead = " ".join(lead_lines).strip()
	# If the lead is huge, keep just the first sentence-ish chunk.
	if len(lead) > 260:
		m = re.split(r"(?<=[.!?])\s+", lead, maxsplit=1)
		lead = (m[0] if m else lead[:260]).strip()
	if not lead:
		lead = "Got it."

	# Add a single, natural follow-up question.
	follow_up = "What would you like to practice next—coding, system design, behavioral, or interview strategy?"
	if lead.endswith("?"):
		return lead
	return f"{lead} {follow_up}"


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
	# Repair common provider formatting glitches before normalization.
	text = self._wrap_loose_sql_blocks(text)
	text = self._drop_empty_example_code_blocks(text)

	# Process thinking tags first to ensure they wrap correctly
	text = self._process_thinking_tags(text)
	# Normalize common bullet rendering issues for markdown UIs
	text = self._normalize_markdown_bullets(text)
	# Convert common 'Label: value' enumerations into proper markdown lists
	text = self._normalize_colon_label_lists(text)
	# Some UIs do not render markdown emphasis; also guard against unbalanced markers
	text = self._fix_unbalanced_markdown_emphasis(text)

	# Sanitize known internal artifacts (Google/Gemini protobuf leakage)
	if "_compiler" in text or "PROTOBUF" in text:
		import re

		# Remove common hallucinated patterns
		text = re.sub(r"_compiler\s+\w+_compiler", "", text)
		text = re.sub(r"malaPROTOBUFiada", "", text)
		text = re.sub(r"asamiadairsi", "", text)
		# Clean up any remaining weirdness if detected
		if "PROTOBUF" in text:
			text = re.sub(r"\S*PROTOBUF\S*", "", text)

	# Remove forbidden side-headings like 'Definition:', 'Why it matters:', 'Concrete example:'
	# This ensures these label-headings are never shown to end users (strip them conservatively).
	text = self._remove_forbidden_side_headings(text)
	# Break wall-of-text paragraphs into readable chunks
	text = self._break_wall_of_text_paragraphs(text)
	# Ensure summary sections are properly formatted (remove bullet conversion logic)
	text = self._format_summary_sections(text)
	# Enforce unlabeled bullets inside Complete Answer
	text = self._strip_labeled_bullets_in_complete_answer(text)

	# If content includes a Mermaid diagram, normalize and return as-is (don't treat as code)
	if self._contains_mermaid(text):
		return self._structural_integrity_check(self._normalize_mermaid_blocks(text))

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
		return self._structural_integrity_check(text)

	# Check if this is explanation content that should use text formatting, not tables
	if self._is_explanation_content(text):
		# For explanation content, convert table-like markdown artifacts conservatively
		text = self._clean_explanation_formatting(text)
		text = self._split_runon_plus_bullets(text)
		# Preserve bold emphasis for headings, side headings, and keywords
		# Ensure headings are still bolded
		text = self._format_headings_bold(text)
		# Remove LaTeX math markers for readability
		text = self._strip_latex_math(text)
		return self._structural_integrity_check(text)

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
	# Split any run-on '+ ' bullets (rare, but ugly in chat UIs)
	text = self._split_runon_plus_bullets(text)
	# Sanitize Mermaid syntax to fix parse errors (must come before normalization)

	text = self._sanitize_mermaid_syntax(text)

	# Normalize any Mermaid code blocks so each statement is on its own line
	text = self._normalize_mermaid_blocks(text)

	# Final cleanup for broken markdown syntax
	text = self._fix_markdown_syntax(text)

	# Guard: rewrite brochure-style onboarding into a natural chat reply.
	text = self._rewrite_onboarding_brochure_if_present(text)

	# Final guardrail: strip any accidental system-prompt leakage.
	text = self._strip_internal_prompt_leakage(text)

	# ── Enterprise-grade structural integrity check ──
	# Validates & auto-repairs: balanced fences, emphasis markers,
	# Mermaid block hygiene, and response size limits.
	text = self._structural_integrity_check(text)

	return text


def _normalize_colon_label_lists(self, text: str) -> str:
	"""Convert consecutive 'Label: value' lines into Markdown list items.

	Many model answers enumerate features like:
	  Platform Independence: ...
	  Object-Oriented: ...
	Without leading '- ' markers, some UIs won't render bullets.

	This pass is conservative:
	- Never touches fenced code blocks.
	- Only converts runs with 2+ label-lines (ignores singletons like 'Confidence: 20%').
	- Does not modify existing list items/headings.
	"""
	if not text:
		return text

	text = text.replace("\r\n", "\n").replace("\r", "\n")
	lines = text.split("\n")
	out: list[str] = []
	in_code = False

	import re

	bold_label_re = re.compile(r"^\s*\*\*([^*]{2,80})\*\*\s*:\s+\S")
	plain_label_re = re.compile(r"^\s*([A-Z][A-Za-z0-9 /&+._-]{1,80}?)\s*:\s+\S")

	def _is_list_or_heading(s: str) -> bool:
		s2 = s.lstrip()
		return s2.startswith(("- ", "* ", "+ ", "#"))

	def _to_bullet(s: str) -> str:
		# Preserve indentation minimally (strip leading spaces).
		s_stripped = s.strip()
		m = bold_label_re.match(s_stripped)
		if m:
			# Preserve bold: '**Label**: rest' → '- **Label:** rest'
			label = m.group(1).strip()
			rest = s_stripped.split(":", 1)[1].strip() if ":" in s_stripped else ""
			return f"- **{label}:** {rest}"
		m2 = plain_label_re.match(s_stripped)
		if m2:
			label = m2.group(1).strip()
			rest = s_stripped.split(":", 1)[1].strip()
			return f"- **{label}:** {rest}"
		return "- " + s_stripped

	i = 0
	while i < len(lines):
		line = lines[i]
		stripped = line.strip()

		if stripped.startswith("```"):
			in_code = not in_code
			out.append(line)
			i += 1
			continue

		if in_code:
			out.append(line)
			i += 1
			continue

		# Start a potential run only when this line looks like a label-line.
		if stripped and (not _is_list_or_heading(line)) and (bold_label_re.match(stripped) or plain_label_re.match(stripped)):
			j = i
			collected: list[str] = []
			# Allow blank lines between items in the run, but don't emit them.
			while j < len(lines):
				l = lines[j]
				s = l.strip()
				if not s:
					j += 1
					continue
				if _is_list_or_heading(l):
					break
				if bold_label_re.match(s) or plain_label_re.match(s):
					collected.append(l)
					j += 1
					continue
				break

			if len(collected) >= 2:
				for item in collected:
					out.append(_to_bullet(item))
				i = j
				continue

		# Default: passthrough
		out.append(line)
		i += 1

	return "\n".join(out)


def _fix_unbalanced_markdown_emphasis(self, text: str) -> str:
	"""Remove obviously broken emphasis markers that show up as raw '**' or '*' in the UI.

	If a non-code line contains an odd number of '**' or '*' markers, markdown renderers
	often display the raw markers. This pass removes the marker characters on that line.

	Never modifies fenced code blocks.
	"""
	if not text:
		return text
	text = text.replace("\r\n", "\n").replace("\r", "\n")
	out: list[str] = []
	in_code = False
	for line in text.split("\n"):
		stripped = line.strip()
		if stripped.startswith("```"):
			in_code = not in_code
			out.append(line)
			continue
		if in_code:
			out.append(line)
			continue

		# Fix unbalanced '**' — only strip the LAST orphan marker, not all
		if line.count("**") % 2 == 1:
			# Find the last '**' and remove only it (preserves earlier valid pairs)
			idx = line.rfind("**")
			if idx >= 0:
				line = line[:idx] + line[idx + 2:]
		# Fix unbalanced single '*' — only strip the last orphan
		# Skip list markers ('* ') and avoid destroying content like *args
		lstrip = line.lstrip()
		if not lstrip.startswith("* "):
			# Count only isolated emphasis markers (not inside words)
			emphasis_count = len(re.findall(r'(?<![\w*])\*(?![\w*])|(?<![\w*])\*(?=\w)', line))
			if emphasis_count % 2 == 1:
				# Remove only the last orphan '*' that looks like a broken emphasis
				line = re.sub(r'\*(?=[^*]*$)', '', line, count=1)

		out.append(line)
	return "\n".join(out)


def _normalize_markdown_bullets(self, text: str) -> str:
	"""Normalize common list formatting issues into Markdown hyphen bullets.

	Fixes patterns that break markdown renderers in chat UIs, e.g.:
	- A unicode bullet marker on its own line ("•"), followed by the bullet text on the next line.
	- Lines that start with unicode bullet markers like "•Item" or "• Item".

	Never modifies fenced code blocks.
	"""
	if not text:
		return text

	# Normalize newlines early; JSON may already contain real newlines, but this keeps behavior stable.
	text = text.replace("\r\n", "\n").replace("\r", "\n")

	bullet_markers = {"•", "·", "‣", "◦"}
	soft_bullet_markers = bullet_markers | {"-", "*"}

	out_lines: list[str] = []
	in_code = False
	pending_bullet = False

	for raw_line in text.split("\n"):
		line = raw_line
		stripped = line.strip()

		# Toggle fenced code blocks; keep content untouched.
		if stripped.startswith("```"):
			in_code = not in_code
			out_lines.append(line)
			pending_bullet = False
			continue

		if in_code:
			out_lines.append(line)
			continue

		# If we saw a dangling bullet marker line, attach the next meaningful line to it.
		if pending_bullet:
			if not stripped:
				# Keep blank lines while waiting for content.
				out_lines.append(line)
				continue
			# If the next line is already a list item/heading, drop the dangling marker.
			if stripped.startswith(("- ", "* ", "+ ", "#")) or stripped in soft_bullet_markers:
				pending_bullet = False
				out_lines.append(line)
				continue
			out_lines.append("- " + stripped)
			pending_bullet = False
			continue

		# Case 1: unicode bullet line by itself.
		if stripped in soft_bullet_markers:
			pending_bullet = True
			continue

		# Case 2: line starts with a unicode bullet marker.
		lstripped = line.lstrip()
		if lstripped.startswith(tuple(bullet_markers)):
			# Preserve indentation where possible.
			indent = line[: len(line) - len(lstripped)]
			after = lstripped[1:].lstrip()
			if after:
				out_lines.append(f"{indent}- {after}")
			else:
				pending_bullet = True
			continue

		out_lines.append(line)

	# If the text ended with a dangling bullet marker, just drop it.
	return "\n".join(out_lines)


def _format_headings_bold(self, text: str) -> str:
	"""Ensure all headings are properly bolded, but never touch fenced code blocks."""
	import re

	lines = text.split("\n")
	formatted_lines = []
	in_code = False

	for line in lines:
		stripped = line.strip()
		# Toggle code fence regions
		if stripped.startswith("```"):
			in_code = not in_code
			formatted_lines.append(line)
			continue

		if not in_code and stripped.startswith(("##", "###", "####")):
			# Extract the heading text (remove the ##, ###, etc.)
			heading_match = re.match(r"^(#{2,4})\s*(.+)$", stripped)
			if heading_match:
				hashes, heading_text = heading_match.groups()
				# Check if already bolded
				if not heading_text.strip().startswith("**") or not heading_text.strip().endswith("**"):
					formatted_line = f"{hashes} **{heading_text.strip()}**"
					formatted_lines.append(formatted_line)
				else:
					formatted_lines.append(line)
			else:
				formatted_lines.append(line)
		else:
			formatted_lines.append(line)

	return "\n".join(formatted_lines)


def _strip_latex_math(self, text: str) -> str:
	r"""Remove LaTeX math markers ($...$, \(...\), \[...\]) from non-code blocks while preserving inner text.
	Skips fenced code blocks entirely."""
	import re

	lines = text.split("\n")
	out: list[str] = []
	in_code = False
	for line in lines:
		stripped = line.strip()
		if stripped.startswith("```"):
			in_code = not in_code
			out.append(line)
			continue
		if in_code:
			out.append(line)
			continue
		# Replace inline math markers
		newline = re.sub(r"\$(.*?)\$", r"\1", line)
		newline = re.sub(r"\\\((.*?)\\\)", r"\1", newline)
		newline = re.sub(r"\\\[(.*?)\\\]", r"\1", newline, flags=re.DOTALL)
		out.append(newline)
	return "\n".join(out)


def _sanitize_mermaid_syntax(self, text: str) -> str:
	"""Sanitize Mermaid diagrams to fix common syntax errors that cause parse failures."""
	import re

	lines = text.split("\n")
	sanitized_lines = []
	in_mermaid = False

	for line in lines:
		# Detect mermaid code blocks
		if line.strip().startswith("```mermaid"):
			in_mermaid = True
			sanitized_lines.append(line)
			continue
		elif line.strip() == "```" and in_mermaid:
			in_mermaid = False
			sanitized_lines.append(line)
			continue

		if in_mermaid:
			# Fix common syntax issues
			# 1. Remove double colons (::) - often causes "Expecting SEMI" errors
			line = line.replace("::", "-")

			# 2. Fix slashes in node labels: [text/with/slashes] -> ["text-with-slashes"]
			line = re.sub(r"\[([^\]]*)/([^\]]*)\]", r"[\"\1-\2\"]", line)
			# Handle multiple slashes
			while "/" in line and "[" in line:
				old_line = line
				line = re.sub(r"\[([^\]]*)/([^\]]*)\]", r"[\"\1-\2\"]", line)
				if old_line == line:
					break

			# 3. Fix colons in node labels: [text:with:colons] -> ["text-with-colons"]
			line = re.sub(r"\[([^\]]*):([^\]]*)\]", r"[\"\1-\2\"]", line)
			# Handle multiple colons
			while ":" in line and "[" in line and "::" not in line:
				old_line = line
				line = re.sub(r"\[([^\]]*):([^\]]*)\]", r"[\"\1-\2\"]", line)
				if old_line == line:
					break

			# 4. Ensure proper spacing around arrows
			line = re.sub(r"(\S)(-->)(\S)", r"\1 \2 \3", line)
			line = re.sub(r"(\S)(--->)(\S)", r"\1 \2 \3", line)

		sanitized_lines.append(line)

	return "\n".join(sanitized_lines)


def _normalize_mermaid_blocks(self, text: str) -> str:
	"""Normalize Mermaid blocks without changing their content semantics."""
	import re

	_NON_FLOWCHART_TYPES = {
		"sequencediagram", "classdiagram", "statediagram",
		"erdiagram", "journey", "gantt", "pie", "gitgraph",
		"mindmap", "timeline",
	}

	def _is_non_flowchart(code: str) -> bool:
		first_line = code.strip().split("\n")[0].strip().lower() if code.strip() else ""
		return any(first_line.startswith(t) for t in _NON_FLOWCHART_TYPES)

	def _basic_cleanup(code: str) -> str:
		"""Light cleanup for non-flowchart diagrams — preserve structure as-is."""
		c = code.strip()
		c = c.replace("`mermaid", "").replace("```", "").replace("`", "")
		# Remove blank lines but keep the diagram intact
		lines = [l for l in c.split("\n") if l.strip()]
		return "\n".join(lines)

	def normalize_block(code: str) -> str:
		"""Bulletproof Mermaid normalizer that completely rebuilds valid syntax."""
		c = code.strip()

		# Remove stray backtick artifacts
		c = c.replace("`mermaid", "").replace("```", "").replace("`", "")

		# Clean up any leading/trailing whitespace and newlines
		c = c.strip()

		# Fix Mermaid syntax issues with special characters in labels
		# Remove parentheses from node labels (Mermaid doesn't handle them well)
		c = re.sub(r"\[([^\]]*?)\(([^)]*?)\)([^\]]*?)\]", r"[\1\2\3]", c)
		# Handle multiple parentheses in the same label
		c = re.sub(r"\[([^\]]*?)\(([^)]*?)\)([^\]]*?)\(([^)]*?)\)([^\]]*?)\]", r"[\1\2\3\4\5]", c)
		# Clean up any remaining parentheses in labels
		c = re.sub(r"\[([^\]]*?)\(([^)]*?)\)([^\]]*?)\]", r"[\1\2\3]", c)
		# Remove parentheses from subgraph names
		c = re.sub(r"subgraph\s+([^[]*?)\(([^)]*?)\)([^[]*?)\[", r"subgraph \1\2\3[", c)
		# Clean up any remaining parentheses in subgraph names
		c = re.sub(r"subgraph\s+([^[]*?)\(([^)]*?)\)([^[]*?)\[", r"subgraph \1\2\3[", c)

		# Extract flowchart type - preserve the original direction
		flowchart_match = re.match(r"^(flowchart\s+[A-Z]{2})", c)
		flowchart_type = flowchart_match.group(1) if flowchart_match else "flowchart LR"

		# Remove flowchart declaration
		remaining = re.sub(r"^flowchart\s+[A-Z]{2}\s*", "", c).strip()

		formatted_lines = [flowchart_type]

		# Extract classDef and class statements first to avoid duplication
		classdef_pattern = r"classDef\s+([^;]+)"
		classdef_matches = re.findall(classdef_pattern, c)
		classdef_statements = [f"classDef {classdef.strip()}" for classdef in classdef_matches]

		class_pattern = r"class\s+([^;]+)"
		class_matches = re.findall(class_pattern, c)
		class_statements = [f"class {class_stmt.strip()}" for class_stmt in class_matches]

		# Remove classDef and class statements from remaining content to avoid duplication
		remaining = re.sub(r"classDef\s+[^;]+;?", "", remaining)
		remaining = re.sub(r"class\s+[^;]+;?", "", remaining)

		# Process the content line by line to preserve structure
		lines = remaining.split("\n")
		in_subgraph = False
		subgraph_depth = 0

		for line in lines:
			line = line.strip()
			if not line:
				continue

			# Skip flowchart declaration as it's already added
			if re.match(r"^(flowchart\s+[A-Z]{2}|sequenceDiagram|classDiagram|erDiagram|stateDiagram|gantt|journey|pie|mindmap|timeline)\s*", line):
				continue

			# Check if this line starts a subgraph
			subgraph_match = re.match(r"subgraph\s+(.+)", line)
			if subgraph_match:
				subgraph_name = subgraph_match.group(1).strip()
				# Ensure subgraph name is properly formatted
				if not subgraph_name.endswith("]") and "[" in subgraph_name:
					# Add missing closing bracket if needed
					subgraph_name += "]"
				formatted_lines.append(f"subgraph {subgraph_name}")
				in_subgraph = True
				subgraph_depth += 1
				continue

			# Check if this line ends a subgraph
			if line == "end" and in_subgraph:
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
		result = "\n".join(formatted_lines)

		# Final cleanup
		result = re.sub(r"\n\s*\n", "\n", result)
		result = re.sub(r"^\s*\n", "", result)
		result = result.strip()

		return result

	lines = text.split("\n")
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
			raw_block = "\n".join(buffer)
			# Only apply the flowchart-specific normalizer to flowchart/graph diagrams.
			# Non-flowchart diagrams (sequence, ER, class, etc.) get light cleanup only.
			if _is_non_flowchart(raw_block):
				normalized = _basic_cleanup(raw_block)
			else:
				normalized = normalize_block(raw_block)
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
	"""Within the '## Complete Answer' section, remove leading label patterns like '**Label:** ' or 'Label:' from each bullet."""
	import re

	lines = text.split("\n")
	out: list[str] = []
	in_complete = False
	for i, line in enumerate(lines):
		header = line.strip().lower()
		if header.startswith("## ") and "complete answer" in header:
			in_complete = True
			out.append(line)
			continue
		# Exit when next top-level header begins
		if in_complete and line.strip().startswith("## ") and "complete answer" not in header:
			in_complete = False
			out.append(line)
			continue
		if in_complete and line.lstrip().startswith(("-", "*")):
			bullet = line
			# Remove patterns like '- **Label:** text' or '- Label: text'
			bullet = re.sub(r"^([\-\*]\s+)(\*\*[^*:]{1,40}\*\*:\s*)", r"\1", bullet)
			bullet = re.sub(r"^([\-\*]\s+)([^*:]{1,40}:\s*)", r"\1", bullet)
			out.append(bullet)
		else:
			out.append(line)
	return "\n".join(out)


def _strip_leading_bold_labels_globally(self, text: str) -> str:
	"""Remove leading bold label patterns at the start of list items anywhere in the document."""
	import re

	lines = text.split("\n")
	out: list[str] = []
	in_code = False
	for line in lines:
		if line.strip().startswith("```"):
			in_code = not in_code
			out.append(line)
			continue
		if in_code:
			out.append(line)
			continue
		m1 = re.match(r"^(\s*[\-\*]\s+)\*\*([^*]{1,80})\*\*\s*:\s*(.*)$", line)
		if m1:
			prefix, label, rest = m1.groups()
			out.append(f"{prefix}{rest}".rstrip())
			continue
		m2 = re.match(r"^(\s*[\-\*]\s+)\*\*([^*]{1,80})\*\*\s+(.*)$", line)
		if m2:
			prefix, label, rest = m2.groups()
			# Keep label plain, drop bold
			out.append(f"{prefix}{label} {rest}".rstrip())
			continue
		out.append(line)
	return "\n".join(out)


def _deplaceholderize(self, text: str) -> str:
	"""Convert bracketed placeholders like [SPECIFIC FEATURE/PROJECT TASK] into neutral, readable text."""
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

	# Replace all [ ... ] occurrences EXCEPT markdown links [text](url)
	return re.sub(r"\[([^\]]{1,80})\](?!\()", repl, text)


def _format_summary_sections(self, text: str) -> str:
	"""Format comprehensive summary sections for interview scenarios - ensure they are prominent and complete"""
	import re

	# Look for comprehensive summary sections and ensure they're properly formatted
	summary_patterns = [
		r"##\s*(Complete\s+Answer|Summary|Overview|Comprehensive\s+Answer)",
		r"###\s*(Complete\s+Answer|Summary|Overview|Comprehensive\s+Answer)",
		r"#\s*(Complete\s+Answer|Summary|Overview|Comprehensive\s+Answer)",
		r"##\s*(Quick\s+Answer|Quick\s+Summary)",  # Keep backward compatibility
		r"###\s*(Quick\s+Answer|Quick\s+Summary)",
		r"#\s*(Quick\s+Answer|Quick\s+Summary)",
	]

	lines = text.split("\n")
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
			if not line.startswith("##"):
				line = re.sub(r"^#+\s*", "## ", line)
			formatted_lines.append(line)

			# Look for the content after the header
			j = i + 1
			summary_content = []
			while j < len(lines) and not lines[j].strip().startswith("#"):
				if lines[j].strip():
					summary_content.append(lines[j].strip())
				j += 1

			# Keep the model's own summary format; do not auto-convert
			if summary_content:
				for line_part in summary_content:
					formatted_lines.append(line_part)
				formatted_lines.append("")

			i = j
		else:
			formatted_lines.append(line)
			i += 1

	return "\n".join(formatted_lines)


def _clean_table_markdown_artifacts(self, text: str) -> str:
	"""Clean up markdown artifacts specifically in table content"""
	lines = text.split("\n")
	cleaned_lines = []

	for line in lines:
		# Check if this line is part of a table
		if self._is_table_line(line):
			# Clean up markdown artifacts in table lines
			cleaned_line = self._clean_table_line(line)
			cleaned_lines.append(cleaned_line)
		else:
			cleaned_lines.append(line)

	return "\n".join(cleaned_lines)


def _is_table_line(self, line: str) -> bool:
	"""Check if a line is part of a table"""
	import re

	# Check for pipe-separated table lines
	if "|" in line and line.count("|") >= 2:
		return True

	# Check for table separator lines
	if re.match(r"^\s*\|[\s\-:]+\|", line):
		return True

	return False


def _clean_table_line(self, line: str) -> str:
	"""Clean up markdown artifacts in a single table line"""
	import re

	# Check if this is a heading line - preserve bold formatting for headings
	if line.strip().startswith(("##", "###", "####")):
		# For headings, preserve bold formatting
		return line

	# Remove all markdown bold formatting (**text**) for non-heading lines
	line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)

	# Remove all markdown italic formatting (*text*)
	line = re.sub(r"\*([^*]+)\*", r"\1", line)

	# Remove any remaining single asterisks
	line = re.sub(r"\*", "", line)

	# Clean up extra spaces around pipes
	line = re.sub(r"\s*\|\s*", "|", line)

	# Ensure proper spacing around pipes for readability
	line = re.sub(r"\|", " | ", line)

	# Remove leading/trailing spaces
	line = line.strip()

	return line


def _format_tables(self, text: str) -> str:
	"""Format tabular data into proper markdown tables"""
	lines = text.split("\n")
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

	return "\n".join(formatted_lines)


def _is_table_row(self, line: str) -> bool:
	"""Check if a line looks like a table row"""
	# Check for pipe-separated values
	if "|" in line and line.count("|") >= 2:
		return True

	return False


def _looks_like_pipe_table(self, text: str) -> bool:
	"""Detect if text contains a valid pipe table with header and at least one row."""
	import re

	lines = [l.strip() for l in text.split("\n")]
	for i in range(len(lines) - 2):
		if "|" in lines[i] and "|" in lines[i + 1]:
			# header and separator or another row
			if re.match(r"^\|?\s*[^|]+\s*(\|[^|]+)+\|?$", lines[i]) and ("---" in lines[i + 1] or "|" in lines[i + 1]):
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
	if "|" in first_line:
		# Clean up existing pipe formatting and remove markdown artifacts
		cleaned_lines = []
		for line in table_lines:
			# Remove markdown formatting first
			cleaned = self._clean_table_line(line)
			# Remove extra spaces around pipes
			cleaned = re.sub(r"\s*\|\s*", "|", cleaned.strip())
			# Remove leading/trailing pipes if they exist
			cleaned = cleaned.strip("|")
			cleaned_lines.append(cleaned)

		# Create markdown table
		if cleaned_lines:
			# Header row
			header = cleaned_lines[0]
			# Determine number of columns
			columns = header.count("|") + 1
			separator = "|" + "|".join(["---"] * columns) + "|"

			# Format header with proper spacing
			formatted_header = "|" + header + "|"

			# Format data rows
			data_rows = []
			for line in cleaned_lines[1:]:
				formatted_row = "|" + line + "|"
				data_rows.append(formatted_row)

			# Combine into markdown table
			table_parts = [formatted_header, separator] + data_rows
			return "\n".join(table_parts)

	# Handle space-separated tables
	else:
		# Parse space-separated data
		rows = []
		for line in table_lines:
			# Clean up markdown artifacts first
			cleaned_line = self._clean_table_line(line)
			# Split by multiple spaces
			columns = re.split(r"\s{2,}", cleaned_line.strip())
			if len(columns) >= 2:
				rows.append(columns)

		if rows:
			# Determine max columns
			max_cols = max(len(row) for row in rows)

			# Pad rows to same length
			padded_rows = []
			for row in rows:
				padded_row = row + [""] * (max_cols - len(row))
				padded_rows.append(padded_row)

			# Create markdown table
			formatted_rows = []
			for i, row in enumerate(padded_rows):
				formatted_row = "|" + "|".join(row) + "|"
				formatted_rows.append(formatted_row)

				# Add separator after header
				if i == 0:
					separator = "|" + "|".join(["---"] * max_cols) + "|"
					formatted_rows.append(separator)

			return "\n".join(formatted_rows)

	# If we can't format it, return original
	return "\n".join(table_lines)


def _is_code_content(self, text: str) -> bool:
	"""Check if the text contains code that should not be formatted as tables"""
	import re

	# Check for code block markers
	if "```" in text:
		return True

	# Check for Python-specific patterns
	python_patterns = [
		r"def\s+\w+\s*\(",
		r"class\s+\w+",
		r"import\s+\w+",
		r"from\s+\w+\s+import",
		r"if\s+__name__\s*==\s*[\"\']__main__[\"\']",
		r"return\s+",
		r"while\s+",
		r"for\s+\w+\s+in\s+",
		r"#\s*[A-Z]",
	]

	for pattern in python_patterns:
		if re.search(pattern, text, re.MULTILINE):
			return True

	# Check for indented code blocks (4+ spaces at start of line)
	lines = text.split("\n")
	indented_lines = 0
	for line in lines:
		if line.strip() and line.startswith("    "):
			indented_lines += 1

	# If more than 30% of non-empty lines are indented, it's likely code
	non_empty_lines = [line for line in lines if line.strip()]
	if non_empty_lines and indented_lines / len(non_empty_lines) > 0.3:
		return True

	return False


def _is_explanation_content(self, text: str) -> bool:
	"""Check if the text is explanation content that should use text formatting, not tables"""
	import re

	explanation_patterns = [
		r"Time\s*Complexity",
		r"Space\s*Complexity",
		r"How\s+it\s+works",
		r"Key\s+Features",
		r"Time:\s*O\(",
		r"Space:\s*O\(",
		r"Input\s+type:",
		r"Output:",
		r"Error\s+handling:",
	]

	for pattern in explanation_patterns:
		if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
			return True

	# Check if text contains table-like formatting but is actually explanation
	lines = text.split("\n")
	table_like_lines = 0
	for line in lines:
		if "|" in line and any(keyword in line.lower() for keyword in ["time", "space", "complexity", "feature", "input", "output"]):
			table_like_lines += 1

	if table_like_lines > 0:
		return True

	return False


def _clean_code_formatting(self, text: str) -> str:
	"""Clean up code formatting issues without converting to tables"""
	import re

	lines = text.split("\n")
	cleaned_lines = []

	for line in lines:
		# Fix lines that look like they were formatted as table rows
		if "|" in line and "=" in line:
			line = re.sub(r"^\s*\|\s*", "", line)
			line = re.sub(r"\s*\|\s*$", "", line)
			line = re.sub(r"\s*\|\s*", " ", line)

			# Fix indentation for code lines
			if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
				if any(keyword in line for keyword in ["def ", "class ", "if ", "while ", "for ", "else:", "elif "]):
					pass
				elif line.strip().startswith(("return", "yield", "break", "continue", "pass")):
					line = "    " + line.strip()
				elif "=" in line and not line.strip().startswith("#"):
					line = "    " + line.strip()

		# Fix comment formatting
		if "|" in line and "#" in line:
			line = re.sub(r"^\s*\|\s*", "", line)
			line = re.sub(r"\s*\|\s*$", "", line)
			line = re.sub(r"\s*\|\s*", " ", line)

		cleaned_lines.append(line)

	return "\n".join(cleaned_lines)


def _clean_explanation_formatting(self, text: str) -> str:
	"""Clean up explanation formatting by converting table-like formatting to proper text"""
	import re

	lines = text.split("\n")
	cleaned_lines = []

	for line in lines:
		if "|" in line and any(keyword in line.lower() for keyword in ["time", "space", "complexity", "feature", "input", "output", "error"]):
			line = line.strip("|")
			parts = [part.strip() for part in line.split("|") if part.strip()]

			if len(parts) >= 2:
				metric = parts[0].strip()
				metric = metric.replace("**", "").replace("*", "")
				metric = metric.rstrip(":").strip()

				value = parts[1].strip()

				if "time" in metric.lower():
					formatted_line = f"**Time Complexity:** {value}"
				elif "space" in metric.lower():
					formatted_line = f"**Space Complexity:** {value}"
				elif "feature" in metric.lower():
					formatted_line = f"**Key Features:** {value}"
				elif "input" in metric.lower():
					formatted_line = f"**Input:** {value}"
				elif "output" in metric.lower():
					formatted_line = f"**Output:** {value}"
				elif "error" in metric.lower():
					formatted_line = f"**Error Handling:** {value}"
				else:
					formatted_line = f"**{metric}:** {value}"

				cleaned_lines.append(formatted_line)
			else:
				cleaned_lines.append(line)
		else:
			# Check for table separator lines and skip them
			if re.match(r"^\s*\|[\s\-:]+\|", line):
				continue
			cleaned_lines.append(line)

	return "\n".join(cleaned_lines)


# ---------------------------------------------------------------------------
# Structural Integrity Validator  (enterprise-grade final assertion layer)
# ---------------------------------------------------------------------------
# Runs as the very last step of _format_response.  Four checks:
#   1. Balanced code fences  — auto-close any unclosed ``` block
#   2. Balanced emphasis      — strip orphan ** / * markers
#   3. Mermaid block hygiene  — ensure diagram header present & no empty blocks
#   4. Response size limit    — truncate cleanly if over 32 KB (≈8k tokens)
# Each check is self-contained and never raises; it repairs in-place.
# ---------------------------------------------------------------------------

# Max response size in characters (~32 KB ≈ 8 000 tokens).
_MAX_RESPONSE_CHARS = 32_000


def _structural_integrity_check(self, text: str) -> str:
	"""Final assertion & auto-repair layer before the response is saved / sent."""
	if not text:
		return text

	text = _repair_balanced_code_fences(text)
	text = _repair_balanced_emphasis(text)
	text = _repair_mermaid_blocks(text)
	text = _enforce_size_limit(text)
	return text


# ── 1. Balanced code fences ──────────────────────────────────────────────

def _repair_balanced_code_fences(text: str) -> str:
	"""If there is an odd number of ``` fence lines, close the last block."""
	fence_indices: list[int] = []
	for i, line in enumerate(text.split("\n")):
		if line.strip().startswith("```"):
			fence_indices.append(i)

	if len(fence_indices) % 2 == 1:
		# Odd count → unclosed code block.  Append a closing fence.
		text = text.rstrip() + "\n```\n"
	return text


# ── 2. Balanced emphasis markers ─────────────────────────────────────────

def _repair_balanced_emphasis(text: str) -> str:
	"""Scan every non-code line for orphan ** or * markers and strip them.

	This is a safety net *after* `_fix_unbalanced_markdown_emphasis` which
	operates earlier.  Here we only touch lines that earlier passes missed
	(e.g. lines introduced by later pipeline steps).
	"""
	lines = text.split("\n")
	out: list[str] = []
	in_code = False
	for line in lines:
		stripped = line.strip()
		if stripped.startswith("```"):
			in_code = not in_code
			out.append(line)
			continue
		if in_code:
			out.append(line)
			continue

		# Double-star orphans
		if line.count("**") % 2 == 1:
			idx = line.rfind("**")
			if idx >= 0:
				line = line[:idx] + line[idx + 2:]

		# Single-star orphans (skip list-marker lines "* item")
		lstrip = line.lstrip()
		if not lstrip.startswith("* "):
			star_count = len(re.findall(r'(?<!\*)\*(?!\*)', line))
			if star_count % 2 == 1:
				# Remove last lone *
				line = re.sub(r'\*(?=[^*]*$)', '', line, count=1)

		out.append(line)
	return "\n".join(out)


# ── 3. Mermaid block hygiene ─────────────────────────────────────────────

_MERMAID_DIAGRAM_HEADERS = re.compile(
	r"^\s*(?:flowchart|graph|sequenceDiagram|classDiagram|stateDiagram|"
	r"erDiagram|journey|gantt|pie|gitGraph|mindmap|timeline)\b",
	re.IGNORECASE,
)


def _repair_mermaid_blocks(text: str) -> str:
	"""Validate every ```mermaid ... ``` block:
	- Must contain a recognised diagram header.
	- Must not be empty.
	If invalid, replace with a markdown comment so the UI isn't broken.
	"""
	if "```mermaid" not in text.lower():
		return text

	parts: list[str] = []
	lines = text.split("\n")
	i = 0
	while i < len(lines):
		line = lines[i]
		# Detect opening ```mermaid fence
		if line.strip().lower().startswith("```mermaid"):
			block_lines: list[str] = [line]
			i += 1
			# Gather until closing ``` or EOF
			while i < len(lines):
				block_lines.append(lines[i])
				if lines[i].strip() == "```":
					i += 1
					break
				i += 1

			# Extract inner content (skip opening/closing fences)
			inner = "\n".join(block_lines[1:-1] if len(block_lines) > 2 else [])
			inner_stripped = inner.strip()

			# Validation: non-empty and has a valid diagram header
			if not inner_stripped:
				# Empty mermaid block → drop silently
				parts.append("<!-- empty diagram removed -->")
			elif not _MERMAID_DIAGRAM_HEADERS.search(inner_stripped):
				# No recognisable header → drop and explain
				parts.append("<!-- invalid diagram removed -->")
			else:
				# Valid — keep the block
				parts.append("\n".join(block_lines))
			continue

		parts.append(line)
		i += 1

	return "\n".join(parts)


# ── 4. Response size limit ───────────────────────────────────────────────

def _enforce_size_limit(text: str) -> str:
	"""Truncate response cleanly at a sentence boundary if it exceeds the max."""
	if len(text) <= _MAX_RESPONSE_CHARS:
		return text

	# Attempt to cut at the last sentence boundary within the limit.
	truncated = text[:_MAX_RESPONSE_CHARS]

	# Walk back to last sentence-ending punctuation followed by whitespace/newline.
	for end_marker in ("\n\n", ".\n", ". ", "!\n", "! ", "?\n", "? "):
		idx = truncated.rfind(end_marker)
		if idx > _MAX_RESPONSE_CHARS * 0.6:  # don't cut more than 40%
			truncated = truncated[: idx + len(end_marker)]
			break

	# If we're inside a code fence, close it
	fence_count = truncated.count("```")
	if fence_count % 2 == 1:
		truncated = truncated.rstrip() + "\n```\n"

	truncated = truncated.rstrip() + "\n\n*... (response truncated for length)*"
	return truncated


def attach_text_postprocess_methods(cls) -> None:
	"""Attach text-postprocess functions as methods on LLMService."""
	cls._process_thinking_tags = _process_thinking_tags
	cls._fix_markdown_syntax = _fix_markdown_syntax
	cls._split_runon_plus_bullets = _split_runon_plus_bullets
	cls._is_internal_prompt_leak_line = _is_internal_prompt_leak_line
	cls._strip_internal_prompt_leakage = _strip_internal_prompt_leakage
	cls._remove_forbidden_side_headings = _remove_forbidden_side_headings
	cls._rewrite_onboarding_brochure_if_present = _rewrite_onboarding_brochure_if_present
	cls._format_response = _format_response
	cls._normalize_colon_label_lists = _normalize_colon_label_lists
	cls._fix_unbalanced_markdown_emphasis = _fix_unbalanced_markdown_emphasis
	cls._normalize_markdown_bullets = _normalize_markdown_bullets
	cls._format_headings_bold = _format_headings_bold
	cls._strip_latex_math = _strip_latex_math
	cls._sanitize_mermaid_syntax = _sanitize_mermaid_syntax
	cls._normalize_mermaid_blocks = _normalize_mermaid_blocks
	cls._contains_mermaid = _contains_mermaid
	cls._strip_labeled_bullets_in_complete_answer = _strip_labeled_bullets_in_complete_answer
	cls._strip_leading_bold_labels_globally = _strip_leading_bold_labels_globally
	cls._deplaceholderize = _deplaceholderize
	cls._format_summary_sections = _format_summary_sections
	cls._clean_table_markdown_artifacts = _clean_table_markdown_artifacts
	cls._is_table_line = _is_table_line
	cls._clean_table_line = _clean_table_line
	cls._format_tables = _format_tables
	cls._is_table_row = _is_table_row
	cls._looks_like_pipe_table = _looks_like_pipe_table
	cls._create_markdown_table = _create_markdown_table
	cls._is_code_content = _is_code_content
	cls._is_explanation_content = _is_explanation_content
	cls._clean_code_formatting = _clean_code_formatting
	cls._clean_explanation_formatting = _clean_explanation_formatting
	cls._structural_integrity_check = _structural_integrity_check
