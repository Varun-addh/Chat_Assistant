from __future__ import annotations

import re
import logging
from typing import Iterable, Tuple, List

logger = logging.getLogger(__name__)

from app.utils.architecture_layers import get_layer_titles
from app.utils.mermaid_sanitizer import MermaidSanitizer


def _to_ascii(text: str) -> str:
    """Normalize to ASCII where possible; drop remaining non-ASCII."""
    if not text:
        return ""
    replacements = {
        "→": "->",
        "←": "<-",
        "⇒": "=>",
        "⇐": "<=",
        "↔": "<->",
        # Normalize the bullet glyph into an ASCII dash so bullets aren't lost.
        "•": "-",
        "–": "-",
        "—": "-",
        "…": "...",
        "“": '"',
        "”": '"',
        "’": "'",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return "".join(ch for ch in text if ord(ch) < 128)


def extract_mermaid_first(text: str) -> Tuple[str, str]:
    """Extract the first ```mermaid ... ``` block from text.

    Returns (mermaid_code, remainder_text_after_block).
    If no mermaid block is found, returns ("", original_text).
    """
    if not text:
        return "", ""

    # Non-greedy: first mermaid block
    m = re.search(r"```mermaid\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return "", text

    code = (m.group(1) or "").strip()
    rest = (text[m.end() :] or "").strip()
    return code, rest


def sanitize_mermaid_subset(code: str) -> str:
    """Keep Mermaid within a safe subset for common renderers.

    NOTE: Single source of truth lives in app.utils.mermaid_sanitizer.
    """
    return MermaidSanitizer.sanitize_subset(code)


def _normalize_bullets(lines: Iterable[str]) -> List[str]:
    out: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        # Normalize common bullet symbols
        # Normalize different bullet markers into a single '• ' marker
        line = re.sub(r"^[\u2022\-\*]\s+", "• ", line)
        line = line.replace("+ ", "• ")
        line = line.replace("•", "• ")
        # If it looks like 'Point 1: ...', drop the prefix
        line = re.sub(r"^(-\s+)?(point\s*\d+\s*:\s*)", "- ", line, flags=re.IGNORECASE)
        if line.startswith("-") and not line.startswith("- "):
            line = "- " + line.lstrip("-").lstrip()

        # Drop empty bullets like "-" or "- " that often come from a lone bullet glyph line.
        if line.strip() in {"•", "• ", "-", "- "}:
            continue
        out.append(line)
    return out


def _strip_redundant_prefixes(text: str) -> str:
    """Remove redundant prefixes that models sometimes include inside bullets."""
    s = (text or "").strip()
    # Remove both start-of-string and mid-sentence occurrences.
    s = re.sub(r"\bwhat happens\s*:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bwhy it exists\s*:\s*", "", s, flags=re.IGNORECASE)
    # New enhanced format prefixes
    s = re.sub(r"\bhow it works\s*:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bwhy this design\s*:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bkey tradeoffs\s*:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bproblem this layer solves\s*:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bhow we solve it\s*:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bmeasurable impact\s*:\s*", "", s, flags=re.IGNORECASE)
    # Common summary artifact.
    s = re.sub(r"\b\*{0,2}bottom line\*{0,2}\s*:\s*", "", s, flags=re.IGNORECASE)
    return s.strip()


def _extract_layer_content(content_lines: List[str], fallback_text: str) -> List[str]:
    """Extract layer content handling both formats:
    - Old: 'What happens' + 'Why it exists'  
    - New: 'How it works' + 'Why this design' + 'Key tradeoffs'
    - Observability: 'Problem this layer solves' + 'How we solve it' + 'Measurable impact'
    
    Returns formatted content lines ready for output.
    """
    if not content_lines and not fallback_text:
        return ["What happens:", "- Layer content not available", "Why it exists:", "- Not specified"]
    
    # Join content and look for subsection markers
    full_text = "\n".join(content_lines)
    
    # Pattern 1: Enhanced format (How + Why + Tradeoffs)
    how_match = re.search(r"how it works\s*:(.+?)(?=why this design|key tradeoffs|$)", full_text, re.IGNORECASE | re.DOTALL)
    why_design_match = re.search(r"why this design\s*:(.+?)(?=key tradeoffs|$)", full_text, re.IGNORECASE | re.DOTALL)
    tradeoffs_match = re.search(r"key tradeoffs\s*:(.+?)(?=$)", full_text, re.IGNORECASE | re.DOTALL)
    
    # Pattern 2: Observability format (Problem + Solution + Impact)
    problem_match = re.search(r"problem this layer solves\s*:(.+?)(?=how we solve it|measurable impact|$)", full_text, re.IGNORECASE | re.DOTALL)
    solution_match = re.search(r"how we solve it\s*:(.+?)(?=measurable impact|$)", full_text, re.IGNORECASE | re.DOTALL)
    impact_match = re.search(r"measurable impact\s*:(.+?)(?=$)", full_text, re.IGNORECASE | re.DOTALL)
    
    # Pattern 3: Old format (What + Why)
    what_match = re.search(r"what happens\s*:(.+?)(?=why it exists|$)", full_text, re.IGNORECASE | re.DOTALL)
    why_match = re.search(r"why it exists\s*:(.+?)(?=$)", full_text, re.IGNORECASE | re.DOTALL)
    
    out_lines = []
    
    # Try enhanced format first
    if how_match:
        out_lines.append("How it works:")
        out_lines.extend(_normalize_bullets(how_match.group(1).strip().splitlines()))
        if why_design_match:
            out_lines.append("")
            out_lines.append("Why this design:")
            out_lines.extend(_normalize_bullets(why_design_match.group(1).strip().splitlines()))
        else:
            logger.warning("Missing 'Why this design' section in enhanced layer format")
        if tradeoffs_match:
            out_lines.append("")
            out_lines.append("Key tradeoffs:")
            out_lines.extend(_normalize_bullets(tradeoffs_match.group(1).strip().splitlines()))
        else:
            # Log warning but provide minimal fallback so UI doesn't break
            logger.warning("Missing 'Key tradeoffs' section - this is critical for FAANG-level depth")
            out_lines.append("")
            out_lines.append("Key tradeoffs:")
            out_lines.append("• Design involves tradeoffs between performance, cost, and complexity")
        return [ln for ln in out_lines if ln.strip()]
    
    # Try observability format
    if problem_match:
        out_lines.append("Problem this layer solves:")
        out_lines.extend(_normalize_bullets(problem_match.group(1).strip().splitlines()))
        if solution_match:
            out_lines.append("")
            out_lines.append("How we solve it:")
            out_lines.extend(_normalize_bullets(solution_match.group(1).strip().splitlines()))
        else:
            logger.warning("Missing 'How we solve it' section in observability format")
        if impact_match:
            out_lines.append("")
            out_lines.append("Measurable impact:")
            out_lines.extend(_normalize_bullets(impact_match.group(1).strip().splitlines()))
        else:
            logger.warning("Missing 'Measurable impact' section - observability needs quantified results")
            out_lines.append("")
            out_lines.append("Measurable impact:")
            out_lines.append("• Improves system observability and reduces time to resolution")
        return [ln for ln in out_lines if ln.strip()]
    
    # Fall back to old format
    if what_match or why_match:
        if what_match:
            out_lines.append("What happens:")
            out_lines.extend(_normalize_bullets(what_match.group(1).strip().splitlines()))
        if why_match:
            out_lines.append("")
            out_lines.append("Why it exists:")
            out_lines.extend(_normalize_bullets(why_match.group(1).strip().splitlines()))
        return [ln for ln in out_lines if ln.strip()]
    
    # If no explicit markers found, structure bullets as old format (backward compatibility)
    bullets = _normalize_bullets(content_lines)
    if bullets:
        # Split bullets: first 2 for "What happens", rest for "Why it exists"
        what_bullets = bullets[:2] if len(bullets) >= 2 else bullets
        why_bullets = bullets[2:3] if len(bullets) >= 3 else []
        
        out_lines.append("What happens:")
        out_lines.extend(what_bullets if what_bullets else ["- Layer processes request"])
        if why_bullets:
            out_lines.append("Why it exists:")
            out_lines.extend(why_bullets)
        else:
            out_lines.append("Why it exists:")
            out_lines.append("• Architectural purpose defined")
        return out_lines
    
    # Absolute fallback
    return ["What happens:", "- Layer details not fully generated", "Why it exists:", "- Architectural purpose"]


def _extract_layer_what_why(content_lines: List[str], fallback_text: str) -> Tuple[List[str], List[str]]:
    """DEPRECATED: Use _extract_layer_content instead.
    
    Extract exactly 2 'What happens' bullets and 1 'Why it exists' bullet.
    Tries to respect explicit prefixes if the model provided them; otherwise uses
    generic bullets/sentences as material. Always returns non-empty bullets.
    """
    # Accept either '- ' or '• ' bullets
    payloads_raw = [b[2:].strip() for b in _normalize_bullets(content_lines) if b.startswith("- ") or b.startswith("• ")]
    payloads_raw = [p for p in payloads_raw if p.strip()]

    payloads = [_strip_redundant_prefixes(p) for p in payloads_raw if p.strip()]

    what_cand: List[str] = []
    why_cand: List[str] = []
    other: List[str] = []

    for raw in payloads_raw:
        cleaned = _strip_redundant_prefixes(raw)
        if re.match(r"^\s*what\s+happens\s*:\s*", raw, flags=re.IGNORECASE):
            what_cand.append(cleaned)
        elif re.match(r"^\s*why\s+it\s+exists\s*:\s*", raw, flags=re.IGNORECASE):
            why_cand.append(cleaned)
        else:
            other.append(cleaned)

    # Derived material (avoid headings)
    derived = [
        _strip_redundant_prefixes(s)
        for s in _first_sentences(" ".join(payloads) or fallback_text or "", 10)
        if s.strip() and "###" not in s and not s.strip().startswith("#")
    ]
    derived = [d for d in derived if d and not d.lower().startswith("layer ")]

    def _pull(src: List[str], dst: List[str], n: int) -> None:
        while len(dst) < n and src:
            v = src.pop(0).strip()
            if not v:
                continue
            if v in dst:
                continue
            dst.append(v)

    # Fill What first from explicit, then other, then derived
    _pull(other, what_cand, 2)
    _pull(derived, what_cand, 2)

    # Fill Why from explicit, then remaining other, then derived
    _pull(other, why_cand, 1)
    _pull(derived, why_cand, 1)

    # Guarantee non-empty with safe fallbacks
    if len(what_cand) == 0:
        what_cand = ["Handle the layer's critical work."]
    if len(what_cand) == 1:
        what_cand.append("Propagate the result to the next step.")
    if len(why_cand) == 0:
        why_cand = ["Preserve correctness and reliability under load."]

    what = [f"• {w}" for w in what_cand[:2]]
    why = [f"• {w}" for w in why_cand[:1]]
    return what, why


def _first_sentences(text: str, n: int) -> List[str]:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if p.strip()][:n]


def _format_numbered_qa(text: str) -> str:
    """Detect numbered Q&A sections (1., 2., ...) and reformat each into a
    consistent template. Returns original text if not a multi-item numbered list.
    """
    items = list(re.finditer(r"(?m)^\s*(\d+)\.\s*(.+)$", text))
    if len(items) < 2:
        return text

    allowed_langs = {
        "python",
        "py",
        "sql",
        "bash",
        "sh",
        "json",
        "yaml",
        "yml",
        "js",
        "javascript",
        "html",
        "css",
        "go",
        "java",
        "c",
        "cpp",
        "rust",
        "ruby",
    }

    out_sections: List[str] = []
    for idx, m in enumerate(items):
        start = m.end()
        end = items[idx + 1].start() if idx + 1 < len(items) else len(text)
        title = m.group(2).strip()
        body = text[start:end].strip()

        # Remove common QA prefixes inside the body
        body = re.sub(r"(?im)^\s*(answer|example|why interviewers ask this|why this is asked|real[- ]world usage|real world usage)\s*:\s*", "", body)

        # Extract first fenced code block
        code_block = None
        code_match = re.search(r"```([^\n]*)\n(.*?)\n```", body, flags=re.DOTALL | re.IGNORECASE)
        if code_match:
            lang = (code_match.group(1) or "").strip().lower()
            code = code_match.group(2).strip()
            if lang and lang in allowed_langs:
                code_block = f"```{lang}\n{code}\n```"
            else:
                code_block = f"```\n{code}\n```"
            body = (body[: code_match.start()] + body[code_match.end() :]).strip()

        expl_sents = _first_sentences(body, 2)
        explanation = expl_sents[0] if expl_sents else ""

        why = ""
        why_m = re.search(r"(?is)why[\w\s]*[:\-]\s*(.*?)(?=real[- ]world|example|$)", body)
        if why_m:
            why = _first_sentences(why_m.group(1).strip(), 1)[0] if _first_sentences(why_m.group(1).strip(), 1) else ""

        rw = ""
        rw_m = re.search(r"(?is)real[- ]world[\s]*usage[:\-]\s*(.*)$", body)
        if rw_m:
            rw = _first_sentences(rw_m.group(1).strip(), 1)[0] if _first_sentences(rw_m.group(1).strip(), 1) else ""
        elif len(expl_sents) > 1:
            rw = expl_sents[1]

        sec_lines: List[str] = [f"{m.group(1)}. {title}", "- Explanation:"]
        if explanation:
            sec_lines.append(f"• {explanation}")

        if code_block:
            sec_lines.append("- Example:")
            sec_lines.append(code_block)
        else:
            ex_m = re.search(r"(?is)example[:\-]\s*(.*?)(?=why|real[- ]world|$)", body)
            if ex_m:
                ex_sent = _first_sentences(ex_m.group(1).strip(), 2)
                for s in ex_sent:
                    sec_lines.append(f"• {s}")

        if why:
            sec_lines.append("- Why interviewers ask this:")
            sec_lines.append(f"• {why}")

        if rw:
            sec_lines.append("- Real-world usage:")
            sec_lines.append(f"• {rw}")

        out_sections.append("\n".join(sec_lines))

    return "\n\n".join(out_sections)


def enforce_story_contract(view_name: str, system_description: str, explanation: str) -> str:
    """Clamp the explanation to the strict, story-driven UX format.

    This is intentionally conservative: it *keeps* what looks compliant and
    discards everything else.
    """
    view = (view_name or "").upper().strip()
    system_description = (system_description or "").strip()
    text = _to_ascii(explanation or "").strip()

    # Strip noisy code-fence language labels that some models inject (e.g. "```TEXT CODE").
    # Preserve common single-token language identifiers (python, sql, bash, etc.)
    def _strip_code_fence_langs(s: str) -> str:
        def _repl(m: re.Match) -> str:
            lang = (m.group(1) or "").strip()
            if not lang:
                return "```\n"
            lang_norm = lang.lower().strip()
            allowed = {
                "python",
                "py",
                "sql",
                "bash",
                "sh",
                "json",
                "yaml",
                "yml",
                "js",
                "javascript",
                "html",
                "css",
                "go",
                "java",
                "c",
                "cpp",
                "rust",
                "ruby",
            }
            # If the language token is a single, allowed identifier, keep it (normalized);
            # otherwise strip the label so the UI won't render noisy subtitles like "TEXT CODE".
            if " " not in lang_norm and lang_norm in allowed:
                return f"```{lang_norm}\n"
            return "```\n"

        return re.sub(r"```([^\n]*)\n", _repl, s)

    text = _strip_code_fence_langs(text)

    # Remove common Q&A-style subheadings that models insert (e.g. "Answer:", "Example:").
    # We strip only the prefix so the actual content remains.
    qa_prefix_pattern = r"(?im)^\s*(answer)\s*:\s*"
    text = re.sub(qa_prefix_pattern, "", text)

    # If the model produced a numbered Q&A list (1., 2., ...), reformat it
    # into the neat interview template automatically.
    if re.search(r"(?m)^\s*\d+\.\s+", text):
        formatted = _format_numbered_qa(text)
        # If formatting produced multiple sections, trust it and return.
        if formatted and formatted.strip():
            return _to_ascii(formatted).strip()

    # Remove common "encyclopedia" sections and UI artifacts
    # NOTE: SINGLE mode intentionally includes executive summary, capacity planning,
    # trade-offs, failure handling, and a small code snippet. Other views should
    # stay compact, so we aggressively drop those markers there.
    base_drop_markers = [
        "key highlights",
        "detailed analysis",
        "requirements analysis",
        "data table",
        "copy table",
        "component design",
        "interview success guide",
        "detailed explanation",
        "part 2",
        # UI artifacts
        "copy source",
        "mermaid diagram",
    ]

    if view == "SINGLE":
        drop_markers = base_drop_markers
    else:
        drop_markers = base_drop_markers + [
            "executive summary",
            "implementation example",
            "python code",
            "copy code",
            "production tip",
            "trade-offs",
            "capacity planning",
            "failure handling",
        ]
    filtered_lines = []
    for line in text.splitlines():
        low = line.strip().lower()
        if any(m in low for m in drop_markers):
            continue
        filtered_lines.append(line)
    text = "\n".join(filtered_lines).strip()

    if view == "SINGLE":
        # SINGLE mode: one diagram + comprehensive interview-ready narrative.
        # Clamp into the exact headings the UI promises.

        allowed_headings = [
            "### Executive Summary",
            "### Requirements",
            "### Architecture",
            "### Data & Storage",
            "### Critical Request Flow",
            "### Capacity Planning",
            "### Trade-offs",
            "### Failure Modes & Mitigations",
            "### Production Considerations",
            "### Example API Contracts",
            "### Example Implementation Snippet",
        ]

        canon_by_lower = {h.lower(): h for h in allowed_headings}
        sections: dict[str, List[str]] = {h: [] for h in allowed_headings}
        current: str | None = None

        lines = [ln.rstrip("\r") for ln in text.splitlines()]

        # Parse by headings. Ignore unknown headings; keep collecting under the last known one.
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("###"):
                h = line.strip()
                norm = h.lower()
                matched = canon_by_lower.get(norm)
                if matched is None:
                    # Try lenient match (ignore hyphen spacing)
                    for k, canon in canon_by_lower.items():
                        if norm.replace("-", " ").strip() == k.replace("-", " ").strip():
                            matched = canon
                            break
                current = matched
                continue
            if current:
                sections[current].append(raw)

        # Fallback: if the model didn't include headings, derive from sentences.
        if all(len(v) == 0 for v in sections.values()):
            blob = "\n".join([ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")])
            sentences = _first_sentences(blob, 30)
            # Distribute conservatively.
            sections["### Executive Summary"] = [f"• {s}" for s in sentences[:5]]
            sections["### Requirements"] = sentences[5:14]
            sections["### Architecture"] = sentences[14:22]
            sections["### Data & Storage"] = sentences[22:26]
            sections["### Critical Request Flow"] = sentences[26:32]

        out_lines: List[str] = []

        def _as_bullets(raw_lines: List[str], n: int) -> List[str]:
            bullets = _normalize_bullets(raw_lines)
            bullets = [b for b in bullets if b.startswith("- ") or b.startswith("• ")]
            if len(bullets) < n:
                derived = [f"• {s}" for s in _first_sentences(" ".join(raw_lines), n)]
                bullets = (bullets + derived)[:n]
            else:
                bullets = bullets[:n]

            clean: List[str] = []
            for b in bullets:
                item = _strip_redundant_prefixes(b[2:].strip())
                item = _first_sentences(item, 1)[0] if _first_sentences(item, 1) else item
                if item and item[-1] not in ".!?":
                    item += "."
                clean.append(f"- {item}")
            return clean

        def _as_numbered_steps(raw_lines: List[str], min_n: int = 5, max_n: int = 8) -> List[str]:
            # Accept existing numbering; else derive from sentences.
            steps: List[str] = []
            for ln in raw_lines:
                m = re.match(r"^\s*(\d+)\s*[\.)]\s+(.+)$", ln.strip())
                if m:
                    steps.append(m.group(2).strip())
            if not steps:
                steps = _first_sentences(" ".join(raw_lines), max_n)
            steps = [s for s in steps if s.strip()][:max_n]
            if len(steps) < min_n:
                steps = (steps + _first_sentences(system_description, min_n))[:min_n]
            out: List[str] = []
            for i, s in enumerate(steps[:max_n], 1):
                s2 = _strip_redundant_prefixes(s.strip())
                if s2 and s2[-1] not in ".!?":
                    s2 += "."
                out.append(f"{i}. {s2}")
            return out

        def _as_short_lines(raw_lines: List[str], n: int) -> List[str]:
            cleaned = [ln.strip() for ln in raw_lines if ln.strip() and not ln.lstrip().startswith("#")]
            if not cleaned:
                cleaned = _first_sentences(" ".join(raw_lines), n)
            cleaned = [_strip_redundant_prefixes(c) for c in cleaned if c.strip()]
            return cleaned[:n]

        # Build output in exact order.
        out_lines.append("### Executive Summary")
        out_lines.extend(_as_bullets(sections.get("### Executive Summary") or lines, 5))
        out_lines.append("")

        out_lines.append("### Requirements")
        req_lines = sections.get("### Requirements") or []
        # Split into functional/non-functional heuristically.
        func = []
        nonfunc = []
        for ln in req_lines:
            low = ln.lower()
            if any(k in low for k in ["latency", "p95", "p99", "qps", "rps", "availability", "slo", "throughput", "cost"]):
                nonfunc.append(ln)
            else:
                func.append(ln)
        out_lines.append("Functional:")
        out_lines.extend(_as_bullets(func or req_lines, 3))
        out_lines.append("Non-Functional:")
        out_lines.extend(_as_bullets(nonfunc or req_lines, 3))
        out_lines.append("")

        out_lines.append("### Architecture")
        out_lines.extend(_as_bullets(sections.get("### Architecture") or [], 8))
        out_lines.append("")

        out_lines.append("### Data & Storage")
        out_lines.extend(_as_bullets(sections.get("### Data & Storage") or [], 4))
        out_lines.append("")

        out_lines.append("### Critical Request Flow")
        out_lines.extend(_as_numbered_steps(sections.get("### Critical Request Flow") or []))
        out_lines.append("")

        out_lines.append("### Capacity Planning")
        out_lines.extend(_as_bullets(sections.get("### Capacity Planning") or [], 4))
        out_lines.append("")

        out_lines.append("### Trade-offs")
        out_lines.extend(_as_bullets(sections.get("### Trade-offs") or [], 4))
        out_lines.append("")

        out_lines.append("### Failure Modes & Mitigations")
        out_lines.extend(_as_bullets(sections.get("### Failure Modes & Mitigations") or [], 4))
        out_lines.append("")

        out_lines.append("### Production Considerations")
        out_lines.extend(_as_bullets(sections.get("### Production Considerations") or [], 6))
        out_lines.append("")

        out_lines.append("### Example API Contracts")
        out_lines.extend(_as_short_lines(sections.get("### Example API Contracts") or [], 4))
        out_lines.append("")

        out_lines.append("### Example Implementation Snippet")
        snippet_lines = sections.get("### Example Implementation Snippet") or []
        # Preserve an existing short python block if present; else provide a minimal placeholder.
        snippet_text = "\n".join(snippet_lines).strip()
        m = re.search(r"```python\s*(.*?)```", snippet_text, flags=re.DOTALL | re.IGNORECASE)
        if m and m.group(1).strip():
            code = m.group(1).strip().splitlines()[:15]
            out_lines.append("```python")
            out_lines.extend(code)
            out_lines.append("```")
        else:
            out_lines.append("```python")
            out_lines.append("# Example: idempotent create request with request_id")
            out_lines.append("def handle_request(request_id: str, payload: dict) -> dict:")
            out_lines.append("    if redis.get(f\"req:{request_id}\"):")
            out_lines.append("        return redis.get(f\"req:{request_id}\")")
            out_lines.append("    result = process(payload)")
            out_lines.append("    redis.setex(f\"req:{request_id}\", 3600, result)")
            out_lines.append("    return result")
            out_lines.append("```")

        return "\n".join(out_lines).strip()

    if view == "SYSTEM_OVERVIEW":
        bullets = _normalize_bullets(text.splitlines())
        bullets = [b for b in bullets if b.startswith("- ") or b.startswith("• ")]

        # Take exactly 5 bullets; if fewer, try to derive from sentences
        if len(bullets) < 5:
            derived = [f"• {s}" for s in _first_sentences(text, 5)]
            bullets = (bullets + derived)[:5]
        else:
            bullets = bullets[:5]

        # Ensure each bullet is one sentence-ish
        clean_bullets: List[str] = []
        for b in bullets:
            item = _strip_redundant_prefixes(b[2:].strip())
            # Trim to first sentence
            item = _first_sentences(item, 1)[0] if _first_sentences(item, 1) else item
            if item and item[-1] not in ".!?":
                item += "."
            clean_bullets.append(f"- {item}")

        goal_line = None
        for line in text.splitlines():
            if line.strip().lower().startswith("goal:"):
                goal_line = "Goal: " + line.split(":", 1)[1].strip()
                break
        if not goal_line:
            # Minimal, neutral goal without placeholders
            goal_line = f"Goal: Deliver {system_description or 'the system'} reliably at scale.".strip()

        return "\n".join(clean_bullets + [goal_line]).strip()

    # Views: enforce a strict, consistent 5-layer contract across ALL multi-view outputs.
    # Layer titles come from a single canonical model to prevent mismatches like:
    # - DATA_MODEL clamped to 3 layers
    # - OBSERVABILITY clamped to 4 layers
    layer_titles: List[str] = get_layer_titles(view)
    if not layer_titles:
        # Unknown / non-layered view: keep a short, safe snippet
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines[:12]).strip()

    final_titles: dict[str, str] = {
        "REQUEST_FLOW": "Final End-to-End Flow Summary",
        "DATA_MODEL": "Final Data Flow Summary",
        "DEPLOYMENT": "Final Deployment Summary",
        "OBSERVABILITY": "Final Observability Summary",
        "ASYNC_PROCESSING": "Final Async Processing Summary",
    }

    max_layer = len(layer_titles)
    final_title = final_titles.get(view, "Final Summary")

    # Parse sections by layer number so minor title differences don't break clamping.
    sections_by_layer: dict[int, List[str]] = {i: [] for i in range(1, max_layer + 1)}
    final_lines: List[str] = []
    seen_layer_heading: dict[int, str] = {}
    seen_final_heading: str | None = None

    current_layer: int | None = None
    in_final = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("###"):
            h = _to_ascii(line)
            low = h.lower()
            # Final heading (accept variations like 'Final End-to-End Summary')
            if "final" in low and "summary" in low:
                in_final = True
                current_layer = None
                seen_final_heading = h
                continue
            m = re.match(r"^###\s*layer\s*(\d+)\b(.*)$", h, flags=re.IGNORECASE)
            if m:
                n = int(m.group(1))
                if 1 <= n <= max_layer:
                    in_final = False
                    current_layer = n
                    # Keep the model's title if it provided one; otherwise use our default.
                    suffix = (m.group(2) or "")
                    # Preserve a leading space so we don't end up with 'Layer 1- Title'.
                    if suffix and not suffix.startswith(" "):
                        suffix = " " + suffix.lstrip()
                    suffix = suffix.rstrip()
                    if suffix:
                        seen_layer_heading[n] = f"### Layer {n}{suffix}"
                    else:
                        seen_layer_heading[n] = f"### Layer {n} - {layer_titles[n-1]}"
                    continue
            # Unknown heading: ignore and keep collecting under current section.
            continue

        if in_final:
            final_lines.append(raw)
        elif current_layer is not None:
            sections_by_layer[current_layer].append(raw)

    # If we didn't detect headings, use the whole text as raw material and distribute.
    if all(len(v) == 0 for v in sections_by_layer.values()) and not final_lines:
        blob_lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        blob = "\n".join(blob_lines)
        sentences = _first_sentences(blob, 16)
        idx = 0
        for n in range(1, max_layer + 1):
            sections_by_layer[n] = sentences[idx : idx + 3]
            idx += 3
        final_lines = sentences[idx : idx + 5]

    out_lines: List[str] = []

    for n in range(1, max_layer + 1):
        out_lines.append(seen_layer_heading.get(n, f"### Layer {n} - {layer_titles[n-1]}") )
        content = sections_by_layer.get(n) or []

        # Use new flexible extraction that handles multiple formats  
        layer_content = _extract_layer_content([ln for ln in content if (ln or "").strip()], text)
        out_lines.extend(layer_content)
        out_lines.append("")

    out_lines.append(f"### {final_title}")

    # Final summary: 3-5 short lines, never headings.
    raw_lines2 = [ln.strip() for ln in final_lines if ln.strip()]

    # Some models emit all summary bullets on one line like:
    #   "* A ... * B ... * C ..."
    # Split those into separate candidate lines.
    lines2: List[str] = []
    for ln in raw_lines2:
        # Split compact star-bullets: "* A ... * B ..."
        if ln.startswith("*") and " * " in ln:
            parts = [p.strip() for p in ln.split("*") if p.strip()]
            lines2.extend(parts)
            continue

        # Split compact dash-bullets: "- A ... - B ... - C ..."
        # Only split when it looks like multiple items.
        if ln.startswith("-") and " - " in ln:
            parts = [p.strip() for p in ln.split(" - ") if p.strip()]
            # If it didn't meaningfully split, keep as-is.
            if len(parts) > 1:
                lines2.extend(parts)
                continue

        lines2.append(ln)
    cleaned: List[str] = []
    for ln in lines2:
        if ln.lstrip().startswith("#"):
            continue
        # Never allow 'Bottom line:' style lines in the final summary.
        if re.match(r"^\s*([\-*]\s*)?(\*\*?)?bottom line(\*\*?)?\s*:", ln, flags=re.IGNORECASE):
            continue
        ln = re.sub(r"^[\-\*]\s+", "", ln)
        if ln:
            cleaned.append(_strip_redundant_prefixes(ln))
    if not cleaned:
        cleaned = [s for s in _first_sentences(text, 8) if "###" not in s and not s.strip().startswith("#")]
        cleaned = [s for s in cleaned if not s.strip().lower().startswith("layer ")]
        cleaned = [_strip_redundant_prefixes(c) for c in cleaned if c.strip()]

    cleaned = cleaned[:5]
    words = " ".join(cleaned).split()
    if view == "REQUEST_FLOW" and len(words) > 70:
        words = words[:70]
        cleaned = [" ".join(words)]
    out_lines.extend(cleaned)
    return "\n".join(out_lines).strip()
