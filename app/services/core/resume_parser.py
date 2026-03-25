"""
Resume Parser Service — Parse → Structure → Embed → Delete Raw.

Extracts structured JSON from uploaded resumes for claim-based interview probing.
Privacy-safe: only the structured representation is stored, never the raw file.
"""

import logging
import json
import re
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured resume schema returned by the parser
# ---------------------------------------------------------------------------

RESUME_PARSE_PROMPT = """You are a world-class resume analyst. Extract structured data from the resume text below.

Return ONLY a valid JSON object with this exact schema (no markdown, no commentary):

{
  "skills": ["skill1", "skill2", ...],
  "projects": [
    {
      "name": "Project name",
      "tech": ["Technology1", "Technology2"],
      "claims": ["Quantifiable claim or achievement from this project"]
    }
  ],
  "experience_summary": "One-paragraph summary of total professional experience",
  "role_titles": ["Most recent role", "Previous role", ...],
  "education": "Degree and institution (or 'Not specified')",
  "achievements": ["Quantifiable achievement 1", "Quantifiable achievement 2", ...],
  "years_of_experience": 0,
  "primary_domain": "Best-guess primary domain (e.g., Backend Engineering, Data Science, Frontend, DevOps, ML)"
}

RULES:
1. Extract ALL quantifiable claims (percentages, metrics, scale). Put them in "achievements" AND in the relevant project's "claims".
2. If a field cannot be determined, use an empty list [] or "Not specified".
3. "years_of_experience" should be an integer estimate from work history dates. Use 0 if unclear.
4. "skills" should list ALL technologies, frameworks, languages, and tools mentioned.
5. Return valid JSON only — no trailing commas, no comments.

RESUME TEXT:
---
{resume_text}
---

JSON:"""


MAX_RESUME_TEXT_FOR_LLM = 6000  # chars — keeps within context budget


class ResumeParseResult:
    """Structured output from resume parsing.

    Thin wrapper around a dict that is fully compatible with
    ``app.schemas.ResumeContext`` (same field names / defaults).
    Keeps backward-compat ``ResumeParseResult(dict)`` constructor
    while also accepting Pydantic-style ``**kwargs``.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None, **kwargs):
        src = data if data is not None else kwargs
        self.skills: List[str] = src.get("skills", [])
        self.projects: List[Dict[str, Any]] = src.get("projects", [])
        self.experience_summary: str = src.get("experience_summary", "Not specified")
        self.role_titles: List[str] = src.get("role_titles", [])
        self.education: str = src.get("education", "Not specified")
        self.achievements: List[str] = src.get("achievements", [])
        self.years_of_experience: int = src.get("years_of_experience", 0)
        self.primary_domain: str = src.get("primary_domain", "General Software Engineering")

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skills": self.skills,
            "projects": self.projects,
            "experience_summary": self.experience_summary,
            "role_titles": self.role_titles,
            "education": self.education,
            "achievements": self.achievements,
            "years_of_experience": self.years_of_experience,
            "primary_domain": self.primary_domain,
        }

    def to_resume_context(self):
        """Convert to the canonical Pydantic ``ResumeContext`` schema."""
        from app.schemas import ResumeContext
        return ResumeContext(**self.to_dict())

    # -- prompt helpers -------------------------------------------------------

    def to_prompt_block(self) -> str:
        """Build a compact prompt block for injection into interview prompts."""
        lines = ["CANDIDATE RESUME CONTEXT:"]

        if self.experience_summary and self.experience_summary != "Not specified":
            lines.append(f"  Summary: {self.experience_summary}")

        if self.role_titles:
            lines.append(f"  Roles: {', '.join(self.role_titles[:5])}")

        if self.skills:
            lines.append(f"  Skills: {', '.join(self.skills[:15])}")

        if self.education and self.education != "Not specified":
            lines.append(f"  Education: {self.education}")

        if self.projects:
            lines.append("  Key Projects:")
            for proj in self.projects[:5]:
                name = proj.get("name", "Unnamed") if isinstance(proj, dict) else getattr(proj, "name", "Unnamed")
                tech_list = proj.get("tech", []) if isinstance(proj, dict) else getattr(proj, "tech", [])
                claims = proj.get("claims", []) if isinstance(proj, dict) else getattr(proj, "claims", [])
                lines.append(f"    - {name} [{', '.join(tech_list[:5])}]")
                for claim in claims[:3]:
                    lines.append(f"      • Claim: {claim}")

        if self.achievements:
            lines.append("  Notable Achievements:")
            for ach in self.achievements[:8]:
                lines.append(f"    • {ach}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Text extraction helpers (same formats the copilot upload already supports)
# ---------------------------------------------------------------------------

# Magic bytes for file-type validation
_PDF_MAGIC = b"%PDF"
_DOCX_MAGIC = b"PK\x03\x04"  # ZIP (Office Open XML)


def _validate_magic_bytes(content: bytes, ext: str) -> None:
    """Ensure file content matches its declared extension."""
    if ext == ".pdf" and not content[:4].startswith(_PDF_MAGIC):
        raise ValueError("File has .pdf extension but does not appear to be a valid PDF")
    if ext == ".docx" and not content[:4].startswith(_DOCX_MAGIC):
        raise ValueError("File has .docx extension but does not appear to be a valid DOCX")


def extract_text_from_bytes(content: bytes, filename: str) -> str:
    """Extract plain text from resume file bytes."""
    ext = Path(filename).suffix.lower()

    if ext in (".txt", ".md"):
        return content.decode("utf-8", errors="replace")

    if ext == ".pdf":
        _validate_magic_bytes(content, ext)
        return _extract_pdf_text(content)

    if ext == ".docx":
        _validate_magic_bytes(content, ext)
        return _extract_docx_text(content)

    raise ValueError(f"Unsupported file type: {ext}. Use .txt, .md, .pdf, or .docx")


def _extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF bytes.  Tries multiple libraries in order."""
    import io

    # 1. PyMuPDF (fastest, best quality)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=content, filetype="pdf")
        try:
            pages = [page.get_text() for page in doc]
        finally:
            doc.close()
        return "\n".join(pages)
    except ImportError:
        logger.debug("PyMuPDF not available, trying next extractor")
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}, trying next extractor")

    # 2. pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        logger.debug("pdfplumber not available, trying next extractor")
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}, trying next extractor")

    # 3. PyPDF2 (already in requirements.txt)
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except ImportError:
        logger.debug("PyPDF2 not available, trying next extractor")
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}, trying next extractor")

    # 4. pdfminer.six (already in requirements.txt)
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        return pdfminer_extract(io.BytesIO(content))
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"pdfminer extraction failed: {e}")

    raise ImportError(
        "No PDF library available. Install one of: pymupdf, pdfplumber, PyPDF2, or pdfminer.six"
    )


def _extract_docx_text(content: bytes) -> str:
    """Extract text from DOCX bytes."""
    try:
        import docx
        import io
        doc = docx.Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        raise ImportError("Install python-docx (`pip install python-docx`) for DOCX support")


# ---------------------------------------------------------------------------
# Core parsing pipeline
# ---------------------------------------------------------------------------

async def parse_resume(
    file_content: bytes,
    filename: str,
    api_key: Optional[str] = None,
) -> ResumeParseResult:
    """
    Parse → Structure → return structured result.
    
    The raw file content is NOT stored anywhere — only the structured JSON is returned.
    The caller is responsible for storing the structured result and discarding the raw file.
    """
    # Step 1: Extract text
    raw_text = extract_text_from_bytes(file_content, filename)
    if not raw_text or len(raw_text.strip()) < 50:
        raise ValueError("Resume appears to be empty or too short to parse")

    logger.info(f"📄 Extracted {len(raw_text)} chars from {filename}")

    # Step 2: Truncate for LLM context budget
    if len(raw_text) > MAX_RESUME_TEXT_FOR_LLM:
        raw_text = raw_text[:MAX_RESUME_TEXT_FOR_LLM]
        logger.info(f"📄 Truncated resume text to {MAX_RESUME_TEXT_FOR_LLM} chars for LLM")

    # Step 3: LLM-parse into structured JSON
    structured = await _llm_parse_resume(raw_text, api_key)

    # Step 4: raw_text goes out of scope here — never stored
    logger.info(
        f"✅ Resume parsed: {len(structured.skills)} skills, "
        f"{len(structured.projects)} projects, "
        f"{len(structured.achievements)} achievements"
    )
    return structured


async def parse_resume_from_text(
    raw_text: str,
    api_key: Optional[str] = None,
) -> ResumeParseResult:
    """Parse resume from already-extracted text (e.g. from QuickStart context field)."""
    if not raw_text or len(raw_text.strip()) < 50:
        raise ValueError("Resume text is too short to parse")

    if len(raw_text) > MAX_RESUME_TEXT_FOR_LLM:
        raw_text = raw_text[:MAX_RESUME_TEXT_FOR_LLM]

    return await _llm_parse_resume(raw_text, api_key)


async def _llm_parse_resume(text: str, api_key: Optional[str] = None) -> ResumeParseResult:
    """Use the configured LLM to extract structured resume data."""
    from app.services.chat.llm_service import llm_service

    prompt = RESUME_PARSE_PROMPT.replace("{resume_text}", text)

    try:
        response = await llm_service.generate_text(
            prompt=prompt,
            api_key=api_key,
            json_mode=True,
            temperature=0.1,  # Low temperature for factual extraction
            max_tokens=3000,
        )
    except Exception as e:
        logger.error(f"LLM resume parsing failed: {e}")
        # Fallback: basic regex extraction
        return _fallback_parse(text)

    if not response:
        logger.warning("LLM returned empty response for resume parse, using fallback")
        return _fallback_parse(text)

    # Parse LLM JSON response
    try:
        data = _extract_json(response)
        return ResumeParseResult(data)
    except Exception as e:
        logger.warning(f"Failed to parse LLM resume response: {e}, using fallback")
        return _fallback_parse(text)


def _extract_json(text: str) -> Dict[str, Any]:
    """Robust JSON extraction from LLM response."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}")


def _fallback_parse(text: str) -> ResumeParseResult:
    """Basic regex-based fallback when LLM is unavailable."""
    # Extract skills using common technology keywords
    tech_keywords = {
        "python", "java", "javascript", "typescript", "react", "angular", "vue",
        "node", "nodejs", "express", "django", "flask", "fastapi", "spring",
        "docker", "kubernetes", "aws", "azure", "gcp", "sql", "postgresql",
        "mongodb", "redis", "kafka", "rabbitmq", "git", "ci/cd", "terraform",
        "go", "rust", "c++", "c#", ".net", "ruby", "rails", "php", "swift",
        "kotlin", "scala", "hadoop", "spark", "tensorflow", "pytorch",
        "machine learning", "deep learning", "data science", "devops",
        "microservices", "rest", "graphql", "grpc",
        "next.js", "nextjs", "svelte", "remix", "astro", "bun", "deno",
        "tailwind", "prisma", "supabase", "vercel", "vite", "playwright",
        "cypress", "langchain", "openai", "llm", "rag", "vector database",
    }

    text_lower = text.lower()
    found_skills = [kw for kw in tech_keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]

    # Extract years of experience
    yoe_match = re.search(r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)", text_lower)
    years = int(yoe_match.group(1)) if yoe_match else 0

    # Extract achievements (lines with numbers/percentages)
    achievements = []
    for line in text.split("\n"):
        line = line.strip()
        if re.search(r"\d+%|\d+x|reduced|improved|increased|scaled|optimized|built|designed|led|managed", line.lower()):
            if 20 < len(line) < 300:
                achievements.append(line.lstrip("•-– "))

    return ResumeParseResult({
        "skills": found_skills[:20],
        "projects": [],
        "experience_summary": text[:300] + "..." if len(text) > 300 else text,
        "role_titles": [],
        "education": "Not specified",
        "achievements": achievements[:10],
        "years_of_experience": years,
        "primary_domain": "General Software Engineering",
    })
