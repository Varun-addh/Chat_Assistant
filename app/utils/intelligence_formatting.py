"""
Formatting utilities for Interview Intelligence responses.

Extracted from app/routers/interview_intelligence.py to keep router files
focused on HTTP glue and maintain single-responsibility principle.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def auto_format_code_blocks(text: str) -> str:
    """
    Code block formatter with precision detection.
    Only wraps ACTUAL code, never regular text or explanations.

    Strategy:
    1. Detect explicit code markers (standalone language names)
    2. Identify continuous code blocks (multiple consecutive code lines)
    3. Require minimum 2-3 lines of code to avoid false positives
    4. Stop immediately when hitting regular text
    5. Preserve existing markdown code blocks
    """
    if not text or not text.strip():
        return text

    # Don't re-process if already has code blocks
    if '```' in text:
        return text

    lines = text.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip().lower()

        # CASE 1: Explicit standalone language marker (python, java, sql, etc.)
        if _is_language_marker(stripped) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and _is_definite_code(next_line):
                lang = _normalize_language(stripped)
                result.append(f'```{lang}')
                i += 1

                code_lines = []
                while i < len(lines):
                    current = lines[i]

                    if not current.strip():
                        if i + 1 < len(lines):
                            peek = lines[i + 1].strip()
                            if peek and (_is_language_marker(peek.lower()) or _is_section_header(peek) or _is_prose(peek)):
                                break
                        code_lines.append(current)
                        i += 1
                        continue

                    if _is_section_header(current.strip()) or _is_prose(current.strip()):
                        break

                    code_lines.append(current)
                    i += 1

                result.extend(code_lines)
                result.append('```')
                continue
            else:
                result.append(line)
                i += 1
                continue

        # CASE 2: Multi-line code block (no explicit marker)
        if _is_definite_code(line.strip()):
            code_count = 0
            temp_i = i
            while temp_i < len(lines) and temp_i < i + 10:
                if _is_definite_code(lines[temp_i].strip()) or not lines[temp_i].strip():
                    if lines[temp_i].strip():
                        code_count += 1
                    temp_i += 1
                else:
                    break

            if code_count >= 3:
                lang = _detect_language_from_code(line)
                result.append(f'```{lang}')

                while i < len(lines):
                    current = lines[i]

                    if not current.strip():
                        if i + 1 < len(lines):
                            peek = lines[i + 1].strip()
                            if peek and not _is_definite_code(peek):
                                result.append(current)
                                i += 1
                                break
                        result.append(current)
                        i += 1
                        continue

                    if _is_section_header(current.strip()) or _is_prose(current.strip()):
                        break

                    if not _is_definite_code(current.strip()):
                        break

                    result.append(current)
                    i += 1

                result.append('```')
                continue

        # CASE 3: Regular text - add as-is
        result.append(line)
        i += 1

    return '\n'.join(result)


def _is_language_marker(text: str) -> bool:
    """Check if text is a standalone language marker."""
    language_markers = {
        'python', 'java', 'javascript', 'typescript', 'sql',
        'cpp', 'c++', 'csharp', 'c#', 'bash', 'shell',
        'go', 'rust', 'ruby', 'php', 'swift', 'kotlin',
        'r', 'matlab', 'scala', 'perl', 'html', 'css'
    }
    return text in language_markers


def _normalize_language(lang: str) -> str:
    """Normalize language name for code block."""
    lang_map = {
        'c++': 'cpp',
        'c#': 'csharp',
        'shell': 'bash',
        'typescript': 'typescript',
        'javascript': 'javascript'
    }
    return lang_map.get(lang, lang)


def _is_definite_code(line: str) -> bool:
    """
    STRICT check: Is this definitely a line of code?
    Must have strong code indicators, not just any text.
    """
    if not line or len(line.strip()) < 2:
        return False

    stripped = line.strip()

    # Definitely NOT code (prose indicators)
    prose_indicators = [
        'example ', 'this is', 'this function', 'the above', 'the code',
        'you can', 'we can', 'to use', 'how to', 'what is',
        'when you', 'if you', 'best practice', 'common mistake',
        'note:', 'important:', 'tip:', 'warning:', 'output:',
        'as mde', 'as shown', 'for example', 'such as',
        'negative sampling', 'performance', 'real-world', 'follow-up'
    ]

    for indicator in prose_indicators:
        if indicator in stripped.lower():
            return False

    # Definitely code (strong indicators)
    definite_code_patterns = [
        r'^(import|from)\s+\w+(\.\w+)*',
        r'^import\s+\w+\.\w+',
        r'^def\s+\w+\s*\(',
        r'^class\s+\w+[\s\(:]',
        r'^function\s+\w+\s*\(',
        r'^(public|private|protected|static)\s+(class|void|int|String|function)',
        r'^\w+\s*=\s*[{\[\(\'\"]',
        r'^\w+\s*=\s*\w+\(',
        r'^\w+\s*=\s*\d+',
        r'^\w+\s*=\s*[\'"]',
        r'^(const|let|var)\s+\w+\s*=',
        r'^\w+\.\w+\(',
        r'^\w+\[\s*[\'\"]',
        r'^(if|for|while|switch|try|catch|except|finally)\s*[\(\:]',
        r'^(return|yield|break|continue)\s+',
        r'^(elif|else|endif)\s*:',
        r'^(SELECT|FROM|WHERE|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s+',
        r'^@\w+',
        r'^\w+\s*<-\s*',
        r'^model\.',
        r'^tf\.',
        r'^np\.',
        r'^pd\.',
        r'^plt\.',
        r'^(print|console\.log|echo|printf)\s*\(',
        r'^#\s*(TODO|FIXME|NOTE|HACK)',
    ]

    for pattern in definite_code_patterns:
        if re.match(pattern, stripped, re.IGNORECASE):
            return True

    if len(stripped.split()) == 1:
        return False

    code_chars = ['(', ')', '{', '}', '[', ']', '=', ';', '::', '->', '=>', '::']
    has_code_char = any(char in stripped for char in code_chars)

    if has_code_char:
        if ' = ' in stripped or '(' in stripped or '[' in stripped:
            if not re.search(r'\b(is|are|the|a|an|and|or|to|for|this|that|with|from)\b', stripped.lower()):
                return True

    return False


def _is_section_header(text: str) -> bool:
    """Check if text is a section header (markdown or caps)."""
    if not text:
        return False

    if text.startswith('#'):
        return True

    if text.isupper() and len(text.split()) <= 6:
        return True

    headers = [
        'example', 'output:', 'best practice', 'common mistake',
        'real-world application', 'follow-up', 'explanation',
        'how it works', 'summary', 'approach', 'solution',
        'practical', 'note:', 'tip:', 'warning:'
    ]

    text_lower = text.lower()
    for header in headers:
        if text_lower.startswith(header):
            return True

    return False


def _is_prose(text: str) -> bool:
    """
    Check if text is regular prose (explanation, not code).
    Prose has normal sentence structure.
    """
    if not text or len(text.strip()) < 10:
        return False

    text_lower = text.lower()

    prose_words = [
        'this is', 'this function', 'the above', 'the code', 'the model',
        'you can', 'we can', 'we use', 'it is', 'they are',
        'to use', 'to create', 'to calculate', 'to prevent',
        'how to', 'what is', 'when you', 'if you', 'why',
        'because', 'since', 'although', 'however', 'therefore',
        'for example', 'such as', 'like', 'similar to',
        'best practice', 'common mistake', 'important to',
        'remember', 'always', 'never', 'should', 'must',
        'negative sampling', 'gives auc', 'but model', 'versus',
        'cold-start', 'pre-train', 'initial weights', 'embedding layers'
    ]

    for word in prose_words:
        if word in text_lower:
            return True

    if text.endswith('.') or text.endswith(':') or text.endswith('?'):
        articles = ['the ', 'a ', 'an ', 'this ', 'that ', 'these ', 'those ']
        if any(article in text_lower for article in articles):
            return True

    return False


def _detect_language_from_code(line: str) -> str:
    """Detect programming language from a code line."""
    line_lower = line.lower()

    if any(x in line_lower for x in ['import ', 'def ', 'from ', 'print(', '__init__', 'self.', 'elif']):
        return 'python'

    if any(x in line_lower for x in ['const ', 'let ', 'var ', 'function ', '=>', 'console.log', 'require(']):
        return 'javascript'

    if any(x in line_lower for x in ['public class', 'private ', 'protected ', 'void ', 'static ', 'new ']):
        if 'using system' in line_lower or 'namespace' in line_lower:
            return 'csharp'
        return 'java'

    if any(x in line_lower for x in ['select ', 'from ', 'where ', 'insert ', 'update ', 'create table']):
        return 'sql'

    if any(x in line_lower for x in ['#include', 'using namespace', 'std::', 'int main(']):
        return 'cpp'

    if any(x in line_lower for x in ['#!/bin/', 'echo ', 'cd ', 'ls ', 'grep ', 'awk ']):
        return 'bash'

    if '<-' in line or any(x in line_lower for x in ['library(', 'ggplot(', 'data.frame(']):
        return 'r'

    return 'python'


def _unescape_common_whitespace_sequences(text: str) -> str:
    """Convert common escaped whitespace sequences into real whitespace."""
    s = text or ""
    if not s:
        return s

    if "\\n" in s and "\n" not in s:
        s = s.replace("\\r\\n", "\n")
        s = s.replace("\\n", "\n")
        s = s.replace("\\r", "")

    if "\\t" in s and "\t" not in s and "\n" in s:
        s = s.replace("\\t", "\t")

    return s


def format_coding_answer_for_interview_tab(
    answer: str,
    code_solution: Optional[str],
    is_coding: bool,
    language: Optional[str],
    time_complexity: Optional[str],
    space_complexity: Optional[str]
) -> str:
    """
    Format answer for Interview Intelligence tab.
    Prevents duplicate code blocks + auto-formats code examples.
    """
    if not is_coding:
        formatted = auto_format_code_blocks(_unescape_common_whitespace_sequences(answer or ""))
        return formatted

    text = _unescape_common_whitespace_sequences((answer or "").strip())
    code_solution = _unescape_common_whitespace_sequences((code_solution or "").strip()) or None
    if not text and not code_solution:
        return ""

    if _has_interview_structure(text):
        return auto_format_code_blocks(text)

    sections = _parse_markdown_sections(text)

    parts = []

    if code_solution:
        lang = (language or "python").lower()
        parts.append(f"## Solution\n\n```{lang}\n{code_solution.strip()}\n```")

        if (time_complexity or space_complexity) and not _answer_has_complexity(text):
            complexity_parts = []
            if time_complexity:
                complexity_parts.append(f"**Time:** {time_complexity}")
            if space_complexity:
                complexity_parts.append(f"**Space:** {space_complexity}")
            parts.append("\n" + " | ".join(complexity_parts))

        parts.append("")

    if text:
        sections = _parse_markdown_sections(text)
        explanation = sections.get('explanation', '')
        approach = sections.get('approach', '')
        summary = sections.get('summary', '')
        formatted_explanation = ''
        if explanation:
            bullets = re.findall(r'^[-*•]\s*(.+)$', explanation, re.MULTILINE)
            if bullets:
                formatted_explanation = '\n'.join([f'- {b.strip()}' for b in bullets])
            else:
                sentences = re.split(r'(?<=[.!?])\s+', explanation)
                formatted_explanation = '\n'.join([f'- {s.strip()}' for s in sentences if len(s.strip()) > 10])
        if summary:
            parts.append(f'**Summary:**\n{summary.strip()}')
        if approach:
            parts.append(f'**Approach:**\n{approach.strip()}')
        if formatted_explanation:
            parts.append(f'**Explanation:**\n{formatted_explanation}')
        else:
            parts.append(text)

    final_output = "\n".join(parts)
    return auto_format_code_blocks(final_output)


def _extract_explanation_only(sections: dict, full_text: str) -> str:
    """Extract ONLY the explanation text, NO CODE."""
    explanation_text = ""

    if sections.get('explanation'):
        explanation_text = sections['explanation']
    elif sections.get('other'):
        explanation_text = sections['other']
    else:
        explanation_text = full_text

    cleaned = _remove_all_code_aggressive(explanation_text)
    return cleaned


def _remove_all_code_aggressive(text: str) -> str:
    """Remove ALL forms of code from text."""
    if not text:
        return ""

    text = re.sub(r'```[\s\S]*?```', '', text, flags=re.DOTALL)

    text = re.sub(r'#+\s*Code\s*Solution[\s\S]*?(?=\n##|\n#|\Z)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'#+\s*Solution[\s\S]*?(?=\n##|\n#|\Z)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Code Solution[\s\S]*?(?=\n##|\n#|\Z)', '', text, flags=re.IGNORECASE)

    lines = text.split('\n')
    filtered_lines = []
    skip_until_blank = False

    for line in lines:
        if re.match(r'^\s{4,}\S', line):
            skip_until_blank = True
            continue

        if skip_until_blank:
            if line.strip():
                continue
            else:
                skip_until_blank = False

        filtered_lines.append(line)

    text = '\n'.join(filtered_lines)

    lines = text.split('\n')
    filtered_lines = []

    for line in lines:
        stripped = line.strip()

        if any([
            re.match(r'^def\s+\w+\s*\(', stripped),
            re.match(r'^function\s+\w+\s*\(', stripped),
            re.match(r'^var\s+\w+\s*=', stripped),
            re.match(r'^const\s+\w+\s*=', stripped),
            re.match(r'^let\s+\w+\s*=', stripped),
            re.match(r'^class\s+\w+', stripped),
            re.match(r'^public\s+\w+\s+\w+\s*\(', stripped),
            re.match(r'^\w+\s*=\s*function\s*\(', stripped),
            re.match(r'^if\s+.*:\s*$', stripped),
            re.match(r'^for\s+.*:\s*$', stripped),
            re.match(r'^while\s+.*:\s*$', stripped),
            stripped.startswith('return '),
            stripped.startswith('console.log'),
            stripped.startswith('print('),
        ]):
            continue

        filtered_lines.append(line)

    text = '\n'.join(filtered_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    lines = text.split('\n')
    filtered_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped in ['{', '}', '(', ')', ';', ':', ','] or re.match(r'^[\{\}\(\);:,\s]+$', stripped):
            continue
        filtered_lines.append(line)

    text = '\n'.join(filtered_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


def _has_interview_structure(text: str) -> bool:
    """Check if answer already has good interview structure."""
    has_approach = bool(re.search(r'\*\*Approach:\*\*|\n## Approach', text))
    has_solution = bool(re.search(r'## Solution|```\w+\n', text))
    has_explanation = bool(re.search(r'## How It Works|## Explanation', text))

    return has_approach and has_solution and has_explanation


def _answer_has_complexity(answer: str) -> bool:
    """Detect if answer already contains time/space complexity notes."""
    if not answer:
        return False
    if re.search(r"\b(time complexity|space complexity|time:|space:|complexity:|O\()\b", answer, re.I):
        return True
    for line in answer.splitlines():
        if re.search(r"\b(time|space)\b.*O\(|\bcomplexity\b", line, re.I):
            return True
    return False


def _parse_markdown_sections(text: str) -> dict:
    """Parse markdown text into logical sections."""
    sections = {
        'summary': '',
        'approach': '',
        'solution': '',
        'explanation': '',
        'complexity': '',
        'edge_cases': '',
        'optimization': '',
        'other': ''
    }

    lines = text.split('\n')
    current_section = 'other'
    current_content = []

    for line in lines:
        lower_line = line.lower().strip()

        if re.match(r'^#{1,3}\s*(complete answer|summary)', lower_line):
            sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'summary'
            current_content = []
        elif re.match(r'^#{1,3}\s*(approach|algorithm|strategy)', lower_line):
            sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'approach'
            current_content = []
        elif re.match(r'^#{1,3}\s*(solution|code)', lower_line):
            sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'solution'
            current_content = []
        elif re.match(r'^#{1,3}\s*(how it works|explanation|walkthrough)', lower_line):
            sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'explanation'
            current_content = []
        elif re.match(r'^#{1,3}\s*complexity', lower_line):
            sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'complexity'
            current_content = []
        elif re.match(r'^#{1,3}\s*(edge case|corner case)', lower_line):
            sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'edge_cases'
            current_content = []
        elif re.match(r'^#{1,3}\s*(optimization|performance)', lower_line):
            sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'optimization'
            current_content = []
        else:
            current_content.append(line)

    sections[current_section] = '\n'.join(current_content).strip()
    return sections


def _extract_approach_summary(sections: dict, full_text: str) -> str:
    """Extract concise 2-3 line approach."""
    for section_key in ['summary', 'approach', 'other']:
        content = sections.get(section_key, '')
        if not content:
            continue

        content = _remove_all_code_aggressive(content)

        bullets = re.findall(r'^\s*[-*•]\s*(.+)$', content, re.MULTILINE)
        if bullets:
            relevant = [b for b in bullets if len(b) > 20][:3]
            if relevant:
                return ' '.join(relevant)

        sentences = re.split(r'(?<=[.!?])\s+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        if len(sentences) >= 2:
            return ' '.join(sentences[:2])

    clean_text = _remove_all_code_aggressive(full_text)
    paragraphs = [p.strip() for p in clean_text.split('\n\n') if p.strip()]
    if paragraphs:
        first = paragraphs[0]
        first = re.sub(r'^#{1,6}\s+', '', first)
        first = re.sub(r'\*\*(.+?)\*\*', r'\1', first)
        sentences = re.split(r'(?<=[.!?])\s+', first)
        return ' '.join(sentences[:2])[:250]

    return ""


def clean_history_metadata(metadata: dict) -> dict:
    """Remove fields that shouldn't be shown in history sidebar."""
    cleaned = dict(metadata)
    cleaned.pop('avg_credibility', None)
    return cleaned


def apply_formatting_to_questions(questions: list) -> list:
    """Apply formatting to a list of question dictionaries."""
    formatted_questions = []

    for qq in (questions or []):
        try:
            formatted = dict(qq)
            formatted['answer'] = format_coding_answer_for_interview_tab(
                qq.get('answer', ''),
                qq.get('code_solution'),
                qq.get('is_coding_question', False),
                qq.get('language'),
                qq.get('time_complexity'),
                qq.get('space_complexity')
            )
            formatted['code_solution'] = None
            formatted_questions.append(formatted)
        except Exception as e:
            logger.error(f"Failed to format question: {e}")
            formatted_questions.append(qq)

    return formatted_questions
