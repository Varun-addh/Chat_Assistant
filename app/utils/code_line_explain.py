from __future__ import annotations

import ast
from typing import Dict, Iterable, Optional, Set


def explain_lines(
    *,
    code: str,
    language: str,
    line_numbers: Iterable[int],
    max_lines: int = 200,
) -> Dict[int, str]:
    """Return a mapping of line_number -> short explanation.

    This is intentionally deterministic and does NOT call an LLM.

    - Python: uses `ast` to generate best-effort explanations.
    - Other languages: falls back to a generic explanation using the source line text.
    """

    lang = (language or "").strip().lower()
    wanted: Set[int] = {int(n) for n in line_numbers if int(n) > 0}
    if not wanted:
        return {}

    # Hard cap to keep payload small.
    if max_lines < 1:
        max_lines = 1
    if max_lines > 2000:
        max_lines = 2000
    if len(wanted) > max_lines:
        wanted = set(sorted(wanted)[:max_lines])

    if lang == "python":
        return _explain_python_lines(code=code, line_numbers=wanted)

    return _explain_generic_lines(code=code, line_numbers=wanted)


def _safe_line_text(code: str, line_no: int) -> str:
    try:
        lines = code.splitlines()
        if 1 <= line_no <= len(lines):
            return lines[line_no - 1].strip()
    except Exception:
        pass
    return ""


def _explain_generic_lines(*, code: str, line_numbers: Set[int]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for ln in sorted(line_numbers):
        text = _safe_line_text(code, ln)
        if text:
            out[ln] = f"Executes: {text}"
        else:
            out[ln] = "Executes this line."
    return out


def _explain_python_lines(*, code: str, line_numbers: Set[int]) -> Dict[int, str]:
    out: Dict[int, str] = {}

    try:
        tree = ast.parse(code)
    except Exception:
        # If code doesn't parse (should be rare if it executed), fall back to generic.
        return _explain_generic_lines(code=code, line_numbers=line_numbers)

    # Map first significant node per line.
    first_node_by_line: Dict[int, ast.AST] = {}

    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", None)
        if not isinstance(lineno, int):
            continue
        if lineno not in line_numbers:
            continue
        if lineno not in first_node_by_line:
            first_node_by_line[lineno] = node

    for ln in sorted(line_numbers):
        node = first_node_by_line.get(ln)
        if node is None:
            # Could be a blank line, comment, or part of a multi-line statement.
            text = _safe_line_text(code, ln)
            out[ln] = f"Continues/executes part of a statement: {text}" if text else "Continues/executes part of a statement."
            continue

        out[ln] = _describe_python_node(node, code=code)

    return out


def _name_of_target(t: ast.AST) -> Optional[str]:
    if isinstance(t, ast.Name):
        return t.id
    if isinstance(t, ast.Attribute):
        base = _name_of_target(t.value)
        if base:
            return f"{base}.{t.attr}"
        return t.attr
    if isinstance(t, ast.Subscript):
        base = _name_of_target(t.value)
        return base or None
    return None


def _describe_python_node(node: ast.AST, *, code: str) -> str:
    # Statements
    if isinstance(node, ast.FunctionDef):
        return f"Defines function `{node.name}`."
    if isinstance(node, ast.AsyncFunctionDef):
        return f"Defines async function `{node.name}`."
    if isinstance(node, ast.ClassDef):
        return f"Defines class `{node.name}`."
    if isinstance(node, ast.Return):
        return "Returns from the current function."
    if isinstance(node, ast.If):
        return "Checks a condition and branches (if/else)."
    if isinstance(node, ast.For):
        return "Starts/continues a for-loop iteration."
    if isinstance(node, ast.While):
        return "Checks loop condition and continues a while-loop."
    if isinstance(node, ast.Break):
        return "Breaks out of the nearest loop."
    if isinstance(node, ast.Continue):
        return "Skips to the next loop iteration."
    if isinstance(node, ast.Import):
        names = ", ".join(n.name for n in node.names)
        return f"Imports module(s): {names}."
    if isinstance(node, ast.ImportFrom):
        mod = node.module or "(unknown)"
        names = ", ".join(n.name for n in node.names)
        return f"Imports {names} from `{mod}`."
    if isinstance(node, ast.Assign):
        targets = [n for n in (_name_of_target(t) for t in node.targets) if n]
        if targets:
            if len(targets) == 1:
                return f"Assigns a value to `{targets[0]}`."
            return f"Assigns values to {', '.join(f'`{t}`' for t in targets)}."
        return "Assigns a value."
    if isinstance(node, ast.AugAssign):
        target = _name_of_target(node.target)
        if target:
            return f"Updates `{target}` in-place (e.g., +=, -=, *=)."
        return "Updates a value in-place (e.g., +=, -=, *=)."

    # Expressions
    if isinstance(node, ast.Expr):
        # Common: function calls like print(...)
        if isinstance(node.value, ast.Call):
            fn = node.value.func
            if isinstance(fn, ast.Name):
                return f"Calls `{fn.id}(...)`."
            if isinstance(fn, ast.Attribute):
                base = _name_of_target(fn.value) or "object"
                return f"Calls `{base}.{fn.attr}(...)`."
            return "Calls a function."
        return "Evaluates an expression."

    # Fallback
    text = _safe_line_text(code, getattr(node, "lineno", 0) or 0)
    return f"Executes: {text}" if text else "Executes this line."