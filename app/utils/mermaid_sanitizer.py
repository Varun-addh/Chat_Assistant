from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


DIAGRAM_TYPES: List[str] = [
    "flowchart",
    "graph",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
    "erDiagram",
    "journey",
    "gantt",
    "pie",
    "gitGraph",
    "mindmap",
    "timeline",
]


@dataclass(frozen=True)
class MermaidSanitizeResult:
    code: str
    changed: bool
    stages: List[str]


class MermaidSanitizer:
    """Single source of truth for Mermaid cleanup.

    Goal: keep Mermaid code within a renderer-friendly subset (Kroki/mermaid.ink)
    while staying deterministic and debuggable.

    Notes:
    - This is intentionally conservative; it removes styling/init directives.
    - This module should NOT import FastAPI routers/services (avoid cycles).
    """

    # --- Public API -----------------------------------------------------

    @staticmethod
    def sanitize_code_block(raw: str) -> str:
        """Remove markdown fences if present, fix escaped newlines, and de-dup diagram declarations."""
        result = MermaidSanitizer.sanitize(raw, mode="code_block")
        return result.code

    @staticmethod
    def sanitize_from_llm(mermaid_code: str) -> str:
        """Cleanup Mermaid returned by an LLM (CSS/HTML/init leakage, basic syntax fixes, header)."""
        result = MermaidSanitizer.sanitize(mermaid_code, mode="llm")
        return result.code

    @staticmethod
    def sanitize_subset(mermaid_code: str) -> str:
        """Keep Mermaid within a safe subset (strip styling/init and non-ascii)."""
        result = MermaidSanitizer.sanitize(mermaid_code, mode="subset")
        return result.code

    @staticmethod
    def ultra_simplify(mermaid_code: str) -> str:
        """Last-resort simplification; should be used only as a fallback."""
        result = MermaidSanitizer.sanitize(mermaid_code, mode="ultra")
        return result.code

    @staticmethod
    def sanitize(
        text: str,
        *,
        mode: str = "render",
    ) -> MermaidSanitizeResult:
        """Run a deterministic sanitize pipeline.

        Modes:
        - code_block: strip markdown fences, unescape, de-dup diagram declarations
        - llm: remove CSS/HTML/init artifacts + basic syntax fix + ensure header
        - subset: ASCII normalize + remove init/style directives + class assignments
        - render: best-effort fixes for renderer compatibility (subset + syntax + edge labels)
        - ultra: extreme simplification to increase chance of rendering
        """
        original = text or ""
        code = original
        stages: List[str] = []

        def mark(stage: str, new_code: str) -> str:
            nonlocal code
            if new_code != code:
                stages.append(stage)
                code = new_code
            return code

        if not code:
            return MermaidSanitizeResult(code="", changed=False, stages=[])

        # Normalize newlines early
        mark("normalize_newlines", code.replace("\r\n", "\n").replace("\r", "\n"))

        if mode in {"code_block", "llm", "render", "subset", "ultra"}:
            mark("unescape_newlines", MermaidSanitizer._unescape_newlines(code))
            mark("strip_markdown_fences", MermaidSanitizer._strip_markdown_fences(code))
            mark("fix_duplicate_declarations", MermaidSanitizer.fix_duplicate_diagram_declarations(code))

        if mode in {"llm", "render", "subset", "ultra"}:
            mark("strip_css_html", MermaidSanitizer._strip_css_html(code))

        if mode in {"subset", "render", "ultra"}:
            mark("to_ascii", MermaidSanitizer.to_ascii(code))
            mark("strip_features", MermaidSanitizer.strip_features(code))

        if mode in {"llm", "render", "ultra"}:
            mark("normalize_arrows", MermaidSanitizer._normalize_basic_arrows(code))
            mark("ensure_header", MermaidSanitizer._ensure_diagram_header(code))

        if mode in {"render", "ultra"}:
            mark("fix_mermaid_syntax_errors", MermaidSanitizer.fix_mermaid_syntax_errors(code))
            mark("sanitize_edge_labels", MermaidSanitizer.sanitize_edge_labels(code))

        if mode == "ultra":
            mark("ultra_simplify", MermaidSanitizer._ultra_simplify_mermaid(code))

        return MermaidSanitizeResult(code=code.strip(), changed=(code != original), stages=stages)

    # --- Building blocks ------------------------------------------------

    @staticmethod
    def _unescape_newlines(text: str) -> str:
        # Fix escaped newlines (\\n -> \n) that LLMs sometimes generate
        return text.replace("\\n", "\n") if "\\n" in text else text

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        s = (text or "").strip()
        if not s.startswith("```"):
            return s

        lines = s.split("\n")
        if not lines:
            return s

        body = "\n".join(lines[1:])
        if body.rstrip().endswith("```"):
            body = body[: body.rfind("```")].rstrip()
        return body

    @staticmethod
    def fix_duplicate_diagram_declarations(text: str) -> str:
        lines = (text or "").split("\n")
        if len(lines) < 2:
            return text

        first_found = False
        cleaned: List[str] = []
        for line in lines:
            stripped = line.strip().lower()
            is_decl = any(stripped.startswith(dtype.lower()) for dtype in DIAGRAM_TYPES)
            if is_decl:
                if not first_found:
                    cleaned.append(line)
                    first_found = True
                # skip subsequent declarations
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    @staticmethod
    def to_ascii(text: str) -> str:
        """Normalize common unicode punctuation/arrows, then drop remaining non-ascii."""
        if not text:
            return ""
        replacements = {
            "→": "->",
            "←": "<-",
            "⇒": "=>",
            "⇐": "<=",
            "↔": "<->",
            "•": "-",
            "“": '"',
            "”": '"',
            "’": "'",
            "–": "-",
            "—": "-",
            "…": "...",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return "".join(ch if ord(ch) < 128 else "" for ch in text)

    @staticmethod
    def strip_features(text: str) -> str:
        """Remove init blocks, classDef/linkStyle, class assignments, and CSS artifacts."""
        if not text:
            return ""

        out_lines: List[str] = []
        for line in text.split("\n"):
            s = line.strip()
            if s.startswith("%%{init"):
                continue
            if s.lower().startswith("classdef"):
                continue
            if s.lower().startswith("linkstyle"):
                continue

            # Skip CSS artifacts that got mixed into Mermaid code
            if any(css_marker in s for css_marker in ["@keyframes", "@import", "#container", ".edge-", "#mermaid-svg"]):
                continue

            if ":::" in line:
                line = re.sub(r":::\w+", "", line)

            out_lines.append(line)

        out = "\n".join(out_lines)
        # Remove htmlLabels directive if present inside init-like remnants
        out = re.sub(r"'htmlLabels'\s*:\s*(true|false)", "", out, flags=re.IGNORECASE)
        return out.strip()

    @staticmethod
    def _strip_css_html(text: str) -> str:
        """Remove common CSS/HTML leak lines from Mermaid text."""
        if not text:
            return ""

        reject_patterns = [
            "@import",
            "@keyframes",
            "@font-face",
            "@media",
            "#mermaid-svg",
            "#container",
            ".edge-",
            ".node-",
            "<style>",
            "</style>",
            "<script>",
            "</script>",
            "font-family:",
            "font-size:",
            "fill:",
            "stroke:",
            "background-color:",
            "color:",
            "opacity:",
            "trebuchet",
            "verdana",
            "sans-serif",
            "}}#",
            "{font-",
            ":root{",
        ]

        cleaned_lines: List[str] = []
        for line in text.split("\n"):
            stripped = line.strip()

            # Skip empty lines at the start
            if not stripped and not cleaned_lines:
                continue

            low = line.lower()

            # Skip init blocks (standalone or multi-line remnants)
            if stripped.startswith("%%{init") or stripped.endswith("}%%"):
                continue

            if any(p in low for p in reject_patterns):
                continue

            # Skip lines that look like pure CSS selectors (contain { } but not node/edge syntax)
            if "{" in line and "}" in line and not any(x in line for x in ["[", "]", "(", ")"]):
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def _normalize_basic_arrows(text: str) -> str:
        """Fix common arrow mistakes: 'A -> B' and 'A - B' to Mermaid-friendly 'A --> B'."""
        if not text:
            return ""
        # Restrict to simple node tokens to avoid rewriting inside labels too aggressively.
        text = re.sub(r"(\w+|\])\s*->\s*(\w+|\[)", r"\1 --> \2", text)
        text = re.sub(r"(\w+|\])\s+-\s+(\w+|\[)", r"\1 --> \2", text)
        return text

    @staticmethod
    def _ensure_diagram_header(code: str) -> str:
        """Ensure the first non-empty line is a supported diagram header; default to flowchart LR."""
        if not code:
            return ""

        lines = code.split("\n")
        first_non_empty_idx: Optional[int] = None
        for i, ln in enumerate(lines):
            if ln.strip():
                first_non_empty_idx = i
                break

        if first_non_empty_idx is None:
            return "flowchart LR\n  A[Empty]"

        first = lines[first_non_empty_idx].strip().lower()
        # Mermaid is case sensitive for some headers; we validate lowercase starts.
        if not any(first.startswith(t.lower()) for t in ["flowchart", "graph", "sequencediagram", "classdiagram", "statediagram", "erdiagram"]):
            return "flowchart LR\n" + code

        return code

    @staticmethod
    def _infer_diagram_kind(code: str) -> str:
        """Best-effort diagram kind inference from the first non-empty line."""
        if not code:
            return "unknown"
        for ln in code.split("\n"):
            s = ln.strip().lower()
            if not s:
                continue
            if s.startswith("sequencediagram"):
                return "sequence"
            if s.startswith("classdiagram"):
                return "class"
            if s.startswith("erdiagram"):
                return "er"
            if s.startswith("flowchart") or s.startswith("graph"):
                return "flowchart"
            break
        return "unknown"

    @staticmethod
    def sanitize_edge_labels(code: str) -> str:
        """Sanitize edge labels like -->| (8) [Mobile Push]| which can break parsing."""
        if not code:
            return ""

        def sanitize_pipe_label(match: re.Match[str]) -> str:
            arrow = match.group(1)
            label = match.group(2)
            sanitized_label = label.replace("[", "(").replace("]", ")")
            return f"{arrow}|{sanitized_label}|"

        out = re.sub(r"([-=\.]+>)\|([^\|]+)\|", sanitize_pipe_label, code)

        # Fix a very common LLM bug: adding an extra '>' after the pipe label.
        # Bad:  A -->|Request Ride|> B
        # Good: A -->|Request Ride| B
        out = re.sub(r"(\|[^\|]{1,200}\|)\s*>\s*", r"\1 ", out)

        return out

    @staticmethod
    def fix_mermaid_syntax_errors(code: str) -> str:
        """Fix common Mermaid syntax errors to prevent rendering failures."""
        if not code:
            return ""

        # Normalize flowchart-only arrow typos before deeper fixes.
        # LLMs often output `-->>` or `->>` (sequence style) inside flowcharts.
        kind = MermaidSanitizer._infer_diagram_kind(code)
        if kind == "flowchart":
            code = re.sub(r"-->>", "-->", code)
            code = re.sub(r"->>", "-->", code)

        # Fix bidirectional arrows: A <--> B becomes A --> B and B --> A
        bidirectional_pattern = re.compile(r"^\s*([A-Za-z0-9_]+)\s*<-->\s*([A-Za-z0-9_]+)\s*$", re.MULTILINE)

        def replace_bidirectional(match: re.Match[str]) -> str:
            from_node = match.group(1)
            to_node = match.group(2)
            return f"  {from_node} --> {to_node}\n  {to_node} --> {from_node}"

        def fix_bidirectional_arrows(text: str) -> str:
            return bidirectional_pattern.sub(replace_bidirectional, text)

        # Fix edge labels: A -- Label --> B becomes A -->|Label| B
        edge_label_pattern = re.compile(
            r"^\s*([A-Za-z0-9_]+)\s*--\s*([^-\n]+?)\s*-->\s*([A-Za-z0-9_]+)\s*$",
            re.MULTILINE,
        )

        def replace_edge_label(match: re.Match[str]) -> str:
            from_node = match.group(1).strip()
            label = match.group(2).strip()
            to_node = match.group(3).strip()
            return f"  {from_node} -->|{label}| {to_node}"

        def fix_edge_labels(text: str) -> str:
            return edge_label_pattern.sub(replace_edge_label, text)

        # Fix malformed edges like A --> B -- Label --> C
        malformed_pattern = re.compile(
            r"^\s*([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)\s*--\s*([^-\n]+?)\s*-->\s*([A-Za-z0-9_]+)\s*$",
            re.MULTILINE,
        )

        def replace_malformed(match: re.Match[str]) -> str:
            from_node = match.group(1).strip()
            middle_node = match.group(2).strip()
            label = match.group(3).strip()
            to_node = match.group(4).strip()
            return f"  {from_node} --> {middle_node}\n  {middle_node} -->|{label}| {to_node}"

        def fix_malformed_edges(text: str) -> str:
            return malformed_pattern.sub(replace_malformed, text)

        # Fix nested parentheses in labels: ID((Label (Text))) becomes ID(("Label (Text)"))
        def fix_nested_labels(text: str) -> str:
            patterns = [
                (r"([A-Za-z0-9_]+)\(\(([^\"]+?)\)\)", r"\1((\"\2\"))"),
                (r"([A-Za-z0-9_]+)\[\(([^\"]+?)\)\]", r"\1([ \"\2\" ])"),
                (r"([A-Za-z0-9_]+)\[([^\"\]]+?)\]", r"\1[\"\2\"]"),
                (r"([A-Za-z0-9_]+)\(([^\"\)]+?)\)", r"\1(\"\2\")"),
                (r"([A-Za-z0-9_]+)\{([^\"\}]+?)\}", r"\1{\"\2\"}"),
            ]

            for pattern, _replacement in patterns:

                def quote_if_needed(match: re.Match[str]) -> str:
                    node_id = match.group(1)
                    label = match.group(2).strip()
                    if any(c in label for c in "()/\\:-"):
                        if "((" in pattern:
                            return f'{node_id}(("{label}"))'
                        if "[(" in pattern:
                            return f'{node_id}(["{label}"])'
                        if "[" in pattern:
                            return f'{node_id}["{label}"]'
                        if "(" in pattern:
                            return f'{node_id}("{label}")'
                        if "{" in pattern:
                            return f'{node_id}{{"{label}"}}'
                    return match.group(0)

                text = re.sub(pattern, quote_if_needed, text)

            return text

        def fix_special_chars(text: str) -> str:
            lines = text.split("\n")
            out_lines: List[str] = []
            for line in lines:
                # Fix double colons (::) BUT preserve class assignments (:::)
                line = re.sub(r"(?<!:)::(?!:)", "-", line)

                # Fix slashes in node labels: [text/with/slashes] -> ["text-with-slashes"]
                while "/" in line and "[" in line:
                    old_line = line
                    line = re.sub(r"\[([^\]]*)/([^\]]*)\]", r"[\"\1-\2\"]", line)
                    if old_line == line:
                        break

                # Fix colons in node labels: [text:with:colons] -> ["text-with-colons"]
                if '["' not in line and '"]' not in line and ":::" not in line:
                    while ":" in line and "[" in line and "::" not in line:
                        old_line = line
                        line = re.sub(r"\[([^\]]*):([^\]]*)\]", r"[\"\1-\2\"]", line)
                        if old_line == line:
                            break

                out_lines.append(line)
            return "\n".join(out_lines)

        def fix_wrong_diagram_type(text: str) -> str:
            """Fix when LLM generates erDiagram but uses flowchart syntax (subgraph, -->)."""
            lines = text.split("\n")
            if not lines:
                return text

            diagram_type_idx = -1
            for i, line in enumerate(lines):
                stripped = line.strip().lower()
                if stripped.startswith("%%{") or stripped.startswith("'") or stripped.startswith("}") or stripped == "":
                    continue
                if stripped in ["{", "},", "}}%%", "}%%"]:
                    continue
                if stripped in ["erdiagram", "er-diagram", "sequencediagram", "classiagram"]:
                    diagram_type_idx = i
                    break
                if stripped.startswith("flowchart") or stripped.startswith("graph"):
                    return text

            if diagram_type_idx >= 0:
                has_flowchart_syntax = any(kw in text.lower() for kw in ["subgraph", "-->", "---", "-.->", ":::"])
                if has_flowchart_syntax:
                    lines[diagram_type_idx] = "flowchart TD"
                    return "\n".join(lines)
            return text

        def fix_invalid_brace_nodes(text: str) -> str:
            pattern = re.compile(
                r"([A-Za-z0-9_]+)\s*\{\s*\n\s*([^\n}]+?)\s*\n\s*\}(:::?\w+)?",
                re.MULTILINE,
            )

            def fix_brace(match: re.Match[str]) -> str:
                node_id = match.group(1)
                label = match.group(2).strip()
                class_def = match.group(3) or ""
                return f'{node_id}["{label}"]{class_def}'

            return pattern.sub(fix_brace, text)

        def find_and_warn_orphaned_subgraphs(text: str) -> str:
            """Warn when a subgraph's nodes take part in no edges.

            Previously this produced false positives on perfectly valid diagrams
            and would have missed genuine orphans, because of two bugs that
            conspired:

            1. ``subgraph\\s+(\\w+)`` stopped at the first non-word character, so
               ``subgraph Global Edge & Routing`` was recorded as "Global".
            2. Node collection only matched lines *starting* with ``id[``, so in
               ``a[A] --> b[B]`` only ``a`` was collected, while the edge scan
               missed ``a`` because the ``]`` sat between it and the arrow. The
               single collected node was therefore the one node the edge scan
               could not see, and the subgraph looked orphaned.

            Now: nodes are collected from anywhere on a line, edges are found by
            splitting on arrows (which is what Mermaid actually does), and the
            subgraph title is captured whole.
            """
            # Subgraph id: `subgraph id[Title]`, `subgraph id["Title"]`, or
            # `subgraph Bare Title Words`. Capture the id when bracketed,
            # otherwise the whole trailing title.
            subgraph_header = re.compile(
                r"^\s*subgraph\s+(?:([A-Za-z0-9_.+#-]+)\s*[\[\(\{]|(.+?)\s*$)",
                re.IGNORECASE,
            )
            if not subgraph_header.search(text) and "subgraph" not in text.lower():
                return text

            # An edge line is any line containing a Mermaid connector. Split on
            # connectors and take the identifier adjacent to each side; strip
            # labels (`|text|`) and node shapes first so `a[A] --> b[B]` yields
            # both `a` and `b`.
            connector = re.compile(r"[-=.]{1,}[->]|[-=.]+[ox>]|~~~")
            ident = re.compile(r"([A-Za-z0-9_.+#-]+)")

            def identifiers_on(line: str) -> List[str]:
                # Drop edge labels and node label text; they are prose, not ids.
                cleaned = re.sub(r"\|[^|]*\|", " ", line)
                cleaned = re.sub(r"[\[\(\{][^\]\)\}]*[\]\)\}]", " ", cleaned)
                return ident.findall(cleaned)

            nodes_in_edges = set()
            for line in text.split("\n"):
                if not connector.search(line):
                    continue
                for part in connector.split(line):
                    ids = identifiers_on(part)
                    if ids:
                        # Endpoints of a connector are the last id before it and
                        # the first id after it.
                        nodes_in_edges.add(ids[0])
                        nodes_in_edges.add(ids[-1])

            subgraph_nodes: Dict[str, List[str]] = {}
            stack: List[str] = []

            for line in text.split("\n"):
                stripped = line.strip()
                header = subgraph_header.match(line)
                if header:
                    name = (header.group(1) or header.group(2) or "").strip()
                    stack.append(name)
                    subgraph_nodes.setdefault(name, [])
                    continue
                if stripped.lower() == "end":
                    if stack:
                        stack.pop()
                    continue
                if not stack or stripped.lower().startswith("direction "):
                    continue
                # Collect every identifier that carries a node shape anywhere on
                # the line, plus bare identifiers on connector lines.
                for node_id in re.findall(r"([A-Za-z0-9_.+#-]+)\s*[\[\(\{]", stripped):
                    subgraph_nodes[stack[-1]].append(node_id)
                if connector.search(stripped):
                    subgraph_nodes[stack[-1]].extend(identifiers_on(stripped))

            orphaned = [
                sg
                for sg, nodes in subgraph_nodes.items()
                if nodes and not any(n in nodes_in_edges for n in nodes)
            ]

            if orphaned:
                logger.warning(f"[MERMAID] Orphaned subgraphs detected (no connections): {orphaned}")

            return text

        fixed_code = code
        fixed_code = fix_wrong_diagram_type(fixed_code)
        fixed_code = fix_invalid_brace_nodes(fixed_code)
        fixed_code = fix_special_chars(fixed_code)
        fixed_code = fix_bidirectional_arrows(fixed_code)
        fixed_code = fix_edge_labels(fixed_code)
        fixed_code = fix_malformed_edges(fixed_code)
        fixed_code = fix_nested_labels(fixed_code)
        fixed_code = find_and_warn_orphaned_subgraphs(fixed_code)

        return fixed_code

    @staticmethod
    def _ultra_simplify_mermaid(text: str) -> str:
        if not text:
            return ""

        code = text.replace("\r\n", "\n").replace("\r", "\n")
        code = MermaidSanitizer.strip_features(code)
        code = MermaidSanitizer.to_ascii(code)

        lines = [ln for ln in code.split("\n") if ln.strip()]
        if not lines:
            return "flowchart TD\n  A[Empty]"

        first = lines[0].strip().lower()
        if not (first.startswith("flowchart") or first.startswith("graph")):
            lines.insert(0, "flowchart LR")

        code = "\n".join(lines)

        # Drop edge labels: -->|label| becomes -->
        code = re.sub(r"-->\|[^\|]{1,80}\|", "-->", code)
        code = re.sub(r"-\.->\|[^\|]{1,80}\|", "-.->", code)
        code = re.sub(r"==>\|[^\|]{1,80}\|", "==>", code)

        # Drop any remaining pipe labels (best-effort)
        code = re.sub(r"\|\s*\([^\)]{1,6}\)\s*\|", "| |", code)
        code = re.sub(r"\|[^\|]{1,80}\|", "| |", code)

        return code.strip()
