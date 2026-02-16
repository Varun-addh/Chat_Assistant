import pytest

from app.utils.mermaid_sanitizer import MermaidSanitizer


def test_fix_duplicate_diagram_declarations_keeps_first():
    raw = """flowchart LR
  graph TD
  A --> B
"""
    out = MermaidSanitizer.fix_duplicate_diagram_declarations(raw)
    assert "flowchart LR" in out
    assert "graph TD" not in out


def test_sanitize_from_llm_strips_css_and_init_and_adds_header():
    raw = """@import url('https://fonts.example');
%%{init: {'theme':'base'}}%%
#container { font-family: Inter; }
A -> B
"""
    out = MermaidSanitizer.sanitize_from_llm(raw)

    # Should remove CSS/init artifacts
    assert "@import" not in out.lower()
    assert "%%{init" not in out
    assert "#container" not in out.lower()

    # Should normalize arrows and add a header
    first_line = next((l.strip().lower() for l in out.splitlines() if l.strip()), "")
    assert first_line.startswith("flowchart") or first_line.startswith("graph")
    assert "-->" in out


def test_sanitize_edge_labels_removes_square_brackets_inside_pipe_label():
    raw = """flowchart TD
  A -->|(8) [Mobile Push]| B
"""
    out = MermaidSanitizer.sanitize_edge_labels(raw)
    assert "-->" in out
    # The bracketed part should be converted to parentheses in the label
    assert "[Mobile Push]" not in out
    assert "(Mobile Push)" in out


def test_sanitize_edge_labels_removes_extra_gt_after_pipe_label():
    raw = """flowchart LR
  A[Rider] -->|Request Ride|> B[Matching Service]
"""
    out = MermaidSanitizer.sanitize(raw, mode="render").code
    assert "|Request Ride|>" not in out
    assert "|Request Ride|" in out


def test_flowchart_double_chevron_arrows_are_normalized():
    raw = """flowchart LR
  A[Rider] -->> B[Matching Service]
  B ->> C[Driver]
"""
    out = MermaidSanitizer.sanitize(raw, mode="render").code
    assert "-->>" not in out
    assert "->>" not in out
    assert "-->" in out


def test_subset_strips_style_directives_and_non_ascii():
    raw = """flowchart TD
  A[Client → API]:::client --> B[Server]
  classDef client fill:#e3f2fd
  linkStyle 0 stroke:#333
"""
    out = MermaidSanitizer.sanitize_subset(raw)
    assert "classdef" not in out.lower()
    assert "linkstyle" not in out.lower()
    assert ":::" not in out
    assert "→" not in out
    assert "->" in out  # unicode arrow replaced
