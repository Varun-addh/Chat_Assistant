"""Regression tests for orphaned-subgraph detection.

The detector fired on valid diagrams and would have missed real orphans, from
two bugs that conspired:

1. ``subgraph\\s+(\\w+)`` stopped at the first non-word character, so
   ``subgraph Global Edge & Routing`` was recorded as "Global".
2. Nodes were only collected from lines *starting* with ``id[``, so in
   ``a[A] --> b[B]`` only ``a`` was collected — while the edge scan missed ``a``
   because the ``]`` sat between it and the arrow. The one node collected was
   precisely the node the edge scan could not see, so the subgraph looked
   orphaned.

Both diagrams below are verbatim from production logs on 2026-07-27, where they
produced false positives.
"""

import logging

import pytest

from app.utils.mermaid_sanitizer import MermaidSanitizer


def _warnings(caplog, code: str):
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="app.utils.mermaid_sanitizer"):
        MermaidSanitizer.fix_mermaid_syntax_errors(code)
    return [r.message for r in caplog.records if "Orphaned subgraphs" in r.message]


# --- production false positives --------------------------------------------

AZURE = """flowchart LR
subgraph Azure
  direction LR
  api_service[API Service] --> cache[Redis Cache]
  cache --> db[Azure SQL Database]
  db --> async_worker[Azure Queue Worker]
  async_worker --> notification_service[Azure Notification Service]
  notification_service --> users
end"""

GLOBAL_EDGE = """flowchart TD
subgraph Global Edge & Routing
  A[User] -->|HTTPS| B[Azure Front Door]
  B --> C[Azure Application Gateway]
  C --> D[Azure Load Balancer]
end
subgraph API Edge & Security
  D --> E[Azure API Management]
  E --> F[Azure Active Directory]
end"""


def test_connected_subgraph_is_not_reported(caplog):
    assert _warnings(caplog, AZURE) == []


def test_multi_word_subgraph_titles_are_not_reported(caplog):
    assert _warnings(caplog, GLOBAL_EDGE) == []


def test_edge_label_does_not_break_node_detection(caplog):
    """`A[User] -->|HTTPS| B[...]` — the label must not hide either endpoint."""
    code = """flowchart TD
subgraph Tier
  A[User] -->|HTTPS| B[Gateway]
end"""
    assert _warnings(caplog, code) == []


# --- genuine orphans must still be caught ----------------------------------

def test_genuinely_orphaned_subgraph_is_reported(caplog):
    code = """flowchart TD
  A[User] --> B[API]
subgraph Detached
  X[Lonely]
  Y[AlsoLonely]
end"""
    found = _warnings(caplog, code)
    assert found, "a subgraph whose nodes appear in no edge must be reported"
    assert "Detached" in found[0]


def test_orphan_with_multi_word_title_reports_the_whole_title(caplog):
    """The title must not be truncated at the first space."""
    code = """flowchart TD
  A[User] --> B[API]
subgraph Cold Storage Tier
  Z[Glacier]
end"""
    found = _warnings(caplog, code)
    assert found
    assert "Cold Storage Tier" in found[0], f"title truncated: {found[0]}"


# --- shapes that must not crash or misreport -------------------------------

@pytest.mark.parametrize(
    "code",
    [
        "flowchart TD\n  A --> B",                      # no subgraphs at all
        "flowchart TD\nsubgraph Empty\nend",            # empty subgraph
        "",                                             # empty input
        "flowchart TD\nsubgraph S\n  direction LR\nend",  # direction only
    ],
)
def test_degenerate_inputs_report_nothing(caplog, code):
    assert _warnings(caplog, code) == []


# --- generalisation across Mermaid syntax -----------------------------------
# The two production diagrams that exposed the bug are only two shapes. These
# pin the detector against the wider syntax the LLM actually emits, so a future
# fix tuned to one diagram cannot regress the rest.

CONNECTED_SHAPES = [
    ("flowchart LR + quoted title", 'flowchart LR\nsubgraph S["Edge & CDN"]\n  a[A] --> b[B]\nend'),
    ("graph keyword", "graph TD\nsubgraph Tier One\n  x[X] --> y[Y]\nend"),
    ("round/diamond/nested shapes", "flowchart TD\nsubgraph S\n  a(Round) --> b{Diamond}\n  b --> c[[Sub]]\nend"),
    ("dotted and thick edges", "flowchart TD\nsubgraph S\n  a -.-> b\n  b ==> c\nend"),
    ("edge label with punctuation", 'flowchart TD\nsubgraph S\n  a[A] -->|"HTTP/2, gRPC"| b[B]\nend'),
    ("bracketed id with cylinder node", "flowchart LR\nsubgraph AWS[AWS Region]\n  api[API] --> db[(DB)]\nend"),
    ("multiple connected subgraphs", "flowchart TD\nsubgraph One\n a[A] --> b[B]\nend\nsubgraph Two\n b --> c[C]\nend"),
]


@pytest.mark.parametrize("name,code", CONNECTED_SHAPES, ids=[c[0] for c in CONNECTED_SHAPES])
def test_connected_diagrams_never_warn(caplog, name, code):
    assert _warnings(caplog, code) == [], f"false positive on {name}"


ORPHAN_SHAPES = [
    ("isolated tier", "flowchart TD\n a[A] --> b[B]\nsubgraph Backup Vault\n  z[Glacier]\nend", "Backup Vault"),
    ("one of two subgraphs", "flowchart TD\nsubgraph Live\n a[A] --> b[B]\nend\nsubgraph Dead Letter Queue\n q[Q]\nend", "Dead Letter Queue"),
    ('quoted title', 'flowchart TD\n a[A] --> b[B]\nsubgraph X["Cold Path"]\n  z[Z]\nend', "X"),
]


@pytest.mark.parametrize("name,code,expected", ORPHAN_SHAPES, ids=[c[0] for c in ORPHAN_SHAPES])
def test_orphans_are_caught_across_shapes(caplog, name, code, expected):
    found = _warnings(caplog, code)
    assert found, f"missed orphan in {name}"
    assert expected in found[0], f"{name}: expected {expected!r} in {found[0]!r}"


def test_bracketed_subgraph_id_uses_the_id(caplog):
    """`subgraph Azure[Azure Layer]` — connected, so nothing to report."""
    code = """flowchart LR
subgraph Azure[Azure Layer]
  api[API] --> db[DB]
end"""
    assert _warnings(caplog, code) == []


def test_nested_subgraphs_do_not_confuse_the_scanner(caplog):
    code = """flowchart TD
  A[User] --> B[API]
subgraph Outer
  subgraph Inner
    B --> C[Cache]
  end
end"""
    assert _warnings(caplog, code) == []
