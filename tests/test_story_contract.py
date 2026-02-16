from app.utils.story_contract import enforce_story_contract


def test_system_overview_clamps_to_5_bullets_and_goal():
    messy = """
    ### Random Heading
    • Point 1: Users send events to gateway
    • Point 2: Core validates preferences
    • Point 3: Queue buffers traffic
    • Point 4: Workers deliver to providers
    • Point 5: Observability monitors
    • Point 6: Extra junk should be removed

    Key Highlights
    - lots of extra

    Goal: Should be kept
    """.strip()

    out = enforce_story_contract("SYSTEM_OVERVIEW", "Design a Notification Service at Scale", messy)
    lines = [ln for ln in out.splitlines() if ln.strip()]

    bullets = [ln for ln in lines if ln.startswith("- ")]
    assert len(bullets) == 5
    assert any(ln.lower().startswith("goal:") for ln in lines)


def test_request_flow_outputs_only_allowed_headings_and_structure():
    messy = """
    ### Layer 1 - Ingestion & Protection
    - Authenticate request
    - Rate limit
    - Extra bullet that should be dropped

    ### Layer 2 - Core Logic & Enrichment
    Some paragraph sentence one. Sentence two.

    ### Layer 3 - Async Buffering & Prioritization
    • Publish to Kafka
    • Separate topics
    • Why this exists: Decouple

    ### Layer 4 - Transaction Finalization
    - Workers call providers
    - Retries

    ### Layer 5 - Post-Processing & Feedback
    - Send notification
    - Update analytics

    ### Final End-to-End Flow Summary
    This is line one.
    This is line two.
    This is line three.
    This is line four.
    This is line five.
    This is line six should be dropped.

    Detailed Analysis
    blah blah
    """.strip()

    out = enforce_story_contract("REQUEST_FLOW", "Design a Notification Service at Scale", messy)

    # Allowed headings appear
    assert "### Layer 1 - Ingestion & Protection" in out
    assert "### Layer 2 - Core Logic & Enrichment" in out
    assert "### Layer 3" in out
    assert "### Layer 4" in out
    assert "### Layer 5" in out
    assert "### Final End-to-End Flow Summary" in out

    # Structure markers appear for layers
    assert "What happens:" in out
    assert "Why it exists:" in out

    # Should not contain encyclopedia sections
    assert "Detailed Analysis" not in out


def test_contract_drops_empty_bullets_and_redundant_prefixes():
    messy = """
    ### Layer 1 - Ingestion & Protection
    •
    • What happens: Gate request
    • Why it exists: Protect downstream

    ### Final End-to-End Summary
    ### Layer 2 - Core Logic & Enrichment
    This summary should not include headings.
    """.strip()

    out = enforce_story_contract("REQUEST_FLOW", "Design a Hotel Booking System", messy)

    # No empty bullet lines
    assert "\n- \n" not in out
    # Redundant prefixes stripped from bullet payload (inside bullets)
    bullet_lines = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("- ")]
    assert all("What happens:" not in ln for ln in bullet_lines)
    assert all("Why it exists:" not in ln for ln in bullet_lines)
    # Headings should not leak into final summary lines
    final_block = out.split("### Final End-to-End Flow Summary", 1)[1]
    assert "###" not in final_block


def test_request_flow_accepts_domain_specific_layer_titles_in_order():
    messy = """
    ### Layer 1 - Ingestion & Protection
    - Authenticate user
    - Route to region

    ### Layer 2 - Core Logic & Enrichment
    - Search rooms
    - Compute pricing

    ### Layer 3 - Reservation & Concurrency Control
    - Acquire lock
    - Create intent

    ### Layer 4 - Transaction Finalization
    - Authorize payment
    - Commit booking

    ### Layer 5 - Post-Processing & Feedback
    - Send confirmation
    - Sync inventory

    ### Final End-to-End Flow Summary
    User -> Search -> Lock -> Pay -> Commit -> Notify
    """.strip()

    out = enforce_story_contract("REQUEST_FLOW", "Design a Hotel Booking System", messy)
    assert "### Layer 1 - Ingestion & Protection" in out
    assert "### Layer 3 - Reservation & Concurrency Control" in out
    assert "### Layer 5 - Post-Processing & Feedback" in out
    assert "### Final End-to-End Flow Summary" in out


def test_request_flow_splits_inline_star_bullets_in_final_summary():
    messy = """
    ### Layer 1 - Ingestion & Protection
    - A
    - B

    ### Layer 2 - Core Logic & Enrichment
    - C
    - D

    ### Layer 3 - Reservation & Concurrency Control
    - E
    - F

    ### Layer 4 - Transaction Finalization
    - G
    - H

    ### Layer 5 - Post-Processing & Feedback
    - I
    - J

    ### Final End-to-End Flow Summary
    * One. * Two. * Three.
    """.strip()

    out = enforce_story_contract("REQUEST_FLOW", "Design a Hotel Booking System", messy)
    final_block = out.split("### Final End-to-End Flow Summary", 1)[1]
    lines = [ln.strip() for ln in final_block.splitlines() if ln.strip()]
    # Should become separate lines without leading '*'
    assert any(ln.startswith("One") for ln in lines)
    assert any(ln.startswith("Two") for ln in lines)
    assert any(ln.startswith("Three") for ln in lines)
    assert all(not ln.startswith("*") for ln in lines)


def test_request_flow_splits_inline_dash_bullets_in_final_summary():
    messy = """
    ### Layer 1 - Ingestion & Protection
    - A
    - B

    ### Layer 2 - Core Logic & Enrichment
    - C
    - D

    ### Layer 3 - Reservation & Concurrency Control
    - E
    - F

    ### Layer 4 - Transaction Finalization
    - G
    - H

    ### Layer 5 - Post-Processing & Feedback
    - I
    - J

    ### Final End-to-End Flow Summary
    - One. - Two. - Three.
    """.strip()

    out = enforce_story_contract("REQUEST_FLOW", "Design a Hotel Booking System", messy)
    final_block = out.split("### Final End-to-End Flow Summary", 1)[1]
    lines = [ln.strip() for ln in final_block.splitlines() if ln.strip()]
    assert any(ln.startswith("One") for ln in lines)
    assert any(ln.startswith("Two") for ln in lines)
    assert any(ln.startswith("Three") for ln in lines)
    assert all(not ln.startswith("-") for ln in lines)


def test_single_outputs_comprehensive_sections_not_multiview_layers():
    messy = """
    ### Executive Summary
    - We accept requests
    - We authenticate
    - We enqueue work
    - Workers deliver
    - Observability watches
    - Extra bullet should be dropped

    ### Requirements
    Functional:
    - Create notification
    - Fanout to providers
    - Store preferences
    Non-Functional:
    - p95 latency 100ms
    - 99.9% availability
    - 10k req/s

    ### Architecture
    - API gateway
    - Auth
    - Queue
    - Workers
    - Provider adapters
    - Status store
    - Rate limiter
    - Metrics

    ### Capacity Planning
    - 10k req/s steady
    - bursts 50k req/s
    - 1M users/day
    - 200ms p99

    ### Trade-offs
    - Kafka vs SQS
    - Postgres vs Dynamo
    - push vs pull
    - at-least-once vs exactly-once

    ### Example Implementation Snippet
    ```python
    def handle_request(req_id, payload):
        return payload
    ```

    Key Highlights
    - should be removed
    """.strip()

    out = enforce_story_contract("SINGLE", "Design a Notification Service at Scale", messy)

    # Expected comprehensive headings
    assert "### Executive Summary" in out
    assert "### Requirements" in out
    assert "### Architecture" in out
    assert "### Capacity Planning" in out
    assert "### Trade-offs" in out
    assert "### Example Implementation Snippet" in out

    # Should NOT look like the multi-view layered template
    assert "### Layer 1" not in out
    assert "### Final End-to-End" not in out

    # Executive summary is clamped to 5 bullets
    summary_block = out.split("### Executive Summary", 1)[1].split("### Requirements", 1)[0]
    bullets = [ln for ln in summary_block.splitlines() if ln.strip().startswith("- ")]
    assert len(bullets) == 5

    # Encyclopedia sections removed
    assert "Key Highlights" not in out
