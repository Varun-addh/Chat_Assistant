from app.services.architecture_generator import get_architecture_generator, ArchitectureViewType


def test_system_overview_always_clamped_tight():
    arch = get_architecture_generator()
    prompt = arch.get_view_prompt(
        ArchitectureViewType.SYSTEM_OVERVIEW,
        "Design a global multi-region payment and inventory platform with strict consistency, auditing, analytics, and low latency",
    )
    assert prompt["max_nodes"] <= 10
    assert prompt["max_edges"] <= 12


def test_request_flow_limits_increase_with_complexity():
    arch = get_architecture_generator()

    simple = arch.get_view_prompt(
        ArchitectureViewType.REQUEST_FLOW,
        "Design a simple URL shortener",
    )
    complex_ = arch.get_view_prompt(
        ArchitectureViewType.REQUEST_FLOW,
        "Design a global multi-region hotel booking system with payments, inventory locking, strong consistency, async notifications, and observability",
    )

    # For complex questions we should allow at least as many nodes/edges as for simple.
    assert complex_["max_nodes"] >= simple["max_nodes"]
    assert complex_["max_edges"] >= simple["max_edges"]
