"""
Offline unit tests for routing logic.
Tests route_execution() and route_after_safety() directly — no LLM calls, no graph execution.
Catches silent misrouting: wrong model used, attack not blocked, decomp not triggered.
"""
import sys
from pathlib import Path

# Add project root so app imports resolve without running the full app
sys.path.insert(0, str(Path(__file__).parents[2]))

from app.graph.nodes.execution import route_execution, route_after_safety
from langgraph.graph import END


# ── route_execution ────────────────────────────────────────────────────────────

def test_low_complexity_routes_to_flash():
    state = {"complexity": "low", "needs_decomp": False, "sub_queries": []}
    assert route_execution(state) == "generate_flash"


def test_high_complexity_routes_to_pro():
    state = {"complexity": "high", "needs_decomp": False, "sub_queries": []}
    assert route_execution(state) == "generate_pro"


def test_needs_decomp_triggers_fanout():
    # needs_decomp=True with 2+ sub_queries should return Send objects (list), not a string
    from langgraph.types import Send
    state = {
        "complexity": "high",
        "needs_decomp": True,
        "sub_queries": ["How do I connect to WiFi?", "How do I transfer books via USB?"],
        "scrubbed_query": "How do I connect to WiFi and transfer books?",
        "retrieved_context": [],
        "session_history": [],
    }
    result = route_execution(state)
    assert isinstance(result, list), "needs_decomp=True should return a list of Send objects"
    assert all(isinstance(r, Send) for r in result), "Fan-out result should be list[Send]"
    assert len(result) == 2, "Two sub_queries should produce two Send objects"


def test_needs_decomp_false_ignores_sub_queries():
    # needs_decomp=False should route by complexity even if sub_queries is populated
    state = {"complexity": "low", "needs_decomp": False, "sub_queries": ["q1", "q2"]}
    assert route_execution(state) == "generate_flash"


def test_needs_decomp_single_query_no_fanout():
    # needs_decomp=True but only 1 sub_query — should not fan out (len check in route_execution)
    state = {"complexity": "high", "needs_decomp": True, "sub_queries": ["only one"]}
    result = route_execution(state)
    assert result == "generate_pro", "Single sub_query should not fan out"


# ── route_after_safety ────────────────────────────────────────────────────────

def test_attack_routes_to_end():
    state = {"is_attack": True}
    assert route_after_safety(state) == END


def test_safe_query_routes_to_query_intelligence():
    state = {"is_attack": False}
    assert route_after_safety(state) == "query_intelligence"


def test_missing_is_attack_treated_as_safe():
    # state.get("is_attack") returns None which is falsy — should continue
    state = {}
    assert route_after_safety(state) == "query_intelligence"
