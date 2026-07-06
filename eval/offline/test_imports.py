"""
Offline import smoke tests.
Verifies all graph nodes and core modules import without errors.
Catches broken dependencies, missing packages, and syntax errors before any request runs.
No server, no LLM calls, no Docker needed.
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))


def test_import_state():
    from app.graph.state import SupportBotState
    assert SupportBotState is not None


def test_import_safety_gate():
    # presidio_analyzer is only installed in Docker — skip gracefully when running locally
    pytest.importorskip("presidio_analyzer", reason="presidio_analyzer only available in Docker")
    from app.graph.nodes.safety_gate import pii_scrub_node, attack_detect_node, safety_merge_node
    assert all([pii_scrub_node, attack_detect_node, safety_merge_node])


def test_import_query_intelligence():
    from app.graph.nodes.query_intelligence import query_intelligence_node
    assert query_intelligence_node is not None


def test_import_context_retrieval():
    # motor and aiobreaker are only installed in Docker — skip gracefully when running locally
    pytest.importorskip("motor", reason="motor only available in Docker")
    pytest.importorskip("aiobreaker", reason="aiobreaker only available in Docker")
    from app.graph.nodes.context_retrieval_vectorless import context_retrieval_node
    assert context_retrieval_node is not None


def test_import_execution():
    from app.graph.nodes.execution import (
        generate_flash_node, generate_pro_node,
        generate_subquery_node, merge_subqueries_node, route_execution,
    )
    assert all([generate_flash_node, generate_pro_node, generate_subquery_node, merge_subqueries_node, route_execution])


def test_import_output_validation():
    from app.graph.nodes.output_validation import (
        faithfulness_node, completeness_node, rag_precision_node, validation_merge_node,
    )
    assert all([faithfulness_node, completeness_node, rag_precision_node, validation_merge_node])


def test_import_metrics():
    from app.metrics.faithfulness import score_faithfulness
    from app.metrics.completeness import score_completeness
    from app.metrics.rag_precision import score_rag_precision
    assert all([score_faithfulness, score_completeness, score_rag_precision])


def test_import_config():
    from app.config import settings
    assert settings is not None
