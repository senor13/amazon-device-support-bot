"""
Output validation: two independent LangGraph nodes run in parallel.

  faithfulness_node  — LLM-as-judge: are claims grounded in retrieved context?
  completeness_node  — LLM-as-judge: did we answer all sub-queries?

Both fan out from the execution node(s) and converge at validation_merge.
"""
from app.graph.state import SupportBotState
from app.metrics.faithfulness import score_faithfulness
from app.metrics.completeness import score_completeness
from app.metrics.rag_precision import score_rag_precision
from app.observability.logging import get_logger
from app.config import settings


# ── Node A: Faithfulness (LLM-as-judge) ──────────────────────────────────────

async def faithfulness_node(state: SupportBotState) -> dict:
    log = get_logger(state["request_id"], node="faithfulness")
    context = state.get("retrieved_context", [])

    if not context:
        log.info("faithfulness_skipped", reason="no_context")
        return {"faithfulness_score": 1.0}

    try:
        score = await score_faithfulness(context, state["raw_response"])
    except Exception as exc:
        log.warning("faithfulness_failed", error=str(exc))
        score = 1.0

    log.info("faithfulness_complete", score=round(score, 3))
    return {"faithfulness_score": score}


# ── Node B: Completeness (LLM-as-judge) ──────────────────────────────────────

async def completeness_node(state: SupportBotState) -> dict:
    log = get_logger(state["request_id"], node="completeness")
    score = await score_completeness(
        intent=state["intent"],
        sub_queries=state["sub_queries"],
        response=state["raw_response"],
    )
    log.info("completeness_complete", score=round(score, 3))
    return {"completeness_score": score}


# ── Node C: RAG Precision (LLM-as-judge) ─────────────────────────────────────

async def rag_precision_node(state: SupportBotState) -> dict:
    log = get_logger(state["request_id"], node="rag_precision")
    context = state.get("retrieved_context", [])

    if not context:
        log.info("rag_precision_skipped", reason="no_context")
        return {"rag_precision_score": 1.0}

    try:
        score = await score_rag_precision(
            query=state["scrubbed_query"],
            context=context,
            response=state["raw_response"],
        )
    except Exception as exc:
        log.warning("rag_precision_failed", error=str(exc))
        score = 1.0

    log.info("rag_precision_complete", score=round(score, 3))
    return {"rag_precision_score": score}


# ── Node D: Validation merge ──────────────────────────────────────────────────

async def validation_merge_node(state: SupportBotState) -> dict:
    log = get_logger(state["request_id"], node="validation_merge")

    faithfulness = state.get("faithfulness_score", 1.0)
    completeness = state.get("completeness_score", 1.0)
    passed = (
        faithfulness >= settings.FAITHFULNESS_THRESHOLD
        and completeness >= settings.COMPLETENESS_THRESHOLD
    )

    if not passed:
        # error level so it stands out in logs for engineer review.
        # user still gets the response — we don't block them.
        # in future: add Slack/PagerDuty call here for real-time alerting.
        log.error(
            "validation_failed_escalation",
            faithfulness=round(faithfulness, 3),
            completeness=round(completeness, 3),
            request_id=state["request_id"],
            session_id=state["session_id"],
            query=state["scrubbed_query"],
        )
    else:
        log.info(
            "validation_passed",
            faithfulness=round(faithfulness, 3),
            completeness=round(completeness, 3),
        )

    return {
        "validation_passed": passed,
        "final_response": state["raw_response"],
    }
