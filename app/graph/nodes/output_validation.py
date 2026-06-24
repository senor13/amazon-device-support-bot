"""
Output validation: two independent LangGraph nodes run in parallel.

  faithfulness_node  — LLM-as-judge: are claims grounded in retrieved context?
  completeness_node  — LLM-as-judge: did we answer all sub-queries?

Both fan out from the execution node(s) and converge at validation_merge.
"""
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from app.graph.state import SupportBotState
from app.metrics.completeness import score_completeness
from app.observability.logging import get_logger
from app.config import settings


_FAITHFULNESS_PROMPT = """You are evaluating whether an AI response is faithful to the provided context.

Faithful means: every claim in the response is supported by the context. The response does not introduce facts not present in the context.

Context:
{context}

Response:
{response}

Score from 0.0 to 1.0 where:
- 1.0 = fully faithful, all claims supported by context
- 0.5 = partially faithful, some claims unsupported
- 0.0 = not faithful, response contradicts or ignores context"""


class FaithfulnessResult(BaseModel):
    score: float


_faithfulness_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
    FaithfulnessResult
)


# ── Node A: Faithfulness (LLM-as-judge) ──────────────────────────────────────

async def faithfulness_node(state: SupportBotState) -> dict:
    log = get_logger(state["request_id"], node="faithfulness")
    context = state.get("retrieved_context", [])

    if not context:
        log.info("faithfulness_skipped", reason="no_context")
        return {"faithfulness_score": 1.0}

    context_text = "\n\n".join(context)
    prompt = _FAITHFULNESS_PROMPT.format(
        context=context_text,
        response=state["raw_response"],
    )

    try:
        result = await _faithfulness_llm.ainvoke(prompt)
        score = max(0.0, min(1.0, result.score))
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


# ── Node C: Validation merge ──────────────────────────────────────────────────

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
