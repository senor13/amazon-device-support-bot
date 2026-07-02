from pathlib import Path
from langchain_openai import ChatOpenAI
from app.resilience.retry import llm_retry

_PROMPT = (Path(__file__).parent.parent / "prompts/v2/completeness_judge.txt").read_text()

_judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)


@llm_retry
async def score_completeness(
    intent: str,
    sub_queries: list[str],
    response: str,
) -> float:
    """
    LLM-as-judge metric: did the response address all sub-queries?
    Returns a float in [0.0, 1.0].
    """
    sub_queries_text = "\n".join(f"- {q}" for q in sub_queries) if sub_queries else "- (single question)"

    prompt = _PROMPT.format(
        intent=intent,
        sub_queries=sub_queries_text,
        response=response,
    )

    result = await _judge.ainvoke(prompt)

    try:
        score = float(result.content.strip())
        return max(0.0, min(1.0, score))
    except (ValueError, AttributeError):
        return 0.5  # safe default on parse failure
