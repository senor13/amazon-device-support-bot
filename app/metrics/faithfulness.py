from pathlib import Path
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from app.resilience.retry import llm_retry

_FAITHFULNESS_PROMPT = (Path(__file__).parent.parent / "prompts/v2/faithfulness_judge.txt").read_text()


class FaithfulnessResult(BaseModel):
    score: float


_faithfulness_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
    FaithfulnessResult
)


@llm_retry
async def score_faithfulness(context: list[str], response: str) -> float:
    """
    LLM-as-judge metric: are all claims in the response grounded in the retrieved context?
    Returns a float in [0.0, 1.0].
    """
    context_text = "\n\n".join(context)
    prompt = _FAITHFULNESS_PROMPT.format(context=context_text, response=response)
    result = await _faithfulness_llm.ainvoke(prompt)
    return max(0.0, min(1.0, result.score))
