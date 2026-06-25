from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from app.resilience.retry import llm_retry

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
