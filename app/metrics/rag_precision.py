from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from app.resilience.retry import llm_retry

_PROMPT = """\
You are evaluating whether retrieved document nodes were useful for generating a support bot response.

Query: {query}

Bot Response:
{response}

Retrieved Nodes:
{nodes}

Score from 0.0 to 1.0:
- 1.0 = every retrieved node contributed information used in the response
- 0.5 = about half the nodes were useful, the rest were irrelevant
- 0.0 = none of the retrieved nodes were used in the response

Reply with only a score."""


class PrecisionResult(BaseModel):
    score: float


_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(PrecisionResult)


@llm_retry
async def score_rag_precision(query: str, context: list[str], response: str) -> float:
    nodes_text = "\n\n---\n\n".join(f"Node {i+1}:\n{c}" for i, c in enumerate(context))
    prompt = _PROMPT.format(query=query, response=response, nodes=nodes_text)
    result = await _llm.ainvoke(prompt)
    return max(0.0, min(1.0, result.score))
