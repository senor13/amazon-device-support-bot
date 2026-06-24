# Amazon Kindle Support Bot

AI-powered Amazon Kindle support chatbot built with LangGraph, FastAPI, and OpenAI. Features parallel safety gates (PII scrubbing + attack detection), tree-based RAG via PageIndex, semantic caching with GPTCache, and LLM-as-judge output validation.

![Architecture](architecture.png)

---

## Architecture

```
POST /query
  │
  ├─ Middleware: JWT auth · rate limiting (slowapi) · input guard
  │
  ├─ Semantic cache check (GPTCache server)
  │     └─ HIT → return immediately
  │
  └─ LangGraph graph (LangSmith traces everything)
       ├─ safety_gate       Presidio PII scrub + LLM attack detection (parallel)
       ├─ query_intelligence  structured LLM call → intent, sub_queries, complexity
       ├─ session_memory    LangGraph PostgresSaver checkpointer
       ├─ context_retrieval PageIndex tree search + MongoDB
       ├─ execution         gpt-4o-mini / gpt-4o / parallel sub-queries (Send API)
       ├─ output_validation LLM-as-judge faithfulness + completeness
       └─ cache_store       GPTCache write + structlog summary
```

## Stack

| Layer | Tool |
|---|---|
| API framework | FastAPI |
| Graph orchestration | LangGraph |
| Auth | python-jose |
| Rate limiting | slowapi |
| PII scrubbing | presidio-analyzer + presidio-anonymizer |
| Attack detection | LLM-as-judge (gpt-4o-mini) |
| Semantic caching | GPTCache (server mode) |
| RAG retrieval | PageIndex + MongoDB (motor) |
| Session memory | LangGraph PostgresSaver + psycopg |
| LLM (low complexity) | gpt-4o-mini |
| LLM (high complexity) | gpt-4o |
| Output validation | LLM-as-judge faithfulness + completeness |
| Observability (LLM) | LangSmith |
| Observability (app) | structlog |
| Retries | tenacity |
| Circuit breaking | aiobreaker |
| HTTP client | httpx |

## Prerequisites

- Python 3.12+
- Docker Desktop (for MongoDB, PostgreSQL, GPTCache)
- An OpenAI account with API key
- A PageIndex account with API key (for indexing documents)
- A LangSmith account with API key (for tracing)

## Quickstart

**1. Clone and configure**

```bash
git clone https://github.com/your-handle/amazon-kindle-support-bot
cd amazon-kindle-support-bot
cp .env.example .env
# Fill in your API keys in .env
```

**2. Index your support documents**

```bash
# Run once before starting the app (requires PAGEINDEX_API_KEY and MONGODB_URI)
python -m prep.index_docs --pdf path/to/kindle-user-guide.pdf --doc-id kindle
```

**3. Start all services**

```bash
docker compose up --build
```

This starts:
- `app` — main FastAPI app on port 8000
- `gptcache` — GPTCache server on port 8001
- `mongodb` — document trees on port 27017
- `postgres` — session memory on port 5432

**4. Generate a JWT token**

```python
from jose import jwt
token = jwt.encode({"sub": "test-user"}, YOUR_JWT_SECRET, algorithm="HS256")
```

**5. Make a request**

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I connect my Kindle to WiFi?", "session_id": "session-abc"}'
```

Or open the interactive Swagger UI at `http://localhost:8000/docs`.

## Project structure

The project has three main areas:

**`main.py`** — FastAPI entry point. Handles the request lifecycle: cache check, graph invocation, session history persistence, and attack rejection.

**`app/graph/`** — The LangGraph graph. `graph.py` wires all nodes together. Each file in `nodes/` is one step in the pipeline — safety gate, query analysis, retrieval, generation, and validation. `state.py` defines the shared state that flows through all nodes.

**`app/prompts/`** — All LLM prompts live here as plain text files, versioned by folder (`v1/`, `v2/`). Separating prompts from code means you can iterate on them without touching Python.

**`app/resilience/`** — Circuit breakers (aiobreaker) and retry logic (tenacity) for all external calls — OpenAI, MongoDB, GPTCache.

**`app/middleware/`** — Three middleware layers that run before every request: JWT auth, input validation, and rate limiting.

**`prep/index_docs.py`** — One-off script to submit a PDF to PageIndex and store the resulting document tree in MongoDB. Run this whenever you add or update a support document.

## Configuration

All configuration is via environment variables. See `.env.example` for the full list.

Key variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `LOW_COMPLEXITY_MODEL` | `gpt-4o-mini` | Model for simple queries |
| `HIGH_COMPLEXITY_MODEL` | `gpt-4o` | Model for complex queries |
| `FAITHFULNESS_THRESHOLD` | `0.7` | Score below this triggers escalation log |
| `COMPLETENESS_THRESHOLD` | `0.6` | Score below this triggers escalation log |
| `MAX_INPUT_CHARS` | `4000` | Queries longer than this are rejected with 400 |
| `JWT_SECRET` | — | Secret for signing/verifying JWT tokens |
| `PAGEINDEX_API_KEY` | — | PageIndex API key for document indexing |
| `MONGODB_URI` | — | MongoDB connection string |
| `POSTGRES_DSN` | — | PostgreSQL connection string |

## Prompt versioning

Prompts live in `app/prompts/v{n}/`. The active version is set in `app/graph/nodes/query_intelligence.py`. The version string is stored in LangGraph state and logged with every request, so you can correlate quality changes with prompt changes in LangSmith.

To make a new prompt version: copy `app/prompts/v1/` to `app/prompts/v2/`, edit, and update `PROMPT_VERSION` in the node.

## Circuit breakers and fallbacks

| Dependency | Breaker opens after | Fallback behaviour |
|---|---|---|
| PageIndex / MongoDB | 5 failures | Empty context, LLM answers from training knowledge |
| GPTCache | 10 failures | Skip cache, continue normally |
| LLM provider | — (tenacity retries x3) | 503 to user |

## Validation and escalation

Every response is scored by two independent LLM-as-judge metrics:
- **Faithfulness** — are all claims grounded in the retrieved documentation?
- **Completeness** — did the response address all parts of the user's question?

If either score falls below threshold, the response is still shown to the user but an `error` level log is emitted with the full request context (request_id, session_id, query) for engineer review.
