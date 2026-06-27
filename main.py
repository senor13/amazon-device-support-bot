import uuid
from contextlib import asynccontextmanager
from psycopg_pool import AsyncConnectionPool
import httpx
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.graph.graph import build_graph
from app.middleware.auth import auth_middleware
from app.middleware.input_guard import input_guard_middleware
from app.middleware.rate_limit import limiter
from app.observability.logging import configure_logging, get_logger

configure_logging()

# ── App state ─────────────────────────────────────────────────────────────────

_graph = None
_pg_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: runs once when the server starts ──────────────────────────────
    # Everything before yield is setup. We do it here (not at module level)
    # because these are async operations that need an event loop to run.
    global _graph, _pg_pool

    # Open a pool of Postgres connections (reused across all requests).
    # min_size=2 keeps 2 connections warm; max_size=10 caps concurrent usage.
    _pg_pool = AsyncConnectionPool(conninfo=settings.POSTGRES_DSN, min_size=2, max_size=10)
    await _pg_pool.open()

    # Compile the LangGraph graph once. This connects to Postgres, creates
    # checkpoint tables, and builds the node/edge structure. Expensive to do
    # per-request, so we build it once and reuse it globally.
    _graph = await build_graph(_pg_pool)

    yield  # ── App runs here, serving all incoming requests ───────────────────

    # ── Shutdown: runs once when the server stops ──────────────────────────────
    # Cleanly close all Postgres connections so nothing is left hanging.
    await _pg_pool.close()


app = FastAPI(title="Amazon Support Bot", lifespan=lifespan)

# ── Middleware (order matters: added last = runs first) ───────────────────────

app.middleware("http")(input_guard_middleware)
app.middleware("http")(auth_middleware)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})



# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None


class QueryResponse(BaseModel):
    response: str
    session_id: str
    request_id: str
    faithfulness_score: float
    completeness_score: float
    validation_passed: bool
    model_used: str


# ── Cache helper ──────────────────────────────────────────────────────────────

async def check_cache(query: str) -> str | None:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.GPTCACHE_URL}/get",
                json={"prompt": query},
                timeout=2.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer")
                if answer:
                    return answer
    except Exception:
        pass
    return None


# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query_endpoint(body: QueryRequest, request: Request):
    request_id = str(uuid.uuid4())
    session_id = body.session_id or str(uuid.uuid4())
    log = get_logger(request_id, session_id=session_id, user_id=request.state.user_id)

    log.info("request_received", query_length=len(body.query))

    # ── Cache check (before graph) ────────────────────────────────────────────
    cached = await check_cache(body.query)
    if cached:
        log.info("cache_hit")
        return QueryResponse(
            response=cached,
            session_id=session_id,
            request_id=request_id,
            faithfulness_score=1.0,
            completeness_score=1.0,
            validation_passed=True,
            model_used="cache",
        )

    log.info("cache_miss")

    # ── LangGraph invocation ──────────────────────────────────────────────────
    config = {"configurable": {"thread_id": session_id}}

    # Load session history from checkpointer if session already exists
    previous = await _graph.aget_state(config)
    previous_history = previous.values.get("session_history", []) if previous else []

    initial_state = {
        "raw_query": body.query,
        "session_id": session_id,
        "request_id": request_id,
        # remaining fields populated by nodes
        "scrubbed_query": "",
        "pii_found": [],
        "is_attack": False,
        "attack_confidence": 0.0,
        "intent": "",
        "sub_queries": [],
        "complexity": "low",
        "needs_decomp": False,
        "prompt_version": "",
        "current_subquery": "",
        "relevant_docs": [],
        "is_cacheable": False,
        "session_history": previous_history,
        "retrieved_context": [],
        "sub_responses": [],
        "raw_response": "",
        "model_used": "",
        "faithfulness_score": 0.0,
        "completeness_score": 0.0,
        "validation_passed": False,
        "final_response": "",
    }

    try:
        result = await _graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        log.error("graph_invocation_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    # Attack was detected — graph ended early, don't save to history
    if result.get("is_attack"):
        log.warning("request_rejected_attack", confidence=result.get("attack_confidence"))
        raise HTTPException(status_code=403, detail="Request rejected")

    # Save current turn to checkpointer for next request
    updated_history = result.get("session_history", []) + [
        {"role": "user", "content": body.query},
        {"role": "assistant", "content": result.get("final_response", "")},
    ]
    await _graph.aupdate_state(config, {"session_history": updated_history})

    log.info("request_success")

    return QueryResponse(
        response=result["final_response"],
        session_id=session_id,
        request_id=request_id,
        faithfulness_score=result.get("faithfulness_score", 0.0),
        completeness_score=result.get("completeness_score", 0.0),
        validation_passed=result.get("validation_passed", False),
        model_used=result.get("model_used", "unknown"),
    )


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "graph_ready": _graph is not None}
