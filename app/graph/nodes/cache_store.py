import httpx
import aiobreaker

from app.graph.state import SupportBotState
from app.observability.logging import get_logger
from app.resilience.breakers import gptcache_breaker
from app.config import settings


async def cache_store_node(state: SupportBotState) -> dict:
    log = get_logger(state["request_id"], node="cache_store")

    # only cache factual context-free answers that passed validation
    # — prevents bad answers and context-dependent responses polluting the cache
    if state.get("is_cacheable") and state.get("validation_passed"):
        async def _store():
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{settings.GPTCACHE_URL}/put",
                    json={
                        "prompt": state["raw_query"],
                        "answer": state["final_response"],
                    },
                    timeout=2.0,
                )

        try:
            await gptcache_breaker.call(_store)
            log.info("cache_stored")
        except aiobreaker.CircuitBreakerError:
            log.warning("gptcache_circuit_open")
        except Exception as exc:
            # Non-fatal — a failed cache write just means the next similar query
            # won't hit the cache, user still gets their response.
            log.warning("cache_store_failed", error=str(exc))
    else:
        log.info("cache_skipped",
                 is_cacheable=state.get("is_cacheable"),
                 validation_passed=state.get("validation_passed"))

    log.info(
        "request_complete",
        model_used=state.get("model_used"),
        faithfulness=round(state.get("faithfulness_score", 0), 3),
        completeness=round(state.get("completeness_score", 0), 3),
        validation_passed=state.get("validation_passed"),
        pii_found=state.get("pii_found", []),
        prompt_version=state.get("prompt_version"),
        num_sub_queries=len(state.get("sub_queries", [])),
        needs_decomp=state.get("needs_decomp"),
        is_cacheable=state.get("is_cacheable"),
    )

    return {}
