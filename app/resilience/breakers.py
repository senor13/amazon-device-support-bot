from datetime import timedelta
import aiobreaker
import structlog

log = structlog.get_logger()


def _on_state_change(breaker, old, new):
    log.warning("circuit_breaker_state_change", breaker=breaker.name, old=str(old), new=str(new))


pageindex_breaker = aiobreaker.CircuitBreaker(
    fail_max=5,
    timeout_duration=timedelta(seconds=30),
    name="pageindex",
    listeners=[aiobreaker.CircuitBreakerListener()],
)

gptcache_breaker = aiobreaker.CircuitBreaker(
    fail_max=10,
    timeout_duration=timedelta(seconds=15),
    name="gptcache",
    listeners=[aiobreaker.CircuitBreakerListener()],
)
