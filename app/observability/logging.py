import logging
import sys
from pathlib import Path
import structlog


_LOG_FILE = Path("/app/logs/app.jsonl")


def configure_logging():
    # ensure log directory exists
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ]

    # write to both stdout (visible in docker logs) and a persistent file
    file_handler = logging.FileHandler(_LOG_FILE)
    stream_handler = logging.StreamHandler(sys.stdout)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(file=_LOG_FILE.open("a")),
        cache_logger_on_first_use=True,
    )


def get_logger(request_id: str, **kwargs):
    """Return a logger pre-bound with request_id and any extra context."""
    return structlog.get_logger().bind(request_id=request_id, **kwargs)
