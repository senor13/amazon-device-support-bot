import logging
import sys
from pathlib import Path
import structlog


_LOG_FILE = Path("/app/logs/app.jsonl")


class _TeeWriter:
    """Writes each log line to both stdout and the JSONL file."""

    def __init__(self, file_path: Path):
        self._file = file_path.open("a")

    def write(self, message: str):
        sys.stdout.write(message)
        self._file.write(message)

    def flush(self):
        sys.stdout.flush()
        self._file.flush()


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

    global _tee_writer
    _tee_writer = _TeeWriter(_LOG_FILE)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(file=_tee_writer),
        cache_logger_on_first_use=True,
    )


_tee_writer: "_TeeWriter | None" = None


def write_separator():
    """Write a blank line to the log output to visually separate requests."""
    if _tee_writer:
        _tee_writer.write("\n")


def get_logger(request_id: str, **kwargs):
    """Return a logger pre-bound with request_id and any extra context."""
    return structlog.get_logger().bind(request_id=request_id, **kwargs)
