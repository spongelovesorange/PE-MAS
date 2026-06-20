"""Runtime helpers for product-grade PE-MAS entrypoints.

Keep this module dependency-light so it can be imported by both API tests and
Studio runtime paths.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import warnings
from pathlib import Path
from threading import RLock
from typing import Iterator


_STDIO_LOCK = RLock()
_FALSE_VALUES = {"0", "false", "no", "off", ""}
_TRUE_VALUES = {"1", "true", "yes", "on"}


class _RuntimeNoiseFilter(logging.Filter):
    """Drop known third-party startup chatter without hiding PE-MAS errors."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        name = record.name
        if name.startswith("mcp") and "Processing request of type" in message:
            return False
        return True


def env_flag(name: str, *, default: bool = False) -> bool:
    """Parse common boolean environment variable values."""

    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return default


def configure_logging(*, default_level: str = "INFO") -> None:
    """Configure quiet, predictable application logging once."""

    level_name = os.getenv("PE_MAS_LOG_LEVEL", default_level).strip().upper() or default_level
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    root_logger = logging.getLogger()
    if not any(isinstance(item, _RuntimeNoiseFilter) for item in root_logger.filters):
        root_logger.addFilter(_RuntimeNoiseFilter())
    for handler in root_logger.handlers:
        if not any(isinstance(item, _RuntimeNoiseFilter) for item in handler.filters):
            handler.addFilter(_RuntimeNoiseFilter())

    for noisy_logger in (
        "httpx",
        "httpcore",
        "ddgs",
        "urllib3",
        "openai",
        "openai._base_client",
        "dotenv",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    for very_noisy_logger in (
        "mcp",
        "mcp.server",
        "mcp.server.lowlevel",
    ):
        logging.getLogger(very_noisy_logger).setLevel(logging.ERROR)


def configure_runtime_warnings() -> None:
    """Suppress known noisy environment warnings in product/demo runs."""

    if env_flag("PE_MAS_SHOW_ENV_WARNINGS", default=False):
        return
    warnings.filterwarnings(
        "ignore",
        message="urllib3 v2 only supports OpenSSL 1.1.1+.*",
    )


@contextlib.contextmanager
def quiet_stdio(log_file: Path, *, enabled: bool = True) -> Iterator[None]:
    """Redirect noisy MAS stdout/stderr to a runtime log file.

    This is a transition tool for MAS nodes that still use ``print``.
    New code should use structured logging instead.
    """

    if not enabled:
        yield
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with _STDIO_LOCK:
        with log_file.open("a", encoding="utf-8") as stream:
            stdout_fd = os.dup(1)
            stderr_fd = os.dup(2)
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(stream.fileno(), 1)
                os.dup2(stream.fileno(), 2)
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(stdout_fd, 1)
                os.dup2(stderr_fd, 2)
                os.close(stdout_fd)
                os.close(stderr_fd)
