"""Entrypoint for the PE-MAS FastAPI application."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from core.runtime import configure_logging, configure_runtime_warnings, env_flag

load_dotenv(Path(__file__).resolve().parent / ".env")
configure_logging()
configure_runtime_warnings()

from app.main import app, create_app  # noqa: E402

__all__ = ["app", "create_app"]


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("PE_MAS_HOST")
    if not host:
        raise RuntimeError("PE_MAS_HOST must be configured locally before starting the server.")
    port = int(os.getenv("PE_MAS_PORT", os.getenv("PORT", "8000")) or "8000")
    access_log = env_flag("PE_MAS_ACCESS_LOG", default=False)
    uvicorn.run(app, host=host, port=port, access_log=access_log, log_level=os.getenv("PE_MAS_LOG_LEVEL", "info").lower())
