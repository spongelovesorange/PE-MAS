"""Application factory for PE-MAS."""

from __future__ import annotations

import os
import warnings

from core.runtime import configure_runtime_warnings, env_flag

configure_runtime_warnings()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.routers import health, plecs, requirements, topologies


def create_api_app() -> FastAPI:
    """Create the API-only application without the Studio frontend."""

    api = FastAPI(title="PE-MAS API")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def modular_home() -> str:
        return """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>PE-MAS Modular API</title>
            <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%231f2328'/%3E%3Cpath d='M17 3 8 18h7l-1 11 10-16h-7l0-10Z' fill='%23ffffff'/%3E%3C/svg%3E">
            <style>
              body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f5f7; color: #1d1d1f; }
              main { max-width: 860px; margin: 0 auto; padding: 72px 24px; }
              h1 { margin: 0 0 12px; font-size: 34px; letter-spacing: 0; }
              p { color: #424245; line-height: 1.6; }
              .panel { margin-top: 28px; padding: 22px; background: #fff; border: 1px solid rgba(0,0,0,.1); border-radius: 8px; box-shadow: 0 18px 50px rgba(0,0,0,.06); }
              code { background: #f0f0f2; border-radius: 6px; padding: 2px 6px; }
              a { color: #0066cc; text-decoration: none; font-weight: 650; }
              ul { padding-left: 20px; color: #424245; }
            </style>
          </head>
          <body>
            <main>
              <h1>PE-MAS API is running</h1>
              <p>The API backend is healthy. For the full Studio UI and Flyback MAS workflow, run <code>make run</code>.</p>
              <section class="panel">
                <p><strong>Useful links</strong></p>
                <ul>
                  <li><a href="/api/health">Health endpoint</a></li>
                  <li><a href="/docs">API docs</a></li>
                  <li><a href="/api/topologies">Topology database</a></li>
                  <li><a href="/api/plecs/models/status">PLECS model status</a></li>
                </ul>
                <p>For the full product UI, install all requirements and run <code>make run</code>.</p>
              </section>
            </main>
          </body>
        </html>
        """
    api.include_router(health.router)
    api.include_router(requirements.router)
    api.include_router(topologies.router)
    api.include_router(plecs.router)
    return api


def create_modular_app() -> FastAPI:
    """Backward-compatible name for the API-only app."""

    return create_api_app()


def create_app(*, include_studio: bool = True) -> FastAPI:
    """Create the production app."""

    if not include_studio:
        return create_api_app()

    from app.studio import app as studio_app

    return studio_app


def create_default_app() -> FastAPI:
    """Create the import-time ASGI app.

    ``PE_MAS_APP_MODE=studio`` serves the product UI and MAS workflow.
    ``PE_MAS_APP_MODE=api`` serves only the API routers.
    ``PE_MAS_APP_MODE=auto`` tries Studio first and falls back to API-only mode
    when optional Studio dependencies are unavailable.
    """

    mode = os.getenv("PE_MAS_APP_MODE", "studio").strip().lower()
    if mode in {"modular", "api"}:
        return create_api_app()
    if mode in {"studio", "product"}:
        return create_app(include_studio=True)
    if mode != "auto":
        warnings.warn(
            f"Unknown PE_MAS_APP_MODE={mode!r}; using auto mode.",
            RuntimeWarning,
            stacklevel=2,
        )

    try:
        return create_app(include_studio=True)
    except ModuleNotFoundError as exc:
        if env_flag("PE_MAS_WARN_ON_FALLBACK", default=False):
            warnings.warn(
                "Falling back to PE-MAS API-only mode because a Studio optional "
                f"dependency is unavailable: {exc.name}",
                RuntimeWarning,
                stacklevel=2,
            )
        return create_api_app()


app = create_default_app()
