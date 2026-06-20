from __future__ import annotations

import os
import shlex
import subprocess
from typing import Any, Dict, List

try:
    from mcp.client.stdio import StdioServerParameters

    MCP_AVAILABLE = True
except Exception:
    StdioServerParameters = None
    MCP_AVAILABLE = False

from .persistent_mcp import PersistentStdioMCPClient


_CLIENTS: Dict[str, PersistentStdioMCPClient] = {}


def parse_node_version() -> tuple[int, int, int]:
    try:
        out = subprocess.run(["node", "-v"], capture_output=True, text=True, check=False)
        version = (out.stdout or "").strip().lstrip("v")
        parts = version.split(".")
        return int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        return (0, 0, 0)


def node_supported_for_chrome_mcp() -> bool:
    major, minor, _ = parse_node_version()
    if major < 20:
        return False
    if major == 20 and minor < 19:
        return False
    if major == 22 and minor < 12:
        return False
    return True


def chrome_mcp_enabled() -> bool:
    val = str(os.getenv("PE_MAS_ENABLE_CHROME_MCP", "0")).strip().lower()
    return val in {"1", "true", "yes", "on"}


def _build_server_params() -> "StdioServerParameters":
    command = str(os.getenv("PE_MAS_CHROME_MCP_COMMAND", "")).strip() or "npx"
    args_env = str(os.getenv("PE_MAS_CHROME_MCP_ARGS", "")).strip()
    if args_env:
        args = shlex.split(args_env)
    else:
        args = ["-y", "chrome-devtools-mcp@latest"]
        if str(os.getenv("PE_MAS_CHROME_MCP_HEADLESS", "1")).strip().lower() in {"1", "true", "yes", "on"}:
            args.append("--headless")
        if str(os.getenv("PE_MAS_CHROME_MCP_ISOLATED", "1")).strip().lower() in {"1", "true", "yes", "on"}:
            args.append("--isolated")
        if str(os.getenv("PE_MAS_CHROME_MCP_USAGE_STATS", "0")).strip().lower() not in {"1", "true", "yes", "on"}:
            args.append("--no-usage-statistics")
    return StdioServerParameters(command=command, args=args)


def _get_client(session_key: str = "default") -> PersistentStdioMCPClient:
    key = str(session_key or "default")
    client = _CLIENTS.get(key)
    if client is None:
        client = PersistentStdioMCPClient(
            name=f"chrome:{key}",
            params=_build_server_params(),
            read_timeout_seconds=float(os.getenv("PE_MAS_CHROME_MCP_TIMEOUT_SEC", "90")),
        )
        _CLIENTS[key] = client
    return client


def browser_session_status(session_key: str = "default") -> Dict[str, Any]:
    client = _get_client(session_key)
    major, minor, patch = parse_node_version()
    return {
        "ok": True,
        "session_key": str(session_key or "default"),
        "mcp_available": MCP_AVAILABLE,
        "chrome_enabled": chrome_mcp_enabled(),
        "node_supported": node_supported_for_chrome_mcp(),
        "node_version": f"{major}.{minor}.{patch}",
        "session_started": client.is_started(),
        "session_mode": "ephemeral_batch",
    }


def ensure_browser_session(session_key: str = "default") -> Dict[str, Any]:
    if not MCP_AVAILABLE:
        return {"ok": False, "error": "mcp package unavailable in current environment"}
    if not chrome_mcp_enabled():
        return {"ok": False, "error": "chrome_mcp_disabled: set PE_MAS_ENABLE_CHROME_MCP=1"}
    if not node_supported_for_chrome_mcp():
        return {"ok": False, "error": "chrome_mcp_node_too_old: need Node.js >= 20.19"}
    return _get_client(session_key).start()


def close_browser_session(session_key: str = "default") -> Dict[str, Any]:
    key = str(session_key or "default")
    client = _CLIENTS.get(key)
    if client is None:
        return {"ok": True, "closed": False}
    result = client.close()
    if result.get("ok"):
        _CLIENTS.pop(key, None)
    return result


def list_browser_tools(session_key: str = "default") -> Dict[str, Any]:
    ready = ensure_browser_session(session_key)
    if not ready.get("ok"):
        return ready
    return _get_client(session_key).list_tools()


def browser_call_tool(tool_name: str, arguments: Dict[str, Any] | None = None, session_key: str = "default") -> Dict[str, Any]:
    ready = ensure_browser_session(session_key)
    if not ready.get("ok"):
        return ready
    return _get_client(session_key).call_tool(tool_name, arguments or {})


def browser_batch(actions: List[Dict[str, Any]], session_key: str = "default", stop_on_error: bool = False) -> Dict[str, Any]:
    ready = ensure_browser_session(session_key)
    if not ready.get("ok"):
        return ready
    return _get_client(session_key).call_batch(actions or [], stop_on_error=stop_on_error)
