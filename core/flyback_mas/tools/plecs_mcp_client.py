from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from mcp.client.stdio import StdioServerParameters

    MCP_AVAILABLE = True
except Exception:
    StdioServerParameters = None
    MCP_AVAILABLE = False

from core.utils.persistent_mcp import PersistentStdioMCPClient


_PLECS_CLIENT: Optional[PersistentStdioMCPClient] = None


def _default_server_script() -> str:
    # core/flyback_mas/tools -> project root at parents[3]
    root = Path(__file__).resolve().parents[3]
    return str(root / "plecs-mcp" / "plecs_mcp_server.py")


def _build_server_params() -> "StdioServerParameters":
    command = str(os.getenv("PE_MAS_PLECS_MCP_COMMAND", "")).strip() or sys.executable
    args_env = str(os.getenv("PE_MAS_PLECS_MCP_ARGS", "")).strip()
    if args_env:
        args = [x for x in args_env.split(" ") if x]
    else:
        args = [_default_server_script()]
    return StdioServerParameters(command=command, args=args)


def _get_client() -> PersistentStdioMCPClient:
    global _PLECS_CLIENT
    if _PLECS_CLIENT is None:
        _PLECS_CLIENT = PersistentStdioMCPClient(
            name="plecs",
            params=_build_server_params(),
            read_timeout_seconds=float(os.getenv("PE_MAS_PLECS_MCP_TIMEOUT_SEC", "120")),
        )
    return _PLECS_CLIENT


def _coerce_json_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"ok": False, "error": "invalid MCP output"}
    if not payload.get("ok"):
        return payload
    parsed = payload.get("json")
    if isinstance(parsed, dict):
        return parsed
    raw_text = str(payload.get("raw_text") or "").strip()
    if not raw_text:
        return {"ok": False, "error": "empty MCP tool response"}
    return {"ok": False, "error": f"non-json MCP response: {raw_text[:500]}"}


def ensure_plecs_mcp_session() -> Dict[str, Any]:
    if not MCP_AVAILABLE:
        return {"ok": False, "error": "mcp package unavailable in current environment"}
    return _get_client().start()


def close_plecs_mcp_session() -> Dict[str, Any]:
    global _PLECS_CLIENT
    if _PLECS_CLIENT is None:
        return {"ok": True, "closed": False}
    result = _PLECS_CLIENT.close()
    if result.get("ok"):
        _PLECS_CLIENT = None
    return result


def list_plecs_mcp_tools() -> Dict[str, Any]:
    ready = ensure_plecs_mcp_session()
    if not ready.get("ok"):
        return ready
    return _get_client().list_tools()


def plecs_call_tool_via_mcp(tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ready = ensure_plecs_mcp_session()
    if not ready.get("ok"):
        return ready
    payload = _get_client().call_tool(tool_name, arguments or {})
    return _coerce_json_response(payload)


def plecs_batch_call_via_mcp(calls: List[Dict[str, Any]], stop_on_error: bool = False) -> Dict[str, Any]:
    ready = ensure_plecs_mcp_session()
    if not ready.get("ok"):
        return ready
    batch = _get_client().call_batch(calls or [], stop_on_error=stop_on_error)
    if not batch.get("ok"):
        return batch
    results: List[Dict[str, Any]] = []
    for row in batch.get("results") or []:
        coerced = _coerce_json_response(row)
        if "index" in row:
            coerced["index"] = row.get("index")
        results.append(coerced)
    return {
        "ok": True,
        "results": results,
        "count": len(results),
        "stopped_early": bool(batch.get("stopped_early")),
    }


def run_plecs_simulation_via_mcp(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run PE-MAS flyback simulation through plecs-mcp compatibility tool."""
    try:
        out = plecs_call_tool_via_mcp("simulate_flyback", {"params": params or {}})
    except Exception as e:
        return {"ok": False, "error": f"mcp_invoke_failed: {e}"}

    if not isinstance(out, dict):
        return {"ok": False, "error": "invalid MCP output"}
    if not out.get("ok"):
        return out

    result = out.get("result")
    if not isinstance(result, dict):
        return {"ok": False, "error": "MCP response missing dict result"}

    return {"ok": True, "result": result, "notes": out.get("notes", [])}


def plecs_rpc_call_via_mcp(method: str, args: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Advanced helper for arbitrary XML-RPC method calls through MCP."""
    try:
        return plecs_call_tool_via_mcp("rpc_call", {"method": method, "args": args or []})
    except Exception as e:
        return {"ok": False, "error": f"mcp_invoke_failed: {e}"}


def plecs_discover_capabilities_via_mcp() -> Dict[str, Any]:
    return plecs_call_tool_via_mcp("discover_capabilities", {})


def plecs_open_model_via_mcp(model_path: str) -> Dict[str, Any]:
    return plecs_call_tool_via_mcp("open_model", {"model_path": model_path})


def plecs_close_model_via_mcp(model_name: str) -> Dict[str, Any]:
    return plecs_call_tool_via_mcp("close_model", {"model_name": model_name})


def plecs_list_open_models_via_mcp() -> Dict[str, Any]:
    return plecs_call_tool_via_mcp("list_open_models", {})


def plecs_simulate_via_mcp(model_name: str, simulation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return plecs_call_tool_via_mcp("simulate", {"model_name": model_name, "simulation": simulation or {}})


def plecs_run_script_via_mcp(script: str, language: str = "plecs") -> Dict[str, Any]:
    return plecs_call_tool_via_mcp("run_script", {"script": script, "language": language})


def plecs_circuit_action_via_mcp(
    model_name: str,
    action: Dict[str, Any],
    allow_script_fallback: bool = True,
) -> Dict[str, Any]:
    return plecs_call_tool_via_mcp(
        "circuit_action",
        {
            "model_name": model_name,
            "action": action,
            "allow_script_fallback": bool(allow_script_fallback),
        },
    )


def plecs_circuit_patch_via_mcp(
    model_name: str,
    actions: List[Dict[str, Any]],
    create_backup: bool = True,
    stop_on_error: bool = False,
) -> Dict[str, Any]:
    return plecs_call_tool_via_mcp(
        "circuit_patch",
        {
            "model_name": model_name,
            "actions": actions or [],
            "create_backup": bool(create_backup),
            "stop_on_error": bool(stop_on_error),
        },
    )
