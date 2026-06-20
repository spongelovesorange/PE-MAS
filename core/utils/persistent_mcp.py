from __future__ import annotations

from datetime import timedelta
import json
from typing import Any, Dict, List, Optional

import anyio

try:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    MCP_AVAILABLE = True
except Exception:
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    MCP_AVAILABLE = False


def _serialize_item(item: Any) -> Dict[str, Any]:
    if item is None:
        return {}
    if hasattr(item, "model_dump"):
        try:
            dumped = item.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    payload: Dict[str, Any] = {}
    for key in ("type", "text", "name", "description", "inputSchema", "annotations"):
        value = getattr(item, key, None)
        if value is not None:
            payload[key] = value
    return payload


def _extract_tool_payload(tool_result: Any) -> Dict[str, Any]:
    content = getattr(tool_result, "content", None) or []
    text_chunks: List[str] = []
    serialized: List[Dict[str, Any]] = []

    for item in content:
        serialized.append(_serialize_item(item))
        text_val = getattr(item, "text", None)
        if text_val:
            text_chunks.append(str(text_val))

    raw_text = "\n".join(text_chunks).strip()
    parsed_json: Optional[Any] = None
    if raw_text:
        try:
            parsed_json = json.loads(raw_text)
        except Exception:
            parsed_json = None

    return {
        "raw_text": raw_text,
        "text_chunks": text_chunks,
        "content": serialized,
        "json": parsed_json,
    }


class PersistentStdioMCPClient:
    def __init__(
        self,
        *,
        name: str,
        params: "StdioServerParameters",
        read_timeout_seconds: float = 60.0,
    ) -> None:
        self.name = name
        self.params = params
        self.read_timeout_seconds = float(read_timeout_seconds or 60.0)

    def is_available(self) -> bool:
        return MCP_AVAILABLE

    def is_started(self) -> bool:
        return False

    def start(self) -> Dict[str, Any]:
        if not MCP_AVAILABLE:
            return {"ok": False, "error": "mcp package unavailable in current environment"}
        return {"ok": True, "started": True, "reused": False, "session_mode": "ephemeral_batch"}

    def close(self) -> Dict[str, Any]:
        return {"ok": True, "closed": False, "session_mode": "ephemeral_batch"}

    async def _with_session(self, fn, *args) -> Dict[str, Any]:
        async with stdio_client(self.params) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=self.read_timeout_seconds)) as session:
                await session.initialize()
                return await fn(session, *args)

    async def _async_list_tools(self, session: Any) -> Dict[str, Any]:
        listing = await session.list_tools()
        tools = getattr(listing, "tools", None) or []
        items: List[Dict[str, Any]] = []
        for tool in tools:
            row = _serialize_item(tool)
            if "name" not in row:
                row["name"] = getattr(tool, "name", "")
            if "description" not in row:
                row["description"] = getattr(tool, "description", "")
            items.append(row)
        return {"ok": True, "tools": items, "count": len(items)}

    def list_tools(self) -> Dict[str, Any]:
        start = self.start()
        if not start.get("ok"):
            return start
        try:
            payload = anyio.run(self._with_session, self._async_list_tools)
        except Exception as e:
            return {"ok": False, "error": f"mcp_list_tools_failed: {e}"}
        payload["session_started"] = False
        payload["session_name"] = self.name
        payload["session_mode"] = "ephemeral_batch"
        return payload

    async def _async_call_tool(self, session: Any, tool_name: str, arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        tool_result = await session.call_tool(tool_name, arguments or {})
        payload = _extract_tool_payload(tool_result)
        return {
            "ok": True,
            "tool_name": tool_name,
            "arguments": arguments or {},
            **payload,
        }

    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        start = self.start()
        if not start.get("ok"):
            return start
        try:
            payload = anyio.run(self._with_session, self._async_call_tool, tool_name, arguments or {})
        except Exception as e:
            return {
                "ok": False,
                "error": f"mcp_call_tool_failed({tool_name}): {e}",
                "tool_name": tool_name,
                "arguments": arguments or {},
            }
        payload["session_started"] = False
        payload["session_name"] = self.name
        payload["session_mode"] = "ephemeral_batch"
        return payload

    async def _async_call_batch(self, session: Any, calls: List[Dict[str, Any]], stop_on_error: bool) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for idx, call in enumerate(calls or []):
            tool_name = str(call.get("tool_name") or call.get("tool") or "").strip()
            arguments = call.get("arguments") or call.get("args") or {}
            if not tool_name:
                row = {"ok": False, "error": "missing tool_name", "index": idx}
            else:
                try:
                    row = await self._async_call_tool(session, tool_name, arguments)
                    row["index"] = idx
                except Exception as e:
                    row = {
                        "ok": False,
                        "error": f"mcp_call_tool_failed({tool_name}): {e}",
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "index": idx,
                    }
            results.append(row)
            if stop_on_error and not row.get("ok"):
                break
        return {
            "ok": True,
            "results": results,
            "count": len(results),
            "stopped_early": bool(stop_on_error and any(not row.get("ok") for row in results)),
        }

    def call_batch(self, calls: List[Dict[str, Any]], stop_on_error: bool = False) -> Dict[str, Any]:
        start = self.start()
        if not start.get("ok"):
            return start
        try:
            payload = anyio.run(self._with_session, self._async_call_batch, calls or [], bool(stop_on_error))
        except Exception as e:
            return {"ok": False, "error": f"mcp_call_batch_failed: {e}"}
        payload["session_name"] = self.name
        payload["session_started"] = False
        payload["session_mode"] = "ephemeral_batch"
        return payload
