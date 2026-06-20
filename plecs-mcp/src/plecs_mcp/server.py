from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .tools import register_tools


def create_server() -> FastMCP:
    mcp = FastMCP("plecs-mcp")
    register_tools(mcp)
    return mcp
