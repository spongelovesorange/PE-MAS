from __future__ import annotations

from typing import Any, Dict


class PlecsClient:
    """Placeholder PLECS client boundary.

    XML-RPC/MCP execution currently lives in the flyback tool layer; this
    adapter is the future stable boundary for direct PLECS clients.
    """

    def simulate(self, model_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "not_implemented",
            "model_path": model_path,
            "params": params,
        }
