from __future__ import annotations

import os


def rpc_url() -> str:
    value = str(os.getenv("PLECS_RPC_URL") or "").strip()
    if not value:
        raise RuntimeError("PLECS_RPC_URL must be configured in the local environment.")
    return value
