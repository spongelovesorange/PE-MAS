from __future__ import annotations

import xmlrpc.client
from typing import Any, Dict, List, Optional, Tuple

from .config import rpc_url


def server() -> xmlrpc.client.ServerProxy:
    return xmlrpc.client.ServerProxy(rpc_url(), allow_none=True)


def resolve_method(root: Any, dotted_name: str) -> Any:
    target = root
    for part in dotted_name.split("."):
        target = getattr(target, part)
    return target


def call(method: str, args: Optional[List[Any]] = None) -> Any:
    s = server()
    fn = resolve_method(s, method)
    return fn(*(args or []))


def call_first_available(candidates: List[str], args: Optional[List[Any]] = None) -> Tuple[bool, str, Any]:
    errors: List[Dict[str, str]] = []
    for method in candidates:
        try:
            result = call(method, args or [])
            return True, method, result
        except Exception as e:
            errors.append({"method": method, "error": str(e)})
    return False, "", errors
