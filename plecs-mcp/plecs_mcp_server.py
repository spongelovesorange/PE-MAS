"""Compatibility entrypoint.

Keeps existing launch path stable for MAS integration:
python plecs-mcp/plecs_mcp_server.py

Internally delegates to the modular package implementation under src/plecs_mcp.
"""

from __future__ import annotations

import os
import sys
from importlib import import_module


def _bootstrap_src_path() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def main() -> None:
    _bootstrap_src_path()
    cli_mod = import_module("plecs_mcp.cli")
    cli_mod.main()


if __name__ == "__main__":
    main()
