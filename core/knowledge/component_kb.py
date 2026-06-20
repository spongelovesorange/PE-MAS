from __future__ import annotations

from typing import Any, Dict, List


class ComponentKnowledgeService:
    """Offline placeholder for future component super-database queries."""

    def search(self, query: str) -> List[Dict[str, Any]]:
        return [{"query": query, "status": "not_implemented_offline_placeholder"}]

