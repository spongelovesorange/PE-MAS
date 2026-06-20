from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .topology_service import TopologyKnowledgeService


class PlecsModelRegistry:
    """Reports local PLECS model coverage without fabricating missing models."""

    def __init__(self, topology_service: Optional[TopologyKnowledgeService] = None, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.topology_service = topology_service or TopologyKnowledgeService(repo_root=self.repo_root)

    def status(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for topo in self.topology_service.list_topologies():
            rel_path = str(topo.get("local_model_path") or "")
            path = self.repo_root / rel_path if rel_path else None
            exists = bool(path and path.exists())
            declared = str(topo.get("plecs_model_status") or "missing")
            effective = declared
            if declared == "available" and not exists:
                effective = "missing"
            rows.append({
                "topology": topo.get("name"),
                "model_family": topo.get("converter_family") or topo.get("conversion_direction") or "unknown",
                "declared_status": declared,
                "status": effective,
                "local_model_path": rel_path,
                "exists": exists,
                "validation_status": "unvalidated" if exists else "not_available",
                "note": "Local PLECS model found." if exists else "No local PLECS model file available; do not simulate as covered.",
                "notes": "Registry reports file coverage only; waveform/model validation is a separate gate." if exists else "Missing/planned model; do not claim simulation support.",
            })
        return rows

    def get(self, topology_name: str) -> Optional[Dict[str, Any]]:
        name = str(topology_name or "").lower()
        for row in self.status():
            if str(row.get("topology") or "").lower() == name:
                return row
        return None
