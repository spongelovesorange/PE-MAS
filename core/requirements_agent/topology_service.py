from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class TopologyKnowledgeService:
    """Offline seed topology knowledge service.

    v1 intentionally uses a local seed file so tests and the first agent node run
    without online crawling. Each record carries provenance.
    """

    def __init__(self, data_path: Optional[Path] = None, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.data_path = data_path or self.repo_root / "data" / "topologies" / "topologies.json"
        self._records: Optional[List[Dict[str, Any]]] = None

    def list_topologies(self) -> List[Dict[str, Any]]:
        if self._records is None:
            self._records = json.loads(self.data_path.read_text(encoding="utf-8"))
        return [dict(row) for row in self._records]

    def get_topology(self, name: str) -> Optional[Dict[str, Any]]:
        needle = str(name or "").strip().lower()
        if not needle:
            return None
        for row in self.list_topologies():
            aliases = [str(x).lower() for x in row.get("aliases", [])]
            if str(row.get("name", "")).lower() == needle or needle in aliases:
                return row
            if needle.replace("-", " ") in str(row.get("name", "")).lower().replace("-", " "):
                return row
        return None

    def candidates_for_requirements(self, prompt: str, specs: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = str(prompt or "").lower()
        records = self.list_topologies()
        scored: List[tuple[int, Dict[str, Any]]] = []
        for row in records:
            score = 0
            aliases = [str(row.get("name", "")).lower(), *[str(x).lower() for x in row.get("aliases", [])]]
            if any(alias and alias in text for alias in aliases):
                score += 5
            if (specs.get("isolation") or specs.get("isolation_requirement")) and row.get("isolated") is True:
                if specs.get("input_type") == "AC":
                    if any(token in str(row.get("name", "")).lower() for token in ["flyback", "llc", "half-bridge", "full-bridge"]):
                        score += 2
                else:
                    score += 2
            if specs.get("input_type") == "AC" and "flyback" in str(row.get("name", "")).lower():
                score += 3
            if specs.get("input_type") != "AC" and specs.get("input_voltage_nominal") and specs.get("output_voltage"):
                try:
                    if float(specs["input_voltage_nominal"]) > float(specs["output_voltage"]) and row.get("conversion_direction") == "step_down":
                        score += 2
                except Exception:
                    pass
            if specs.get("input_type") != "AC" and specs.get("input_voltage_min") and specs.get("output_voltage"):
                try:
                    if float(specs["input_voltage_min"]) > float(specs["output_voltage"]) and row.get("conversion_direction") in {"step_down", "bidirectional"}:
                        score += 2
                except Exception:
                    pass
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [dict(row) | {"candidate_score": score} for score, row in scored if score > 0][:5]
