from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.knowledge.topology_kb import TopologyKnowledgeService


class TopologySelectionService:
    """Task B contract service for topology selection.

    v1 is intentionally deterministic and offline. It consumes the Spec Package
    produced by Task A and returns candidate routes plus the exact information a
    future AI/topology evaluator should receive. It does not freeze a final
    topology.
    """

    def __init__(self, topology_service: Optional[TopologyKnowledgeService] = None) -> None:
        self.topology_service = topology_service or TopologyKnowledgeService()

    def analyze(self, spec_package: Dict[str, Any]) -> Dict[str, Any]:
        topology_input = self._extract_topology_input(spec_package)
        candidates = self._candidate_routes(topology_input)
        return {
            "agent": "topology_selection",
            "task": "Task B",
            "status": "CANDIDATE_COMPARISON_READY" if candidates else "NEEDS_SPEC_CLARIFICATION",
            "final_topology_selected": False,
            "ai_input_contract": self._ai_input_contract(spec_package, topology_input),
            "candidate_routes": candidates,
            "comparison_axes": [
                "isolation fit",
                "input/output voltage fit",
                "rated and peak power fit",
                "efficiency potential",
                "thermal risk",
                "EMI/debug risk",
                "control complexity",
                "magnetics complexity",
                "PLECS model availability",
                "BOM/cost/supply-chain risk",
            ],
            "test_contract": {
                "must_not_select_final_topology_when_architecture_gate_is_tbd": True,
                "must_compare_isolated_and_non_isolated_routes_when_isolation_is_unknown": topology_input.get("isolation_requirement") in {"unknown", "TBD", None},
                "must_report_model_coverage_without_faking_missing_plecs_models": True,
                "must_return_handoff_to_power_stage_after_candidate_shortlist": True,
            },
            "next_handoff": {
                "to_power_stage": "Run stress and operating-point estimates only after candidate route is selected or explicitly evaluated.",
                "to_magnetics": "Generate magnetic requirements only for isolated or inductor-based shortlisted routes.",
                "to_simulation": "Use PLECS only where registry reports available/planned/missing status honestly.",
            },
        }

    @staticmethod
    def _extract_topology_input(spec_package: Dict[str, Any]) -> Dict[str, Any]:
        notes = spec_package.get("handoff_notes", {}) if isinstance(spec_package, dict) else {}
        topology = notes.get("topology_agent", {}) if isinstance(notes, dict) else {}
        if not topology and "topology_selection_agent" in spec_package:
            topology = spec_package.get("topology_selection_agent", {})
        return {
            "spec_gate_status": topology.get("spec_gate_status") or spec_package.get("spec_gate_status"),
            "conversion": topology.get("conversion"),
            "vin_range": topology.get("vin_range"),
            "vin_nominal": topology.get("vin_nominal"),
            "vout": topology.get("vout"),
            "rated_power": topology.get("rated_power"),
            "peak_power": topology.get("peak_power"),
            "efficiency_target": topology.get("efficiency_target"),
            "isolation_requirement": topology.get("isolation_requirement") or "unknown",
            "open_items": topology.get("open_items", []),
        }

    def _candidate_routes(self, topology_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        conversion = str(topology_input.get("conversion") or "").upper()
        isolation = str(topology_input.get("isolation_requirement") or "unknown").lower()
        vin_range = topology_input.get("vin_range") or []
        vout = topology_input.get("vout")
        rated_power = topology_input.get("rated_power")

        records = self.topology_service.list_topologies()
        candidates: List[Dict[str, Any]] = []
        for row in records:
            name = row.get("name")
            include = False
            reason: List[str] = []
            if conversion == "DC/DC":
                if row.get("conversion_direction") in {"step_down", "step_up_down", "bidirectional", "resonant"}:
                    include = True
                    reason.append("DC/DC-compatible route")
                if isolation in {"unknown", "tbd"} and row.get("isolated") in {True, False}:
                    reason.append("kept because isolation is unresolved")
                elif "isolat" in isolation and row.get("isolated") is True:
                    reason.append("matches isolated requirement")
                elif isolation in {"non-isolated", "not required", "no"} and row.get("isolated") is False:
                    reason.append("matches non-isolated requirement")
                elif "isolat" in isolation and row.get("isolated") is False:
                    include = False
            elif conversion == "AC/DC":
                if name in {"Flyback Converter", "Half-Bridge Converter", "Full-Bridge Converter", "LLC Resonant Converter"}:
                    include = True
                    reason.append("offline AC/DC candidate family")

            if include:
                candidates.append({
                    "name": name,
                    "converter_family": row.get("converter_family"),
                    "isolated": row.get("isolated"),
                    "conversion_direction": row.get("conversion_direction"),
                    "plecs_model_status": row.get("plecs_model_status"),
                    "local_model_path": row.get("local_model_path") or "",
                    "why_candidate": reason,
                    "main_design_tasks": row.get("main_design_tasks", []),
                    "common_risks": row.get("common_risks", []),
                    "fit_notes": self._fit_notes(row, vin_range, vout, rated_power),
                })
        return candidates[:6]

    @staticmethod
    def _fit_notes(row: Dict[str, Any], vin_range: Any, vout: Any, rated_power: Any) -> List[str]:
        notes: List[str] = []
        if vin_range and vout:
            try:
                vin_min, vin_max = float(vin_range[0]), float(vin_range[1])
                vout_f = float(vout)
                if vin_min > vout_f and row.get("conversion_direction") == "step_down":
                    notes.append("input range stays above output; step-down route is plausible")
                if vin_min <= vout_f <= vin_max and row.get("conversion_direction") == "step_up_down":
                    notes.append("input range crosses output; buck-boost style route may be relevant")
            except Exception:
                pass
        if rated_power:
            notes.append(f"rated power passed from Spec Agent: {rated_power} W")
        return notes

    @staticmethod
    def _ai_input_contract(spec_package: Dict[str, Any], topology_input: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "give_ai": [
                "Layer-1 minimum converter definition",
                "Layer-2 constraints/preferences",
                "derived quantities and formulas",
                "engineering-derived checks",
                "open questions/TBD items",
                "risk flags",
                "non-assumptions and must-not-assume list",
            ],
            "do_not_give_ai_as_confirmed": [
                "final topology",
                "unconfirmed isolation decision",
                "unconfirmed EMI standard",
                "unconfirmed mechanical envelope",
                "unconfirmed cooling method",
                "unconfirmed protection thresholds",
            ],
            "normalized_topology_input": topology_input,
            "source_spec_gate_status": spec_package.get("spec_gate_status"),
        }
