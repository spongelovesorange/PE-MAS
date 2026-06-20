from __future__ import annotations

import json
import time
from typing import Any, Dict, List


class StudioRequirementsGateService:
    """Build Studio SSE payloads for the requirement gate.

    This service preserves the current frontend event shape while keeping
    payload assembly out of ``app.studio``. It does not run agents, mutate
    sessions, call RAG, or perform engineering analysis.
    """

    step_key = "requirements"

    def event(self, event_name: str, payload: Dict[str, Any]) -> Dict[str, str]:
        return {
            "event": event_name,
            "data": json.dumps(payload, ensure_ascii=False),
        }

    def status_event(self, sid: str, message: str) -> Dict[str, str]:
        return self.event(
            "status",
            {
                "sid": sid,
                "step_key": self.step_key,
                "message": message,
            },
        )

    def thought_keypoints_event(self, sid: str, items: List[str]) -> Dict[str, str]:
        return self.event(
            "thought_keypoints",
            {
                "sid": sid,
                "step_key": self.step_key,
                "items": list(items or []),
            },
        )

    def search_progress_event(self, sid: str, item: Dict[str, Any]) -> Dict[str, str]:
        return self.event(
            "search_progress",
            {
                "sid": sid,
                "step_key": self.step_key,
                **dict(item or {}),
            },
        )

    def node_result_event(
        self,
        sid: str,
        intake: Dict[str, Any],
        checkpoint_details: Dict[str, Any],
        decision_options: List[Dict[str, Any]],
        requirement_sections: List[Dict[str, Any]],
        review_pack: Dict[str, Any],
    ) -> Dict[str, str]:
        return self.event(
            "node_result",
            {
                "sid": sid,
                "node": "requirements",
                "step_key": self.step_key,
                "title": "Power Electronics Requirement Analysis",
                "summary": (
                    "Structured the user request into explicit specs, derived specs, prioritized missing inputs, "
                    "feasibility risks, downstream tasks, and a handoff package. Schematic/BOM generation remains blocked."
                ),
                "result_status": "scaffold",
                "result_label": "Requirement Analysis + RAG + Review",
                "sections": list(requirement_sections or []),
                "reasoning": [
                    {
                        "label": "NLP path",
                        "value": "Deterministic requirement parser first; LLM reviewer is called when configured and shown explicitly in trace.",
                    },
                    {
                        "label": "Gate policy",
                        "value": "A release schematic is blocked until specs, assumptions, topology, magnetics, loop, simulation, layout, and test evidence are explicitly gated.",
                    },
                    {
                        "label": "Engineer logic",
                        "value": "The first action is requirements and assumption control, not automatic schematic/BOM generation.",
                    },
                    {"label": "Reviewer status", "value": review_pack.get("status") or "review"},
                ],
                "evidence": [
                    {"label": "Workflow", "value": "Human-in-the-loop requirements gate"},
                    {
                        "label": "RAG / sources",
                        "value": f"{len(review_pack.get('search_items') or [])} local/web source card(s) attached.",
                    },
                    {"label": "Reviewer trace", "value": " | ".join(review_pack.get("trace") or [])},
                    {
                        "label": "Traceability",
                        "value": "Specs, assumptions, decisions, risks, and artifacts are stored in design_meta.",
                    },
                ],
                "content": json.dumps(
                    {
                        "checkpoint_details": checkpoint_details,
                        "decision_options": decision_options,
                        **dict(intake or {}),
                    },
                    ensure_ascii=False,
                ),
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    def checkpoint_event(
        self,
        sid: str,
        intake: Dict[str, Any],
        checkpoint_details: Dict[str, Any],
        decision_options: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        return self.event(
            "checkpoint",
            {
                "sid": sid,
                "phase": "requirements",
                "title": "Requirements Gate",
                "question": (
                    "Review the extracted design envelope. PE-MAS will not generate a release schematic until "
                    "critical specs and assumptions are accepted or edited."
                ),
                "options": [row.get("option", "") for row in decision_options],
                "commands": [row.get("command", "") for row in decision_options],
                "context": self.checkpoint_context(intake, checkpoint_details, decision_options),
            },
        )

    def checkpoint_context(
        self,
        intake: Dict[str, Any],
        checkpoint_details: Dict[str, Any],
        decision_options: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "requirements_intake": intake,
            "checkpoint_details": checkpoint_details,
            "decision_options": decision_options,
            "specifications": intake.get("specs", {}),
            "assumptions": intake.get("assumptions", []),
            "missing_inputs": intake.get("missing_inputs", []),
            "topology_candidates": intake.get("topology_candidates", []),
            "decisions": intake.get("decisions", []),
            "risks": intake.get("risks", []),
            "artifacts": intake.get("artifacts", []),
            "recommended_patch": {
                "specifications": intake.get("specs", {}),
                "design_overrides": {
                    "assumptions": intake.get("assumptions", []),
                    "topology_candidates": intake.get("topology_candidates", []),
                },
            },
        }

    def done_event(
        self,
        sid: str,
        intake: Dict[str, Any],
        checkpoint_details: Dict[str, Any],
        decision_options: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        return self.event(
            "done",
            {
                "sid": sid,
                "summary": "Requirements Gate is ready. Adopt defaults to continue, or edit assumptions before running the PE design workflow.",
                "report": "",
                "design_meta": {
                    "mode": "requirements_intake",
                    "status": "waiting_spec_lock",
                    "checkpoint_details": checkpoint_details,
                    "decision_options": decision_options,
                    **dict(intake or {}),
                },
            },
        )
