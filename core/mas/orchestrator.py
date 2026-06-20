from __future__ import annotations

import uuid
from typing import Any, Dict

from core.agents.registry import AgentRegistry
from core.agents.requirement_analysis.agent import RequirementAnalysisMasAgent

from .artifacts import ArtifactStore
from .state import MasState
from .trace import TraceRecord


class MasOrchestrator:
    """Minimal plan-execute-observe orchestrator.

    v1 executes the requirement-analysis step and writes artifacts. Later agents
    can be registered without changing API routers.
    """

    def __init__(self, registry: AgentRegistry | None = None, artifact_store: ArtifactStore | None = None) -> None:
        self.registry = registry or AgentRegistry()
        if self.registry.get("requirement_analysis") is None:
            self.registry.register(RequirementAnalysisMasAgent())
        self.artifact_store = artifact_store or ArtifactStore()

    def run_requirement_analysis(self, prompt: str, project_id: str | None = None) -> MasState:
        project_id = project_id or str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        state = MasState(project_id=project_id, run_id=run_id, input_prompt=prompt)
        agent = self.registry.get("requirement_analysis")
        if agent is None:
            raise RuntimeError("requirement_analysis agent is not registered")

        state.trace.append(TraceRecord(step="requirement_analysis", status="running", message="Starting requirement analysis."))
        self.artifact_store.write_json(project_id, run_id, "input.json", {"prompt": prompt})
        result: Dict[str, Any] = agent.run({"prompt": prompt, "project_id": project_id})
        state.data["requirement_analysis"] = result
        state.artifacts["requirement_analysis.json"] = self.artifact_store.write_json(project_id, run_id, "requirement_analysis.json", result)
        state.artifacts["requirement_analysis.md"] = self.artifact_store.run_dir(project_id, run_id).joinpath("requirement_analysis.md").as_posix()
        self.artifact_store.run_dir(project_id, run_id).joinpath("requirement_analysis.md").write_text(
            str(result.get("human_readable_report") or ""),
            encoding="utf-8",
        )
        state.trace.append(TraceRecord(step="requirement_analysis", status="done", message="Requirement analysis completed."))
        for record in state.trace:
            self.artifact_store.append_trace(project_id, run_id, record.model_dump(mode="json"))
        return state
