from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RequirementAnalysisRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    project_id: Optional[str] = None


class RequirementField(BaseModel):
    name: str
    value: Any
    unit: str = ""
    source_text: str = ""
    requirement_type: str = "other"
    status: str = "confirmed"
    source_type: str = "explicit"


class DerivedRequirement(BaseModel):
    name: str
    value: Any
    unit: str = ""
    formula: str
    based_on: List[str] = Field(default_factory=list)
    status: str = "derived"


class QualitativeRequirement(BaseModel):
    name: str
    description: str
    status: str = "requires_clarification"
    source_type: str = "explicit_qualitative"


class MissingInformation(BaseModel):
    item: str
    priority: str
    why_it_matters: str
    affected_design_tasks: List[str] = Field(default_factory=list)
    clarification_question: str
    architecture_critical: bool = False


class DesignTask(BaseModel):
    task_name: str
    purpose: str
    required_inputs: List[str] = Field(default_factory=list)
    expected_outputs: List[str] = Field(default_factory=list)
    downstream_agent: str


class FeasibilityIssue(BaseModel):
    issue: str
    risk_level: str
    explanation: str
    quantitative_basis: str = ""
    affected_requirements: List[str] = Field(default_factory=list)
    recommended_action: str


class WorkflowStep(BaseModel):
    step: int
    action: str
    reason: str
    depends_on: List[str] = Field(default_factory=list)


class AssumptionRecord(BaseModel):
    assumption: str
    reason: str
    risk_if_wrong: str
    must_confirm: bool = True


class QualityCheck(BaseModel):
    no_unsupported_numbers: bool
    explicit_specs_have_source: bool
    derived_values_have_formula: bool
    critical_missing_info_checked: bool
    no_final_topology_fabricated: bool


class RequirementAnalysisResult(BaseModel):
    agent_name: str = "Power_Electronics_Requirement_Analysis_Agent"
    task_type: str = "requirement_analysis"
    project_id: Optional[str] = None
    human_readable_report: str
    spec_gate_status: str = "PARTIAL"
    layer1_minimum_converter_definition: List[Dict[str, Any]] = Field(default_factory=list)
    layer2_constraints_preferences: List[Dict[str, Any]] = Field(default_factory=list)
    engineering_derived_checks: List[Dict[str, Any]] = Field(default_factory=list)
    spec_package: Dict[str, Any] = Field(default_factory=dict)
    normalized_specs: Dict[str, Any] = Field(default_factory=dict)
    project_summary: Dict[str, Any]
    application_analysis: Dict[str, Any]
    extracted_specifications: Dict[str, List[Dict[str, Any]]]
    missing_information: List[Dict[str, Any]]
    clarification_questions: List[Dict[str, Any]]
    preliminary_design_task_decomposition: List[Dict[str, Any]]
    feasibility_and_conflict_check: List[Dict[str, Any]]
    refined_design_objectives: Dict[str, List[str]]
    recommended_workflow: List[Dict[str, Any]]
    assumption_register: List[Dict[str, Any]]
    handoff_package: Dict[str, Any]
    readiness_status: str
    quality_check: Dict[str, bool]
    frontend_response: Dict[str, Any]
    source_provenance: List[Dict[str, Any]] = Field(default_factory=list)

    def to_api_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")
