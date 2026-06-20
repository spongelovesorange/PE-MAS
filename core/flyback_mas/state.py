from typing import TypedDict, List, Optional, Literal, Dict, Any, Union
from typing_extensions import Annotated
import operator

# --- DeepRare-Inspired Traceability ---
class ReasoningTraceItem(TypedDict):
    """A single step in the transparent reasoning chain."""
    step: int
    agent: str
    action: str  # e.g., "Web Search", "Physics Calculation", "Self-Reflection"
    evidence: str # e.g., "Found datasheet for IPP60R190P6", "App Note AN-1024 Equation 5"
    conclusion: str # e.g., "Selected VOR=80V based on standard practice for 230Vac"
    confidence: float # 0.0 - 1.0

class EvidenceSource(TypedDict):
    """A verified external knowledge source."""
    source_type: Literal["Paper", "Datasheet", "AppNote", "Standard", "Web"]
    title: str
    url: Optional[str]
    relevance_score: float # 0.0 - 1.0
    key_insight: str # e.g., "Recommended VOR = 100V for high efficiency"

# --- Sub-structures ---

class DesignSpecs(TypedDict):
    """Design specifications extracted from user input."""
    is_chitchat: bool
    response_text: Optional[str]
    input_voltage_min: float  # V_ac_min
    input_voltage_max: float  # V_ac_max
    output_voltage: float     # V_out
    output_current: float     # I_out
    efficiency_target: float  # e.g., 0.85
    max_ripple_voltage: float # V_pp
    isolation: bool           # True/False
    application_type: str     # e.g., "Adapter", "Industrial"

class DesignRequestProfile(TypedDict, total=False):
    """Higher-level intent extracted from richer prompts and follow-up requests."""
    raw_request: str
    conversation_intent: Literal[
        "new_design",
        "modify_existing",
        "follow_up_qa",
        "follow_up_report",
        "follow_up_review",
        "skill_request",
    ]
    topology: str
    reuse_previous_result: bool
    preserve_prior_specs: bool
    session_context_available: bool
    requested_outputs: List[str]
    report_requirements: List[str]
    workflow_preferences: List[str]
    component_grounding_required: bool
    magnetics_detail_requested: bool
    simulation_corner_preference: Literal["auto", "low_line", "high_line", "nominal", "custom"]
    requested_input_ac_rms_v: float
    requested_input_dc_bus_v: float
    experiment_intent: Literal["none", "corner_rerun"]

class TheoreticalDesign(TypedDict):
    """Theoretical design parameters calculated from formulas."""
    topology: Literal["Flyback"]
    switching_frequency: float  # Hz
    primary_inductance: float   # Henrys
    primary_peak_current: float # Amps
    turns_ratio: float          # Np/Ns
    max_duty_cycle: float       # D_max
    ripple_factor: float        # k_rf (dimensionless current ripple factor)
    magnetizing_current_ripple: float # Delta_I
    snubber_r: float            # Ohms
    snubber_c: float            # Farads
    reflected_output_voltage: float # Vor

class BillOfMaterials(TypedDict):
    """Bill of Materials: Selected real components."""
    mosfet: Dict[str, Any]      # {mpn, vds, rds_on, package...}
    diode: Dict[str, Any]       # {mpn, vr, vf...}
    controller: Dict[str, Any]  # {mpn, freq_range...}
    transformer: Dict[str, Any] # {core_shape, core_material, Np, Ns, gap...}
    input_cap: Dict[str, Any]   # {mpn, value, voltage...}
    output_cap: Dict[str, Any]  # {mpn, value, esr...}
    input_protection: Dict[str, Any]  # {fuse, ntc, mov, bridge_rectifier...}
    emi_filter: Dict[str, Any]  # {cm_choke, x_cap, y_cap, dm_inductor...}
    clamp_snubber: Dict[str, Any]  # {rcd_clamp, tvs, target_vclamp...}

class SimulationMetrics(TypedDict):
    """Simulation results extracted from PLECS."""
    efficiency_measured: float
    v_out_ripple_measured: float
    v_ds_spike_max: float       # MOSFET Vds spike
    transformer_temp_est: float # Transformer est. temp
    is_converged: bool          # Did simulation converge?
    waveforms_path: str         # Path to waveform files

class VerificationResult(TypedDict):
    """Verification conclusion."""
    status: Literal["PASS", "FAIL", "WARN", "REVIEW_NEEDED", "NEEDS_HUMAN_REVIEW"]
    failed_items: List[str]     # e.g., ["Efficiency: 82% < 85%"]
    correction_strategy: str    # e.g., "DECREASE_FSW_FOR_EFFICIENCY"

class CorrectionReview(TypedDict):
    """Post-validation alignment check against original user intent and application."""
    status: Literal["ALIGNED", "MISMATCH", "REVIEW_NEEDED"]
    summary: str
    mismatches: List[str]
    recommendations: List[str]

class SkillState(TypedDict):
    """State for the active skill."""
    active_skill_id: Optional[str]
    skill_output: Optional[Dict[str, Any]]
    skill_history: List[str]

# --- Main State Definition ---

class PowerSupplyState(TypedDict):
    # Message history
    messages: Annotated[List[str], operator.add]
    
    # Structured Engineering Data
    specifications: Optional[DesignSpecs]
    request_profile: Optional[DesignRequestProfile]
    theoretical_design: Optional[TheoreticalDesign]
    bom: Optional[BillOfMaterials]
    simulation_results: Optional[SimulationMetrics]
    verification: Optional[VerificationResult]
    correction_review: Optional[CorrectionReview]
    
    # [DeepRare] Knowledge & Transparency
    reasoning_trace: Annotated[List[Any], operator.add] 
    literature_references: Annotated[List[EvidenceSource], operator.add]
    hard_guardrails_prompt: Optional[str]
    retrieved_knowledge_context: Optional[str]
    retrieved_knowledge_references: Optional[List[Dict[str, Any]]]
    
    # [NEW] Skills Support
    active_skill: Optional[str] # ID of the active skill
    skill_state: Optional[SkillState]
    
    # [NEW] Reasoning Traces (Manus-style)
    reasoning_logs: Annotated[Dict[str, List[str]], lambda a, b: {**a, **b}]
    formula_checks: Optional[Dict[str, Any]]
    node_verification: Optional[Dict[str, Any]]
    design_overrides: Optional[Dict[str, Any]]
    execution_plan: Optional[Dict[str, Any]]
    planning_summary: Optional[str]

    # Control Flow Metadata
    iteration: int              # Current iteration count
    max_iterations: int         # Max allowed iterations
    error_log: List[str]        # System error log
    config: Optional[Dict[str, Any]]
    thread_id: Optional[str]

    # [NEW] Best Effort Tracking
    best_design_candidate: Optional[Dict[str, Any]] # Snapshot of the best design so far (params, sim, bom)
    
    # [NEW] Final Report Artifact
    report_content: Optional[str] # The generated markdown report content

    # [NEW] Skill-driven quality artifacts
    evidence_grade: Optional[Dict[str, Any]]
    peer_review_findings: Optional[Dict[str, Any]]
    simulation_consistency: Optional[Dict[str, Any]]
    param_sensitivity_plan: Optional[Dict[str, Any]]
    citation_pack: Optional[Dict[str, Any]]
    magnetic_design: Optional[Dict[str, Any]]

    # [NEW] Lifelong learning artifacts
    memory_context: Optional[Dict[str, Any]]
    memory_writeback: Optional[Dict[str, Any]]
    memory_insights: Optional[Dict[str, Any]]
    iteration_learning: Optional[Dict[str, Any]]
    learning_context: Optional[Dict[str, Any]]
    curriculum_context: Optional[Dict[str, Any]]
    skill_catalog: Optional[List[Dict[str, Any]]]
    skill_recommendations: Optional[List[Dict[str, Any]]]
