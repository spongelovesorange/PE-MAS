import json
import os
import re
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ..state import PowerSupplyState, DesignSpecs, DesignRequestProfile
from ..knowledge_guardrails import get_hard_guardrails_prompt, retrieve_flyback_context
from ..formula_guardrails import normalize_and_validate_specs
from ..planning import build_execution_plan, summarize_execution_plan
try:
    from core.utils.component_rag_bridge import retrieve_component_rag_context
except Exception:
    retrieve_component_rag_context = None


def _normalize_user_message(msg_obj: Any) -> str:
    """Convert message tuple/object into plain user text for search/classification."""
    try:
        if isinstance(msg_obj, tuple) and len(msg_obj) >= 2:
            return str(msg_obj[1])
        content = getattr(msg_obj, "content", msg_obj)
        return str(content)
    except Exception:
        return str(msg_obj)
from ..skills_manager import SkillManager
from .research_helper import collect_node_research
# try:
#     from ...llm.llm import get_agent_reasoning
# except ImportError:
#     from core.llm.llm import get_agent_reasoning


def get_llm():
    """Lazy initialization of LLM to allow env var setting after import."""
    from core.llm.llm import get_llm_runtime_config

    runtime = get_llm_runtime_config(
        preferred_model=os.environ.get("PE_MAS_REQUIREMENTS_MODEL") or "gpt-4o"
    )
    api_key = runtime["api_key"] or "MISSING_PROVIDER_KEY"

    kwargs = {
        "model": runtime["model"],
        "temperature": 0.1,
        "api_key": api_key,
    }
    if runtime.get("api_base"):
        kwargs["base_url"] = runtime["api_base"]
    return ChatOpenAI(**kwargs)

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional

# Pydantic Model for Robust Extraction
class DesignSpecsModel(BaseModel):
    """Design specifications extraction model."""
    is_chitchat: bool = Field(description="True if casual chat OR checking info; False if design task.", default=False)
    response_text: Optional[str] = Field(description="If casual chat or Q&A, put the helpful answer here. If design task, leave empty.", default=None)
    input_voltage_min: Optional[float] = Field(description="Minimum input voltage (V).", default=None)
    input_voltage_max: Optional[float] = Field(description="Maximum input voltage (V).", default=None)
    output_voltage: Optional[float] = Field(description="Target output voltage (V).", default=None)
    output_current: Optional[float] = Field(description="Target output current (A).", default=None)
    efficiency_target: float = Field(description="Target efficiency (0.0 to 1.0), default 0.85", default=0.85)
    max_ripple_voltage: float = Field(description="Max allowable ripple (V), default 1% of Vout", default=0.2)
    isolation: bool = Field(description="Is isolation required?", default=True)
    application_type: str = Field(description="Application type context", default="Adapter")
    topology: str = Field(description="Requested topology. Default Flyback.", default="Flyback")
    reuse_previous_result: bool = Field(description="True if the user explicitly asks to continue from the current session result instead of starting from scratch.", default=False)
    requested_outputs: List[str] = Field(description="Explicit outputs the user wants in the final answer/report.", default_factory=list)
    workflow_preferences: List[str] = Field(description="Workflow or evidence preferences, for example local DigiKey grounding or full workflow execution.", default_factory=list)


def _extract_first_number(text: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, str(text or ""), flags=re.I)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


_REQUESTED_OUTPUT_HINTS = [
    ("switching frequency", ["switching frequency", "fsw", "f_sw", "开关频率"]),
    ("primary inductance", ["primary inductance", "magnetizing inductance", "lp", "初级电感", "励磁电感"]),
    ("turns ratio", ["turns ratio", "np/ns", "匝比"]),
    ("measured efficiency", ["measured efficiency", "efficiency", "效率"]),
    ("output ripple", ["output ripple", "ripple", "纹波"]),
    ("peak drain stress", ["peak drain stress", "drain stress", "vds", "漏极应力"]),
    ("dominant loss contributors", ["dominant loss", "loss contributor", "loss breakdown", "损耗构成", "损耗分解"]),
    ("bom blockers or provisional parts", ["bom blocker", "provisional", "part blocker", "器件阻塞", "临时器件"]),
    ("magnetic advisor mode", ["magnetic advisor", "magnetics recommendation", "heuristic fallback", "磁设计", "磁性件建议"]),
    ("node verification summary", ["node verification", "verification summary", "节点校验", "节点验证"]),
]

_WORKFLOW_PREFERENCE_HINTS = [
    ("full workflow execution", ["full workflow", "run the full", "完整流程", "全流程"]),
    ("local DigiKey grounding", ["digikey", "grounded corpus", "local corpus", "本地器件库", "真实器件库"]),
    ("realistic component selection", ["realistic component", "realistic selection", "真实器件选型", "真实选型"]),
    ("explicit magnetics recommendation", ["transformer recommendation", "magnetics recommendation", "磁性件建议", "变压器建议"]),
    ("proof-of-concept reporting", ["proof of concept", "poc", "概念验证"]),
]

_SIMULATION_CORNER_HINTS = {
    "low_line": ["low-line", "low line", "minimum line", "min line", "低线", "最低输入", "低压角点"],
    "high_line": ["high-line", "high line", "maximum line", "max line", "高线", "最高输入", "高压角点"],
    "nominal": ["nominal", "mid-line", "mid line", "rated line", "额定输入", "中间角点"],
}

_EXPERIMENT_RERUN_VERBS = [
    "rerun", "re-run", "run again", "simulate", "re-simulate", "evaluate", "verify", "check", "test", "sweep",
    "再跑", "重跑", "重新仿真", "仿真", "评估", "验证", "测试", "扫角点",
]

_FOLLOW_UP_HINTS = [
    "previously generated",
    "previous design",
    "current design",
    "current result",
    "this design",
    "this result",
    "using the previous",
    "using the current",
    "follow-up",
    "follow up",
    "reuse the previous",
    "base it on the previous",
    "上一个结果",
    "上一轮结果",
    "当前设计",
    "当前结果",
    "这个设计",
    "这个结果",
    "沿着上一个结果",
    "基于上一个结果",
]


def _has_prior_design_context(state: Dict[str, Any]) -> bool:
    return bool(
        state.get("theoretical_design")
        or state.get("bom")
        or state.get("simulation_results")
        or state.get("verification")
        or state.get("report_content")
    )


def _looks_like_fresh_spec_request(user_text: str) -> bool:
    text = str(user_text or "")
    low = text.lower()
    score = 0
    if re.search(r"\b\d+(?:\.\d+)?\s*[-~to]{1,3}\s*\d+(?:\.\d+)?\s*v(?:ac|dc)?\b", low):
        score += 2
    if re.search(r"\b\d+(?:\.\d+)?\s*v(?:dc)?\s*[/x* ]\s*\d+(?:\.\d+)?\s*a\b", low):
        score += 2
    if re.search(r"\b\d+(?:\.\d+)?\s*v(?:dc)?\s+\d+(?:\.\d+)?\s*a\b", low):
        score += 2
    if re.search(r"\b\d+(?:\.\d+)?\s*w\b", low):
        score += 1
    if any(k in low for k in ["design", "redesign", "optimize", "design a", "设计", "重设计", "优化"]):
        score += 1
    return score >= 3


def _looks_like_follow_up_request(user_text: str, state: Dict[str, Any]) -> bool:
    if not _has_prior_design_context(state):
        return False
    text = str(user_text or "")
    low = text.lower()
    if any(hint in low for hint in _FOLLOW_UP_HINTS):
        return True
    if any(hint in text for hint in _FOLLOW_UP_HINTS):
        return True
    follow_up_actions = [
        "review",
        "verify",
        "validation",
        "report",
        "summary",
        "explain",
        "why",
        "how",
        "analyze",
        "compare",
        "strict",
        "复核",
        "验证",
        "总结",
        "报告",
        "解释",
        "分析",
        "为什么",
        "如何",
    ]
    return any(k in low for k in follow_up_actions) or any(k in text for k in follow_up_actions)


def _extract_requested_outputs(user_text: str) -> List[str]:
    text = str(user_text or "")
    low = text.lower()
    outputs: List[str] = []
    for canonical, hints in _REQUESTED_OUTPUT_HINTS:
        if any(h in low for h in hints) or any(h in text for h in hints):
            outputs.append(canonical)
    return list(dict.fromkeys(outputs))


def _extract_workflow_preferences(user_text: str) -> List[str]:
    text = str(user_text or "")
    low = text.lower()
    prefs: List[str] = []
    for canonical, hints in _WORKFLOW_PREFERENCE_HINTS:
        if any(h in low for h in hints) or any(h in text for h in hints):
            prefs.append(canonical)
    return list(dict.fromkeys(prefs))


def _extract_requested_input_targets(user_text: str, state: Dict[str, Any]) -> Dict[str, Optional[float]]:
    text = str(user_text or "").strip()
    low = text.lower()
    requested_input_ac_rms_v: Optional[float] = None
    requested_input_dc_bus_v: Optional[float] = None

    has_range_expr = bool(re.search(r"\b\d+(?:\.\d+)?\s*[-~to]{1,3}\s*\d+(?:\.\d+)?\s*v(?:ac|dc)\b", low))
    if not has_range_expr:
        ac_candidates = re.findall(r"(\d+(?:\.\d+)?)\s*v(?:ac)\b", low)
        if len(ac_candidates) == 1:
            try:
                requested_input_ac_rms_v = float(ac_candidates[0])
            except Exception:
                requested_input_ac_rms_v = None

    dc_match = re.search(r"(?:dc\s*bus|bus|vin\s*bus|input\s*bus|at|@)\s*(\d+(?:\.\d+)?)\s*vdc\b", low)
    if dc_match:
        try:
            requested_input_dc_bus_v = float(dc_match.group(1))
        except Exception:
            requested_input_dc_bus_v = None

    specs = state.get("specifications") or {}
    try:
        vin_min = float(specs.get("input_voltage_min") or 0.0)
        vin_max = float(specs.get("input_voltage_max") or vin_min)
    except Exception:
        vin_min, vin_max = 0.0, 0.0

    if requested_input_ac_rms_v is not None and vin_max > 0.0:
        if requested_input_ac_rms_v < max(1.0, vin_min * 0.5) or requested_input_ac_rms_v > max(vin_max * 1.5, vin_max + 50.0):
            requested_input_ac_rms_v = None

    return {
        "requested_input_ac_rms_v": requested_input_ac_rms_v,
        "requested_input_dc_bus_v": requested_input_dc_bus_v,
    }



def _extract_simulation_corner_preference(user_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
    text = str(user_text or "").strip()
    low = text.lower()
    requested_targets = _extract_requested_input_targets(text, state)
    requested_input_ac_rms_v = requested_targets.get("requested_input_ac_rms_v")
    requested_input_dc_bus_v = requested_targets.get("requested_input_dc_bus_v")
    preference = "auto"

    for key, hints in _SIMULATION_CORNER_HINTS.items():
        if any(h in low for h in hints) or any(h in text for h in hints):
            preference = key
            break

    specs = state.get("specifications") or {}
    try:
        vin_min = float(specs.get("input_voltage_min") or 0.0)
        vin_max = float(specs.get("input_voltage_max") or vin_min)
    except Exception:
        vin_min, vin_max = 0.0, 0.0

    if requested_input_dc_bus_v is not None:
        preference = "custom"
    elif requested_input_ac_rms_v is not None:
        if vin_max > vin_min:
            tol = max(3.0, 0.03 * max(vin_min, vin_max, 1.0))
            if abs(requested_input_ac_rms_v - vin_min) <= tol:
                preference = "low_line"
            elif abs(requested_input_ac_rms_v - vin_max) <= tol:
                preference = "high_line"
            else:
                preference = "custom"
        else:
            preference = "custom"

    return {
        "simulation_corner_preference": preference,
        "requested_input_ac_rms_v": requested_input_ac_rms_v,
        "requested_input_dc_bus_v": requested_input_dc_bus_v,
    }



def _looks_like_experiment_rerun_request(user_text: str, state: Dict[str, Any]) -> bool:
    if not _has_prior_design_context(state):
        return False
    text = str(user_text or "")
    low = text.lower()
    has_verb = any(v in low for v in _EXPERIMENT_RERUN_VERBS) or any(v in text for v in _EXPERIMENT_RERUN_VERBS)
    if not has_verb:
        return False
    has_corner_hint = any(
        any(h in low for h in hints) or any(h in text for h in hints)
        for hints in _SIMULATION_CORNER_HINTS.values()
    )
    has_explicit_target = bool(
        re.search(r"(?:at|@|under|with|vin|input|line|bus)\s*(\d+(?:\.\d+)?)\s*v(?:ac|dc)\b", low)
        or re.search(r"\b\d+(?:\.\d+)?\s*vdc\b", low)
    )
    return has_corner_hint or has_explicit_target



def _build_request_profile(
    user_text: str,
    state: Dict[str, Any],
    *,
    active_skill: Optional[str] = None,
    is_chitchat: bool = False,
) -> DesignRequestProfile:
    text = str(user_text or "").strip()
    low = text.lower()
    prior_context = _has_prior_design_context(state)
    follow_up_mode = _looks_like_follow_up_request(text, state)
    experiment_rerun_mode = _looks_like_experiment_rerun_request(text, state)
    report_outputs = _extract_requested_outputs(text)
    workflow_preferences = _extract_workflow_preferences(text)
    corner_request = _extract_simulation_corner_preference(text, state)

    if experiment_rerun_mode and "corner-specific validation" not in workflow_preferences:
        workflow_preferences.append("corner-specific validation")

    if is_chitchat and prior_context and not experiment_rerun_mode:
        conversation_intent = "follow_up_qa"
    elif experiment_rerun_mode and prior_context:
        conversation_intent = "modify_existing"
    elif active_skill == "final_report_writer":
        conversation_intent = "follow_up_report" if prior_context else "skill_request"
    elif active_skill == "design_peer_review":
        conversation_intent = "follow_up_review" if prior_context else "skill_request"
    elif follow_up_mode and prior_context:
        conversation_intent = "modify_existing" if _looks_like_fresh_spec_request(text) else "follow_up_qa"
    elif active_skill:
        conversation_intent = "skill_request"
    else:
        conversation_intent = "new_design"

    return {
        "raw_request": text,
        "conversation_intent": conversation_intent,
        "topology": "Flyback" if ("flyback" in low or "反激" in text or not prior_context) else str(((state.get("request_profile") or {}).get("topology") or "Flyback")),
        "reuse_previous_result": bool(prior_context and (follow_up_mode or experiment_rerun_mode)),
        "preserve_prior_specs": bool(prior_context and (follow_up_mode or experiment_rerun_mode) and not _looks_like_fresh_spec_request(text)),
        "session_context_available": bool(prior_context),
        "requested_outputs": report_outputs,
        "report_requirements": report_outputs,
        "workflow_preferences": workflow_preferences,
        "component_grounding_required": any("digikey" in p.lower() or "grounding" in p.lower() for p in workflow_preferences),
        "magnetics_detail_requested": any("magnet" in p.lower() for p in workflow_preferences) or any("magnetic" in o or "magnet" in o for o in report_outputs),
        "simulation_corner_preference": str(corner_request.get("simulation_corner_preference") or "auto"),
        "requested_input_ac_rms_v": corner_request.get("requested_input_ac_rms_v"),
        "requested_input_dc_bus_v": corner_request.get("requested_input_dc_bus_v"),
        "experiment_intent": "corner_rerun" if experiment_rerun_mode else "none",
    }


def _compact_part_summary(part: Any) -> str:
    if not isinstance(part, dict):
        return str(part or "-")
    for key in ("Part Number", "Mfr Part #", "part_number", "description", "title", "name"):
        if part.get(key):
            return str(part.get(key))
    return "-"


def _follow_up_context_bundle(state: Dict[str, Any]) -> str:
    specs = state.get("specifications") or {}
    design = state.get("theoretical_design") or {}
    sim = state.get("simulation_results") or {}
    verification = state.get("verification") or {}
    bom = state.get("bom") or {}
    magnetic = state.get("magnetic_design") or {}
    request_profile = state.get("request_profile") or {}
    sim_corner = sim.get("simulation_corner") if isinstance(sim.get("simulation_corner"), dict) else {}
    return (
        f"Specs: Vin={specs.get('input_voltage_min')}-{specs.get('input_voltage_max')}Vac, "
        f"Vout={specs.get('output_voltage')}V, Iout={specs.get('output_current')}A, "
        f"eff_target={specs.get('efficiency_target')}, ripple_target={specs.get('max_ripple_voltage')}V\n"
        f"Design: fsw={design.get('switching_frequency')}Hz, Lp={design.get('primary_inductance')}H, "
        f"turns_ratio={design.get('turns_ratio')}, Dmax={design.get('max_duty_cycle')}\n"
        f"Simulation: eff={sim.get('efficiency_measured')}, ripple={sim.get('v_out_ripple_measured')}, "
        f"vds_peak={sim.get('v_ds_spike_max')}, source={sim.get('source')}, scope={sim.get('efficiency_scope')}, "
        f"corner={sim_corner.get('corner_label') or sim_corner.get('requested_corner') or '-'}\n"
        f"Verification: status={verification.get('status')}, strategy={verification.get('correction_strategy')}, "
        f"failed_items={(verification.get('failed_items') or [])[:4]}\n"
        f"BOM: mosfet={_compact_part_summary(bom.get('mosfet'))}, diode={_compact_part_summary(bom.get('diode'))}, "
        f"controller={_compact_part_summary(bom.get('controller'))}, transformer={_compact_part_summary(bom.get('transformer'))}\n"
        f"Magnetics: status={magnetic.get('status')}, engine={magnetic.get('engine')}, api_strategy={magnetic.get('api_strategy')}\n"
        f"Prior request profile: {request_profile}"
    )


def _fallback_follow_up_answer(user_text: str, state: Dict[str, Any]) -> str:
    q = str(user_text or "").lower()
    specs = state.get("specifications") or {}
    design = state.get("theoretical_design") or {}
    sim = state.get("simulation_results") or {}
    verification = state.get("verification") or {}
    magnetic = state.get("magnetic_design") or {}

    if any(k in q for k in ["efficiency", "效率"]):
        return (
            f"Current measured power-stage efficiency is {sim.get('efficiency_measured', '-')}, "
            f"against a target of {specs.get('efficiency_target', '-')}. "
            f"The validator status is {verification.get('status', 'UNKNOWN')}."
        )
    if any(k in q for k in ["ripple", "纹波"]):
        return (
            f"Current output ripple is {sim.get('v_out_ripple_measured', '-')} V, "
            f"with target {specs.get('max_ripple_voltage', '-')} V."
        )
    if any(k in q for k in ["vds", "stress", "应力", "漏极"]):
        return f"The current simulated drain-stress peak is {sim.get('v_ds_spike_max', '-')} V."
    if any(k in q for k in ["magnet", "磁", "transformer", "变压器"]):
        return (
            f"Magnetics advisory status is {magnetic.get('status', '-')}, engine={magnetic.get('engine', '-')}, "
            f"with turns ratio {design.get('turns_ratio', '-')} and primary inductance {design.get('primary_inductance', '-')} H."
        )
    return (
        f"The current session already has a flyback design snapshot: "
        f"{specs.get('output_voltage', '-')} V / {specs.get('output_current', '-')} A, "
        f"fsw={design.get('switching_frequency', '-')}, efficiency={sim.get('efficiency_measured', '-')}, "
        f"ripple={sim.get('v_out_ripple_measured', '-')}, validation={verification.get('status', 'UNKNOWN')}."
    )


def _answer_follow_up_from_state(user_text: str, state: Dict[str, Any], retrieved_context: str = "") -> str:
    if not _has_prior_design_context(state):
        return _fallback_follow_up_answer(user_text, state)

    try:
        llm = get_llm()
        context_blob = _follow_up_context_bundle(state)
        system_prompt = (
            "You are a senior power electronics reviewer answering a follow-up about the CURRENT session result. "
            "Use exact values from the provided session snapshot. Do not invent missing data. "
            "If a requested datum is unavailable, say so explicitly. Keep the answer concise and engineering-focused."
        )
        human_prompt = (
            f"User follow-up: {user_text}\n\n"
            f"Current session snapshot:\n{context_blob}\n\n"
            f"Retrieved knowledge context:\n{retrieved_context[:4000]}\n\n"
            "Answer the follow-up using the current result, not a fresh design."
        )
        answer = (ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ]) | llm).invoke({"input": human_prompt})
        content = getattr(answer, "content", "")
        if str(content).strip():
            return str(content).strip()
    except Exception:
        pass
    return _fallback_follow_up_answer(user_text, state)


def _heuristic_extract_specs(user_text: str) -> Optional[Dict[str, Any]]:
    text = str(user_text or "").strip()
    if not text:
        return None
    low = text.lower()
    if _is_qa_prompt(text):
        return None

    specs: Dict[str, Any] = {
        "is_chitchat": False,
        "response_text": None,
        "input_voltage_min": 85.0,
        "input_voltage_max": 265.0,
        "output_voltage": None,
        "output_current": None,
        "efficiency_target": 0.85,
        "max_ripple_voltage": 0.2,
        "isolation": True,
        "application_type": "Adapter",
    }

    ac_range = re.search(r"(\d+(?:\.\d+)?)\s*[-~to]{1,3}\s*(\d+(?:\.\d+)?)\s*v(?:ac)?(?:\s*input)?", low)
    if ac_range:
        specs["input_voltage_min"] = float(ac_range.group(1))
        specs["input_voltage_max"] = float(ac_range.group(2))
    else:
        single_input = _extract_first_number(low, r"\b(\d+(?:\.\d+)?)\s*v(?:ac)?\s*input\b")
        if single_input is None:
            single_input = _extract_first_number(low, r"input[^.\n]*?(\d+(?:\.\d+)?)\s*v(?:ac)?")
        if single_input is not None:
            specs["input_voltage_min"] = single_input
            specs["input_voltage_max"] = single_input

    out_pair = re.search(
        r"(\d+(?:\.\d+)?)\s*v(?:dc)?\s*[/x* ]\s*(\d+(?:\.\d+)?)\s*a\b|(\d+(?:\.\d+)?)\s*v(?:dc)?\s*(\d+(?:\.\d+)?)\s*a\b",
        low,
        flags=re.I,
    )
    if out_pair:
        vout = out_pair.group(1) or out_pair.group(3)
        iout = out_pair.group(2) or out_pair.group(4)
        specs["output_voltage"] = float(vout)
        specs["output_current"] = float(iout)
    else:
        specs["output_voltage"] = _extract_first_number(low, r"output[^.\n]*?(\d+(?:\.\d+)?)\s*v(?:dc)?")
        specs["output_current"] = _extract_first_number(low, r"output[^.\n]*?(\d+(?:\.\d+)?)\s*a\b")

    if specs["output_voltage"] is None:
        specs["output_voltage"] = _extract_first_number(low, r"\b(\d+(?:\.\d+)?)\s*v(?:dc)?\b")
    if specs["output_current"] is None:
        specs["output_current"] = _extract_first_number(low, r"\b(\d+(?:\.\d+)?)\s*a\b")

    power_w = _extract_first_number(low, r"(\d+(?:\.\d+)?)\s*w\b")
    if power_w is not None:
        if specs["output_voltage"] and not specs["output_current"] and specs["output_voltage"] > 0:
            specs["output_current"] = power_w / float(specs["output_voltage"])
        elif specs["output_current"] and not specs["output_voltage"] and specs["output_current"] > 0:
            specs["output_voltage"] = power_w / float(specs["output_current"])

    eff_pct = _extract_first_number(low, r"efficiency[^.\n<>:=]*[>:=< ]\s*(\d+(?:\.\d+)?)\s*%")
    if eff_pct is None:
        eff_pct = _extract_first_number(low, r"\beta[^.\n<>:=]*[>:=< ]\s*(\d+(?:\.\d+)?)\s*%")
    if eff_pct is not None:
        specs["efficiency_target"] = eff_pct / 100.0 if eff_pct > 1.0 else eff_pct

    ripple_mv = _extract_first_number(low, r"ripple[^.\n<>:=]*[<:= >]\s*(\d+(?:\.\d+)?)\s*mv")
    ripple_v = _extract_first_number(low, r"ripple[^.\n<>:=]*[<:= >]\s*(\d+(?:\.\d+)?)\s*v\b")
    if ripple_mv is not None:
        specs["max_ripple_voltage"] = ripple_mv / 1000.0
    elif ripple_v is not None:
        specs["max_ripple_voltage"] = ripple_v

    if any(k in low for k in ["medical", "iec 60601", "mopp"]):
        specs["application_type"] = "Medical"
    elif any(k in low for k in ["industrial", "factory", "din rail"]):
        specs["application_type"] = "Industrial"
    elif any(k in low for k in ["automotive", "car", "vehicle", "iso 7637"]):
        specs["application_type"] = "Automotive"

    if any(k in low for k in ["non-isolated", "non isolated", "不隔离"]):
        specs["isolation"] = False

    if specs["output_voltage"] is None or specs["output_current"] is None:
        return None
    return specs


def _infer_active_skill(user_text: str) -> Optional[str]:
    text = str(user_text or "").lower()
    # If this looks like a full converter design request, do not short-circuit into
    # a single skill (for example "final report" as a deliverable section).
    design_terms = [
        "design", "redesign", "optimize", "flyback", "topology", "workflow",
        "input", "output", "vin", "vout", "iout", "efficiency", "ripple",
        "设计", "重设计", "优化", "拓扑", "流程", "输入", "输出", "效率", "纹波", "仿真",
    ]
    has_design_intent = any(term in text for term in design_terms)

    datasheet_terms = ["datasheet", "data sheet", "pdf", "手册", "规格书"]
    report_terms = [
        "final report", "generate report", "write report", "engineering report", "updated report", "updated final report",
        "summary report", "strict report", "总结报告", "最终报告", "更新报告", "报告",
    ]
    explicit_report_request_terms = [
        "generate report", "write report", "report only", "final report only", "updated final report", "generate the updated final report",
        "生成报告", "写报告", "仅生成报告", "只生成报告", "只要报告", "更新最终报告",
    ]
    peer_review_terms = [
        "peer review", "design review", "engineering review", "validation-oriented", "strict validation",
        "verify current design", "review the current design", "同行评审", "评审", "复核", "验证当前设计",
    ]
    evidence_terms = ["evidence grade", "source quality", "证据打分", "证据质量"]
    citation_terms = ["citation", "reference audit", "引用", "参考文献"]
    consistency_terms = [
        "consistency checker", "consistency report", "cross-corner consistency", "corner consistency",
        "一致性检查", "一致性分析", "角点一致性",
    ]
    sensitivity_terms = ["sensitivity", "what-if", "参数敏感性", "灵敏度"]
    web_terms = [
        "search", "find", "paper", "arxiv", "blog", "forum", "reddit", "post",
        "元件", "器件", "文献", "论文", "帖子", "博客", "搜索",
    ]
    if any(term in text for term in datasheet_terms) and not has_design_intent:
        return "datasheet_analysis"
    if any(term in text for term in report_terms):
        if has_design_intent and not any(term in text for term in explicit_report_request_terms):
            return None
        return "final_report_writer"
    if any(term in text for term in peer_review_terms):
        return "design_peer_review"
    if any(term in text for term in evidence_terms):
        return "evidence_grader"
    if any(term in text for term in citation_terms):
        return "engineering_citation_manager"
    if any(term in text for term in consistency_terms):
        return "simulation_consistency_checker"
    if any(term in text for term in sensitivity_terms):
        return "param_sensitivity_planner"
    if any(term in text for term in web_terms):
        return "web_research"
    return None


def _is_qa_prompt(user_text: str) -> bool:
    text = str(user_text or "").strip()
    low = text.lower()
    qa_markers = [
        "what is", "who are you", "what can you do", "how does", "explain",
    ]
    qa_markers_cn = [
        "什么是", "是什么", "是啥", "什么事", "你是谁", "你能做什么", "怎么", "如何", "解释", "介绍", "为啥", "为何", "为什么",
    ]
    design_markers = [
        "design", "redesign", "optimize", "spec", "vin", "vout", "iout", "efficiency", "ripple",
        "设计", "重设计", "优化", "输入", "输出", "效率", "纹波",
    ]

    has_qa = any(k in low for k in qa_markers) or any(k in text for k in qa_markers_cn) or ("?" in text) or ("？" in text)
    has_design = any(k in low for k in design_markers) or any(ch.isdigit() for ch in text)
    return bool(has_qa and not has_design)


def _run_evidence_grader(refs_local: list, refs_lit: list) -> Dict[str, Any]:
    skills_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "skills")
    skill = SkillManager(skills_dir).get_skill("evidence_grader")
    if not skill or not skill.tools_module or not hasattr(skill.tools_module, "grade_evidence"):
        return {
            "aggregate_confidence": 0.3,
            "evidence_grade": "LOW",
            "graded_sources": [],
            "bias_flags": ["evidence_grader skill unavailable"],
        }
    try:
        return skill.tools_module.grade_evidence({
            "retrieved_knowledge_references": refs_local or [],
            "literature_references": refs_lit or [],
        })
    except Exception as err:
        return {
            "aggregate_confidence": 0.3,
            "evidence_grade": "LOW",
            "graded_sources": [],
            "bias_flags": [f"evidence_grader failed: {err}"],
        }


def _plan_payload(state: Dict[str, Any], specs: Dict[str, Any]) -> Dict[str, Any]:
    plan = build_execution_plan({**state, "specifications": specs})
    return {
        "execution_plan": plan,
        "planning_summary": summarize_execution_plan(plan),
    }

def requirements_agent_node(state: PowerSupplyState) -> Dict[str, Any]:
    """
    Node 1: Requirements Analysis
    Extracts structured DesignSpecs from natural language messages.
    Initializes Traceable Reasoning (DeepRare).
    """
    print("\n" + "="*50)
    print("[Requirements Agent] START")
    print("="*50)
    
    # Initialize Global Reasoning Trace
    reasoning_trace = [{
        "step": "System Initialization",
        "agent": "Requirements",
        "action": "Analyze User Intent",
        "evidence": "User Input Messages",
        "confidence": 1.0,
        "citation": "Conversation History"
    }]
    
    llm = get_llm()
    messages = state.get('messages', [])
    
    print(f"Input Messages ({len(messages)}):")
    for msg in messages:
        if isinstance(msg, str):
            print(f"  - Message: {msg}")
        else:
            # Fallback for objects that might not have type/content attributes
            m_type = getattr(msg, 'type', 'ActiveObject')
            m_content = getattr(msg, 'content', str(msg))
            print(f"  - {m_type}: {m_content}")

    # Ensure messages exist
    if not messages:
        print("Error: No messages found.")
        print("="*50 + "\n")
        return {"error_log": ["No messages found to analyze."], "reasoning_trace": reasoning_trace}

    print("\nProcessing with LLM to extract specifications...")
    
    # --- SYSTEM 2 REASONING START ---
    print("[System 2] Requirements Analysis...")
    reasoning_logs = [
        "[PLAN] Parse user intent and classify mode (design task vs Q&A).",
        "[EXECUTION] Normalizing conversation messages into plain requirement text.",
        "[CHECK] Applying hard guardrails before parameter extraction.",
    ]
    try:
        from core.llm.llm import get_agent_reasoning
        reasoning_logs.extend(get_agent_reasoning(
            agent_role="Technical Project Manager",
            task="Analyze user intent and formulate technical assumptions.",
            context_data={"messages": [str(m) for m in messages[-3:]]}
        ))
    except Exception as e:
        print(f"Warning: reasoning engine unavailable: {e}") 
        reasoning_logs.append("[THOUGHT] Initializing standard analysis protocols (Fallback mode).")
    # --- SYSTEM 2 REASONING END ---

    last_msg = messages[-1] if messages else ""
    print(f"\nAnalyzing Last Message: {getattr(last_msg, 'content', last_msg)}")

    last_msg_text = _normalize_user_message(last_msg)
    reasoning_logs.append(f"[INPUT] Last user request normalized: {last_msg_text[:180]}")
    active_skill = _infer_active_skill(last_msg_text)
    prior_context_available = _has_prior_design_context(state)
    follow_up_mode = _looks_like_follow_up_request(last_msg_text, state)
    fresh_spec_request = _looks_like_fresh_spec_request(last_msg_text)
    experiment_rerun_mode = _looks_like_experiment_rerun_request(last_msg_text, state)
    if experiment_rerun_mode:
        active_skill = None
        reasoning_logs.append("[DECISION] Corner-rerun request detected; bypassing report-only skill routing and returning to the design/simulation path.")
    request_profile = _build_request_profile(
        last_msg_text,
        state,
        active_skill=active_skill,
        is_chitchat=False,
    )

    # Fast path for post-validation retry: keep prior constraints and skip brittle re-extraction.
    if str(last_msg_text).strip().upper() == "RETRY_REQUESTED":
        prev_specs = state.get("specifications") or {}
        if isinstance(prev_specs, dict) and prev_specs:
            reasoning_logs.append("[DECISION] Retry requested; reusing previously validated requirement snapshot.")
            reasoning_logs.append(
                f"[RESULT] Reused specs: Vin={prev_specs.get('input_voltage_min')}-{prev_specs.get('input_voltage_max')}V, "
                f"Vout={prev_specs.get('output_voltage')}V, Iout={prev_specs.get('output_current')}A"
            )
            return {
                "specifications": prev_specs,
                "request_profile": request_profile,
                "active_skill": None,
                "messages": ["Retry requested. Reusing previous specifications for next iteration."],
                "formula_checks": state.get("formula_checks", {}) | {"requirements": {"checks": [], "warnings": [], "fatal": []}},
                "reasoning_logs": state.get("reasoning_logs", {}) | {"requirements": reasoning_logs},
                "reasoning_trace": reasoning_trace,
                **_plan_payload(state, prev_specs),
            }
        reasoning_logs.append("[WARNING] Retry requested but no prior specifications found; falling back to extraction.")

    if str(last_msg_text).strip().upper() == "BOM_RESELECT_REQUESTED":
        prev_specs = dict(state.get("specifications") or {})
        if isinstance(prev_specs, dict) and prev_specs:
            prev_specs["is_chitchat"] = False
            prev_specs["response_text"] = None
            prev_profile = dict(state.get("request_profile") or {})
            prev_profile["conversation_intent"] = "new_design"
            prev_profile["component_grounding_required"] = True
            prev_profile["workflow_preferences"] = list(dict.fromkeys(
                (prev_profile.get("workflow_preferences") or [])
                + [
                    "strict BOM reselection",
                    "real orderable MPNs",
                    "source evidence required",
                    "avoid generic fallback",
                ]
            ))
            reasoning_logs.append("[DECISION] BOM reselection requested; reusing specs and forcing design path, not skill execution.")
            reasoning_logs.append(
                f"[RESULT] Reselect specs: Vin={prev_specs.get('input_voltage_min')}-{prev_specs.get('input_voltage_max')}Vac, "
                f"Vout={prev_specs.get('output_voltage')}V, Iout={prev_specs.get('output_current')}A"
            )
            return {
                "specifications": prev_specs,
                "request_profile": prev_profile,
                "active_skill": None,
                "messages": ["BOM reselection requested. Reusing previous specifications with strict component evidence requirements."],
                "hard_guardrails_prompt": state.get("hard_guardrails_prompt") or get_hard_guardrails_prompt(),
                "retrieved_knowledge_context": state.get("retrieved_knowledge_context") or "",
                "retrieved_knowledge_references": state.get("retrieved_knowledge_references") or [],
                "literature_references": state.get("literature_references") or [],
                "evidence_grade": state.get("evidence_grade") or {},
                "formula_checks": state.get("formula_checks", {}) | {"requirements": {"checks": [], "warnings": [], "fatal": []}},
                "reasoning_logs": state.get("reasoning_logs", {}) | {"requirements": reasoning_logs},
                "reasoning_trace": reasoning_trace,
                "design_overrides": state.get("design_overrides") or {},
                "iteration_learning": state.get("iteration_learning") or {},
                "learning_context": state.get("learning_context") or {},
                "curriculum_context": state.get("curriculum_context") or {},
                **_plan_payload(state, prev_specs),
            }
        reasoning_logs.append("[WARNING] BOM reselection requested but no prior specifications found; falling back to extraction.")

    locked_override = state.get("design_overrides") if isinstance(state.get("design_overrides"), dict) else {}
    locked_specs = dict(locked_override.get("locked_specs") or state.get("specifications") or {})
    if (
        locked_specs
        and str(locked_override.get("requirements_gate") or "") == "framework_scaffold_locked"
        and "CONTINUE FROM GATED ENGINEERING SCAFFOLD" in last_msg_text
    ):
        locked_specs["is_chitchat"] = False
        locked_specs["response_text"] = None
        spec_check = normalize_and_validate_specs(locked_specs)
        specs = spec_check["normalized"]
        request_profile = {
            **request_profile,
            "conversation_intent": "new_design",
            "preserve_prior_specs": True,
            "reuse_previous_result": False,
            "workflow_preferences": list(dict.fromkeys(
                (request_profile.get("workflow_preferences") or [])
                + ["gated engineering workflow", "preserve locked specs", "release evidence required"]
            )),
        }
        reasoning_logs.append("[DECISION] Framework scaffold continuation detected; preserving locked specs instead of re-parsing defaults.")
        reasoning_logs.append(
            f"[RESULT] Locked specs: Vin={specs.get('input_voltage_min')}-{specs.get('input_voltage_max')}Vac, "
            f"Vout={specs.get('output_voltage')}V, Iout={specs.get('output_current')}A, Pout={specs.get('output_power')}W"
        )
        return {
            "specifications": specs,
            "request_profile": request_profile,
            "active_skill": None,
            "messages": ["Framework scaffold continuation. Reusing locked specifications for real agent/tool workflow."],
            "hard_guardrails_prompt": state.get("hard_guardrails_prompt") or get_hard_guardrails_prompt(),
            "retrieved_knowledge_context": state.get("retrieved_knowledge_context") or "",
            "retrieved_knowledge_references": state.get("retrieved_knowledge_references") or [],
            "literature_references": state.get("literature_references") or [],
            "evidence_grade": state.get("evidence_grade") or {},
            "formula_checks": state.get("formula_checks", {}) | {"requirements": spec_check},
            "node_verification": state.get("node_verification", {}) | {"requirements": {"status": "PASS" if not spec_check.get("fatal") else "FAIL", "fatal": spec_check.get("fatal", []), "warnings": spec_check.get("warnings", []), "locked_spec_reused": True}},
            "reasoning_logs": state.get("reasoning_logs", {}) | {"requirements": reasoning_logs},
            "reasoning_trace": reasoning_trace,
            **_plan_payload(state, specs),
        }

    if experiment_rerun_mode and prior_context_available and not fresh_spec_request:
        reused_specs = dict(state.get("specifications") or {})
        if reused_specs:
            reused_specs["is_chitchat"] = False
            reused_specs["response_text"] = None
        reasoning_logs.append(
            "[DECISION] Follow-up experiment rerun detected; reusing current specification snapshot and applying requested corner override."
        )
        return {
            "specifications": reused_specs,
            "request_profile": request_profile,
            "active_skill": None,
            "messages": ["Experiment rerun detected. Reusing current session design with a new simulation corner request."],
            "hard_guardrails_prompt": state.get("hard_guardrails_prompt") or get_hard_guardrails_prompt(),
            "retrieved_knowledge_context": state.get("retrieved_knowledge_context") or "",
            "retrieved_knowledge_references": state.get("retrieved_knowledge_references") or [],
            "literature_references": state.get("literature_references") or [],
            "evidence_grade": state.get("evidence_grade") or {},
            "formula_checks": state.get("formula_checks", {}) | {"requirements": {"checks": [], "warnings": [], "fatal": []}},
            "reasoning_logs": state.get("reasoning_logs", {}) | {"requirements": reasoning_logs},
            "reasoning_trace": reasoning_trace,
            "design_overrides": state.get("design_overrides") or {},
            **_plan_payload(state, reused_specs),
        }

    # Follow-up skill/report/review requests should reuse the current session snapshot
    # instead of defaulting to a fresh 12V/2A spec parse.
    if active_skill and prior_context_available and follow_up_mode and not fresh_spec_request and not experiment_rerun_mode:
        reused_specs = dict(state.get("specifications") or {})
        if reused_specs:
            reused_specs["is_chitchat"] = False
            reused_specs["response_text"] = None
        reasoning_logs.append(
            f"[DECISION] Follow-up skill request detected ({active_skill}); reusing prior specification snapshot."
        )
        return {
            "specifications": reused_specs,
            "request_profile": request_profile,
            "active_skill": active_skill,
            "messages": [f"Follow-up request detected. Reusing current session result for {active_skill}."],
            "hard_guardrails_prompt": state.get("hard_guardrails_prompt") or get_hard_guardrails_prompt(),
            "retrieved_knowledge_context": state.get("retrieved_knowledge_context") or "",
            "retrieved_knowledge_references": state.get("retrieved_knowledge_references") or [],
            "literature_references": state.get("literature_references") or [],
            "evidence_grade": state.get("evidence_grade") or {},
            "formula_checks": state.get("formula_checks", {}) | {"requirements": {"checks": [], "warnings": [], "fatal": []}},
            "reasoning_logs": state.get("reasoning_logs", {}) | {"requirements": reasoning_logs},
            "reasoning_trace": reasoning_trace,
            **_plan_payload(state, reused_specs),
        }

    if prior_context_available and follow_up_mode and not fresh_spec_request and not active_skill and not experiment_rerun_mode:
        qa_text = _answer_follow_up_from_state(
            last_msg_text,
            state,
            state.get("retrieved_knowledge_context") or "",
        )
        reused_specs = dict(state.get("specifications") or {})
        specs = {
            **reused_specs,
            "is_chitchat": True,
            "response_text": qa_text,
        }
        request_profile = _build_request_profile(
            last_msg_text,
            state,
            active_skill=None,
            is_chitchat=True,
        )
        reasoning_logs.append("[DECISION] General follow-up request detected; answering from current session snapshot.")
        return {
            "specifications": specs,
            "request_profile": request_profile,
            "messages": [qa_text],
            "active_skill": None,
            "formula_checks": state.get("formula_checks", {}) | {"requirements": {"checks": [], "warnings": [], "fatal": []}},
            "reasoning_logs": state.get("reasoning_logs", {}) | {"requirements": reasoning_logs},
            "reasoning_trace": reasoning_trace,
            **_plan_payload(state, specs),
        }

    # Deterministic QA short-circuit to avoid misclassification by LLM
    if _is_qa_prompt(last_msg_text) and not experiment_rerun_mode:
        reasoning_logs.append("[DECISION] Classified as Q&A/chitchat. Skip design pipeline and return direct answer.")
        if prior_context_available:
            qa_text = _answer_follow_up_from_state(
                last_msg_text,
                state,
                state.get("retrieved_knowledge_context") or "",
            )
            reused_specs = dict(state.get("specifications") or {})
            specs = {
                **reused_specs,
                "is_chitchat": True,
                "response_text": qa_text,
            }
            request_profile = _build_request_profile(
                last_msg_text,
                state,
                active_skill=active_skill,
                is_chitchat=True,
            )
            reasoning_logs.append("[DECISION] Contextual follow-up QA selected using current session snapshot.")
        elif (
            any(term in last_msg_text.lower() for term in ["flyback", "flybakc", "flayback", "flaybakc"])
            or "反激" in last_msg_text
        ):
            qa_text = (
                "Flyback（反激变换器）是一种隔离型开关电源拓扑：\n"
                "1) 开关导通时，能量先储存在变压器磁化电感；\n"
                "2) 开关关断时，能量转移到次级给负载与输出电容。\n\n"
                "它常用于中小功率适配器，因为结构简单、易隔离、成本低。"
            )
            specs = {
                "is_chitchat": True,
                "response_text": qa_text,
                "input_voltage_min": None,
                "input_voltage_max": None,
                "output_voltage": None,
                "output_current": None,
                "efficiency_target": 0.85,
                "max_ripple_voltage": 0.2,
                "isolation": True,
                "application_type": "Adapter",
            }
        else:
            qa_text = "这是问答模式。我可以解释原理、对比拓扑、分析你当前设计结果，或再进入完整设计流程。"
            specs = {
                "is_chitchat": True,
                "response_text": qa_text,
                "input_voltage_min": None,
                "input_voltage_max": None,
                "output_voltage": None,
                "output_current": None,
                "efficiency_target": 0.85,
                "max_ripple_voltage": 0.2,
                "isolation": True,
                "application_type": "Adapter",
            }

        if not prior_context_available:
            request_profile = _build_request_profile(
                last_msg_text,
                state,
                active_skill=active_skill,
                is_chitchat=True,
            )

        reasoning_logs.append("[DECISION] Deterministic QA mode selected (non-design prompt).")
        return {
            "specifications": specs,
            "request_profile": request_profile,
            "messages": [qa_text],
            "active_skill": None,
            "formula_checks": state.get("formula_checks", {}) | {"requirements": {"checks": [], "warnings": [], "fatal": []}},
            "reasoning_logs": state.get("reasoning_logs", {}) | {"requirements": reasoning_logs},
            "reasoning_trace": reasoning_trace,
            **_plan_payload(state, specs),
        }

    hard_guardrails_prompt = get_hard_guardrails_prompt()
    retrieved_bundle = retrieve_flyback_context(last_msg_text, top_k=6)
    retrieved_context = retrieved_bundle.get("context_text", "")
    retrieved_refs = retrieved_bundle.get("references", [])
    component_bundle = {"context_text": "", "references": []}
    if retrieve_component_rag_context:
        try:
            component_bundle = retrieve_component_rag_context(last_msg_text, top_k=8)
            c_refs = component_bundle.get("references") or []
            c_types = component_bundle.get("inferred_categories") or []
            if c_refs:
                reasoning_logs.append(f"[SEARCH] DigiKey local component RAG references: {len(c_refs)}")
            if c_types:
                reasoning_logs.append("[SEARCH] DigiKey inferred categories: " + ", ".join(c_types))
        except Exception as component_err:
            reasoning_logs.append(f"[WARNING] DigiKey local component RAG failed: {component_err}")

    component_context = component_bundle.get("context_text", "")
    if component_context:
        retrieved_context = (retrieved_context + "\n\n" + component_context).strip()

    component_refs = component_bundle.get("references", [])
    if component_refs:
        retrieved_refs = list(retrieved_refs) + list(component_refs)

    web_research = collect_node_research(
        "requirements",
        f"flyback requirements constraints standards papers blogs forums: {last_msg_text}",
        max_results=6,
    )
    reasoning_logs.extend(web_research.get("logs", []))
    reasoning_logs.append("[SEARCH] Merging local RAG references with web references for requirement confidence grading.")
    merged_refs = list(retrieved_refs)
    for ref in web_research.get("references", []):
        if isinstance(ref, dict):
            merged_refs.append(ref)
    evidence_pack = _run_evidence_grader(merged_refs, web_research.get("references", []))
    reasoning_logs.append(
        f"[EVIDENCE] grade={evidence_pack.get('evidence_grade')} confidence={evidence_pack.get('aggregate_confidence')}"
    )
    for flag in evidence_pack.get("bias_flags", [])[:4]:
        reasoning_logs.append(f"[EVIDENCE-WARN] {flag}")
    
    system_prompt = f"""{hard_guardrails_prompt}

    You are a highly intelligent Power Electronics AI Agent. 
    Your job is to classify the USER'S INTENT based on the conversation history.

    ### CLASSIFICATION RULES:

    **TYPE A: CHITCHAT / Q&A (Set `is_chitchat = True`)**
    - User asks a question about the PREVIOUS design result (e.g., "Is this efficiency reasonable?", "Why is ripple so high?").
    - User asks general knowledge questions (e.g., "What is a Flyback?", "Who are you?").
    - User says greetings ("Hi", "Thanks").
    - **CRITICAL**: If the user is doubting, questioning, or discussing the CURRENT results, it is Q&A. Do NOT start a new design.

    **TYPE B: NEW DESIGN / MODIFICATION (Set `is_chitchat = False`)**
    - User explicitly asks to START a design (e.g., "Design a 12V 5V converter").
    - User explicitly CHANGES a parameter to RESTART (e.g., "Change input to 24V", "Redesign with 100kHz", "Make Iout 3A").
    - User asks for a stricter review/report based on the current session result; in that case set `reuse_previous_result = True` and preserve requested outputs/preferences.
    - User asks to rerun or verify the CURRENT design at low-line, high-line, nominal, or a specified input corner; that is NOT chitchat.
    
    ### RESPONSE INSTRUCTIONS:
    - If Type A: Write a helpful, professional answer in `response_text`.
    - If Type B: Extract parameters significantly. Leave `response_text` empty.
    - Also capture `requested_outputs` and `workflow_preferences` when the user explicitly asks for specific report fields, realistic component grounding, or magnetics recommendations.

    ### DEFAULTS (Only for Type B):
    - Input: Universal (85-265V AC)
    - Efficiency: 0.85

    RETRIEVED DOMAIN CONTEXT (Use when extracting engineering constraints):
    {retrieved_context[:5000]}
    
    IMPORTANT: Return strict JSON matching the schema."""
    
    if str(os.getenv("PE_MAS_DEBUG_PROMPTS") or "").strip().lower() in {"1", "true", "yes", "on"}:
        print("\nSystem Prompt:\n", system_prompt)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Structured output abstraction
    # Fallback to standard model if structured fails? No, try structured first.
    try:
        print("\nInvoking LLM with Structured Output (DesignSpecsModel)...")
        reasoning_logs.append("[EXECUTION] Invoking structured LLM extraction for normalized specification fields.")
        # Use full conversation context
        input_text = "\n".join([_normalize_user_message(m) for m in messages])
        structured_llm = llm.with_structured_output(DesignSpecsModel)
        response_model = structured_llm.invoke(
            f"Current Request: {last_msg}\n"
            f"History: {input_text}\n"
            f"Hard Guardrails:\n{hard_guardrails_prompt}\n"
            f"Retrieved Knowledge:\n{retrieved_context[:8000]}"
        )
        
        # Convert Pydantic to Dict
        specs = response_model.dict()
        merged_requested_outputs = list(dict.fromkeys((request_profile.get("requested_outputs") or []) + list(specs.get("requested_outputs") or [])))
        merged_workflow_preferences = list(dict.fromkeys((request_profile.get("workflow_preferences") or []) + list(specs.get("workflow_preferences") or [])))
        request_profile = {
            **request_profile,
            "reuse_previous_result": bool(request_profile.get("reuse_previous_result") or specs.get("reuse_previous_result")),
            "requested_outputs": merged_requested_outputs,
            "report_requirements": list(dict.fromkeys((request_profile.get("report_requirements") or []) + merged_requested_outputs)),
            "workflow_preferences": merged_workflow_preferences,
            "topology": str(specs.get("topology") or request_profile.get("topology") or "Flyback"),
            "component_grounding_required": any("digikey" in p.lower() or "grounding" in p.lower() for p in merged_workflow_preferences),
            "magnetics_detail_requested": any("magnet" in p.lower() for p in merged_workflow_preferences) or any("magnetic" in o or "magnet" in o for o in merged_requested_outputs),
            "simulation_corner_preference": request_profile.get("simulation_corner_preference") or "auto",
            "requested_input_ac_rms_v": request_profile.get("requested_input_ac_rms_v"),
            "requested_input_dc_bus_v": request_profile.get("requested_input_dc_bus_v"),
            "experiment_intent": request_profile.get("experiment_intent") or "none",
        }
        if prior_context_available and request_profile.get("preserve_prior_specs") and not fresh_spec_request:
            prev_specs = dict(state.get("specifications") or {})
            if prev_specs:
                prev_specs["is_chitchat"] = False
                prev_specs["response_text"] = None
                specs = prev_specs
                reasoning_logs.append("[DECISION] Preserved prior specification snapshot for follow-up request without new numeric constraints.")
        if experiment_rerun_mode:
            specs["is_chitchat"] = False
            specs["response_text"] = None
            reasoning_logs.append("[DECISION] Forcing design-mode extraction because the user requested a corner-specific experiment rerun.")
        reasoning_logs.append(
            f"[RESULT] Parsed specs candidate: Vin={specs.get('input_voltage_min')}-{specs.get('input_voltage_max')}V, "
            f"Vout={specs.get('output_voltage')}V, Iout={specs.get('output_current')}A"
        )
        import json
        
        # [NEW] Chitchat / Q&A Handler
        if specs.get('is_chitchat') and not experiment_rerun_mode:
            print("Mode: Chitchat/QA Detected.")
            
            # Use the response_text if the model generated one (Preferred)
            final_response = specs.get('response_text')
            
            # Fallback if the structured model output did not include chat text.
            if not final_response:
                 print("No response_text in structured output, invoking chat fallback...")
                 chat_chain = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful Power Electronics AI Assistant. Respond warmly to the user's casual message."),
                    ("human", "{input}")
                 ]) | llm
                 response = chat_chain.invoke({"input": last_msg})
                 final_response = response.content
            
            request_profile = _build_request_profile(
                last_msg_text,
                state,
                active_skill=active_skill,
                is_chitchat=True,
            )
            return {
                "specifications": specs,
                "request_profile": request_profile,
                "messages": [final_response], 
                "active_skill": active_skill,
                "hard_guardrails_prompt": hard_guardrails_prompt,
                "retrieved_knowledge_context": retrieved_context,
                "retrieved_knowledge_references": merged_refs,
                "literature_references": web_research.get("references", []),
                "evidence_grade": evidence_pack,
                "formula_checks": state.get("formula_checks", {}) | {"requirements": {"checks": [], "warnings": [], "fatal": []}},
                "reasoning_logs": state.get("reasoning_logs", {}) | {"requirements": reasoning_logs},
                "reasoning_trace": reasoning_trace,
                **_plan_payload(state, specs),
            }

        spec_check = normalize_and_validate_specs(specs)
        specs = spec_check["normalized"]
        plan_payload = _plan_payload(state, specs)
        if plan_payload.get("planning_summary"):
            reasoning_logs.append(f"[PLAN] {plan_payload.get('planning_summary')}")
        for w in spec_check.get("warnings", []):
            reasoning_logs.append(f"[FORMULA] {w}")
        for f in spec_check.get("fatal", []):
            reasoning_logs.append(f"[FORMULA-FAIL] {f}")
        derived = spec_check.get("derived", {})
        if derived:
            reasoning_logs.append(
                f"[FORMULA] Pout=Vout*Iout={derived.get('pout_w', 0.0):.2f}W, Pin_est=Pout/eta={derived.get('pin_est_w', 0.0):.2f}W"
            )

        print("\nExtracted Specifications:")
        print(json.dumps(specs, indent=2))
        
        print("\n" + "="*50)
        print("[Requirements Agent] END")
        print("="*50 + "\n")
        
        reasoning_trace.append({
            "step": "Requirements Extraction",
            "agent": "Requirements",
            "action": f"Extracted Specs: {specs.get('output_voltage')}V/{specs.get('output_current')}A",
            "evidence": "LLM Analysis",
            "confidence": 0.95
        })

        return {
            "specifications": specs,
            "request_profile": request_profile,
            "active_skill": active_skill,
            "hard_guardrails_prompt": hard_guardrails_prompt,
            "retrieved_knowledge_context": retrieved_context,
            "retrieved_knowledge_references": merged_refs,
            "literature_references": web_research.get("references", []),
            "evidence_grade": evidence_pack,
            "formula_checks": state.get("formula_checks", {}) | {"requirements": spec_check},
            "node_verification": state.get("node_verification", {}) | {"requirements": {"status": "PASS" if not spec_check.get("fatal") else "FAIL", "fatal": spec_check.get("fatal", []), "warnings": spec_check.get("warnings", []), "evidence_grade": evidence_pack.get("evidence_grade"), "evidence_confidence": evidence_pack.get("aggregate_confidence")}},
            "reasoning_logs": state.get("reasoning_logs", {}) | {"requirements": reasoning_logs},
            "reasoning_trace": reasoning_trace,
            **plan_payload,
        }

    except Exception as e:

        print(f"ERROR: Requirements LLM Call Failed: {e}")
        error_msg = str(e)

        # Reliability fallback: keep workflow alive by reusing known-good specs.
        prev_specs = state.get("specifications") or {}
        if isinstance(prev_specs, dict) and prev_specs and not (
            "401" in error_msg or "Incorrect API key" in error_msg
        ):
            reasoning_logs.append("[FALLBACK] LLM extraction failed; reusing previous specifications from checkpoint.")
            return {
                "specifications": prev_specs,
                "request_profile": request_profile,
                "active_skill": active_skill,
                "messages": ["Requirements extraction temporarily failed. Reused previous specifications."],
                "formula_checks": state.get("formula_checks", {}) | {"requirements": {"checks": [], "warnings": ["LLM extraction failed; reused previous specifications"], "fatal": []}},
                "reasoning_logs": state.get("reasoning_logs", {}) | {"requirements": reasoning_logs},
                "reasoning_trace": reasoning_trace,
                **_plan_payload(state, prev_specs),
            }
        
        heuristic_specs = _heuristic_extract_specs(last_msg_text)
        if heuristic_specs:
            quota_related = "quota" in error_msg.lower() or "rate limit" in error_msg.lower() or "insufficient_quota" in error_msg.lower()
            auth_related = "401" in error_msg or "Incorrect API key" in error_msg or "invalid_api_key" in error_msg.lower()
            if auth_related:
                reasoning_logs.append("[FALLBACK] OpenAI API key invalid; using heuristic requirements extraction.")
            elif quota_related:
                reasoning_logs.append("[FALLBACK] LLM quota/rate limit reached; using heuristic requirements extraction.")
            else:
                reasoning_logs.append("[FALLBACK] Structured LLM extraction failed; using heuristic requirements extraction.")

            spec_check = normalize_and_validate_specs(heuristic_specs)
            specs = spec_check["normalized"]
            plan_payload = _plan_payload(state, specs)
            if auth_related:
                spec_check["warnings"] = list(spec_check.get("warnings", [])) + [
                    "OpenAI API key invalid; requirements were parsed with local heuristic extraction."
                ]
            elif quota_related:
                spec_check["warnings"] = list(spec_check.get("warnings", [])) + [
                    "LLM quota or rate limit reached; requirements were parsed with local heuristic extraction."
                ]
            for w in spec_check.get("warnings", []):
                reasoning_logs.append(f"[FORMULA-WARN] {w}")
            return {
                "specifications": specs,
                "request_profile": request_profile,
                "active_skill": active_skill,
                "messages": [
                    "Requirements extracted with local fallback parser."
                    + (" OpenAI API key is invalid, so reasoning quality may be reduced." if auth_related else "")
                ],
                "hard_guardrails_prompt": hard_guardrails_prompt if 'hard_guardrails_prompt' in locals() else get_hard_guardrails_prompt(),
                "retrieved_knowledge_context": retrieved_context if 'retrieved_context' in locals() else "",
                "retrieved_knowledge_references": merged_refs if 'merged_refs' in locals() else [],
                "literature_references": web_research.get("references", []) if 'web_research' in locals() and isinstance(web_research, dict) else [],
                "evidence_grade": evidence_pack if 'evidence_pack' in locals() and isinstance(evidence_pack, dict) else {},
                "formula_checks": state.get("formula_checks", {}) | {"requirements": spec_check},
                "node_verification": state.get("node_verification", {}) | {
                    "requirements": {
                        "status": "PASS" if not spec_check.get("fatal") else "FAIL",
                        "fatal": spec_check.get("fatal", []),
                        "warnings": spec_check.get("warnings", []),
                        "fallback_mode": "heuristic",
                    }
                },
                "reasoning_logs": state.get("reasoning_logs", {}) | {"requirements": reasoning_logs},
                "reasoning_trace": reasoning_trace,
                **plan_payload,
            }

        auth_related = "401" in error_msg or "Incorrect API key" in error_msg or "invalid_api_key" in error_msg.lower()
        failure_message = (
            "OpenAI API key is invalid and local heuristic parsing could not recover the requirements."
            if auth_related
            else f"Requirements analysis failed: {str(e)}"
        )
        return {
            "error_log": [failure_message],
            "messages": [f"SYSTEM: {failure_message}"],
            "node_verification": state.get("node_verification", {}) | {
                "requirements": {
                    "status": "FAIL",
                    "fatal": [failure_message],
                    "warnings": [],
                }
            },
            "reasoning_logs": state.get("reasoning_logs", {}) | {"requirements": reasoning_logs + [f"[ERROR] {failure_message}"]},
            "reasoning_trace": reasoning_trace,
        }
        
    return {
        "specifications": specs,
        "request_profile": request_profile,
        "active_skill": active_skill,
        "messages": [f"Spec Analysis Complete: Targeted {specs.get('output_voltage')}V / {specs.get('output_current')}A"],
        "formula_checks": state.get("formula_checks", {}) | {"requirements": normalize_and_validate_specs(specs)},
        **_plan_payload(state, specs),
    }
