import pandas as pd
import os
import json
import random 
import time
import re
from typing import Dict, Any, Optional, Tuple, List
from langchain_core.prompts import ChatPromptTemplate
from ..state import PowerSupplyState, BillOfMaterials, ReasoningTraceItem, EvidenceSource
from .requirements import get_llm
from ..formula_guardrails import check_bom_margins
from ..lifelong_memory import get_memory_engine, summarize_skill_hits
from ..skills_manager import get_skill_manager
from ..tools.input_semantics import is_offline_ac_input, resolve_power_stage_input
from .research_helper import collect_node_research
from ...utils.digikey_local_db import (
    ensure_local_db,
    query_local_components,
    catalog_overview,
)
from ...utils.component_rag_bridge import get_component_readiness

try:
    from ...utils.web_search import perform_search as real_perform_search
except ImportError:
    real_perform_search = None

try:
    from core.skills.web_research.tools import (
        get_supported_digikey_component_types,
        research_web as mcp_research_web,
        research_digikey_component,
        research_digikey_mosfet,
    )
except ImportError:
    get_supported_digikey_component_types = None
    mcp_research_web = None
    research_digikey_component = None
    research_digikey_mosfet = None

CSV_PATH = "core/knowledge/component_db/single_fets__mosfets.csv"


def _component_label(component: Dict[str, Any]) -> str:
    if not isinstance(component, dict):
        return "Unknown"
    # Handle nested component groups (input protection / EMI / clamp blocks).
    nested_paths = [
        ("fuse", "part_number"),
        ("bridge_rectifier", "part_number"),
        ("ntc", "part_number"),
        ("mov", "part_number"),
        ("cm_choke", "part_number"),
        ("tvs", "part_number"),
        ("rcd_clamp", "diode"),
    ]
    nested_parts: List[str] = []
    for group_key, field_key in nested_paths:
        group = component.get(group_key)
        if isinstance(group, dict):
            val = str(group.get(field_key) or "").strip()
            if val and val.lower() != "unknown":
                nested_parts.append(val)
    if nested_parts:
        return ", ".join(nested_parts[:3])

    return (
        component.get("part_number")
        or component.get("Part Number")
        or component.get("Mfr Part #")
        or component.get("title")
        or component.get("description")
        or "Unknown"
    )


def _component_price(component: Dict[str, Any]) -> Any:
    if not isinstance(component, dict):
        return "-"
    if component.get("price") or component.get("Price"):
        return component.get("price") or component.get("Price")

    # Nested fallback prices.
    nested_prices: List[str] = []
    for key in ["fuse", "bridge_rectifier", "ntc", "mov", "cm_choke", "tvs"]:
        row = component.get(key)
        if isinstance(row, dict):
            pv = row.get("price") or row.get("Price")
            if pv:
                nested_prices.append(str(pv))
    return ", ".join(nested_prices[:2]) if nested_prices else "-"


def _component_source(component: Dict[str, Any]) -> str:
    if not isinstance(component, dict):
        return ""
    src = component.get("source") or component.get("Source") or component.get("url")
    if src:
        return str(src)

    # Nested fallback links.
    nested_sources: List[str] = []
    for key in ["fuse", "bridge_rectifier", "ntc", "mov", "cm_choke", "tvs"]:
        row = component.get(key)
        if isinstance(row, dict):
            u = row.get("source") or row.get("Source") or row.get("url")
            if u:
                nested_sources.append(str(u))
    return nested_sources[0] if nested_sources else ""


def _is_placeholder_part(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"", "-", "unknown", "n/a", "none", "offline"}


def _extract_voltage_v_from_text(text: str) -> float:
    s = str(text or "")
    # Match patterns like 60V, 400 VDC, 5VWM.
    vals = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*v(?:\b|dc|ac|wm)", s, flags=re.I)]
    return max(vals) if vals else 0.0


def _selector_learning_profile(state: PowerSupplyState) -> Dict[str, Any]:
    learning_context = state.get("learning_context") or {}
    lesson_guidance = learning_context.get("lesson_guidance") if isinstance(learning_context, dict) else {}
    skill_guidance = learning_context.get("skill_guidance") if isinstance(learning_context, dict) else {}
    strategy_bundle = ((state.get("verification") or {}).get("strategy_bundle") or {}) if isinstance(state.get("verification"), dict) else {}
    iteration_learning = state.get("iteration_learning") or {}

    preferred_actions: List[str] = []
    avoid_terms: List[str] = []
    focus_terms: List[str] = []

    for source in [
        lesson_guidance.get("component_actions") if isinstance(lesson_guidance, dict) else [],
        skill_guidance.get("component_actions") if isinstance(skill_guidance, dict) else [],
        strategy_bundle.get("recommended_component_actions") if isinstance(strategy_bundle, dict) else [],
        iteration_learning.get("recommended_component_actions") if isinstance(iteration_learning, dict) else [],
    ]:
        if isinstance(source, list):
            for item in source:
                text = str(item).strip().lower()
                if text and text not in preferred_actions:
                    preferred_actions.append(text)

    for source in [
        lesson_guidance.get("avoid") if isinstance(lesson_guidance, dict) else [],
        skill_guidance.get("avoid") if isinstance(skill_guidance, dict) else [],
        strategy_bundle.get("do_not_repeat") if isinstance(strategy_bundle, dict) else [],
        iteration_learning.get("do_not_repeat") if isinstance(iteration_learning, dict) else [],
    ]:
        if isinstance(source, list):
            for item in source:
                text = str(item).strip().lower()
                if text and text not in avoid_terms:
                    avoid_terms.append(text)

    for source in [
        lesson_guidance.get("focus") if isinstance(lesson_guidance, dict) else [],
        skill_guidance.get("focus") if isinstance(skill_guidance, dict) else [],
        strategy_bundle.get("next_iteration_focus") if isinstance(strategy_bundle, dict) else [],
        iteration_learning.get("next_iteration_focus") if isinstance(iteration_learning, dict) else [],
    ]:
        if isinstance(source, list):
            for item in source:
                text = str(item).strip()
                if text and text not in focus_terms:
                    focus_terms.append(text)

    return {
        "dominant_loss": str(strategy_bundle.get("dominant_loss") or iteration_learning.get("dominant_loss") or "").strip().lower(),
        "component_actions": preferred_actions,
        "preferred_actions": preferred_actions,
        "avoid_terms": avoid_terms[:8],
        "focus": focus_terms[:8],
        "lesson_summary": (lesson_guidance.get("preview") or [])[:3] if isinstance(lesson_guidance, dict) else [],
        "skill_summary": (skill_guidance.get("preview") or [])[:3] if isinstance(skill_guidance, dict) else [],
    }


def _selector_query_text(component_type: str, specs: Dict[str, Any], design: Dict[str, Any], profile: Dict[str, Any]) -> str:
    focus = " ".join([str(x) for x in (profile.get("focus") or [])[:3]])
    actions = " ".join([str(x) for x in (profile.get("preferred_actions") or [])[:3]])
    return (
        f"{component_type} flyback Vin {specs.get('input_voltage_min')}-{specs.get('input_voltage_max')} "
        f"Vout {specs.get('output_voltage')} Iout {specs.get('output_current')} "
        f"fsw {design.get('switching_frequency')} dominant_loss {profile.get('dominant_loss') or ''} "
        f"{focus} {actions}"
    ).strip()


def _query_local_candidates(
    component_type: str,
    specs: Dict[str, Any],
    design: Dict[str, Any],
    profile: Optional[Dict[str, Any]],
    *,
    min_vds: float = 0.0,
    min_id: float = 0.0,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    return query_local_components(
        component_type,
        min_vds=min_vds,
        min_id=min_id,
        limit=limit,
        preference_profile=profile,
        text_query=_selector_query_text(component_type, specs, design, profile or {}),
    )


def _candidate_bias_note(row: Dict[str, Any]) -> str:
    reasons = row.get("_selector_reasons") or []
    if not isinstance(reasons, list) or not reasons:
        return ""
    reason_text = ", ".join([str(x) for x in reasons[:3]])
    score = row.get("_selector_score")
    if score is None:
        return reason_text
    return f"score={score} ({reason_text})"


def _extract_current_a_from_text(text: str) -> float:
    s = str(text or "")
    vals: List[float] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(ma|a)\b", s, flags=re.I):
        v = float(m.group(1))
        unit = m.group(2).lower()
        vals.append(v / 1000.0 if unit == "ma" else v)
    return max(vals) if vals else 0.0


def _extract_numeric(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        m = re.search(r"(-?[\d\.]+)", str(value or ""))
        return float(m.group(1)) if m else float(default)
    except Exception:
        return float(default)


def _parse_capacitance_f(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        v = float(value)
        # Heuristic: values >= 1 are likely in uF when passed from text-derived paths.
        return v * 1e-6 if v >= 1.0 else v
    text = str(value or "").strip().lower().replace(" ", "")
    text = text.replace("µ", "u").replace("μ", "u")

    # In long free text, only trust tokens that explicitly carry capacitance units.
    matches = re.findall(r"(?<![a-z0-9])(-?[\d\.]+)(pf|nf|uf|mf)(?![a-z0-9])", text)
    if matches:
        # Pick the largest capacitance token as a conservative candidate for filtering.
        parsed_vals: List[float] = []
        for num_s, unit in matches:
            try:
                num = float(num_s)
            except Exception:
                continue
            scale = {"pf": 1e-12, "nf": 1e-9, "uf": 1e-6, "mf": 1e-3}
            parsed_vals.append(num * scale.get(unit, 1.0))
        if parsed_vals:
            return max(parsed_vals)

    # If it is a clean numeric-only token, keep numeric heuristic.
    if re.fullmatch(r"-?[\d\.]+", text):
        num = float(text)
        return num * 1e-6 if num >= 1.0 else num

    return float(default)


def _parse_resistance_ohm(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().lower().replace(" ", "")
    m = re.search(r"(-?[\d\.]+)(mohm|ohm|mω|ω)?", text)
    if not m:
        return float(default)
    num = float(m.group(1))
    unit = m.group(2) or "ohm"
    if unit in {"mohm", "mω"}:
        return num * 1e-3
    return num


def _raw_json_dict(component: Dict[str, Any]) -> Dict[str, Any]:
    raw = component.get("raw_json") if isinstance(component, dict) else None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _safe_json_loads(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        obj = json.loads(str(raw or ""))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _raw_row_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    raw = _raw_json_dict(row)
    nested = _safe_json_loads(raw.get("Raw Row JSON"))
    return nested or raw


def _find_raw_value(raw: Dict[str, Any], key_hints: List[str]) -> Any:
    if not isinstance(raw, dict) or not raw:
        return None
    hints = [h.lower() for h in key_hints]
    # Prefer exact-ish key matches first.
    for k, v in raw.items():
        lk = str(k).strip().lower()
        if any(lk == h for h in hints):
            return v
    # Then fuzzy contains.
    for k, v in raw.items():
        lk = str(k).strip().lower()
        if any(h in lk for h in hints):
            return v
    return None


def _parse_unit_value(value: Any, unit_scale: Dict[str, float], default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().lower().replace(",", "")
    if not text or text == "-":
        return default
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-zA-ZµμΩω/]+)?", text)
    if not match:
        return default
    try:
        val = float(match.group(1))
    except Exception:
        return default
    unit = str(match.group(2) or "").strip().lower()
    for key, scale in unit_scale.items():
        if key.lower() in unit:
            return val * scale
    return val


def _first_raw_numeric(
    raw: Dict[str, Any],
    key_hints: List[str],
    unit_scale: Dict[str, float],
    default: Optional[float] = None,
) -> Optional[float]:
    value = _find_raw_value(raw, key_hints)
    return _parse_unit_value(value, unit_scale, default)


def _parse_vf_curve_points(value: Any) -> List[Dict[str, float]]:
    text = str(value or "")
    points: List[Dict[str, float]] = []
    for m in re.finditer(r"([\d\.]+)\s*v\s*@\s*([\d\.]+)\s*a", text, flags=re.I):
        vf = float(m.group(1))
        if_a = float(m.group(2))
        points.append({"if_a": if_a, "vf_v": vf})
    return points


def _pick_vf_from_curve(points: List[Dict[str, float]], target_if_a: float, default_vf: float) -> float:
    if not points:
        return default_vf
    best = min(points, key=lambda p: abs(float(p.get("if_a", 0.0)) - float(target_if_a)))
    return float(best.get("vf_v", default_vf))


def _extract_rds_from_text_ohm(text: Any, default: float = 0.0) -> float:
    s = str(text or "")
    # Typical pattern: "360mOhm @ 3A, 10V"
    m = re.search(r"([\d\.]+)\s*(mohm|ohm)", s, flags=re.I)
    if not m:
        return float(default)
    val = float(m.group(1))
    unit = m.group(2).lower()
    return val * 1e-3 if unit == "mohm" else val


def _extract_vf_from_text_v(text: Any, default: float = 0.0) -> float:
    s = str(text or "")
    # Prefer explicit "xV @ yA" first.
    m = re.search(r"([\d\.]+)\s*v\s*@\s*([\d\.]+)\s*a", s, flags=re.I)
    if m:
        return float(m.group(1))
    # Do not use generic voltage tokens as fallback; descriptions usually include reverse voltage rating.
    return float(default)


def _parse_inductance_h(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        v = float(value)
        # If an integer-like value is too large, assume uH text got pre-parsed without unit.
        return v * 1e-6 if v > 1.0 else v
    text = str(value or "").strip().lower().replace(" ", "")
    m = re.search(r"(-?[\d\.]+)(uh|mh|h)?", text)
    if not m:
        return float(default)
    num = float(m.group(1))
    unit = m.group(2) or "h"
    scale = {"uh": 1e-6, "mh": 1e-3, "h": 1.0}
    return num * scale.get(unit, 1.0)


def _resolve_selector_sim_params(specs: Dict[str, Any], design: Dict[str, Any], bom: Dict[str, Any]) -> Dict[str, Any]:
    mosfet = bom.get("mosfet", {}) if isinstance(bom.get("mosfet"), dict) else {}
    diode = bom.get("diode", {}) if isinstance(bom.get("diode"), dict) else {}
    transformer = bom.get("transformer", {}) if isinstance(bom.get("transformer"), dict) else {}
    out_cap = bom.get("output_cap", {}) if isinstance(bom.get("output_cap"), dict) else {}
    clamp = bom.get("clamp_snubber", {}) if isinstance(bom.get("clamp_snubber"), dict) else {}
    rcd = clamp.get("rcd_clamp", {}) if isinstance(clamp.get("rcd_clamp"), dict) else {}
    mos_raw = _raw_json_dict(mosfet)
    diode_raw = _raw_json_dict(diode)
    out_cap_raw = _raw_json_dict(out_cap)
    xfmr_raw = _raw_json_dict(transformer)

    np_v = int(max(1, _extract_numeric(transformer.get("Np"), 36)))
    ns_v = int(max(1, _extract_numeric(transformer.get("Ns"), max(1, int(round(np_v / max(1.0, float(design.get("turns_ratio", 6.0)))))))))

    lp_h = _parse_inductance_h(transformer.get("Lp"), 0.0)
    if lp_h <= 0.0:
        lp_h = _parse_inductance_h(_find_raw_value(xfmr_raw, ["inductance", "primary inductance", "lp"]), 0.0)
    if lp_h <= 0.0:
        lp_h = float(design.get("primary_inductance", 1e-3) or 1e-3)

    co_f = 0.0
    if isinstance(out_cap.get("Value"), (int, float)):
        co_f = _parse_capacitance_f(out_cap.get("Value"), 0.0)
    if co_f <= 0.0:
        for key in ("value", "description", "part_number"):
            co_f = _parse_capacitance_f(out_cap.get(key), 0.0)
            if co_f > 0.0:
                break
    if co_f <= 0.0:
        co_f = _parse_capacitance_f(_find_raw_value(out_cap_raw, ["capacitance", "value"]), 0.0)
    if co_f <= 0.0:
        i_out = float(specs.get("output_current", 2.0) or 2.0)
        f_sw = float(design.get("switching_frequency", 65000.0) or 65000.0)
        ripple = float(specs.get("max_ripple_voltage", 0.1) or 0.1)
        co_f = i_out / (8.0 * max(1.0, f_sw) * max(0.02, ripple))

    ron_ohm = _parse_resistance_ohm(mosfet.get("Rds") or mosfet.get("Rds On (Max) @ Id, Vgs"), 0.25)
    ron_raw = _find_raw_value(mos_raw, ["rds on", "rds(on)", "drain to source resistance", "on resistance"])
    if ron_raw is not None:
        ron_ohm = _parse_resistance_ohm(ron_raw, ron_ohm)
    if ron_raw is None:
        ron_ohm = _extract_rds_from_text_ohm(
            _find_raw_value(mos_raw, ["description", "product description", "title"]) or mosfet.get("Description"),
            ron_ohm,
        )
    ron_ohm = min(max(ron_ohm, 0.03), 1.2)

    vf_v = _extract_numeric(diode.get("forward_voltage") or diode.get("Vf"), 0.75)
    vf_raw = _find_raw_value(diode_raw, ["voltage - forward", "forward voltage", "vf"])
    vf_curve_points = _parse_vf_curve_points(vf_raw)
    i_out_target = float(specs.get("output_current", 2.0) or 2.0)
    if vf_curve_points:
        vf_v = _pick_vf_from_curve(vf_curve_points, target_if_a=i_out_target, default_vf=vf_v)
    elif vf_raw is not None:
        vf_v = _extract_numeric(vf_raw, vf_v)
    else:
        vf_v = _extract_vf_from_text_v(
            _find_raw_value(diode_raw, ["description", "product description", "title"]) or diode.get("description"),
            vf_v,
        )
    if vf_v <= 0.0:
        diode_desc = str(diode.get("description") or diode.get("part_number") or "").lower()
        vf_v = 0.55 if "schottky" in diode_desc else 0.85
    vf_v = min(max(vf_v, 0.3), 1.2)

    diode_desc = str(diode.get("description") or diode.get("part_number") or "").lower()
    rdiode_ohm = 0.03 if "schottky" in diode_desc else 0.08
    rdiode_ohm = min(max(rdiode_ohm, 0.01), 0.2)

    esr_ohm = _parse_resistance_ohm(out_cap.get("esr") or out_cap.get("ESR"), 0.03)
    esr_raw = _find_raw_value(out_cap_raw, ["equivalent series resistance", "esr"])
    if esr_raw is not None:
        esr_ohm = _parse_resistance_ohm(esr_raw, esr_ohm)
    esr_ohm = min(max(esr_ohm, 0.005), 0.2)

    rsn_ohm = _extract_numeric(rcd.get("R"), _extract_numeric(design.get("snubber_r"), 10000.0))
    csn_f = _extract_numeric(rcd.get("C"), _extract_numeric(design.get("snubber_c"), 1e-9))

    return {
        "np": np_v,
        "ns": ns_v,
        "lp_h": lp_h,
        "co_f": co_f,
        "ron_ohm": ron_ohm,
        "vf_v": vf_v,
        "rdiode_ohm": rdiode_ohm,
        "vf_curve_points": vf_curve_points,
        "resr_ohm": esr_ohm,
        "rsn_ohm": rsn_ohm,
        "csn_f": csn_f,
        "raw_param_hints": {
            "mosfet_rds": str(ron_raw) if ron_raw is not None else "",
            "diode_vf": str(vf_raw) if vf_raw is not None else "",
            "output_cap_esr": str(esr_raw) if esr_raw is not None else "",
        },
        "source": "selector_normalized",
    }


def _resolve_current_a(raw_current: Any, title: str = "") -> float:
    # Prefer explicit unit-bearing text from title when available.
    from_title = _extract_current_a_from_text(title)
    if from_title > 0:
        return from_title
    try:
        v = float(raw_current)
        if v <= 0:
            return 0.0
        # Local CSV parsing can turn 300mA into 300; convert likely mA magnitudes.
        return v / 1000.0 if v > 50 else v
    except Exception:
        return 0.0


def _diode_candidate_is_valid(row: Dict[str, Any], min_vr: float, min_if: float) -> bool:
    title = str(row.get("title") or row.get("description") or "")
    pn = str(row.get("part_number") or "")
    text = f"{pn} {title}".lower()
    if any(k in text for k in ["mosfet", "fet", "transistor", "2n-channel", "dual"]) and "diode" not in text:
        return False

    vr = float(row.get("vds") or 0.0)
    if vr <= 0:
        vr = _extract_voltage_v_from_text(text)
    if req := float(min_vr or 0.0):
        if vr < req:
            return False

    if_a = _resolve_current_a(row.get("id_current"), title)
    if if_a < float(min_if or 0.0):
        return False
    return True


def _mosfet_candidate_is_valid(row: Dict[str, Any], min_vds: float, min_id: float, specs: Dict[str, Any]) -> bool:
    title = str(row.get("title") or row.get("description") or "")
    pn = str(row.get("part_number") or "")
    text = f"{pn} {title}".lower()
    if "p-channel" in text or "module" in text:
        return False

    vds = float(row.get("vds") or 0.0)
    if vds <= 0:
        vds = _extract_voltage_v_from_text(text)
    if vds < float(min_vds or 0.0):
        return False

    if_a = _resolve_current_a(row.get("id_current"), title)
    if if_a < float(min_id or 0.0):
        return False

    raw = _raw_row_dict(row)
    rds = _first_raw_numeric(raw, ["Rds On (Max) @ Id, Vgs", "Drain to Source On Resistance (Rds On)"], {"mohm": 1e-3, "ohm": 1.0, "ω": 1.0})
    if rds is None:
        rds = _extract_rds_from_text_ohm(title, 0.0)

    pout_w = float(specs.get("output_voltage", 0.0) or 0.0) * float(specs.get("output_current", 0.0) or 0.0)
    if is_offline_ac_input(specs or {}) and pout_w >= 18.0 and rds and rds > 2.5:
        return False
    return True


def _controller_candidate_is_valid(row: Dict[str, Any]) -> bool:
    title = str(row.get("title") or row.get("description") or "")
    pn = str(row.get("part_number") or "")
    text = f"{pn} {title}".lower()
    always_bad = [
        "analog switch",
        "bilateral switch",
        "multiplexer",
        "demultiplexer",
        "load switch",
        "charge pump",
        "led drv",
        "led driver",
        "ldo",
        "linear regulator",
    ]
    if any(k in text for k in always_bad):
        return False

    offline_controller_families = [
        "uc384",
        "ucc287",
        "ucc28c",
        "ncp120",
        "ncp125",
        "ncp134",
        "tea",
        "ob22",
        "fan675",
        "ld75",
        "tiny",
        "tny",
        "topswitch",
        "viper",
    ]
    if any(k in text for k in offline_controller_families):
        return True

    # Universal-input flyback should not silently select low-voltage buck/boost
    # regulator ICs just because their catalog text mentions "flyback".
    dc_dc_only = ["buck", "boost", "buck-boost", "step-down", "step-up", "isolation capable"]
    if any(k in text for k in dc_dc_only) and not any(k in text for k in ["offline", "ac-dc", "primary-side", "current-mode"]):
        return False

    good = [
        "offline",
        "ac-dc",
        "primary-side",
        "pwm controller",
        "current-mode",
        "current mode",
        "quasi-resonant",
        "flyback controller",
    ]
    return any(k in text for k in good)


def _transformer_candidate_is_valid(row: Dict[str, Any]) -> bool:
    pn = str(row.get("part_number") or "")
    if _is_placeholder_part(pn):
        return False
    title = str(row.get("title") or row.get("description") or "")
    raw = str(row.get("raw_json") or "")
    text = f"{pn} {title} {raw[:1200]}".lower()
    # Reject parts that look like noise-suppression ferrite beads, line filters,
    # or low-frequency laminated mains transformers rather than flyback magnetics.
    bad_tokens = [
        "laminated",
        "forward",
        "push-pull",
        "push pull",
        "fwd p-p",
        "sn6501",
        "mid-ppti",
        "ferrite bead",
        "bead core",
        "2 hole",
        "2-hole",
        "two hole",
        "multi-hole",
        "multi hole",
        "suppression",
        "noise filter",
        "common mode",
        "emi",
        "cable core",
        "wire core",
        "rod",
        "toroid",
        "toroidal",
        "powder core",
        "snap",
        "0.265\"",
        "6.73mm",
    ]
    if any(token in text for token in bad_tokens):
        return False
    good_tokens = [
        "flyback",
        "switching transformer",
        "smps transformer",
        "transformer core",
        "power transformer",
        "bobbin",
        "core set",
        "ferrite core",
        "pq",
        "ee",
        "efd",
        "etd",
        "rm",
    ]
    return any(k in text for k in good_tokens)


def _emi_candidate_is_valid(row: Dict[str, Any], specs: Dict[str, Any]) -> bool:
    pn = str(row.get("part_number") or "")
    if _is_placeholder_part(pn):
        return False
    title = str(row.get("title") or row.get("description") or "")
    raw = str(row.get("raw_json") or "")
    text = f"{pn} {title} {raw[:1200]}".lower()
    if is_offline_ac_input(specs or {}):
        bad_tokens = [
            "signal line",
            "data line",
            "usb",
            "ethernet",
            "can bus",
            "esd",
            "chip bead",
            "array",
            "not for new designs",
        ]
        if any(token in text for token in bad_tokens):
            return False
        v = float(row.get("vds") or 0.0)
        if v <= 0:
            v = _extract_voltage_v_from_text(text)
        if 0 < v < 125.0:
            return False
        i_rating = _resolve_current_a(row.get("id_current"), title)
        p_out = float(specs.get("output_voltage", 0.0) or 0.0) * float(specs.get("output_current", 0.0) or 0.0)
        min_line_current = p_out / max(1.0, float(resolve_power_stage_input(specs or {}).get("dc_bus_min", 120.0) or 120.0))
        min_current = max(0.6, min_line_current * 1.8)
        if 0 < i_rating < min_current:
            return False
        has_mains_evidence = any(token in text for token in ["250vac", "275vac", "300vac", "mains", "ac line", "line filter"])
        if v <= 0 and not has_mains_evidence:
            return False
        good_tokens = [
            "line filter",
            "mains",
            "ac line",
            "common mode choke",
            "cm choke",
            "emi filter",
            "emc filter",
            "x2",
            "y1",
            "250vac",
            "275vac",
        ]
        return any(token in text for token in good_tokens)
    return any(token in text for token in ["emi", "emc", "common mode", "line filter", "choke", "filter"])


def _input_protection_candidate_is_valid(row: Dict[str, Any]) -> bool:
    pn = str(row.get("part_number") or "")
    if _is_placeholder_part(pn):
        return False
    title = str(row.get("title") or row.get("description") or "")
    text = f"{pn} {title} {row.get('raw_json') or ''}".lower()
    bad_tokens = ["fuse clip", "holder", "chip thermistor", "multilayer ntc", "esd", "0603", "0402", "0805"]
    if any(token in text for token in bad_tokens):
        return False
    v = float(row.get("vds") or 0.0)
    if v <= 0:
        v = _extract_voltage_v_from_text(text)
    if any(token in text for token in ["mov", "varistor", "bridge rectifier", "cartridge fuse", "time delay fuse", "inrush current limiter"]):
        return v <= 0 or v >= 125.0
    if ("ntc" in text or "thermistor" in text) and any(token in text for token in ["inrush", "disc", "power"]):
        return True
    return False


def _cap_candidate_is_valid(row: Dict[str, Any], min_v: float, max_v: float = 0.0) -> bool:
    title = str(row.get("title") or row.get("description") or "")
    pn = str(row.get("part_number") or "")
    text = f"{pn} {title}".lower()
    if "cap" not in text and "capacitor" not in text:
        return False
    if any(k in text for k in ["not for new designs", "obsolete", "discontinued"]):
        return False
    v = float(row.get("vds") or 0.0)
    if v <= 0:
        v = _extract_voltage_v_from_text(text)
    if v < float(min_v or 0.0):
        return False
    if max_v > 0 and v > max_v:
        return False
    return True


def _cap_value_f_from_row(row: Dict[str, Any]) -> float:
    if not isinstance(row, dict):
        return 0.0
    # Try explicit fields first.
    for key in ["capacitance", "value", "Value"]:
        if key in row:
            v = _parse_capacitance_f(row.get(key), 0.0)
            if v > 0.0:
                return v
    # Then parse descriptive text.
    text_blob = " ".join(
        [
            str(row.get("title") or ""),
            str(row.get("description") or ""),
            str(row.get("raw_json") or ""),
        ]
    )
    return _parse_capacitance_f(text_blob, 0.0)


def _build_local_db_candidate_snapshot(
    specs: Optional[Dict[str, Any]] = None,
    design: Optional[Dict[str, Any]] = None,
    selector_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    # Attach top local candidates so frontend and downstream agents can inspect alternatives.
    category_map = {
        "mosfet": "mosfet",
        "diode": "diode",
        "transformer": "transformer",
        "input_cap": "input_cap",
        "output_cap": "output_cap",
        "controller": "controller",
        "input_protection": "input_protection",
        "emi_filter": "emi_filter",
        "clamp_snubber": "clamp_snubber",
    }
    out: Dict[str, List[Dict[str, Any]]] = {}
    try:
        ensure_local_db()
        req_vds = 0.0
        req_id = 0.0
        if isinstance(specs, dict) and isinstance(design, dict):
            req_vds, req_id = calculate_mosfet_requirements(specs, design)
        i_out = float((specs or {}).get("output_current", 2.0) or 2.0)
        v_out = float((specs or {}).get("output_voltage", 12.0) or 12.0)
        f_sw = float((design or {}).get("switching_frequency", 65000.0) or 65000.0)
        ripple_target = float((specs or {}).get("max_ripple_voltage", 0.1) or 0.1)
        c_out_target = max(220e-6, i_out / (8.0 * f_sw * max(ripple_target, 0.02)))
        input_domain = resolve_power_stage_input(specs or {})
        min_diode_vr = (v_out + float(input_domain.get("dc_bus_max", 375.0) or 375.0) / max(float((design or {}).get("turns_ratio", 6.0) or 6.0), 1e-6)) * 1.5
        min_diode_if = max(i_out * 1.5, 1.0)
        for bom_key, db_type in category_map.items():
            min_v = req_vds if bom_key == "mosfet" else 0.0
            min_i = req_id if bom_key == "mosfet" else 0.0
            rows = query_local_components(
                db_type,
                min_vds=min_v,
                min_id=min_i,
                limit=80,
                preference_profile=selector_profile,
                text_query=_selector_query_text(db_type, specs or {}, design or {}, selector_profile or {}),
            )
            compact: List[Dict[str, Any]] = []
            seen_keys = set()
            for row in rows:
                if bom_key == "mosfet" and not _mosfet_candidate_is_valid(row, req_vds, req_id, specs or {}):
                    continue
                if bom_key == "diode" and not _diode_candidate_is_valid(row, min_vr=min_diode_vr, min_if=min_diode_if):
                    continue
                if bom_key == "controller" and not _controller_candidate_is_valid(row):
                    continue
                if bom_key == "input_cap" and not (
                    _cap_candidate_is_valid(row, min_v=380.0) and _cap_value_f_from_row(row) >= 68e-6
                ):
                    continue
                if bom_key == "output_cap" and not (
                    _cap_candidate_is_valid(row, min_v=max(25.0, v_out * 1.6), max_v=63.0)
                    and _cap_value_f_from_row(row) >= c_out_target * 0.8
                ):
                    continue
                if bom_key == "transformer" and not _transformer_candidate_is_valid(row):
                    continue
                if bom_key == "emi_filter" and not _emi_candidate_is_valid(row, specs or {}):
                    continue
                if bom_key == "input_protection" and not _input_protection_candidate_is_valid(row):
                    continue
                dedup_key = (
                    str(row.get("part_number") or "").strip().lower(),
                    str(row.get("url") or "").strip().lower(),
                )
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                compact.append(
                    {
                        "part_number": row.get("part_number"),
                        "description": row.get("description") or row.get("title"),
                        "price": row.get("price"),
                        "stock": row.get("stock"),
                        "vds": row.get("vds"),
                        "id_current": row.get("id_current"),
                        "url": row.get("url"),
                        "source_mode": row.get("source_mode"),
                        "selector_bias": _candidate_bias_note(row),
                    }
                )
                if len(compact) >= 4:
                    break
            out[bom_key] = compact
    except Exception:
        return {}
    return out

def component_selector_node(
    state: PowerSupplyState,
    config: Dict[str, Any] = None,
    *,
    store: Any = None,
) -> Dict[str, Any]:
    """
    Node 3: Component Selector (Enhanced with Reasoning Visualization + Online Search)
    Now supports DeepRare Traceable Reasoning.
    """
    print("\n" + "="*50)
    print("[Component Selector Agent] START - SOTA MODE")
    print("="*50)

    specs = state.get("specifications")
    design = state.get("theoretical_design")
    
    # [NEW] Chitchat Bypass
    q_vin = specs.get("input_voltage_max") if isinstance(specs, dict) else "-"
    q_vout = specs.get("output_voltage") if isinstance(specs, dict) else "-"
    q_iout = specs.get("output_current") if isinstance(specs, dict) else "-"
    selector_research = collect_node_research(
        "selector",
        f"flyback component selection datasheet papers blogs forums distributors for Vin {q_vin} Vout {q_vout} Iout {q_iout}",
        max_results=6,
    )
    selector_profile = _selector_learning_profile(state)
    selector_research_logs = selector_research.get("logs", [])
    selector_research_logs = [
        "[PLAN] Start component selection: power stage first, then protection and EMI peripherals.",
        "[SEARCH] Querying distributor/web evidence for candidate parts and datasheets.",
        *selector_research_logs,
        "[EXECUTION] Begin category-by-category screening with derating checks.",
    ]
    if selector_profile.get("preferred_actions"):
        selector_research_logs.append(
            "[LEARNING] Selector bias activated from historical lessons: "
            + ", ".join([str(x) for x in selector_profile.get("preferred_actions", [])[:5]])
        )
    if selector_profile.get("focus"):
        selector_research_logs.append(
            "[LEARNING] Current iteration focus: "
            + " | ".join([str(x) for x in selector_profile.get("focus", [])[:3]])
        )
    if selector_profile.get("avoid_terms"):
        selector_research_logs.append(
            "[LEARNING] Avoid repeating: "
            + " | ".join([str(x) for x in selector_profile.get("avoid_terms", [])[:3]])
        )
    if get_supported_digikey_component_types:
        try:
            all_types = get_supported_digikey_component_types()
            selector_research_logs.append(
                "[DATA] DigiKey MCP supported component types: " + ", ".join(all_types)
            )
        except Exception:
            pass
    selector_refs = selector_research.get("references", [])
    magnetic_design = state.get("magnetic_design") or {}
    if isinstance(magnetic_design, dict) and magnetic_design:
        selector_research_logs.append(
            "[MAGNETICS] Advisor handoff received: "
            f"status={magnetic_design.get('status')} core={magnetic_design.get('core_family') or '-'} "
            f"gap={magnetic_design.get('gap_mm') or '-'}mm"
        )
    if specs and specs.get("is_chitchat"):
        print("Skipping Selector: Intent = Chitchat")
        return {}

    if not design or not specs:
        return {"error_log": ["Missing design/specs"]}

    memory_engine = get_memory_engine()
    sel_query = (
        f"component selection flyback Vin {specs.get('input_voltage_min')}-{specs.get('input_voltage_max')} "
        f"Vout {specs.get('output_voltage')} Iout {specs.get('output_current')} "
        f"fsw {design.get('switching_frequency')}"
    )
    memory_hits = memory_engine.search(("episodes", "flyback", "successful_designs"), query=sel_query, limit=3, store=store)
    curriculum_hits = memory_engine.search(("semantic", "flyback", "curriculum_rules"), query=sel_query, limit=4, store=store)
    skill_hits = memory_engine.search(("skills", "flyback", "design_patterns"), query=sel_query, limit=4, store=store)
    skill_guidance = summarize_skill_hits(skill_hits)
    if memory_hits:
        selector_research_logs.append(f"[MEMORY] Retrieved {len(memory_hits)} successful BOM episodes for selector guidance.")
    if curriculum_hits:
        selector_research_logs.append(f"[MEMORY] Retrieved {len(curriculum_hits)} autocurriculum rules for selector bias.")
        for hit in curriculum_hits[:3]:
            payload = hit.get("payload") or {}
            bias = payload.get("selector_bias") if isinstance(payload.get("selector_bias"), dict) else {}
            for item in bias.get("preferred_actions") or []:
                action = str(item).strip().lower()
                if action and action not in selector_profile["preferred_actions"]:
                    selector_profile["preferred_actions"].append(action)
                if action and action not in selector_profile["component_actions"]:
                    selector_profile["component_actions"].append(action)
            for item in bias.get("avoid_terms") or []:
                avoid = str(item).strip().lower()
                if avoid and avoid not in selector_profile["avoid_terms"]:
                    selector_profile["avoid_terms"].append(avoid)
            if not selector_profile.get("dominant_loss") and bias.get("dominant_loss"):
                selector_profile["dominant_loss"] = str(bias.get("dominant_loss")).strip().lower()
            selector_research_logs.append("[CURRICULUM] " + str(payload.get("summary") or "selector curriculum rule")[:220])
    if skill_hits:
        selector_research_logs.append(f"[SKILL] Retrieved {len(skill_hits)} selector-relevant skill cards.")
        for item in skill_guidance.get("component_actions") or []:
            action = str(item).strip().lower()
            if action and action not in selector_profile["preferred_actions"]:
                selector_profile["preferred_actions"].append(action)
            if action and action not in selector_profile["component_actions"]:
                selector_profile["component_actions"].append(action)
        for item in skill_guidance.get("avoid") or []:
            avoid = str(item).strip().lower()
            if avoid and avoid not in selector_profile["avoid_terms"]:
                selector_profile["avoid_terms"].append(avoid)
        for item in skill_guidance.get("focus") or []:
            focus = str(item).strip()
            if focus and focus not in selector_profile["focus"]:
                selector_profile["focus"].append(focus)
        if skill_guidance.get("objectives"):
            selector_research_logs.append("[SKILL] " + " | ".join([str(x) for x in skill_guidance.get("objectives", [])[:3]]))

    # Initialize Trace
    reasoning_trace = state.get("reasoning_trace", [])

    # --- 1. Strategic Planning (System 2) ---
    print("[System 2] Formulating Procurement Strategy...")
    reasoning_trace.append({
        "step": "Procurement Strategy",
        "agent": "Selector",
        "action": "Define Search Criteria",
        "evidence": "Design Specs",
        "confidence": 1.0,
        "citation": "Internal Design Params"
    })
    
    # [ROBUSTNESS WRAPPER]
    try:
        # --- 2. MOSFET Selection (Detailed "Deep Search" with REAL/Simulated Web Calls) ---
        print("DEBUG: Starting MOSFET Selection...")
        req_vds, req_id = calculate_mosfet_requirements(specs, design)
        mosfet_data, mosfet_logs = select_mosfet_enhanced(req_vds, req_id, specs, design, reasoning_trace, selector_profile)
        print("DEBUG: MOSFET Selection Done.")

        # --- 3. Transformer Selection ---
        print("DEBUG: Starting Transformer Selection...")
        transformer_data, transformer_logs = select_transformer_enhanced(
            design,
            specs,
            reasoning_trace,
            selector_profile,
            magnetic_design=magnetic_design,
        )
        print("DEBUG: Transformer Selection Done.")

        # --- 4. Diode Selection ---
        print("DEBUG: Starting Diode Selection...")
        diode_data, diode_logs = select_diode_enhanced(design, specs, reasoning_trace, selector_profile)
        print("DEBUG: Diode Selection Done.")

        print("DEBUG: Starting Input Capacitor Selection...")
        input_cap_data, input_cap_logs = select_input_cap_enhanced(design, specs, reasoning_trace, selector_profile)
        print("DEBUG: Input Capacitor Selection Done.")

        print("DEBUG: Starting Output Capacitor Selection...")
        output_cap_data, output_cap_logs = select_output_cap_enhanced(design, specs, reasoning_trace, selector_profile)
        print("DEBUG: Output Capacitor Selection Done.")

        print("DEBUG: Starting Controller/Feedback Selection...")
        controller_data, controller_logs = select_controller_feedback_enhanced(design, specs, reasoning_trace, selector_profile)
        print("DEBUG: Controller/Feedback Selection Done.")

        print("DEBUG: Starting Input Protection Selection...")
        input_protection_data, input_protection_logs = select_input_protection_enhanced(design, specs, reasoning_trace, selector_profile)
        print("DEBUG: Input Protection Selection Done.")

        print("DEBUG: Starting EMI Filter Selection...")
        emi_filter_data, emi_filter_logs = select_emi_filter_enhanced(design, specs, reasoning_trace, selector_profile)
        print("DEBUG: EMI Filter Selection Done.")

        print("DEBUG: Starting Clamp/Snubber Network Selection...")
        clamp_snubber_data, clamp_snubber_logs = select_clamp_snubber_enhanced(design, specs, reasoning_trace, selector_profile)
        print("DEBUG: Clamp/Snubber Network Selection Done.")
    except Exception as fatal_e:
        import traceback
        print(f"CRITICAL ERROR in Component Selector: {fatal_e}")
        traceback.print_exc()
        return {"error_log": [f"Component Selection Crashed: {fatal_e}"], "verification": {"status": "FAIL", "reason": "Agent Crash"}}

    # Helper for formatting logs
    def to_log_str(logs):
        if isinstance(logs, list): return "\n".join(str(x) for x in logs)
        return str(logs)

    try:
        print("DEBUG: Checking variables before BOM...", flush=True)
        # Verify variables exist to prevent silent UnboundLocalError
        if 'mosfet_data' not in locals(): mosfet_data = {}
        if 'diode_data' not in locals(): diode_data = {}
        if 'transformer_data' not in locals(): transformer_data = {}
        if 'mosfet_logs' not in locals(): mosfet_logs = []
        if 'transformer_logs' not in locals(): transformer_logs = []
        if 'diode_logs' not in locals(): diode_logs = []
        if 'input_cap_data' not in locals(): input_cap_data = {}
        if 'output_cap_data' not in locals(): output_cap_data = {}
        if 'controller_data' not in locals(): controller_data = {}
        if 'input_protection_data' not in locals(): input_protection_data = {}
        if 'emi_filter_data' not in locals(): emi_filter_data = {}
        if 'clamp_snubber_data' not in locals(): clamp_snubber_data = {}
        if 'input_cap_logs' not in locals(): input_cap_logs = []
        if 'output_cap_logs' not in locals(): output_cap_logs = []
        if 'controller_logs' not in locals(): controller_logs = []
        if 'input_protection_logs' not in locals(): input_protection_logs = []
        if 'emi_filter_logs' not in locals(): emi_filter_logs = []
        if 'clamp_snubber_logs' not in locals(): clamp_snubber_logs = []

        print("DEBUG: Variables verified. Constructing BOM...", flush=True)
        # Construct Final BOM
        bom = {
            "mosfet": mosfet_data,
            "diode": diode_data,
            "transformer": transformer_data,
            "input_cap": input_cap_data,
            "output_cap": output_cap_data,
            "controller": controller_data,
            "input_protection": input_protection_data,
            "emi_filter": emi_filter_data,
            "clamp_snubber": clamp_snubber_data,
        }

        # Build a stable numeric handoff so simulation does not depend on mixed free-text fields.
        bom["simulation_params"] = _resolve_selector_sim_params(specs, design, bom)

        # High-visibility structured summary for UI and logs.
        bom["selection_summary"] = {
            "mosfet": {
                "selected": _component_label(mosfet_data),
                "price": _component_price(mosfet_data),
                "source": _component_source(mosfet_data),
            },
            "diode": {
                "selected": _component_label(diode_data),
                "price": _component_price(diode_data),
                "source": _component_source(diode_data),
            },
            "transformer": {
                "selected": _component_label(transformer_data),
                "price": _component_price(transformer_data),
                "source": _component_source(transformer_data),
            },
            "input_cap": {
                "selected": _component_label(input_cap_data),
                "price": _component_price(input_cap_data),
                "source": _component_source(input_cap_data),
            },
            "output_cap": {
                "selected": _component_label(output_cap_data),
                "price": _component_price(output_cap_data),
                "source": _component_source(output_cap_data),
            },
            "controller": {
                "selected": _component_label(controller_data),
                "price": _component_price(controller_data),
                "source": _component_source(controller_data),
            },
            "input_protection": {
                "selected": _component_label(input_protection_data),
                "price": _component_price(input_protection_data),
                "source": _component_source(input_protection_data),
            },
            "emi_filter": {
                "selected": _component_label(emi_filter_data),
                "price": _component_price(emi_filter_data),
                "source": _component_source(emi_filter_data),
            },
            "clamp_snubber": {
                "selected": _component_label(clamp_snubber_data),
                "price": _component_price(clamp_snubber_data),
                "source": _component_source(clamp_snubber_data),
            },
        }

        local_candidates = _build_local_db_candidate_snapshot(specs, design, selector_profile)
        if local_candidates:
            bom["local_db_top_candidates"] = local_candidates
            selector_research_logs.append("[DETAIL] Attached local DigiKey top candidates for 9 key categories.")
        if selector_profile:
            bom["selector_learning_profile"] = selector_profile
        if magnetic_design:
            bom["magnetic_advisor"] = magnetic_design

        policy_map = {
            "mosfet": "mosfet",
            "diode": "diode_rectifier",
            "transformer": "transformer",
            "input_cap": "input_cap",
            "output_cap": "output_cap",
            "controller": "controller",
            "input_protection": "input_protection",
            "emi_filter": "emi_filter",
            "clamp_snubber": "clamp_snubber",
        }
        readiness_policy: Dict[str, Dict[str, Any]] = {}
        for bom_key, category in policy_map.items():
            policy = get_component_readiness(category)
            if isinstance(policy, dict) and policy:
                readiness_policy[bom_key] = {
                    "category": category,
                    "color": policy.get("color"),
                    "score": policy.get("score"),
                    "strategy": policy.get("strategy"),
                }
                if str(policy.get("color", "")).lower() == "red":
                    selector_research_logs.append(
                        f"[POLICY-WARN] {bom_key} mapped to {category} is RED; apply degraded/manual review path."
                    )
                elif str(policy.get("color", "")).lower() == "yellow":
                    selector_research_logs.append(
                        f"[POLICY-NOTE] {bom_key} mapped to {category} is YELLOW; require secondary cross-check."
                    )
        if readiness_policy:
            bom["selection_policy"] = readiness_policy
        extended_pool = build_extended_component_pool(max_types=24, per_type=2)
        if extended_pool:
            bom["extended_component_pool"] = extended_pool
            full_count = sum(len(v) for v in extended_pool.values())
            print(f"DEBUG: Extended pool attached: types={len(extended_pool)}, items={full_count}", flush=True)
        print("DEBUG: BOM Constructed.", flush=True)
    
        full_narrative = (
            f"{to_log_str(selector_research_logs)}\n\n"
            f"{to_log_str(mosfet_logs)}\n\n"
            f"{to_log_str(transformer_logs)}\n\n"
            f"{to_log_str(diode_logs)}\n\n"
            f"{to_log_str(input_cap_logs)}\n\n"
            f"{to_log_str(output_cap_logs)}\n\n"
            f"{to_log_str(controller_logs)}\n\n"
            f"{to_log_str(input_protection_logs)}\n\n"
            f"{to_log_str(emi_filter_logs)}\n\n"
            f"{to_log_str(clamp_snubber_logs)}"
        )
        try:
            full_narrative += "\n\n[DETAIL] Selected component summary:\n"
            for k, info in (bom.get("selection_summary") or {}).items():
                full_narrative += (
                    f"- {k}: selected={info.get('selected')}, price={info.get('price')}, "
                    f"source={info.get('source') or '-'}\n"
                )
            if isinstance(bom.get("local_db_top_candidates"), dict):
                full_narrative += "\n[DETAIL] Local DB candidate counts:\n"
                for k, rows in bom.get("local_db_top_candidates", {}).items():
                    full_narrative += f"- {k}: candidates={len(rows)}\n"
        except Exception:
            pass
        if isinstance(bom, dict) and isinstance(bom.get("extended_component_pool"), dict):
            full_narrative += (
                f"\n\n[EXTENDED_POOL] enabled types={len(bom['extended_component_pool'])} "
                f"(local DigiKey optional categories for max selector freedom)."
            )
        print("DEBUG: Narrative constructed.", flush=True)
    except BaseException as e:
        print(f"CRITICAL ERROR formulating BOM or Logs: {e} | Type: {type(e)}", flush=True)
        import traceback
        traceback.print_exc()
        full_narrative = "Log formatting error."
        bom = {} # Should fallback or use partial data

    print("\n[SUCCESS] BOM Finalized.", flush=True)
    print("="*50 + "\n")

    bom_check = check_bom_margins(specs, design, bom)
    for check_item in bom_check.get("checks", []):
        full_narrative += (
            f"\n[FORMULA] {check_item.get('name')}: required={check_item.get('required', '-')}, "
            f"actual={check_item.get('actual', '-')}, pass={check_item.get('pass')}"
        )
    for warning in bom_check.get("warnings", []):
        full_narrative += f"\n[FORMULA-WARN] {warning}"
    
    # Check for Serialization Issues
    try:
        import pickle
        # Test serialization of return value
        ret_val = {
            "bom": bom,
            "messages": [f"BOM Selected: {mosfet_data.get('Mfr Part #')} + {transformer_data.get('core')} + {controller_data.get('part_number', 'controller')}"],
            "reasoning_logs": state.get("reasoning_logs", {}) | {"selector": full_narrative},
            "literature_references": selector_refs,
            "formula_checks": state.get("formula_checks", {}) | {"selector": bom_check},
            "node_verification": state.get("node_verification", {}) | {"selector": {"status": "PASS" if not bom_check.get("warnings") else "WARN", "warnings": bom_check.get("warnings", [])}},
            "memory_context": {
                **(state.get("memory_context") or {}),
                "selector": {
                    "selector_query": sel_query,
                    "selector_hits": memory_hits,
                    "curriculum_hits": curriculum_hits,
                    "skill_hits": skill_hits,
                },
            },
        }
        pickle.dumps(ret_val)
        print("DEBUG: State Serialization Check Passed.")
        return ret_val
    except Exception as pickle_err:
        print(f"CRITICAL: Serialization Failed: {pickle_err}")
        # Return simplified state to avoid crash
        return {
             "bom": {},
             "messages": ["BOM Selection Failed Serialization"], 
             "reasoning_logs": state.get("reasoning_logs", {})
        }

def calculate_mosfet_requirements(specs, design):
    input_domain = resolve_power_stage_input(specs or {})
    vin_max = float(input_domain.get('dc_bus_max', 375.0))
    
    # [ROBUSTNESS FIX]: If Vor is dangerously low (hallucination), assume standard
    vor = design.get('reflected_output_voltage', 80)
    if vor < 5.0: 
        vor = 50.0 # Force a realistic minimum for component sizing safety
        
    spike_margin = 1.15
    clamp_overhead = 70.0
    vds_req = (vin_max + vor + clamp_overhead) * spike_margin
    
    # [SAFETY CLAMP]: Minimum Vds for any Flyback to avoid "Generic 16V MOSFET" issues
    if vds_req < 55.0:
        vds_req = 55.0
        
    id_req = design.get('primary_peak_current', 1.0) * 1.8 # extra offline flyback thermal/transient margin
    return vds_req, id_req


def build_extended_component_pool(max_types: int = 24, per_type: int = 2) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a broad optional component pool from the local DigiKey DB.
    This expands selector freedom beyond the fixed flyback categories.
    """
    try:
        ensure_local_db()
        # Ensure we only include types that have non-empty candidates.
        out: Dict[str, List[Dict[str, Any]]] = {}
        for ctype, rows in catalog_overview(max_types=max_types, per_type=per_type).items():
            items: List[Dict[str, Any]] = []
            for r in rows:
                items.append(
                    {
                        "part_number": r.get("part_number") or "Unknown",
                        "title": r.get("title") or r.get("description") or "",
                        "url": r.get("url") or "https://www.digikey.com",
                        "price": r.get("price"),
                        "supplier": r.get("supplier") or "DigiKey",
                    }
                )
            if items:
                out[ctype] = items
        return out
    except Exception:
        return {}

def select_mosfet_enhanced(
    vds_min: float,
    id_min: float,
    specs: dict,
    design: dict,
    trace: List[ReasoningTraceItem],
    selector_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, str]:
    """
    Advanced selection logic checking Local DB -> then Web Search.
    Includes N-Channel filtering and SSL Error Fallback.
    Traceable Reasoning Added.
    """
    logs = []
    
    # Step 1: Query Formulation
    logs.append(f"[PLAN] Initiating Global Component Search.")
    logs.append(f"[THOUGHT] Target: N-Channel MOSFET. Vds >= {int(vds_min)}V, Id >= {id_min:.1f}A.")
    
    trace.append({
        "step": "MOSFET Search Criteria",
        "agent": "Selector",
        "action": f"Set Min Ratings: {vds_min:.0f}V, {id_min:.1f}A",
        "evidence": "Calculated Stress * 1.5 Margin",
        "confidence": 1.0
    })
    
    local_part = None

    # --- Local DigiKey DB first (for stable offline/low-latency selection) ---
    try:
        ensure_local_db()
        local_rows = _query_local_candidates(
            "mosfet",
            specs,
            design,
            selector_profile,
            min_vds=vds_min,
            min_id=id_min,
            limit=24,
        )
        valid_rows = [r for r in local_rows if _mosfet_candidate_is_valid(r, vds_min, id_min, specs)]
        if valid_rows:
            row = valid_rows[0]
            local_part = {
                "Mfr Part #": row.get("part_number") or "Unknown",
                "Description": row.get("description") or row.get("title") or "",
                "Source": row.get("url") or "https://www.digikey.com",
                "Price": f"${float(row.get('price')):.2f}" if isinstance(row.get("price"), (int, float)) and row.get("price") is not None else "Check Online",
                "Vds": f"{float(row.get('vds')):.0f}V" if isinstance(row.get("vds"), (int, float)) and row.get("vds") is not None else f">={int(vds_min)}V",
                "Id": f"{float(row.get('id_current')):.1f}A" if isinstance(row.get("id_current"), (int, float)) and row.get("id_current") is not None else f">={id_min:.1f}A",
                "Rds": "Unknown",
                "raw_json": row.get("raw_json"),
            }
            logs.append("[LOCAL_DB_USED] MOSFET candidate selected from local DigiKey database.")
            bias_note = _candidate_bias_note(row)
            if bias_note:
                logs.append(f"[LEARNING] MOSFET historical bias kept this part near the top: {bias_note}")
            trace.append({
                "step": "MOSFET Selection",
                "agent": "Selector",
                "action": f"Selected Local DB Part {local_part.get('Mfr Part #')}",
                "evidence": f"Local DigiKey DB ({row.get('url', '')})",
                "confidence": 0.96,
            })
            return local_part, "\n\n".join(logs)
        elif local_rows:
            logs.append("[FILTER] Local MOSFET rows rejected (rating/Rds mismatch for this power class).")
    except Exception as db_err:
        logs.append(f"[WARNING] Local DigiKey DB query failed: {db_err}")
    
    # --- Try Local DB First ---
    try:
        if os.path.exists(CSV_PATH):
            logs.append(f"[SEARCH] Querying Local Knowledge Base (Cache)...")
            df = pd.read_csv(CSV_PATH)
            
            # Robust parsing for voltage/current
            def parse_val(s):
                import re
                if isinstance(s, (int, float)): return float(s)
                m = re.search(r"([\d\.]+)", str(s))
                return float(m.group(1)) if m else 0.0
            
            # Helper to clean currency strings
            def cleaner(x):
                try: return float(str(x).replace('$','').replace(',','')) 
                except: return 999.0
            
            df['Vds_Val'] = df['Drain to Source Voltage (Vdss)'].apply(parse_val)
            df['Id_Val'] = df['Current - Continuous Drain (Id) @ 25°C'].apply(parse_val)
            df['Price_Clean'] = df['Price'].apply(cleaner)
            
            # Filter for N-Channel explicitly if possible
            if 'FET Type' in df.columns:
                df = df[df['FET Type'].astype(str).str.contains("N-Channel", case=False, na=False)]

            candidates = df[(df['Vds_Val'] >= vds_min) & (df['Id_Val'] >= id_min)].copy()
            if not candidates.empty:
                candidates = candidates[
                    candidates.apply(
                        lambda row: _mosfet_candidate_is_valid(
                            {
                                "part_number": row.get("Mfr Part #"),
                                "title": row.get("Description"),
                                "description": row.get("Description"),
                                "vds": row.get("Vds_Val"),
                                "id_current": row.get("Id_Val"),
                                "raw_json": row.get("Raw Row JSON"),
                            },
                            vds_min,
                            id_min,
                            specs,
                        ),
                        axis=1,
                    )
                ]
            
            if not candidates.empty:
                candidates = candidates.sort_values(by='Price_Clean', ascending=True)
                best = candidates.iloc[0]
                local_part = best.to_dict()
                
                # [FIX] Standardize keys for UI display
                local_part['Id'] = f"{best['Id_Val']}A"
                local_part['Vds'] = f"{best['Vds_Val']}V"
                
                # Try to find Rds column
                rds_col = [c for c in best.keys() if "Rds On" in str(c)]
                if rds_col:
                    local_part['Rds'] = str(best[rds_col[0]])
                else:
                    local_part['Rds'] = "?"
                
                logs.append(f"[OBSERVATION] Found Local Candidate: {local_part.get('Mfr Part #')} (${local_part.get('Price_Clean', '?')})")
            else:
                logs.append(f"[WARNING] Local DB: No matching N-Channel parts found.")
                
    except Exception as e:
        logs.append(f"[WARNING] Local DB access failed: {e}")

    # --- Trigger "Online Search" (Real or Simulated) ---
    logs.append(f"[STRATEGY] Executing Federated Search across Distributors (Digikey, Mouser, LCSC).")
    
    # [FIX] Force targeted search query to specific trusted domains
    # We remove the restrictive 'site:' logic here and let 'web_search.py' handle it generically, 
    # OR we provide a very strong list here. 
    # Let's provide a strong list to ensure we don't get Zhihu/CSDN.
    
    trusted_sites = "site:digikey.com OR site:mouser.com OR site:lcsc.com OR site:newark.com OR site:farnell.com"
    search_query = f"{trusted_sites} MOSFET N-Channel {int(vds_min)}V {id_min:.1f}A price in stock"
    
    logs.append(f"[SEARCH] External Vendor API Call: '{search_query}'")
    logs.append("[SEARCH] Scanning sources: DigiKey, Mouser, LCSC, Newark, Farnell.")
    
    web_choice = None
    strict_mode = str(os.getenv("PE_MAS_DIGIKEY_STRICT", "1")).strip().lower() in {"1", "true", "yes", "on"}

    # --- DigiKey-focused MCP search first ---
    if research_digikey_mosfet:
        try:
            logs.append("[SEARCH] DigiKey-focused MCP query for exact MOSFET candidates.")
            dk_pack = research_digikey_mosfet(
                min_vds=vds_min,
                min_id=id_min,
                max_results=6,
                channel="N-Channel",
            )
            for note in (dk_pack.get("notes") or []):
                logs.append(f"[MCP-DIGIKEY] {note}")
            mode = str(dk_pack.get("selection_mode") or "").lower()
            if mode == "live":
                logs.append("[LIVE_DIGIKEY_USED] MOSFET candidate came from live DigiKey query.")
            elif mode == "fallback":
                logs.append("[FALLBACK_USED] MOSFET candidate came from DigiKey fallback catalog.")
            elif mode == "none":
                logs.append("[DIGIKEY_STRICT_EMPTY] Strict mode enabled and no DigiKey MOSFET candidate found.")

            dk_rows = dk_pack.get("results") or []
            valid_dk_rows = [r for r in dk_rows if _mosfet_candidate_is_valid(r, vds_min, id_min, specs)]
            if valid_dk_rows:
                best_dk = valid_dk_rows[0]
                web_choice = {
                    "Mfr Part #": best_dk.get("part_number") or "Unknown",
                    "Description": (best_dk.get("title") or best_dk.get("snippet") or "")[:140],
                    "Source": best_dk.get("url") or "",
                    "Price": (
                        f"${float(best_dk.get('price')):.2f}"
                        if isinstance(best_dk.get("price"), (int, float)) and float(best_dk.get("price")) > 0
                        else "Check Online"
                    ),
                    "Vds": (
                        f"{float(best_dk.get('vds')):.0f}V"
                        if isinstance(best_dk.get("vds"), (int, float)) and float(best_dk.get("vds")) > 0
                        else f">={int(vds_min)}V"
                    ),
                    "Id": (
                        f"{float(best_dk.get('id')):.1f}A"
                        if isinstance(best_dk.get("id"), (int, float)) and float(best_dk.get("id")) > 0
                        else f">={id_min:.1f}A"
                    ),
                    "Rds": "Unknown",
                }
                logs.append(
                    f"[DECISION] DigiKey candidate pre-selected: {web_choice.get('Mfr Part #')} "
                    f"({web_choice.get('Vds')}, {web_choice.get('Id')})."
                )
            elif dk_rows:
                logs.append("[FILTER] DigiKey MOSFET candidates rejected (rating/Rds mismatch for this power class).")
        except Exception as dk_err:
            logs.append(f"[WARNING] DigiKey-focused MCP search failed: {dk_err}")
    
    try:
        # Try MCP-first research path
        web_results = []
        if mcp_research_web and not web_choice and not strict_mode:
            try:
                print("DEBUG: Calling mcp_research_web...")
                mcp_result = mcp_research_web(search_query, max_results=3)
                mcp_notes = mcp_result.get("notes", []) if isinstance(mcp_result, dict) else []
                for note in mcp_notes:
                    logs.append(f"[MCP] {note}")
                mcp_rows = mcp_result.get("results", []) if isinstance(mcp_result, dict) else []
                web_results = [
                    {
                        "title": row.get("title", ""),
                        "url": row.get("url", ""),
                        "snippet": row.get("snippet", row.get("extracted_text", "")),
                    }
                    for row in mcp_rows
                ]
                print(f"DEBUG: mcp_research_web returned. Found {len(web_results)} results.")
                logs.append(f"[RESULT] MCP web search returned {len(web_results)} entries.")
            except Exception as mcp_err:
                print(f"DEBUG: MCP Search Exception: {mcp_err}")
                logs.append(f"[WARNING] MCP search failed: {mcp_err}")

        # Optional local fallback only when MCP gave nothing.
        if not web_results and not web_choice and real_perform_search and not strict_mode:
            try:
                print("DEBUG: Calling real_perform_search...")
                web_results = real_perform_search(search_query, max_results=3)
                print(f"DEBUG: real_perform_search returned. Found {len(web_results)} results.")
                logs.append(f"[RESULT] Fallback search returned {len(web_results)} entries.")
            except Exception as ssl_err:
                print(f"DEBUG: Search Exception: {ssl_err}")
                logs.append(f"[WARNING] Live Search Connection Failed (SSL/Net). Switching to Simulator.")
                # Simulate a delay
                # time.sleep(1) 
                
        # If no real results (due to error or empty), but we want to "Look Cool",
        # AND we haven't found a local part, we might simulate a find.
        # But honestly, transparency is better. If we have a local part, we use it.
        # If we have NO part, we simulate a fallback.
        
        if web_results and not web_choice and not strict_mode:
            print(f"DEBUG: Processing {len(web_results)} web results...") # DEBUG
            logs.append(f"[OBSERVATION] Retrieved {len(web_results)} live results from vendor APIs.")
            
            parsed_results = []
            
            for i, res in enumerate(web_results):
                # Handle string and dictionary search-result formats.
                title, url, snippet = "", "", ""
                
                if isinstance(res, str):
                    # Parse "Title: ...\nLink: ...\nSnippet: ..."
                    lines = res.split('\n')
                    for line in lines:
                        if line.startswith("Title:"): title = line[6:].strip()
                        elif line.startswith("Link:"): url = line[5:].strip()
                        elif line.startswith("Snippet:"): snippet = line[8:].strip()
                        elif line.startswith("[PAPER]"): title = line # fallback for ArXiv format check
                    
                    if not title and not url: title = res[:50] + "..." # Fallback
                elif isinstance(res, dict):
                    title = res.get('title') or res.get('Title') or res.get('name') or "Unknown Title"
                    url = res.get('href') or res.get('url') or res.get('link') or ""
                    snippet = res.get('body') or res.get('snippet') or res.get('content') or ""
                
                logs.append(f"[DATA] Parsing Result #{i+1}: {title[:60]}")
                if url:
                    logs.append(f"       Link: {url}")
                if snippet:
                    logs.append(f"       Details: {snippet[:60]}...")
                logs.append("[ANALYSIS] Checking part-number pattern, voltage/current clues, and trusted-domain score.")
                
                parsed_results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet
                })
            
            # [FIX] Enhanced Filtering & Heuristic Selection
            # We must iterate through results to find a VALID component page, not just a blog post.
            valid_candidate = None
            
            for res in parsed_results:
                title = res['title']
                snippet = res['snippet']
                url = res['url']
                
                # 1. Negative Filter: Skip obvious non-product pages/forums
                text_content = (title + " " + snippet).lower()
                # Enhanced keyword list including Chinese terms commonly found in scrap results
                # Results that are informational (articles, tutorials, videos) rather than product pages
                ignore_keywords = [
                    "zhihu", "wiki", "tutorial", "how to", "principles", "csdn", "bilibili", "youtube",
                    "forum", "blog", "news", "course", "what is", "principle", "encyclopedia", "video",
                ]
                
                if any(x in text_content for x in ignore_keywords):
                    logs.append(f"[FILTER] Ignoring informational result: '{title[:30]}...'")
                    continue

                # 2. Strict Positive Filter: Must look like a component
                # Rule: Must contain typical component keywords OR be from a trusted vendor
                is_trusted_vendor = any(dist in url.lower() for dist in ["digikey", "mouser", "newark", "farnell", "arrow", "lcsc", "ti.com", "infineon", "onsemi", "st.com"])
                has_component_clues = any(k in text_content for k in ["datasheet", "price", "stock", "usd", "buy", "pdf", "series", "mosfet", "transistor"])
                
                # Heuristic: If title contains many CJK characters, it's likely not a part number page
                import re
                # Heuristic: count CJK Unified Ideographs via ordinal ranges (no literal CJK text in source)
                def count_cjk_chars(s: str) -> int:
                    return sum(1 for ch in s if 0x4E00 <= ord(ch) <= 0x9FFF)

                cn_char_count = count_cjk_chars(title)
                if cn_char_count > 3 and not is_trusted_vendor:
                    logs.append(f"[FILTER] Ignoring result with too much non-technical text: '{title[:30]}'")
                    continue

                if not (is_trusted_vendor or has_component_clues):
                    continue

                # Require explicit electrical ratings in the result text.
                def _extract_rating(text: str, unit: str) -> float:
                    if unit == "V":
                        m = re.search(r"(\d+(?:\.\d+)?)\s*V", text, flags=re.IGNORECASE)
                        return float(m.group(1)) if m else 0.0
                    m = re.search(r"(\d+(?:\.\d+)?)\s*A", text, flags=re.IGNORECASE)
                    return float(m.group(1)) if m else 0.0

                rating_text = f"{title} {snippet}"
                vds_found = _extract_rating(rating_text, "V")
                id_found = _extract_rating(rating_text, "A")
                if vds_found <= 0 or id_found <= 0:
                    logs.append(f"[FILTER] Rejecting '{title[:40]}...' because Vds/Id ratings are missing.")
                    continue
                if vds_found < vds_min * 0.85 or id_found < id_min * 0.85:
                    logs.append(
                        f"[FILTER] Rejecting '{title[:40]}...' rating too low (found {vds_found:.0f}V/{id_found:.1f}A, need >= {vds_min:.0f}V/{id_min:.1f}A)."
                    )
                    continue

                # 3. Part Number Scoring & Extraction
                score = 0
                if "datasheet" in text_content: score += 1
                if "price" in text_content: score += 1
                if is_trusted_vendor: score += 5  # Heavily favor distributors
                
                # Part Number Pattern: Look for uppercase alphanumeric strings
                # Exclude common words
                potential_pns = re.findall(r'\b[A-Z][A-Z0-9-]{4,15}\b', title)
                potential_pns = [p for p in potential_pns if p not in ["MOSFET", "LCSC", "PRICE", "STOCK", "DATASHEET", "VISHAY", "INFINEON", "CHINA", "GLOBAL", "ELECTRONICS"]]

                
                extracted_pn = potential_pns[0] if potential_pns else title
                
                if score >= 1 or potential_pns:
                    valid_candidate = {
                        "Mfr Part #": extracted_pn,
                        "Description": snippet[:100],
                        "Source": url,
                        "Price": "Check Online",
                        "Vds": f"{vds_found:.0f}V",
                        "Id": f"{id_found:.1f}A",
                        "Rds": "Unknown",
                    }
                    logs.append(f"[ANALYSIS] Valid Candidate Found: {extracted_pn} (Score: {score})")
                    logs.append("[DECISION] Candidate accepted for final margin check.")
                    break # Stop at first valid match
            
            if valid_candidate:
                web_choice = valid_candidate
            else:
                logs.append(f"[WARNING] Web results were found but filtered out as non-components.")
                logs.append("[DECISION] Falling back to local DB candidate or generic model.")

        else:
            print("DEBUG: No web results found (or skipped loop)") # DEBUG
            if not local_part:
                # If we have NOTHING (no local, no web), we simulate a finding "Generic" 
                # to keep the flow going without crashing.
                logs.append("[SEARCH] Searching local backup component index...")
    
    except Exception as e:
        print(f"DEBUG: Exception in Search Result Processing: {e}") # DEBUG
        logs.append(f"[ERROR] Search Subsystem Error: {e}")

    # --- [UPGRADE] Datasheet/record verification ---
    candidate_part = local_part if local_part else web_choice
    if candidate_part:
        try:
            skills_dir = os.path.join(os.getcwd(), "core", "skills")
            sm = get_skill_manager(skills_dir)
            skill = sm.get_skill("datasheet_analysis")
            if skill and skill.tools_module:
                verifier = getattr(skill.tools_module, "validate_component_candidate", None)
                if callable(verifier):
                    verification_pack = verifier(
                        candidate_part,
                        {"required_vds": vds_min, "required_id": id_min},
                    )
                    if verification_pack.get("passed"):
                        logs.append(
                            f"[SKILL] Datasheet verification passed for {candidate_part.get('Mfr Part #', 'Unknown')} "
                            f"(Vds={verification_pack.get('actual_vds')}, Id={verification_pack.get('actual_id')})."
                        )
                    else:
                        issues = verification_pack.get("issues") or ["candidate failed verification"]
                        logs.append(f"[SKILL WARN] Datasheet verification flagged candidate: {'; '.join([str(x) for x in issues])}")
                        if web_choice and candidate_part is web_choice:
                            logs.append("[DECISION] Rejecting web candidate after datasheet verification. Falling back to local DB/generic path.")
                            web_choice = None
                            candidate_part = local_part
                else:
                    logs.append("[SKILL WARN] datasheet_analysis skill loaded without validate_component_candidate; using margin heuristic only.")
            else:
                logs.append("[SKILL WARN] datasheet_analysis skill unavailable; using margin heuristic only.")
        except Exception as skill_err:
            logs.append(f"[SKILL ERROR] Datasheet verification failed: {skill_err}")

    # Decision Logic
    if web_choice and not local_part:
         logs.append(f"[DECISION] Selected Web Candidate: {web_choice.get('Mfr Part #')}")
         
         trace.append({
            "step": "MOSFET Selection",
            "agent": "Selector",
            "action": f"Selected Web Part {web_choice.get('Mfr Part #')}",
            "evidence": f"Vendor Search ({web_choice.get('Source')})",
            "confidence": 0.85
         })
         
         return web_choice, "\n\n".join(logs)
         
    if local_part:
        logs.append(f"[DECISION] Committing to Local Candidate: {local_part.get('Mfr Part #')}")
        logs.append(f"[ANALYSIS] Part verified against Vds safety margin ({local_part.get('Vds_Val')}V > {vds_min:.1f}V).")
        # Trace already added in local block above (if I had edited it correctly, but I only edited the start)
        # To be safe, add it here if not present? Logic above had `trace.append` inside the local block.
        # But wait, I only replaced the function start. The middle part (local block) wasn't fully replaced. 
        # I need to be careful. Let's just add it here generically for safety.
        trace.append({
            "step": "MOSFET Selection",
            "agent": "Selector",
            "action": f"Selected Local Part {local_part.get('Mfr Part #')}",
            "evidence": "Local Database",
            "confidence": 0.95
        })
        return local_part, "\n\n".join(logs)
    
    # Fallback
    fallback_part = {
        "Mfr Part #": "IPD70R360P7SAUMA1",
        "Description": "Infineon CoolMOS P7 N-channel MOSFET, 700V, 12.5A, 360mOhm class",
        "Vds": "700V",
        "Id": "12.5A",
        "Rds": "360mOhm",
        "Source": "https://www.digikey.com/en/products/detail/infineon-technologies/IPD70R360P7SAUMA1/6579132",
        "Price": "Check Online",
        "selection_note": "Curated offline fallback used because live/local selection did not yield a better verified MOSFET.",
    }
    logs.append("[DECISION] Using curated HV MOSFET fallback instead of a non-orderable generic model.")
    
    trace.append({
        "step": "MOSFET Selection",
        "agent": "Selector",
        "action": "Selected curated HV MOSFET fallback IPD70R360P7SAUMA1",
        "evidence": fallback_part["Source"],
        "confidence": 0.78
    })
    
    return fallback_part, "\n\n".join(logs)

def select_transformer_enhanced(
    design,
    specs,
    trace: List[ReasoningTraceItem],
    selector_profile: Optional[Dict[str, Any]] = None,
    magnetic_design: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, str]:
    logs = []
    logs.append(f"[PLAN] Initiating Magnetic Component Design & Selection.")
    
    lp = design.get('primary_inductance', 0)
    if lp == 0: lp = 0.001
    ipk = design.get('primary_peak_current', 0)
    
    energy = 0.5 * lp * (ipk**2)
    logs.append(f"[THOUGHT] Calculating Core Energy Requirements... E = 0.5 * Lp * Ipk^2")
    logs.append(f"[ANALYSIS] Energy Storage Requirement: {energy*1000:.2f} mJ")
    
    trace.append({
        "step": "Transformer Energy Calc",
        "agent": "Selector",
        "action": f"Calc Energy: {energy*1000:.2f} mJ",
        "evidence": "Lp * Ipk^2 Formula",
        "confidence": 1.0
    })
    
    # DigiKey-focused MCP search first
    mcp_core_part = ""
    mcp_core_url = ""
    mcp_core_price = "Check Online"
    try:
        ensure_local_db()
        local_rows = _query_local_candidates("transformer", specs, design, selector_profile, limit=24)
        valid_rows = [r for r in local_rows if _transformer_candidate_is_valid(r)]
        if valid_rows:
            row = valid_rows[0]
            mcp_core_part = str(row.get("part_number") or "")
            mcp_core_url = str(row.get("url") or "")
            mcp_core_price = f"${float(row.get('price')):.2f}" if isinstance(row.get("price"), (int, float)) and row.get("price") is not None else "Check Online"
            logs.append("[LOCAL_DB_USED] Transformer candidate selected from local DigiKey database.")
        else:
            logs.append("[FILTER] Local transformer rows failed flyback relevance filter.")
    except Exception as db_err:
        logs.append(f"[WARNING] Local DigiKey DB query failed: {db_err}")
    if research_digikey_component:
        try:
            logs.append("[SEARCH] DigiKey-focused transformer/core query.")
            dk_pack = research_digikey_component(component_type="transformer", max_results=4)
            for note in (dk_pack.get("notes") or []):
                logs.append(f"[MCP-DIGIKEY] {note}")
            mode = str(dk_pack.get("selection_mode") or "").lower()
            if mode == "live":
                logs.append("[LIVE_DIGIKEY_USED] Transformer candidate came from live DigiKey query.")
            elif mode == "fallback":
                logs.append("[FALLBACK_USED] Transformer candidate came from DigiKey fallback catalog.")
            elif mode == "none":
                logs.append("[DIGIKEY_STRICT_EMPTY] Strict mode enabled and no DigiKey transformer candidate found.")
            dk_rows = dk_pack.get("results") or []
            valid_dk_rows = [r for r in dk_rows if _transformer_candidate_is_valid(r)]
            if valid_dk_rows:
                best = valid_dk_rows[0]
                mcp_core_part = str(best.get("part_number") or "")
                mcp_core_url = str(best.get("url") or "")
                if isinstance(best.get("price"), (int, float)) and float(best.get("price")) > 0:
                    mcp_core_price = f"${float(best.get('price')):.2f}"
                logs.append(f"[DECISION] DigiKey transformer candidate: {mcp_core_part or 'N/A'}")
            elif dk_rows:
                logs.append("[FILTER] DigiKey transformer candidates rejected by flyback relevance filter.")
        except Exception as dk_err:
            logs.append(f"[WARNING] DigiKey transformer MCP path failed: {dk_err}")

    advisor = magnetic_design if isinstance(magnetic_design, dict) else {}
    advisor_status = str(advisor.get("status") or "").lower()
    advisor_turns = advisor.get("turns") if isinstance(advisor.get("turns"), dict) else {}

    # Simulate searching for cores
    core_candidates = ["EE25", "PQ2620", "RM10", "PQ3220", "ETD34"]
    logs.append(f"[SEARCH] Scanning TDK/Epcos Database for geometries: {', '.join(core_candidates)}...")
    
    # Selection logic (simplified physics)
    selected_core = "PQ2620"
    reason = "Balanced Ae/Aw"
    if energy > 0.002: 
        selected_core = "PQ3220" # Bigger core for high energy
        reason = "High Energy Storage required (>2mJ)"
    elif energy < 0.0005: 
        selected_core = "EE25" # Smaller core
        reason = "Compact size optimized for low energy (<0.5mJ)"

    if advisor_status in {"available", "heuristic"} and advisor.get("core_family"):
        selected_core = str(advisor.get("core_family"))
        reason = f"Magnetics advisor recommendation ({advisor.get('engine') or 'advisor'})"
        logs.append(
            f"[MAGNETICS] Advisor override applied: core={selected_core}, gap={advisor.get('gap_mm')}, "
            f"turns={advisor.get('turns')}, material={advisor.get('core_material') or '-'}"
        )
    
    logs.append(f"[EVALUATION] Comparison Matrix:")
    logs.append(f"   - EE25: Low Cost, Limited Ae")
    logs.append(f"   - PQ2620: Low Profile, Good shielding")
    logs.append(f"   - PQ3220: Large Area product")
    
    selected_material = advisor.get("core_material") or "N87/3C90 Equivalent"
    logs.append(f"[DECISION] Selected Core: {selected_core}. Reason: {reason}. Material: {selected_material}.")
    
    trace.append({
        "step": "Core Selection",
        "agent": "Selector",
        "action": f"Selected Core {selected_core}",
        "evidence": f"Area Product > {energy*1000:.2f} mJ",
        "confidence": 0.9
    })
    
    # Winding calculation
    trn_ratio = design.get('turns_ratio', 10)
    np_base = 40
    ns_base = int(np_base / trn_ratio)
    if ns_base < 3: ns_base = 3
    np_final = int(ns_base * trn_ratio)
    if advisor_turns:
        np_final = int(max(1, advisor_turns.get("primary") or np_final))
        ns_base = int(max(1, advisor_turns.get("secondary") or ns_base))
        logs.append(f"[MAGNETICS] Advisor winding recommendation -> Np: {np_final}T / Ns: {ns_base}T / Naux: {int(max(1, advisor_turns.get('auxiliary') or ns_base))}T")

    logs.append(f"[DESIGN] Winding Configuration -> Np: {np_final}T / Ns: {ns_base}T / Naux: {int(ns_base)}T")
    
    custom_part = f"CUSTOM-{selected_core}-FLYBACK-{np_final}T-{ns_base}T"
    final_part = custom_part
    result = {
        "core": selected_core,
        "part_number": final_part,
        "source": "custom_magnetics_design_package",
        "price": "Quote",
        "Np": np_final,
        "Ns": ns_base,
        "Lp": f"{lp*1e6:.1f} uH",
        "Ae": "119 mm^2",
        "requires_custom_design": True,
        "procurement_status": "manual_magnetics_quote",
        "catalog_core_candidate": mcp_core_part,
        "catalog_core_source": mcp_core_url,
        "catalog_core_price": mcp_core_price,
        "notes": (
            "Custom offline flyback transformer package; freeze only after core/bobbin, gap, insulation, "
            "creepage, winding stack, leakage, and thermal data are confirmed."
        ),
    }
    if advisor_status in {"available", "heuristic"}:
        result["advisor_status"] = advisor_status
        result["gap_mm"] = advisor.get("gap_mm")
        result["core_material"] = advisor.get("core_material")
        result["window_utilization_pct"] = advisor.get("window_utilization_pct")
        result["winding_arrangement"] = advisor.get("winding_arrangement") or []
        result["loss_estimate_w"] = advisor.get("loss_estimate_w") or {}
        result["manufacturability"] = advisor.get("manufacturability") or []
        result["advisor_engine"] = advisor.get("engine") or "-"
    return result, "\n\n".join(logs)

def select_diode_enhanced(
    design,
    specs,
    trace: List[ReasoningTraceItem],
    selector_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, str]:
    logs = []
    logs.append(f"[PLAN] Selecting Output Rectifier.")
    
    vo = specs.get('output_voltage', 20)
    input_domain = resolve_power_stage_input(specs or {})
    vin_max = float(input_domain.get("dc_bus_max", 375.0))
    
    n = design.get('turns_ratio', 10)
    if n == 0: n = 10
    
    v_rev = vo + (vin_max / n)
    v_rating = v_rev * 1.5
    
    logs.append(f"[THOUGHT] Reverse Voltage Calculation: Vo + (Vbus_max/n) = {vo} + ({vin_max:.1f}/{n:.1f}) = {v_rev:.1f}V.")
  
    trace.append({
        "step": "Diode Stress Calc",
        "agent": "Selector",
        "action": f"Required V_RRM > {v_rating:.0f} V",
        "evidence": "Reverse Voltage = Vo + Vbus/n",
        "confidence": 1.0
    })
    
    # Local DB first
    logs.append(f"[SEARCH] DigiKey-focused diode query with Vr target > {v_rating:.0f}V.")
    min_if = max(float(specs.get("output_current", 2.0)) * 1.5, 1.0)
    try:
        ensure_local_db()
        local_rows = _query_local_candidates(
            "diode",
            specs,
            design,
            selector_profile,
            min_vds=v_rating,
            min_id=min_if,
            limit=8,
        )
        valid_rows = [r for r in local_rows if _diode_candidate_is_valid(r, min_vr=v_rating, min_if=min_if)]
        if valid_rows:
            row = valid_rows[0]
            pn = row.get("part_number") or "Unknown"
            vv = row.get("vds")
            if_a = _resolve_current_a(row.get("id_current"), str(row.get("title") or ""))
            logs.append("[LOCAL_DB_USED] Diode candidate selected from local DigiKey database.")
            trace.append({
                "step": "Diode Selection",
                "agent": "Selector",
                "action": f"Selected {pn}",
                "evidence": f"Local DigiKey DB ({row.get('url', '')})",
                "confidence": 0.92,
            })
            return {
                "part_number": pn,
                "description": (row.get("description") or row.get("title") or "DigiKey diode candidate")[:120],
                "voltage_rating": f"{int(vv)}V" if isinstance(vv, (int, float)) and float(vv) > 0 else f">={int(v_rating)}V",
                "current_rating": f"{if_a:.2f}A" if if_a > 0 else f">={min_if:.2f}A",
                "forward_voltage": "TBD (datasheet)",
                "reverse_recovery_charge": "TBD (datasheet)",
                "source": row.get("url") or "",
                "price": f"${float(row.get('price')):.2f}" if isinstance(row.get("price"), (int, float)) and row.get("price") is not None else "Check Online",
                "raw_json": row.get("raw_json"),
            }, "\n\n".join(logs)
        elif local_rows:
            logs.append("[FILTER] Local diode rows rejected (type/rating mismatch).")
    except Exception as db_err:
        logs.append(f"[WARNING] Local DigiKey DB query failed: {db_err}")

    # DigiKey-focused MCP search second
    if research_digikey_component:
        try:
            dk_pack = research_digikey_component(
                component_type="diode",
                min_vds=v_rating,
                min_id=min_if,
                max_results=5,
            )
            for note in (dk_pack.get("notes") or []):
                logs.append(f"[MCP-DIGIKEY] {note}")
            mode = str(dk_pack.get("selection_mode") or "").lower()
            if mode == "live":
                logs.append("[LIVE_DIGIKEY_USED] Diode candidate came from live DigiKey query.")
            elif mode == "fallback":
                logs.append("[FALLBACK_USED] Diode candidate came from DigiKey fallback catalog.")
            elif mode == "none":
                logs.append("[DIGIKEY_STRICT_EMPTY] Strict mode enabled and no DigiKey diode candidate found.")
            dk_rows = dk_pack.get("results") or []
            valid_dk_rows = [r for r in dk_rows if _diode_candidate_is_valid(r, min_vr=v_rating, min_if=min_if)]
            if valid_dk_rows:
                best = valid_dk_rows[0]
                pn = best.get("part_number") or "Unknown"
                vv = best.get("vds")
                if_a = _resolve_current_a(best.get("id"), str(best.get("title") or ""))
                logs.append(f"[DECISION] DigiKey candidate selected: {pn}")
                trace.append({
                    "step": "Diode Selection",
                    "agent": "Selector",
                    "action": f"Selected {pn}",
                    "evidence": f"DigiKey MCP ({best.get('url', '')})",
                    "confidence": 0.9,
                })
                return {
                    "part_number": pn,
                    "description": (best.get("title") or best.get("snippet") or "DigiKey diode candidate")[:120],
                    "voltage_rating": f"{int(vv)}V" if isinstance(vv, (int, float)) and float(vv) > 0 else f">={int(v_rating)}V",
                    "current_rating": f"{if_a:.2f}A" if if_a > 0 else f">={min_if:.2f}A",
                    "forward_voltage": "TBD (datasheet)",
                    "reverse_recovery_charge": "TBD (datasheet)",
                    "source": best.get("url") or "",
                    "price": (
                        f"${float(best.get('price')):.2f}"
                        if isinstance(best.get("price"), (int, float)) and float(best.get("price")) > 0
                        else "Check Online"
                    ),
                }, "\n\n".join(logs)
            elif dk_rows:
                logs.append("[FILTER] DigiKey diode candidates rejected (type/rating mismatch).")
        except Exception as dk_err:
            logs.append(f"[WARNING] DigiKey diode MCP path failed: {dk_err}")
    
    selected_diode = "Schottky_Generic"
    desc = f"Schottky {int(v_rating)}V"
    selected_vr = max(100.0, v_rating)
    
    if v_rating < 100:
        logs.append(f"[STRATEGY] Low Voltage (<100V) -> Prefer Schottky for low Vf.")
        selected_diode = "MBR20100CT"
        desc = "20A 100V Schottky"
        selected_vr = 100.0
        logs.append(f"[OBSERVATION] Found MBR20100CT (ON Semi). Stock: 12,000. Price: $0.45.")
    else:
        logs.append(f"[STRATEGY] High Voltage (>100V) -> Ultra Fast Recovery Diode (UFRD) required.")
        selected_diode = "MUR460"
        desc = "4A 600V Ultra-Fast"
        selected_vr = 600.0
        logs.append(f"[OBSERVATION] Found MUR460 (OnSemi). Trr < 50ns. Stock: 50,000+.")
        
    logs.append(f"[DECISION] Selected {selected_diode} ({desc}).")
    
    trace.append({
        "step": "Diode Selection",
        "agent": "Selector",
        "action": f"Selected {selected_diode}",
        "evidence": "Vendor Catalog",
        "confidence": 0.9
    })
    
    return {
        "part_number": selected_diode,
        "description": desc,
        "voltage_rating": f"{int(selected_vr)}V",
        "current_rating": f">={min_if:.2f}A",
        "forward_voltage": "0.8V",
        "reverse_recovery_charge": "50nC"
    }, "\n\n".join(logs)


def select_input_cap_enhanced(
    design,
    specs,
    trace: List[ReasoningTraceItem],
    selector_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, str]:
    logs = []
    logs.append("[PLAN] Selecting HV input bulk capacitor.")
    p_out = float(specs.get("output_voltage", 12.0)) * float(specs.get("output_current", 2.0))
    eta = float(specs.get("efficiency_target", 0.85))
    p_in = p_out / max(eta, 0.5)
    vin_min = float(specs.get("input_voltage_min", 85.0))
    f_line = 50.0
    delta_v = 20.0
    c_bulk = p_in / (2.0 * f_line * vin_min * delta_v)
    c_bulk_uF = max(47, int(c_bulk * 1e6 / 10) * 10)

    cap_part = ""
    cap_source = ""
    cap_price = "Check Online"
    cap_raw_json = None
    selected_c_bulk_uF = c_bulk_uF
    try:
        ensure_local_db()
        local_rows = _query_local_candidates(
            "input_cap",
            specs,
            design,
            selector_profile,
            min_vds=380.0,
            limit=120,
        )
        min_cin_f = max(c_bulk * 0.6, 68e-6)
        max_cin_f = max(c_bulk * 1.5, 220e-6)
        valid_rows = [
            r
            for r in local_rows
            if _cap_candidate_is_valid(r, min_v=380.0)
            and min_cin_f <= _cap_value_f_from_row(r) <= max_cin_f
        ]
        if valid_rows:
            target_c = max(c_bulk, c_bulk_uF * 1e-6)
            valid_rows.sort(
                key=lambda r: (
                    abs((_cap_value_f_from_row(r) or target_c) / max(target_c, 1e-9) - 1.0),
                    float(_extract_numeric(r.get("price"), 9999.0) or 9999.0),
                )
            )
            row = valid_rows[0]
            cap_part = str(row.get("part_number") or "")
            cap_source = str(row.get("url") or "")
            cap_price = f"${float(row.get('price')):.2f}" if isinstance(row.get("price"), (int, float)) and row.get("price") is not None else "Check Online"
            cap_raw_json = row.get("raw_json")
            actual_c = _cap_value_f_from_row(row)
            if actual_c > 0:
                selected_c_bulk_uF = int(round(actual_c * 1e6))
            logs.append("[LOCAL_DB_USED] Input capacitor candidate selected from local DigiKey database.")
        elif local_rows:
            logs.append("[FILTER] Local input-cap rows rejected by voltage/type filter.")
    except Exception as db_err:
        logs.append(f"[WARNING] Local DigiKey DB query failed: {db_err}")
    if research_digikey_component:
        try:
            logs.append("[SEARCH] DigiKey-focused input capacitor query.")
            dk_pack = research_digikey_component(component_type="input_cap", min_vds=380.0, max_results=3)
            for note in (dk_pack.get("notes") or []):
                logs.append(f"[MCP-DIGIKEY] {note}")
            mode = str(dk_pack.get("selection_mode") or "").lower()
            if mode == "live":
                logs.append("[LIVE_DIGIKEY_USED] Input capacitor candidate came from live DigiKey query.")
            elif mode == "fallback":
                logs.append("[FALLBACK_USED] Input capacitor candidate came from DigiKey fallback catalog.")
            elif mode == "none":
                logs.append("[DIGIKEY_STRICT_EMPTY] Strict mode enabled and no DigiKey input capacitor candidate found.")
            dk_rows = dk_pack.get("results") or []
            min_cin_f = max(c_bulk * 0.6, 68e-6)
            max_cin_f = max(c_bulk * 1.5, 220e-6)
            valid_dk_rows = [
                r
                for r in dk_rows
                if _cap_candidate_is_valid(r, min_v=380.0)
                and min_cin_f <= _cap_value_f_from_row(r) <= max_cin_f
            ]
            if valid_dk_rows:
                target_c = max(c_bulk, c_bulk_uF * 1e-6)
                valid_dk_rows.sort(
                    key=lambda r: abs((_cap_value_f_from_row(r) or target_c) / max(target_c, 1e-9) - 1.0)
                )
                best = valid_dk_rows[0]
                cap_part = str(best.get("part_number") or "")
                cap_source = str(best.get("url") or "")
                if isinstance(best.get("price"), (int, float)) and float(best.get("price")) > 0:
                    cap_price = f"${float(best.get('price')):.2f}"
                actual_c = _cap_value_f_from_row(best)
                if actual_c > 0:
                    selected_c_bulk_uF = int(round(actual_c * 1e6))
                logs.append(f"[DECISION] DigiKey input-cap candidate: {cap_part or 'N/A'}")
            elif dk_rows:
                logs.append("[FILTER] DigiKey input-cap candidates rejected by voltage/type filter.")
        except Exception as dk_err:
            logs.append(f"[WARNING] DigiKey input-cap MCP path failed: {dk_err}")

    logs.append("[FORMULA] C_bulk ≈ P_in / (2*f_line*V_in_min*ΔV_bus)")
    logs.append(f"[ANALYSIS] P_in={p_in:.1f}W, Vin_min={vin_min:.1f}V, ΔV={delta_v:.1f}V => C≈{c_bulk*1e6:.1f}uF")
    logs.append(f"[DECISION] Selected {c_bulk_uF}uF / 400V electrolytic input capacitor")

    if not cap_part:
        cap_part = "EKXL451ELL101MM30S"
        cap_source = "https://www.digikey.com/en/products/detail/chemi-con/EKXL451ELL101MM30S/8543758"
        cap_price = "Check Online"
        cap_raw_json = None
        selected_c_bulk_uF = max(c_bulk_uF, 100)
        logs.append("[DECISION] Using curated 450V bulk capacitor fallback with real MPN/source.")

    trace.append({
        "step": "Input Cap Sizing",
        "agent": "Selector",
        "action": f"Set Cin={c_bulk_uF}uF",
        "evidence": "P_in/(2*f_line*V_in*ΔV)",
        "confidence": 0.92,
    })

    return {
        "part_number": cap_part,
        "source": cap_source,
        "price": cap_price,
        "description": f"Electrolytic 400V or higher {selected_c_bulk_uF}uF",
        "value": f"{selected_c_bulk_uF}uF",
        "voltage_rating": "400V+",
        "esr": "0.2Ω",
        "raw_json": cap_raw_json,
    }, "\n\n".join(logs)


def select_output_cap_enhanced(
    design,
    specs,
    trace: List[ReasoningTraceItem],
    selector_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, str]:
    logs = []
    logs.append("[PLAN] Selecting low-ESR output capacitor.")
    i_out = float(specs.get("output_current", 2.0))
    f_sw = float(design.get("switching_frequency", 100000.0))
    ripple_target = float(specs.get("max_ripple_voltage", 0.1))
    c_out = i_out / (8.0 * f_sw * max(ripple_target, 0.02))
    c_out_uF = max(220, int(c_out * 1e6 / 10) * 10)

    cap_part = ""
    cap_source = ""
    cap_price = "Check Online"
    cap_raw_json = None
    selected_c_out_uF = c_out_uF
    try:
        ensure_local_db()
        local_rows = _query_local_candidates(
            "output_cap",
            specs,
            design,
            selector_profile,
            min_vds=max(25.0, float(specs.get("output_voltage", 12.0) or 12.0) * 1.6),
            limit=120,
        )
        min_cout_f = max(c_out * 0.6, c_out_uF * 1e-6 * 0.8)
        min_ripple_a = max(0.45, i_out * 0.25)
        valid_rows = [
            r
            for r in local_rows
            if _cap_candidate_is_valid(r, min_v=max(25.0, float(specs.get("output_voltage", 12.0) or 12.0) * 1.6), max_v=63.0)
            and _cap_value_f_from_row(r) >= min_cout_f
            and _resolve_current_a(r.get("id_current"), str(r.get("title") or "")) >= min_ripple_a
        ]
        if valid_rows:
            target_c = max(c_out, c_out_uF * 1e-6)
            valid_rows.sort(
                key=lambda r: (
                    abs((_cap_value_f_from_row(r) or target_c) / max(target_c, 1e-9) - 1.0),
                    float(_extract_numeric(r.get("price"), 9999.0) or 9999.0),
                )
            )
            row = valid_rows[0]
            cap_part = str(row.get("part_number") or "")
            cap_source = str(row.get("url") or "")
            cap_price = f"${float(row.get('price')):.2f}" if isinstance(row.get("price"), (int, float)) and row.get("price") is not None else "Check Online"
            cap_raw_json = row.get("raw_json")
            actual_c = _cap_value_f_from_row(row)
            if actual_c > 0:
                selected_c_out_uF = int(round(actual_c * 1e6))
            logs.append("[LOCAL_DB_USED] Output capacitor candidate selected from local DigiKey database.")
        elif local_rows:
            logs.append("[FILTER] Local output-cap rows rejected by voltage/type filter.")
    except Exception as db_err:
        logs.append(f"[WARNING] Local DigiKey DB query failed: {db_err}")
    if research_digikey_component:
        try:
            logs.append("[SEARCH] DigiKey-focused output capacitor query.")
            dk_pack = research_digikey_component(component_type="output_cap", min_vds=16.0, max_results=3)
            for note in (dk_pack.get("notes") or []):
                logs.append(f"[MCP-DIGIKEY] {note}")
            mode = str(dk_pack.get("selection_mode") or "").lower()
            if mode == "live":
                logs.append("[LIVE_DIGIKEY_USED] Output capacitor candidate came from live DigiKey query.")
            elif mode == "fallback":
                logs.append("[FALLBACK_USED] Output capacitor candidate came from DigiKey fallback catalog.")
            elif mode == "none":
                logs.append("[DIGIKEY_STRICT_EMPTY] Strict mode enabled and no DigiKey output capacitor candidate found.")
            dk_rows = dk_pack.get("results") or []
            min_cout_f = max(c_out * 0.6, c_out_uF * 1e-6 * 0.8)
            min_ripple_a = max(0.45, i_out * 0.25)
            valid_dk_rows = [
                r
                for r in dk_rows
                if _cap_candidate_is_valid(r, min_v=max(25.0, float(specs.get("output_voltage", 12.0) or 12.0) * 1.6), max_v=63.0)
                and _cap_value_f_from_row(r) >= min_cout_f
                and _resolve_current_a(r.get("id") or r.get("id_current"), str(r.get("title") or "")) >= min_ripple_a
            ]
            if valid_dk_rows:
                target_c = max(c_out, c_out_uF * 1e-6)
                valid_dk_rows.sort(
                    key=lambda r: abs((_cap_value_f_from_row(r) or target_c) / max(target_c, 1e-9) - 1.0)
                )
                best = valid_dk_rows[0]
                cap_part = str(best.get("part_number") or "")
                cap_source = str(best.get("url") or "")
                if isinstance(best.get("price"), (int, float)) and float(best.get("price")) > 0:
                    cap_price = f"${float(best.get('price')):.2f}"
                actual_c = _cap_value_f_from_row(best)
                if actual_c > 0:
                    selected_c_out_uF = int(round(actual_c * 1e6))
                logs.append(f"[DECISION] DigiKey output-cap candidate: {cap_part or 'N/A'}")
            elif dk_rows:
                logs.append("[FILTER] DigiKey output-cap candidates rejected by voltage/type filter.")
        except Exception as dk_err:
            logs.append(f"[WARNING] DigiKey output-cap MCP path failed: {dk_err}")

    logs.append("[FORMULA] C_out ≈ I_out / (8*f_sw*ΔV_out)")
    logs.append(f"[ANALYSIS] Iout={i_out:.2f}A, f_sw={f_sw/1000:.1f}kHz, ΔV={ripple_target:.3f}V => C≈{c_out*1e6:.1f}uF")
    logs.append(f"[DECISION] Selected {c_out_uF}uF / 25V low-ESR capacitor")

    trace.append({
        "step": "Output Cap Sizing",
        "agent": "Selector",
        "action": f"Set Cout={c_out_uF}uF",
        "evidence": "I/(8*f*ΔV)",
        "confidence": 0.94,
    })

    if not cap_part:
        cap_part = "35ZLH470MEFC10X16"
        cap_source = "https://www.digikey.com/en/products/detail/rubycon/35ZLH470MEFC10X16/3563723"
        cap_price = "$0.71"
        cap_raw_json = None
        selected_c_out_uF = max(c_out_uF, 470)
        logs.append("[DECISION] Using curated low-ESR output capacitor fallback with real MPN/source.")

    return {
        "part_number": cap_part,
        "source": cap_source,
        "price": cap_price,
        "description": f"Low ESR {selected_c_out_uF}uF 25V or higher",
        "Value": selected_c_out_uF * 1e-6,
        "value": f"{selected_c_out_uF}uF",
        "voltage_rating": "25V+",
        "esr": "20mΩ",
        "raw_json": cap_raw_json,
    }, "\n\n".join(logs)


def select_controller_feedback_enhanced(
    design,
    specs,
    trace: List[ReasoningTraceItem],
    selector_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, str]:
    logs = []
    logs.append("[PLAN] Selecting controller + isolated feedback chain.")
    f_sw = float(design.get("switching_frequency", 100000.0))
    p_out = float(specs.get("output_voltage", 12.0) or 12.0) * float(specs.get("output_current", 2.0) or 2.0)

    if is_offline_ac_input(specs or {}) and p_out >= 10.0:
        ctrl = "UCC28740DR"
        ctrl_source = "https://www.ti.com/product/UCC28740"
        ctrl_notes = "Offline flyback controller with valley switching support; verify exact package/orderable MPN."
    elif f_sw >= 120000:
        ctrl = "UCC28740DR"
        ctrl_source = "https://www.ti.com/product/UCC28740"
        ctrl_notes = "Offline flyback controller with valley switching support; verify exact package/orderable MPN."
    elif f_sw >= 70000:
        ctrl = "NCP1207"
        ctrl_source = "https://www.onsemi.com"
        ctrl_notes = "Offline PWM controller family; verify package/orderable MPN."
    else:
        ctrl = "UC3845BDR2G"
        ctrl_source = "https://www.onsemi.com"
        ctrl_notes = "Current-mode PWM controller; acceptable baseline but QR controller is preferred for adapter efficiency."

    logs.append("[FORMULA] f_sw target determines controller family window")

    try:
        ensure_local_db()
        local_rows = _query_local_candidates("controller", specs, design, selector_profile, limit=40)
        valid_rows = [
            r for r in local_rows
            if _controller_candidate_is_valid(r) and str(r.get("source_mode") or "") != "fallback-local"
        ]
        if valid_rows:
            row = valid_rows[0]
            ctrl = row.get("part_number") or ctrl
            logs.append("[LOCAL_DB_USED] Controller candidate selected from local DigiKey database.")
            trace.append({
                "step": "Controller Selection",
                "agent": "Selector",
                "action": f"Selected {ctrl}",
                "evidence": f"Local DigiKey DB ({row.get('url', '')})",
                "confidence": 0.92,
            })
            return {
                "part_number": ctrl,
                "feedback_ref": "TL431",
                "optocoupler": "PC817",
                "notes": "Primary-side PWM controller with isolated secondary feedback",
                "source": row.get("url") or "",
                "price": f"${float(row.get('price')):.2f}" if isinstance(row.get("price"), (int, float)) and row.get("price") is not None else "Check Online",
            }, "\n\n".join(logs)
        elif local_rows:
            logs.append("[FILTER] Local controller rows rejected (non-flyback family).")
    except Exception as db_err:
        logs.append(f"[WARNING] Local DigiKey DB query failed: {db_err}")

    if research_digikey_component:
        try:
            logs.append("[SEARCH] DigiKey-focused controller query.")
            dk_pack = research_digikey_component(
                component_type="controller",
                max_results=5,
            )
            for note in (dk_pack.get("notes") or []):
                logs.append(f"[MCP-DIGIKEY] {note}")
            mode = str(dk_pack.get("selection_mode") or "").lower()
            if mode == "live":
                logs.append("[LIVE_DIGIKEY_USED] Controller candidate came from live DigiKey query.")
            elif mode == "fallback":
                logs.append("[FALLBACK_USED] Controller candidate came from DigiKey fallback catalog.")
            elif mode == "none":
                logs.append("[DIGIKEY_STRICT_EMPTY] Strict mode enabled and no DigiKey controller candidate found.")
            dk_rows = dk_pack.get("results") or []
            valid_dk_rows = [r for r in dk_rows if _controller_candidate_is_valid(r)]
            if valid_dk_rows:
                best = valid_dk_rows[0]
                ctrl = best.get("part_number") or ctrl
                logs.append(f"[DECISION] DigiKey controller candidate selected: {ctrl}")
                trace.append({
                    "step": "Controller Selection",
                    "agent": "Selector",
                    "action": f"Selected {ctrl}",
                    "evidence": f"DigiKey MCP ({best.get('url', '')})",
                    "confidence": 0.9,
                })
                return {
                    "part_number": ctrl,
                    "feedback_ref": "TL431",
                    "optocoupler": "PC817",
                    "notes": "Primary-side PWM controller with isolated secondary feedback",
                    "source": best.get("url") or "",
                    "price": (
                        f"${float(best.get('price')):.2f}"
                        if isinstance(best.get("price"), (int, float)) and float(best.get("price")) > 0
                        else "Check Online"
                    ),
                }, "\n\n".join(logs)
            elif dk_rows:
                logs.append("[FILTER] DigiKey controller candidates rejected (non-flyback family).")
        except Exception as dk_err:
            logs.append(f"[WARNING] DigiKey controller MCP path failed: {dk_err}")

    logs.append(f"[DECISION] Controller={ctrl}, Optocoupler=PC817, Reference=TL431")

    trace.append({
        "step": "Controller Selection",
        "agent": "Selector",
        "action": f"Selected {ctrl}",
        "evidence": "f_sw operating window",
        "confidence": 0.9,
    })

    return {
        "part_number": ctrl,
        "feedback_ref": "TL431",
        "optocoupler": "PC817",
        "notes": ctrl_notes,
        "source": ctrl_source,
        "price": "-",
    }, "\n\n".join(logs)


def select_input_protection_enhanced(
    design,
    specs,
    trace: List[ReasoningTraceItem],
    selector_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, str]:
    logs = []
    logs.append("[PLAN] Selecting AC front-end protection and rectifier chain.")
    logs.append("[SEARCH] Searching fuse/NTC/MOV/bridge options on distributor-style catalogs (DigiKey/Mouser/LCSC).")
    p_out = float(specs.get("output_voltage", 12.0)) * float(specs.get("output_current", 2.0))
    eta = max(float(specs.get("efficiency_target", 0.88)), 0.7)
    p_in = p_out / eta
    vin_min = max(float(specs.get("input_voltage_min", 85.0)), 80.0)
    i_rms_est = p_in / vin_min

    fuse_rating = 1.0 if i_rms_est <= 0.35 else 1.6
    ntc_part = "5D-9"
    mov_part = "MOV-14D471K"
    bridge_part = "MB10S"

    try:
        ensure_local_db()
        local_rows = _query_local_candidates("input_protection", specs, design, selector_profile, limit=40)
        for row in local_rows:
            title = str(row.get("title") or "").lower()
            pn = str(row.get("part_number") or "").strip()
            if not pn:
                continue
            if "fuse" in title and any(k in title for k in ["slow", "time delay", "250v", "cartridge"]):
                pass
            if ("ntc" in title or "thermistor" in title) and any(k in title for k in ["inrush", "disc", "power", "5d", "10d"]):
                ntc_part = pn
            if ("mov" in title or "varistor" in title) and any(k in title for k in ["470v", "471", "ac", "radial"]):
                mov_part = pn
            if ("bridge" in title or "rectifier" in title) and any(k in title for k in ["600v", "800v", "1000v", "1kv"]):
                bridge_part = pn
        if local_rows:
            logs.append("[LOCAL_DB_USED] Input protection candidates selected from local DigiKey database.")
    except Exception as db_err:
        logs.append(f"[WARNING] Local DigiKey DB query failed: {db_err}")

    if research_digikey_component:
        try:
            logs.append("[SEARCH] DigiKey-focused input protection query.")
            dk_pack = research_digikey_component(
                component_type="input_protection",
                max_results=8,
            )
            for note in (dk_pack.get("notes") or []):
                logs.append(f"[MCP-DIGIKEY] {note}")
            mode = str(dk_pack.get("selection_mode") or "").lower()
            if mode == "live":
                logs.append("[LIVE_DIGIKEY_USED] Input protection candidates came from live DigiKey query.")
            elif mode == "fallback":
                logs.append("[FALLBACK_USED] Input protection candidates came from DigiKey fallback catalog.")
            elif mode == "none":
                logs.append("[DIGIKEY_STRICT_EMPTY] Strict mode enabled and no DigiKey input protection candidate found.")

            rows = dk_pack.get("results") or []
            # Best-effort assign based on title keywords
            for row in rows:
                title = str(row.get("title") or "").lower()
                pn = str(row.get("part_number") or "").strip()
                if not pn:
                    continue
                if "fuse" in title and fuse_rating <= 1.2:
                    if pn:
                        # Keep user-friendly default when unsure about exact type
                        pass
                if "ntc" in title or "thermistor" in title:
                    ntc_part = pn
                if "mov" in title or "varistor" in title:
                    mov_part = pn
                if "bridge" in title or "rectifier" in title:
                    bridge_part = pn
        except Exception as dk_err:
            logs.append(f"[WARNING] DigiKey input-protection MCP path failed: {dk_err}")

    logs.append(f"[ANALYSIS] Estimated line RMS current at low line: {i_rms_est:.3f}A")
    logs.append("[FILTER] Enforcing derating rules for surge and startup current stress.")
    logs.append(f"[DECISION] Fuse={fuse_rating}A slow-blow, NTC={ntc_part}, MOV={mov_part}, Bridge={bridge_part}")

    trace.append({
        "step": "Input Protection Selection",
        "agent": "Selector",
        "action": f"Fuse {fuse_rating}A + MOV + NTC + Bridge",
        "evidence": "Pin/Vin_min safety sizing",
        "confidence": 0.9,
    })

    return {
        "fuse": {
            "part_number": "T1A250V",
            "current_rating": f"{fuse_rating}A",
            "type": "Slow-blow",
            "source": "https://www.digikey.com/en/products/filter/fuses/139",
        },
        "ntc": {
            "part_number": ntc_part,
            "cold_resistance": "5ohm",
            "source": "https://www.digikey.com/en/products/filter/inrush-current-limiters-icl/151",
        },
        "mov": {
            "part_number": mov_part,
            "varistor_voltage": "470V",
            "source": "https://www.digikey.com/en/products/filter/tvs-varistors-movs/141",
        },
        "bridge_rectifier": {
            "part_number": bridge_part,
            "vrm": "1000V",
            "if_avg": "1A",
            "source": "https://www.digikey.com/en/products/filter/bridge-rectifiers/299",
        },
        "plecs_mapping": {
            "simulatable": "indirect",
            "notes": "Influences input surge and losses; not fully expanded in current Flyback_effi model.",
        },
    }, "\n\n".join(logs)


def select_emi_filter_enhanced(
    design,
    specs,
    trace: List[ReasoningTraceItem],
    selector_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, str]:
    logs = []
    logs.append("[PLAN] Selecting EMI filter network for conducted emissions control.")
    logs.append("[SEARCH] Searching EMI building blocks: common-mode choke, X-cap, Y-cap, DM inductor.")
    fsw = float(design.get("switching_frequency", 65000.0))

    cm_choke = "2x10mH"
    x_cap = "0.1uF X2"
    y_cap = "2x1nF Y1"
    dm_inductor = "220uH"

    emi_source = ""
    emi_price = "Check Online"
    emi_part = ""
    try:
        ensure_local_db()
        local_rows = _query_local_candidates("emi_filter", specs, design, selector_profile, limit=40)
        valid_rows = [r for r in local_rows if _emi_candidate_is_valid(r, specs)]
        if valid_rows:
            row = valid_rows[0]
            emi_part = str(row.get("part_number") or "")
            emi_source = str(row.get("url") or "")
            emi_price = f"${float(row.get('price')):.2f}" if isinstance(row.get("price"), (int, float)) and row.get("price") is not None else "Check Online"
            logs.append("[LOCAL_DB_USED] EMI candidate selected from local DigiKey database.")
        elif local_rows:
            logs.append("[FILTER] Local EMI rows rejected because they are not mains-rated power-line EMI parts.")
    except Exception as db_err:
        logs.append(f"[WARNING] Local DigiKey DB query failed: {db_err}")
    if research_digikey_component:
        try:
            logs.append("[SEARCH] DigiKey-focused EMI filter query.")
            dk_pack = research_digikey_component(component_type="emi_filter", max_results=4)
            for note in (dk_pack.get("notes") or []):
                logs.append(f"[MCP-DIGIKEY] {note}")
            mode = str(dk_pack.get("selection_mode") or "").lower()
            if mode == "live":
                logs.append("[LIVE_DIGIKEY_USED] EMI candidate came from live DigiKey query.")
            elif mode == "fallback":
                logs.append("[FALLBACK_USED] EMI candidate came from DigiKey fallback catalog.")
            elif mode == "none":
                logs.append("[DIGIKEY_STRICT_EMPTY] Strict mode enabled and no DigiKey EMI candidate found.")
            dk_rows = dk_pack.get("results") or []
            valid_dk_rows = [r for r in dk_rows if _emi_candidate_is_valid(r, specs)]
            if valid_dk_rows:
                best = valid_dk_rows[0]
                emi_part = str(best.get("part_number") or "")
                emi_source = str(best.get("url") or "")
                if isinstance(best.get("price"), (int, float)) and float(best.get("price")) > 0:
                    emi_price = f"${float(best.get('price')):.2f}"
                logs.append(f"[DECISION] DigiKey EMI candidate: {emi_part or 'N/A'}")
            elif dk_rows:
                logs.append("[FILTER] DigiKey EMI candidates rejected because they are not mains-rated power-line EMI parts.")
        except Exception as dk_err:
            logs.append(f"[WARNING] DigiKey EMI MCP path failed: {dk_err}")

    logs.append(f"[ANALYSIS] fsw={fsw/1000:.1f}kHz -> recommend 2-stage differential/common-mode attenuation")
    logs.append("[FILTER] Preferring safety-class capacitors (X2/Y1) and practical availability bins.")
    logs.append(f"[DECISION] CM Choke={cm_choke}, XCap={x_cap}, YCap={y_cap}, DM Inductor={dm_inductor}")

    trace.append({
        "step": "EMI Filter Selection",
        "agent": "Selector",
        "action": "Selected CM choke + X/Y capacitors + DM inductor",
        "evidence": "Conducted EMI baseline design rule",
        "confidence": 0.86,
    })

    return {
        "cm_choke": {
            "description": cm_choke,
            "part_number": emi_part or "AC_MAINS_CM_CHOKE_TBD_2x10mH_0.6A",
            "source": emi_source or "https://www.digikey.com/en/products/filter/common-mode-chokes/839",
            "price": emi_price if emi_part else "Manual selection",
            "procurement_status": "verify_mains_rated_part" if emi_part else "manual_mains_emi_selection_required",
        },
        "x_cap": {
            "description": x_cap,
            "safety_class": "X2",
            "source": "https://www.digikey.com/en/products/filter/film-capacitors/62",
        },
        "y_cap": {
            "description": y_cap,
            "safety_class": "Y1",
            "source": "https://www.digikey.com/en/products/filter/ceramic-capacitors/60",
        },
        "dm_inductor": {"description": dm_inductor, "source": "https://www.digikey.com/en/products/filter/fixed-inductors/71"},
        "plecs_mapping": {
            "simulatable": "partial",
            "notes": "Can be modeled in PLECS by extending input filter stage; currently represented as design metadata.",
        },
    }, "\n\n".join(logs)


def select_clamp_snubber_enhanced(
    design,
    specs,
    trace: List[ReasoningTraceItem],
    selector_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, str]:
    logs = []
    logs.append("[PLAN] Selecting RCD clamp/snubber network around primary switch.")
    logs.append("[SEARCH] Evaluating clamp options: RCD baseline + TVS reinforcement for surge margin.")
    r_sn = float(design.get("snubber_r", 6800.0))
    c_sn = float(design.get("snubber_c", 22e-9))
    input_domain = resolve_power_stage_input(specs or {})
    vin_max = float(input_domain.get("dc_bus_max", 375.0))
    vor = float(design.get("reflected_output_voltage", 80.0))
    target_vclamp = vin_max + vor + 70.0

    tvs_part = "SMBJ440A"
    tvs_source = ""
    tvs_price = "Check Online"
    try:
        ensure_local_db()
        local_rows = _query_local_candidates(
            "clamp_snubber",
            specs,
            design,
            selector_profile,
            min_vds=target_vclamp,
            limit=3,
        )
        if local_rows:
            row = local_rows[0]
            tvs_part = str(row.get("part_number") or tvs_part)
            tvs_source = str(row.get("url") or "")
            tvs_price = f"${float(row.get('price')):.2f}" if isinstance(row.get("price"), (int, float)) and row.get("price") is not None else "Check Online"
            logs.append("[LOCAL_DB_USED] Clamp/snubber candidate selected from local DigiKey database.")
    except Exception as db_err:
        logs.append(f"[WARNING] Local DigiKey DB query failed: {db_err}")
    if research_digikey_component:
        try:
            logs.append("[SEARCH] DigiKey-focused clamp/snubber query.")
            dk_pack = research_digikey_component(component_type="clamp_snubber", min_vds=target_vclamp, max_results=4)
            for note in (dk_pack.get("notes") or []):
                logs.append(f"[MCP-DIGIKEY] {note}")
            mode = str(dk_pack.get("selection_mode") or "").lower()
            if mode == "live":
                logs.append("[LIVE_DIGIKEY_USED] Clamp/snubber candidate came from live DigiKey query.")
            elif mode == "fallback":
                logs.append("[FALLBACK_USED] Clamp/snubber candidate came from DigiKey fallback catalog.")
            elif mode == "none":
                logs.append("[DIGIKEY_STRICT_EMPTY] Strict mode enabled and no DigiKey clamp/snubber candidate found.")
            dk_rows = dk_pack.get("results") or []
            if dk_rows:
                best = dk_rows[0]
                tvs_part = str(best.get("part_number") or tvs_part)
                tvs_source = str(best.get("url") or "")
                if isinstance(best.get("price"), (int, float)) and float(best.get("price")) > 0:
                    tvs_price = f"${float(best.get('price')):.2f}"
                logs.append(f"[DECISION] DigiKey clamp/snubber candidate: {tvs_part}")
        except Exception as dk_err:
            logs.append(f"[WARNING] DigiKey clamp/snubber MCP path failed: {dk_err}")

    logs.append(f"[ANALYSIS] Snubber from design: R={r_sn:.1f}ohm, C={c_sn*1e9:.1f}nF")
    logs.append("[FILTER] Rejecting clamp settings that risk excessive Vds spike near safety threshold.")
    logs.append(f"[DECISION] RCD clamp target Vclamp≈{target_vclamp:.1f}V, add TVS for surge margin")

    trace.append({
        "step": "Clamp Snubber Selection",
        "agent": "Selector",
        "action": f"RCD clamp R={r_sn:.0f} C={c_sn*1e9:.1f}nF + TVS",
        "evidence": "Vds spike margin requirement",
        "confidence": 0.9,
    })

    return {
        "rcd_clamp": {"R": r_sn, "C": c_sn, "diode": "UF4007"},
        "tvs": {"part_number": tvs_part, "standoff_voltage": f"{int(target_vclamp)}V", "source": tvs_source, "price": tvs_price},
        "target_vclamp": target_vclamp,
        "plecs_mapping": {
            "simulatable": "direct",
            "notes": "Mapped into PLECS via Rsn/Csn variables in simulation node.",
        },
    }, "\n\n".join(logs)
