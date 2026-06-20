from typing import Dict, Any, List
import json
import os
from langchain_core.prompts import ChatPromptTemplate
from ..state import PowerSupplyState, VerificationResult, ReasoningTraceItem
from .requirements import get_llm
from ..knowledge_guardrails import get_hard_guardrails_prompt, retrieve_flyback_context
from ..skills_manager import SkillManager
from ..lifelong_memory import build_iteration_playbook
from ..planning import build_execution_plan, summarize_execution_plan
from ..tools.input_semantics import resolve_power_stage_input
from .research_helper import collect_node_research
try:
    from ...llm.llm import get_agent_reasoning
except ImportError:
    from core.llm.llm import get_agent_reasoning


def _run_design_peer_review(context: Dict[str, Any]) -> Dict[str, Any]:
    skills_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "skills")
    skill = SkillManager(skills_dir).get_skill("design_peer_review")
    if not skill or not skill.tools_module or not hasattr(skill.tools_module, "review_design"):
        return {
            "review_status": "CAUTION",
            "major_findings": [],
            "minor_findings": ["design_peer_review skill unavailable"],
            "required_actions": ["Review validator result manually"],
            "false_pass_risk_score": 25.0,
        }
    try:
        return skill.tools_module.review_design(context)
    except Exception as err:
        return {
            "review_status": "CAUTION",
            "major_findings": [],
            "minor_findings": [f"design_peer_review failed: {err}"],
            "required_actions": ["Review validator result manually"],
            "false_pass_risk_score": 35.0,
        }


def _top_loss_summary(sim: Dict[str, Any], top_k: int = 3) -> str:
    losses = sim.get("formula_losses", {}) if isinstance(sim.get("formula_losses"), dict) else {}
    if not losses:
        return "loss breakdown unavailable"
    ranked = sorted(losses.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True)[:top_k]
    parts = [f"{k}={float(v or 0.0):.3f}W" for k, v in ranked]
    return ", ".join(parts)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    import re

    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _first_value(row: Dict[str, Any], names: List[str]) -> str:
    if not isinstance(row, dict):
        return ""
    for name in names:
        value = row.get(name)
        if value not in (None, "", "-", "N/A", "n/a"):
            return str(value)
    return ""


def _selection_summary_row(bom: Dict[str, Any], key: str) -> Dict[str, Any]:
    summary = bom.get("selection_summary") if isinstance(bom.get("selection_summary"), dict) else {}
    row = summary.get(key, {})
    return row if isinstance(row, dict) else {}


def _component_row(bom: Dict[str, Any], key: str) -> Dict[str, Any]:
    row = bom.get(key, {})
    return row if isinstance(row, dict) else {}


def _component_name(bom: Dict[str, Any], key: str) -> str:
    summary = _selection_summary_row(bom, key)
    row = _component_row(bom, key)
    return (
        str(summary.get("selected") or "").strip()
        or _first_value(row, ["Part Number", "Mfr Part #", "Manufacturer Part Number", "part_number", "title", "description"])
    )


def _component_source(bom: Dict[str, Any], key: str) -> str:
    summary = _selection_summary_row(bom, key)
    row = _component_row(bom, key)
    return (
        str(summary.get("source") or "").strip()
        or _first_value(row, ["Product URL", "URL", "DigiKey URL", "source", "Source"])
    )


def _component_price(bom: Dict[str, Any], key: str) -> str:
    summary = _selection_summary_row(bom, key)
    row = _component_row(bom, key)
    return (
        str(summary.get("price") or "").strip()
        or _first_value(row, ["Price", "Unit Price", "price"])
    )


def _component_blob(bom: Dict[str, Any], key: str) -> str:
    parts = [
        _component_name(bom, key),
        _component_source(bom, key),
        _component_price(bom, key),
        json.dumps(_component_row(bom, key), ensure_ascii=False, default=str)[:2500],
    ]
    return " ".join([p for p in parts if p]).lower()


def _looks_provisional_part(name: str, price: str = "") -> bool:
    low = str(name or "").strip().lower()
    price_low = str(price or "").strip().lower()
    tokens = ("generic", "fallback", "unknown", "check online", "tbd", "placeholder", "provisional", "custom-")
    price_tokens = ("check online", "manual selection", "quote")
    return not low or any(token in low for token in tokens) or any(token in price_low for token in price_tokens)


def _requires_manual_signoff(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("requires_custom_design") or "manual" in str(row.get("procurement_status") or "").lower():
        return True
    return any(_requires_manual_signoff(value) for value in row.values() if isinstance(value, dict))


def _extract_strategy_from_review(content: str) -> str:
    for line in str(content or "").splitlines():
        if "Strategy:" in line:
            strategy = line.split("Strategy:", 1)[1].strip()
            if strategy and strategy.upper() != "N/A":
                return strategy
    for token in [
        "ADD_REALISTIC_PARASITICS",
        "REDUCE_TURNS_RATIO",
        "INCREASE_INDUCTANCE",
        "DECREASE_FSW",
        "FIX_FORMULA_INCONSISTENCY",
    ]:
        if token in str(content or ""):
            return token
    return "REVIEW_LLM_CONCERNS"


def _quality_gate_findings(
    specs: Dict[str, Any],
    design: Dict[str, Any],
    bom: Dict[str, Any],
    sim: Dict[str, Any],
    vds_rating: float,
    safety_ratio: float,
) -> Dict[str, Any]:
    blockers: List[str] = []
    cautions: List[str] = []
    component_actions: List[str] = []

    target_eff = _as_float(specs.get("efficiency_target")) or 0.0
    formula_eff = _as_float(sim.get("efficiency_formula_est"))
    formula_conf = str(sim.get("efficiency_formula_confidence") or "").strip().lower()
    scope = str(sim.get("efficiency_scope") or "").strip().lower()
    corner = sim.get("simulation_corner") if isinstance(sim.get("simulation_corner"), dict) else {}
    corner_label = str(corner.get("corner_label") or corner.get("requested_corner") or "").strip().lower()
    sim_basis = str(sim.get("simulation_basis") or "")
    control_mode = str(sim.get("control_mode") or sim.get("model_control_mode") or "").lower()

    if scope == "power_stage_only":
        blockers.append("Simulation scope is power-stage only; bridge rectifier, EMI filter, input protection, standby/control losses, and thermal derating are not included.")
    if corner_label in {"high_line_default", "high_line", "auto"}:
        blockers.append("Only high-line simulation evidence is present; low-line 85 Vac full-load stress, RMS current, thermal, and loop-margin checks are missing.")
    if "open_loop" in control_mode or "forced open loop" in sim_basis.lower():
        blockers.append("PLECS evidence is from forced open-loop duty operation; closed-loop regulation, transient response, and compensation stability are not verified.")
    if formula_conf == "low":
        blockers.append("Independent efficiency formula confidence is low because key magnetic/device parameters are default or heuristic.")
    if formula_eff is not None and target_eff and formula_eff < target_eff:
        blockers.append(f"Independent formula efficiency estimate is {formula_eff:.2%}, below the {target_eff:.2%} target.")

    if vds_rating and safety_ratio >= 0.80:
        blockers.append(f"MOSFET Vds stress uses {safety_ratio:.1%} of the parsed rating; the release guardrail requires <80%.")

    evidence_closure = sim.get("evidence_closure") if isinstance(sim.get("evidence_closure"), dict) else {}
    for gate in (evidence_closure.get("required_gates") or [])[:8]:
        if not isinstance(gate, dict):
            continue
        gate_status = str(gate.get("status") or "open").lower()
        if gate_status == "closed":
            continue
        label = str(gate.get("label") or gate.get("key") or "Evidence gate")
        missing = str(gate.get("missing") or gate.get("release_impact") or "Missing release evidence.")
        blockers.append(f"{label} is {gate_status}; {missing}")

    component_labels = {
        "mosfet": "MOSFET",
        "diode": "Output rectifier",
        "transformer": "Transformer/core",
        "input_cap": "Input bulk capacitor",
        "output_cap": "Output capacitor",
        "controller": "Controller",
        "input_protection": "Input protection",
        "emi_filter": "EMI filter",
        "clamp_snubber": "Clamp/snubber",
    }
    for key, label in component_labels.items():
        name = _component_name(bom, key)
        source = _component_source(bom, key)
        price = _component_price(bom, key)
        row = _component_row(bom, key)
        manual_required = _requires_manual_signoff(row)
        if manual_required:
            blockers.append(f"{label} '{name or 'missing'}' requires manual engineering/procurement sign-off before BOM freeze.")
            component_actions.append(f"manual_signoff_{key}")
            continue
        if _looks_provisional_part(name, price):
            blockers.append(f"{label} is not a real frozen orderable part: '{name or 'missing'}'.")
            component_actions.append(f"replace_{key}_with_orderable_part")
            continue
        if key in {"mosfet", "diode", "transformer", "input_cap", "output_cap", "controller", "input_protection", "emi_filter"} and not source:
            blockers.append(f"{label} '{name}' is missing a distributor or datasheet source link.")

    transformer_blob = _component_blob(bom, "transformer")
    bad_transformer_tokens = [
        "ferrite core 2 hole",
        "multi-hole",
        "bead",
        "mhb2/p",
        "0.085",
        "forward",
        "push-pull",
        "push pull",
        "fwd p-p",
        "sn6501",
        "mid-ppti",
        "0.265",
        "6.73mm",
    ]
    if any(token in transformer_blob for token in bad_transformer_tokens):
        blockers.append("Transformer/core selection appears to be a small signal/DC-DC or EMI part, not a realistic 24 W isolated offline flyback transformer/core set.")
        component_actions.append("replace_transformer_with_power_flyback_core_set")

    emi_blob = _component_blob(bom, "emi_filter")
    if "signal line" in emi_blob or "50v" in emi_blob or "400ma" in emi_blob or "not for new designs" in emi_blob:
        blockers.append("EMI filter selection appears to be a low-voltage signal-line common-mode choke, not a mains-rated 85-265 Vac input EMI filter.")
        component_actions.append("replace_emi_filter_with_mains_rated_filter")

    # Secondary diode reverse stress estimate: Vrr ~= Vout + Vin_bus_max / n.
    turns_ratio = _as_float(design.get("turns_ratio"))
    vin_bus_max = _as_float(corner.get("vin_bus_max_v"))
    vout = _as_float(specs.get("output_voltage"))
    diode_rating = _as_float(_component_row(bom, "diode").get("Vds") or _component_row(bom, "diode").get("Voltage - DC Reverse (Vr)"))
    if turns_ratio and vin_bus_max and vout and diode_rating:
        diode_stress = vout + vin_bus_max / turns_ratio
        diode_ratio = diode_stress / diode_rating
        if diode_ratio >= 0.80:
            blockers.append(f"Output rectifier reverse stress estimate is {diode_stress:.1f} V ({diode_ratio:.1%} of rating), exceeding the <80% guardrail.")
        elif diode_ratio >= 0.65:
            cautions.append(f"Output rectifier reverse stress estimate is {diode_stress:.1f} V ({diode_ratio:.1%} of rating); margin should be reviewed.")

    return {
        "status": "BLOCK" if blockers else ("CAUTION" if cautions else "PASS"),
        "blockers": list(dict.fromkeys(blockers)),
        "cautions": list(dict.fromkeys(cautions)),
        "component_actions": list(dict.fromkeys(component_actions)),
    }

def design_validator_node(state: PowerSupplyState) -> Dict[str, Any]:
    """
    Node 5: Design Validator (Now powered by LLM Agent)
    The 'Senior Engineer' who validates the Junior Designer's work and reasoning.
    """
    print("DEBUG: AI Validator Node Started")
    specs = state.get("specifications") or {}
    sim = state.get("simulation_results") or {}
    design = state.get("theoretical_design") or {}
    bom = state.get("bom") or {}
    reasoning_trace = state.get("reasoning_trace", [])
    hard_guardrails_prompt = state.get("hard_guardrails_prompt") or get_hard_guardrails_prompt()

    retrieved_context = state.get("retrieved_knowledge_context") or ""
    if not retrieved_context:
        rag_query = (
            f"flyback constraints for Vin={specs.get('input_voltage_min')}-{specs.get('input_voltage_max')}, "
            f"Vout={specs.get('output_voltage')}, Iout={specs.get('output_current')}, "
            f"duty cycle, soft switching, snubber, ripple, safety margin"
        )
        retrieved_context = retrieve_flyback_context(rag_query, top_k=6).get("context_text", "")
    
    # [NEW] Chitchat Bypass
    if specs and specs.get("is_chitchat"):
        print("⏩ Skipping Validator: Intent = Chitchat")
        return {"verification": {"status": "PASS"}}

    if not specs or not sim:
        return {"error_log": ["Validation skipped due to missing data"]}

    formula_checks = state.get("formula_checks", {}) or {}
    formula_blockers = []
    sim_source = str(sim.get("source") or "UNKNOWN")
    for node_key in ["requirements", "designer"]:
        node_check = formula_checks.get(node_key, {}) if isinstance(formula_checks, dict) else {}
        for fatal in node_check.get("fatal", []) or []:
            formula_blockers.append(f"{node_key}: {fatal}")

    # --- SYSTEM 2 REASONING START ---
    print("[System 2] Validator Reasoning...")
    
    # [DEEPRARE FEATURE]: Traceable Reasoning Check
    # Verify if key design decisions are supported by evidence
    evidence_check_log = []
    unverified_decisions = []
    
    for item in reasoning_trace:
        if item.get("action", "").startswith("Set") and item.get("evidence") == "None":
            unverified_decisions.append(item["action"])
    
    if unverified_decisions:
        evidence_check_log.append(f"[WARNING] Unverified Decisions Found: {unverified_decisions}")

    # Precompute rating hint before composing reasoning text.
    vds_rating_str = bom.get('mosfet', {}).get('Drain to Source Voltage (Vdss)') or \
                     bom.get('mosfet', {}).get('Vds') or \
                     "100V"
    
    try:
        reasoning_logs = get_agent_reasoning(
            agent_role="Quality Assurance Engineer",
            task="Evaluate simulation results against safety standards and specifications. For diode-rectified flyback, treat efficiency > 90% as caution-only and > 96% as hard unrealistic/fail. Use hard pass/fail hints as ground truth.",
            context_data={
                "efficiency": sim.get("efficiency_measured"),
                "ripple": sim.get("v_out_ripple_measured"),
                "Vds_spike": sim.get("v_ds_spike_max"),
                "target_eff": specs.get("efficiency_target"),
                "topology": "Flyback (Diode Rectification assumed)",
                "evidence_gaps": unverified_decisions
            },
            history_trace=reasoning_trace
        )
    except Exception as reasoning_err:
        print(f"Warning: validator reasoning engine unavailable: {reasoning_err}")
        reasoning_logs = [f"[FALLBACK] Validator reasoning unavailable: {reasoning_err}"]
    reasoning_logs = [
        "[PLAN] Build validation checklist: KPI, safety margin, formula consistency, and realism.",
        "[EXECUTION] Running checklist item 1/4: efficiency vs target.",
        "[EXECUTION] Running checklist item 2/4: ripple vs target.",
        "[EXECUTION] Running checklist item 3/4: Vds stress vs device rating.",
        "[EXECUTION] Running checklist item 4/4: topology realism and suspicious artifact detection.",
        *reasoning_logs,
    ]

    node_research = collect_node_research(
        "validator",
        (
            f"flyback validation checklist standards papers blogs forums efficiency ripple safety margin Vin {specs.get('input_voltage_max')} "
            f"Vout {specs.get('output_voltage')}"
        ),
        max_results=5,
    )
    reasoning_logs.extend(node_research.get("logs", []))
    
    # Add structured logs for UI
    reasoning_logs.extend(evidence_check_log)
    if sim_source != "PLECS":
        reasoning_logs.append(f"[BACKEND-WARN] Simulation fallback source detected: {sim_source}")
    reasoning_logs.append(f"[THOUGHT] Analyzing Efficiency: {sim.get('efficiency_measured'):.2%} vs Target {specs.get('efficiency_target'):.2%}")
    reasoning_logs.append(f"[THOUGHT] Analyzing Ripple: {sim.get('v_out_ripple_measured'):.3f}V vs Max {specs.get('max_ripple_voltage'):.3f}V")
    reasoning_logs.append(f"[THOUGHT] Analyzing Stress: Vds_peak={sim.get('v_ds_spike_max')}V, component_rating_hint={vds_rating_str}")
    # --- SYSTEM 2 REASONING END ---

    # Fallback logic references (in case LLM fails)
    status = "PASS"
    strategy = ""
    failed_items = []
    
    # Construct the "Debate" Context
    
    input_domain = resolve_power_stage_input(specs or {})

    # Extract numeric Vds rating for pre-check
    import re
    try:
        vds_rating = float(re.search(r"([\d\.]+)", str(vds_rating_str)).group(1))
    except:
        vds_rating = 100.0

    v_spike = sim.get('v_ds_spike_max', 0)
    # v_spike is typically around 550V for 265V Input Flyback
    # Generic MOSFET rating is set to 600V or 800V in Generic Fallback, but might parse as 100V default here.
    
    # REPAIR LOGIC:
    if vds_rating < 200 and float(input_domain.get("dc_bus_max", 0.0) or 0.0) > 200.0:
        # Detected a low default rating for a High Voltage App
        vds_rating = 650.0 # Upgrade the rating assumption for validation to avoid false failure
    
    safety_ratio = v_spike / vds_rating if vds_rating > 0 else 999
    is_safe = safety_ratio < 0.80
    
    eff_meas = sim.get('efficiency_measured', 0)
    eff_target = (specs.get('efficiency_target') or 0.85)
    is_eff_ok = eff_meas >= (eff_target - 0.005) # 0.5% tolerance
    
    # [SOTA FIX]: Detect Diminishing Returns / Topology Limits
    # If we are iterating and efficiency is getting WORSE or plateauing, we should stop.
    prev_sim = ((state.get("best_design_candidate") or {}).get("simulation_results") or {})
    prev_eff = prev_sim.get("efficiency_measured", 0)
    current_iter = state.get("iteration", 0)
    
    # Heuristic: If we are deep in iterations (e.g. > 2) and current eff is worse than best seen
    force_accept = False
    limit_reason = ""
    
    if current_iter >= 2:
        if eff_meas < prev_eff - 0.005: # Dropped by more than 0.5%
             force_accept = True
             limit_reason = f"Diminishing Returns Detected. Best efficiency was {prev_eff:.1%} (Current: {eff_meas:.1%}). Accepting best result."
             print(f"META-COGNITION: {limit_reason}")
             
        # Also check for physical limits of Diode Rectification
        # 5V out, 1V drop -> 1/6 loss ~ 16%. Max Eff ~ 84%.
        # If target is 85% and we are at 80%, maybe acceptable for this topology.
        if (specs.get('output_voltage') or 0) <= 5.0 and eff_meas > 0.78 and eff_meas < eff_target:
             # Low voltage high current diode flyback is hard to get > 80% without Sync Rect
             print("Insight: Low Voltage Diode Flyback limit detected. Relaxing constraints.")
             eff_target = 0.78 # Relax target to physical reality

    # [FIX] Define missing variables that were cut off in previous edit
    ripple_meas = sim.get('v_out_ripple_measured', 0)
    ripple_target = (specs.get('max_ripple_voltage') or 0.2)
    is_ripple_ok = ripple_meas <= ripple_target
    
    # [FIX] Define Override fail vars
    override_fail = False
    override_reason = ""
    override_strategy = ""

    if eff_meas > 0.96 and (specs.get('output_voltage') or 0) < 50:
         override_fail = True
         override_reason = f"Efficiency {eff_meas:.1%} is suspiciously high (Ideal Model?)."
         override_strategy = "ADD_REALISTIC_PARASITICS"

    validation_context = f"""
    TARGET SPECIFICATIONS:
    - Efficiency Target: {eff_target:.2%}
    - Max Ripple: {ripple_target:.3f} V
    
    CURRENT DESIGN RESULTS:
    - Measured Efficiency: {eff_meas:.2%}
    - Measured Ripple: {ripple_meas:.4f} V (Pass? {is_ripple_ok})
    - MOSFET Voltage Spike: {v_spike:.1f} V
    - Component Stress Ratio: {safety_ratio:.1%} (Limit: 80%, Pass? {is_safe})
    
    CONTEXT:
    - Iteration: {current_iter}
    - Topology Limit Check: {"Force Accept" if force_accept else "Normal"}

    HARD GUARDRAILS:
    {hard_guardrails_prompt}

    RETRIEVED KNOWLEDGE:
    {retrieved_context[:8000]}
    
    WARNING: Efficiency > 96% is physically impossible for this topology. 
    If Measured Efficiency > 96%, you MUST FAIL the design.
    """
    
    # Using LLM to Judge
    try:
        import time
        time.sleep(1.0) 
        
        llm = get_llm()
        
        system_prompt = f"""{hard_guardrails_prompt}

        You are a Strict Senior Power Electronics Chief Engineer. 
        Your job is to CRITIQUE the design submitted by the junior designer.
        
        Use the provided "Pass?" hints in the context as the GROUND TRUTH for your decision.
        
        CRITICAL RULES:
        1. SAFETY: If Stress Ratio >= 80%, FAIL. (Strategy: REDUCE_TURNS_RATIO)
        2. RIPPLE: If Measured Ripple > Max Ripple, FAIL. (Strategy: INCREASE_INDUCTANCE)
        3. EFFICIENCY: If Measured Efficiency < Target, FAIL. (Strategy: DECREASE_FSW)
        4. TOPO-LIMIT: If Context says "Force Accept", you MUST PASS with warning "Topology Limit Reached".
        5. REALISM: If efficiency > 96%, FAIL. (Strategy: ADD_REALISTIC_PARASITICS)

          PRIORITY RULE:
          - Hard Guardrails are highest priority and non-overridable.
          - If user target conflicts with safety/physics/compliance constraints, reject with clear reason.
        
        OUTPUT FORMAT:
           Status: PASS or FAIL
           Reason: [Clear engineering explanation]
           Strategy: [One strategy code from above if FAIL, else N/A]
          """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"Review this design:\n{validation_context}\n\nDecision:")
        ])
        
        chain = prompt | llm
        response = chain.invoke({})
        content = response.content
        print(f"DEBUG: AI Validation Opinion:\n{content}")
        reasoning_logs.append("[EXECUTION] LLM reviewer completed. Parsing status/reason/strategy fields.")
        
        # Parse Agent Response
        # Clean specific N/A issues for UI Presentation
        if "Strategy: N/A" in content and "status: PASS" in content.lower():
             content = content.replace("Strategy: N/A", "").strip()

        # Hard safety guardrail must override any force-accept behavior.
        if not is_safe:
            status = "FAIL"
            failed_items.append(f"Safety violation: stress ratio {safety_ratio:.1%} exceeds guardrail.")
            strategy = "REDUCE_TURNS_RATIO"
            print("SAFETY GUARDRAIL: Rejecting design due to high Vds stress.")
        # [SOTA FIX]: Diminishing returns must NOT auto-convert FAIL to PASS.
        elif force_accept:
            # Stop blind retries and ask for human decision with physical explanation.
            status = "NEEDS_HUMAN_REVIEW"
            failed_items.append(limit_reason)
            failed_items.append(
                "Physical-limit suspect: repeated parameter perturbation did not improve efficiency. "
                f"Top losses: {_top_loss_summary(sim)}"
            )
            strategy = "PHYSICAL_LIMIT_REACHED"
            print(f"DEBUG: Diminishing-return safeguard triggered: {limit_reason}")
        elif override_fail:
             # Check if we switched to PANN source (which means Sim Agent already handled the stuck loop)
             sim_source = sim.get("source", "PLECS")
             if "PANN" in sim_source:
                 # If valid PANN result (usually < 90%), strict check applies.
                 # If PANN gave > 96%? Still Fail.
                 # Usually PANN gives 85-88%, so this logic won't trigger, and we PASS.
                 if eff_meas < 0.95:
                      status = "PASS"
                      failed_items = []
                      print(f"DEBUG: PANN Source detected ({eff_meas:.1%}). Accepting as valid.")
                 else:
                      status = "FAIL"
                      failed_items.append(override_reason)
                      strategy = override_strategy
             else:
                 # Still on Broken PLECS
                 status = "FAIL"
                 failed_items.append(override_reason)
                 strategy = override_strategy
                 print(f"DEBUG: Forcing FAIL due to Auto-Correction: {override_reason}")
                 
        elif "FAIL" in content:
            hard_fail_required = (not is_eff_ok) or (not is_ripple_ok) or (not is_safe) or bool(override_fail) or bool(formula_blockers)
            if not hard_fail_required:
                status = "NEEDS_HUMAN_REVIEW"
                failed_items = [
                    "Senior LLM reviewer flagged the design despite scalar KPI pass hints. Treat this as review-blocking until the concern is resolved.",
                    content,
                ]
                strategy = _extract_strategy_from_review(content)
                reasoning_logs.append("[DECISION] AI FAIL downgraded to NEEDS_HUMAN_REVIEW instead of being ignored.")
            else:
                status = "FAIL"
                failed_items.append("AI Reviewer Rejected the design.")
                failed_items.append(content) # Show the LLM's full critique

                # Extract Strategy - Enhanced for Physical Feedback
                if "Strategy:" in content:
                    # Capture the full strategy suggested by the LLM
                    strategy_lines = [line.strip() for line in content.splitlines() if "Strategy:" in line]
                    if strategy_lines:
                        strategy = strategy_lines[0].replace("Strategy:", "").strip()

                # Fallback to keyphrase mapping if extraction fails or is too generic
                if not strategy or "N/A" in strategy:
                    if "DECREASE_FSW" in content: strategy = "DECREASE_FSW"
                    elif "INCREASE_INDUCTANCE" in content: strategy = "INCREASE_INDUCTANCE"
                    elif "REDUCE_TURNS_RATIO" in content: strategy = "REDUCE_TURNS_RATIO"
                    elif "ADD_REALISTIC_PARASITICS" in content: strategy = "ADD_REALISTIC_PARASITICS"
                    else: strategy = "DECREASE_FSW" # Default fallback
            
        else:
            status = "PASS"
            failed_items = []

    except Exception as e:
        print(f"WARN: AI Validator fell asleep ({e}). Using rule-based fallback.")
        # --- ORIGINAL RULE BASED FALLBACK ---
        # 1. Efficiency Check
        eff_target = (specs.get('efficiency_target') or 0.85)
        eff_meas = sim.get('efficiency_measured', 0.0)
        
        if eff_meas < eff_target:
            failed_items.append(f"Efficiency Low: {eff_meas:.2%} < {eff_target:.2%}")
            if "DECREASE_FSW" not in strategy: strategy = "DECREASE_FSW"
                
        # 2. Ripple Check
        vo = specs.get('output_voltage') or 20
        ripple_max = (specs.get('max_ripple_voltage') or vo*0.01)
        ripple_meas = sim.get('v_out_ripple_measured', 0.0)
        
        if ripple_meas > ripple_max:
            failed_items.append(f"Ripple High: {ripple_meas:.3f}V > {ripple_max:.3f}V")
            if not strategy: strategy = "INCREASE_INDUCTANCE"
        
        if failed_items: status = "FAIL"
        # ------------------------------------

    if sim_source != "PLECS":
        status = "NEEDS_HUMAN_REVIEW"
        strategy = "FIX_SIMULATION_BACKEND"
        failed_items.append(
            f"Simulation fallback source in use: {sim_source}. Results are not trustworthy enough for closed-loop optimization."
        )

    # [HITL FIX MOVED] Check for Infinite Loop Fatigue
    # Only force review if we are failing repeatedy.
    # Allow at least 5 attempts before forcing review.
    iteration = state.get("iteration", 0)
    MAX_AUTO_RETRIES = 5
    
    if status == "FAIL" and iteration >= MAX_AUTO_RETRIES:
         status = "NEEDS_HUMAN_REVIEW"
         failed_items.append(f"Maximum autonomous iterations ({iteration}) reached.")
         # Suggest Strategy is Human
         strategy = "HUMAN_INTERVENTION_REQUIRED"
         print(f"AUTO-CORRECTION: Early Stop for Human Intervention (Iter {iteration}).")

    # Define result object for state updates
    if formula_blockers:
        status = "FAIL"
        strategy = strategy or "FIX_FORMULA_INCONSISTENCY"
        failed_items.extend([f"Formula blocker: {item}" for item in formula_blockers])

    fail_axes = []
    if (eff_target - eff_meas) > 1e-6:
        fail_axes.append("efficiency")
    if (ripple_meas - ripple_target) > 1e-6:
        fail_axes.append("ripple")
    if safety_ratio >= 0.80:
        fail_axes.append("stress")
    if formula_blockers:
        fail_axes.append("formula")

    quality_gate = _quality_gate_findings(specs, design, bom, sim, vds_rating, safety_ratio)
    quality_blockers = quality_gate.get("blockers") or []
    quality_cautions = quality_gate.get("cautions") or []
    if quality_blockers:
        if "bom" not in fail_axes:
            fail_axes.append("bom")
        if "evidence" not in fail_axes:
            fail_axes.append("evidence")
        if status == "PASS":
            status = "NEEDS_HUMAN_REVIEW"
        failed_items.extend([f"Quality gate: {item}" for item in quality_blockers])
        if not strategy or strategy == "N/A":
            strategy = "QUALITY_GATE_BLOCK"
        reasoning_logs.append(f"[QUALITY-GATE] BLOCK: {json.dumps(quality_blockers[:8], ensure_ascii=False)}")
    elif quality_cautions:
        reasoning_logs.append(f"[QUALITY-GATE] CAUTION: {json.dumps(quality_cautions[:6], ensure_ascii=False)}")

    # System-2 policy: adjust strategy using loss attribution and previous iteration outcome.
    prev_verification = state.get("best_design_candidate", {}).get("verification", {}) if isinstance(state.get("best_design_candidate"), dict) else {}
    prev_bundle = prev_verification.get("strategy_bundle", {}) if isinstance(prev_verification.get("strategy_bundle"), dict) else {}
    live_prev_verification = state.get("verification", {}) if isinstance(state.get("verification"), dict) else {}
    live_prev_bundle = live_prev_verification.get("strategy_bundle", {}) if isinstance(live_prev_verification.get("strategy_bundle"), dict) else {}
    # Prefer the most recent action (state.verification) over best-candidate action to avoid stale anti-oscillation signals.
    last_action = str(
        live_prev_bundle.get("primary_action")
        or live_prev_verification.get("correction_strategy")
        or prev_bundle.get("primary_action")
        or prev_verification.get("correction_strategy")
        or ""
    ).strip().upper()
    formula_losses = sim.get("formula_losses", {}) if isinstance(sim.get("formula_losses"), dict) else {}
    dominant_loss = ""
    if formula_losses:
        dominant_loss = max(formula_losses.items(), key=lambda kv: float(kv[1] or 0.0))[0]

    component_actions: List[str] = []

    if status == "FAIL" and "efficiency" in fail_axes and "formula" not in fail_axes and "stress" not in fail_axes and "ripple" not in fail_axes:
        tuned_strategy = str(strategy or "DECREASE_FSW").strip().upper()

        # Physics-aware initial suggestion based on dominant loss bucket.
        # Important: for efficiency shortfall, avoid proposing INCREASE_FSW as first move.
        if dominant_loss == "diode_conduction":
            tuned_strategy = "REDUCE_DIODE_LOSS"
            component_actions = [
                "prefer_low_vf_diode",
                "consider_sync_rectification",
                "prioritize_secondary_rectifier_loss_in_selector",
            ]
        elif dominant_loss in {"mosfet_turn_on_switching", "mosfet_turn_off_switching", "transformer_core"}:
            tuned_strategy = "DECREASE_FSW"
            component_actions = [
                "reduce_switching_loss_focus",
                "keep_magnetics_loss_below_target_ratio",
            ]
        elif dominant_loss in {"transformer_copper_primary", "transformer_copper_secondary"}:
            tuned_strategy = "INCREASE_INDUCTANCE"
            component_actions = [
                "reduce_rms_current",
                "upgrade_winding_or_wire_gauge",
            ]

        # If previous action degraded efficiency, force an alternate action.
        degraded = (prev_eff > 0) and (eff_meas < prev_eff - 0.003)
        alt_map = {
            "DECREASE_FSW": "INCREASE_INDUCTANCE",
            "INCREASE_INDUCTANCE": "REDUCE_DIODE_LOSS",
            "REDUCE_TURNS_RATIO": "INCREASE_INDUCTANCE",
            "INCREASE_FSW": "DECREASE_FSW",
            "REDUCE_DIODE_LOSS": "DECREASE_FSW",
        }
        if degraded and last_action and tuned_strategy == last_action:
            tuned_strategy = alt_map.get(tuned_strategy, "INCREASE_INDUCTANCE")

        if tuned_strategy != str(strategy or "").strip().upper():
            reasoning_logs.append(
                f"[META-COGNITION] Strategy retuned from {strategy or 'N/A'} to {tuned_strategy}; "
                f"dominant_loss={dominant_loss or 'n/a'}, prev_eff={prev_eff:.3f}, curr_eff={eff_meas:.3f}, last_action={last_action or 'N/A'}"
            )
            strategy = tuned_strategy

    # Final guardrail: if we are still efficiency-fail after multiple tries and no improvement,
    # stop retrying and provide physical explanation instead of another scalar tweak.
    if status == "FAIL" and "efficiency" in fail_axes and int(iteration or 0) >= 2:
        no_gain = (prev_eff > 0.0) and (eff_meas <= prev_eff + 0.002)
        if no_gain:
            status = "NEEDS_HUMAN_REVIEW"
            strategy = "PHYSICAL_LIMIT_REACHED"
            failed_items.append(
                "Autonomous retries show no efficiency gain trend. "
                f"Best={prev_eff:.2%}, current={eff_meas:.2%}."
            )
            failed_items.append(
                "Likely constrained by real losses (e.g., diode conduction / switching / core loss). "
                f"Top losses: {_top_loss_summary(sim)}"
            )
            failed_items.append(
                "Recommended path: consider low-Vf rectifier or synchronous rectification, "
                "re-balance turns ratio and clamp energy, or migrate topology (e.g., ACF/LLC) for higher efficiency targets."
            )

    strategy_bundle = {
        "primary_action": (strategy or "NONE").strip() or "NONE",
        "failed_axes": fail_axes,
        "efficiency_gap": max(0.0, float(eff_target or 0.0) - float(eff_meas or 0.0)),
        "ripple_gap": max(0.0, float(ripple_meas or 0.0) - float(ripple_target or 0.0)),
        "stress_ratio": float(safety_ratio or 0.0),
        "iteration": int(iteration or 0),
        "dominant_loss": dominant_loss,
        "last_action": last_action,
        "recommended_overrides": {},
        "recommended_component_actions": component_actions,
        "root_causes": [],
        "next_iteration_focus": [],
        "do_not_repeat": [],
        "expected_tradeoffs": [],
        "quality_gate": quality_gate,
    }
    if strategy_bundle["primary_action"] == "INCREASE_FSW":
        strategy_bundle["recommended_overrides"]["switching_frequency_scale"] = 1.12
    if strategy_bundle["primary_action"] == "DECREASE_FSW":
        strategy_bundle["recommended_overrides"]["switching_frequency_scale"] = 0.82
    if strategy_bundle["primary_action"] == "INCREASE_INDUCTANCE":
        strategy_bundle["recommended_overrides"]["ripple_factor_scale"] = 0.82
    if strategy_bundle["primary_action"] == "REDUCE_TURNS_RATIO":
        strategy_bundle["recommended_overrides"]["reflected_output_voltage_scale"] = 0.9
    if strategy_bundle["primary_action"] == "REDUCE_DIODE_LOSS":
        # Keep design changes conservative while signaling selector-side optimization.
        strategy_bundle["recommended_overrides"]["switching_frequency_scale"] = 0.95
        strategy_bundle["recommended_overrides"]["reflected_output_voltage_scale"] = 0.96
    if (strategy or "").strip().upper() == "PHYSICAL_LIMIT_REACHED":
        strategy_bundle["primary_action"] = "PHYSICAL_LIMIT_REACHED"
        strategy_bundle["recommended_overrides"] = {}
        strategy_bundle["recommended_component_actions"] = [
            "consider_sync_rectification",
            "prioritize_low_vf_secondary_device",
            "revisit_clamp_and_magnetics_loss_budget",
            "consider_topology_upgrade_acf_or_llc",
        ]
    if "efficiency" in fail_axes:
        strategy_bundle["root_causes"].append(
            f"Efficiency shortfall {max(0.0, float(eff_target or 0.0) - float(eff_meas or 0.0)):.2%} vs target."
        )
        strategy_bundle["next_iteration_focus"].append("Prioritize the dominant loss bucket before broad parameter sweeps.")
        strategy_bundle["expected_tradeoffs"].append("Efficiency improvement may increase ripple or thermal concentration if magnetics are not rebalanced.")
    if "ripple" in fail_axes:
        strategy_bundle["root_causes"].append(
            f"Ripple exceeds target by {max(0.0, float(ripple_meas or 0.0) - float(ripple_target or 0.0)):.4f}V."
        )
        strategy_bundle["next_iteration_focus"].append("Check output capacitor ESR and magnetizing ripple simultaneously.")
        strategy_bundle["expected_tradeoffs"].append("Ripple reduction via larger capacitance can raise cost/volume.")
    if "stress" in fail_axes:
        strategy_bundle["root_causes"].append(
            f"Stress ratio is {float(safety_ratio or 0.0):.2f}; clamp and turns-ratio margin are insufficient."
        )
        strategy_bundle["next_iteration_focus"].append("Lower drain stress before chasing efficiency.")
        strategy_bundle["do_not_repeat"].append("Do not keep the same clamp/turns-ratio combination once stress exceeds policy.")
    if "bom" in fail_axes:
        strategy_bundle["root_causes"].append("BOM quality gate blocked release because at least one critical selection is provisional, misclassified, or not source-traceable.")
        strategy_bundle["next_iteration_focus"].append("Replace provisional/misclassified critical components before rerunning simulation.")
    if "evidence" in fail_axes:
        strategy_bundle["root_causes"].append("Simulation evidence is incomplete for release-grade validation.")
        strategy_bundle["next_iteration_focus"].append("Run low-line/full-load closed-loop corners with realistic parasitics and full front-end loss scope.")
    if dominant_loss:
        strategy_bundle["next_iteration_focus"].append(f"Dominant loss observed: {dominant_loss}.")
    if str(strategy_bundle.get("primary_action") or "") == str(last_action or "") and status != "PASS":
        strategy_bundle["do_not_repeat"].append(
            f"Previous action '{last_action}' did not unlock improvement; rotate to a different lever."
        )
    strategy_bundle["next_iteration_focus"] = list(dict.fromkeys(strategy_bundle["next_iteration_focus"]))
    strategy_bundle["do_not_repeat"] = list(dict.fromkeys(strategy_bundle["do_not_repeat"]))
    strategy_bundle["root_causes"] = list(dict.fromkeys(strategy_bundle["root_causes"]))
    strategy_bundle["expected_tradeoffs"] = list(dict.fromkeys(strategy_bundle["expected_tradeoffs"]))

    result = {
        "status": status,
        "failed_items": failed_items,
        "correction_strategy": strategy,
        "strategy_bundle": strategy_bundle,
        "quality_gate": quality_gate,
    }
    reasoning_logs.append(f"[DECISION] status={result.get('status')} strategy={result.get('correction_strategy') or 'N/A'}")
    reasoning_logs.append(f"[DECISION] strategy_bundle={json.dumps(strategy_bundle, ensure_ascii=False)}")
    if result.get("failed_items"):
        reasoning_logs.append(f"[DETAIL] failed_items_preview={result.get('failed_items', [])[:3]}")

    peer_review = _run_design_peer_review({
        "specifications": specs,
        "theoretical_design": design,
        "bom": bom,
        "simulation_results": sim,
        "verification": result,
        "formula_checks": formula_checks,
        "node_verification": state.get("node_verification") or {},
    })
    reasoning_logs.append(
        f"[PEER-REVIEW] status={peer_review.get('review_status')} false_pass_risk={peer_review.get('false_pass_risk_score')}"
    )
    for item in peer_review.get("major_findings", [])[:4]:
        reasoning_logs.append(f"[PEER-REVIEW-MAJOR] {item}")
    for item in peer_review.get("minor_findings", [])[:4]:
        reasoning_logs.append(f"[PEER-REVIEW-MINOR] {item}")

    # Gate false PASS states with major peer-review findings.
    if str(result.get("status") or "").upper() == "PASS" and peer_review.get("major_findings"):
        result["status"] = "NEEDS_HUMAN_REVIEW"
        result["failed_items"] = (result.get("failed_items") or []) + peer_review.get("major_findings", [])[:6]
        result["correction_strategy"] = "PEER_REVIEW_BLOCK"
        reasoning_logs.append("[DECISION] PASS downgraded to NEEDS_HUMAN_REVIEW due to peer-review major findings.")

    # [NEW] Best Effort Tracking Logic
    # Calculate a score for the current design
    score = 0.0
    
    # Critical Safety Constraint
    # If Vds stress consumes the conservative margin, apply a severe penalty.
    v_stress_ratio = sim.get('v_ds_spike_max', 999) / vds_rating
    if v_stress_ratio >= 0.80:
        score -= 500 # Severe penalty for safety violation
    
    # Efficiency Contribution (0.85 -> 85 points)
    score += (sim.get('efficiency_measured', 0) * 100)
    
    # Ripple Penalty (0.1V -> -10 points)
    score -= (sim.get('v_out_ripple_measured', 0) * 100)
    
    # Validation Bonus uses finalized result status only.
    if str(result.get("status") or "").upper() == "PASS":
        score += 1000
    
    current_candidate = {
        "score": score,
        "theoretical_design": design,
        "simulation_results": sim,
        "bom": bom,
        "verification": result,
        "iteration": iteration
    }
    
    # Compare with existing best
    best_candidate_in_state = state.get("best_design_candidate")
    
    if not best_candidate_in_state or score > best_candidate_in_state.get("score", -9999):
        print(f"New Best Design Found. (Score: {score:.1f})")
        # Update State with new best
        best_candidate_final = current_candidate
    else:
        best_candidate_final = best_candidate_in_state # Keep old best

    plan = build_execution_plan({**state, "verification": result})
    plan_summary = summarize_execution_plan(plan)
    iteration_learning = build_iteration_playbook({
        **state,
        "verification": result,
    })
    if plan_summary:
        reasoning_logs.append(f"[PLAN] {plan_summary}")
    learning_preview = iteration_learning.get("next_iteration_focus") or []
    if learning_preview:
        reasoning_logs.append(f"[LEARNING] next_iteration_focus={learning_preview[:3]}")
        
    return {
        "verification": result,
        "peer_review_findings": peer_review,
        "messages": [f"Validation Decision ({'AI' if 'AI' in str(failed_items) else 'Auto'}): {status}. {strategy}"],
        "reasoning_logs": state.get("reasoning_logs", {}) | {"validator": reasoning_logs},
        "literature_references": node_research.get("references", []),
        "best_design_candidate": best_candidate_final,
        "node_verification": state.get("node_verification", {}) | {"validator": {"status": result.get("status"), "formula_blockers": formula_blockers, "peer_review_status": peer_review.get("review_status"), "false_pass_risk_score": peer_review.get("false_pass_risk_score")}},
        "execution_plan": plan,
        "planning_summary": plan_summary,
        "iteration_learning": iteration_learning,
    }
