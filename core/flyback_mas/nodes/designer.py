from typing import Dict, Any, List
from ..state import PowerSupplyState, TheoreticalDesign, ReasoningTraceItem, EvidenceSource
from ..tools.flyback_math import calculate_flyback_params
from core.llm.llm import get_agent_reasoning
from core.utils.web_search import perform_search
from ..knowledge_guardrails import retrieve_flyback_context
from ..formula_guardrails import check_design_equations
from ..lifelong_memory import get_memory_engine, summarize_learning_hits, summarize_skill_hits
from ..planning import build_execution_plan, summarize_execution_plan
from ..tools.input_semantics import resolve_power_stage_input
from .research_helper import collect_node_research
import re
import json

def circuit_designer_node(
    state: PowerSupplyState,
    config: Dict[str, Any] = None,
    *,
    store: Any = None,
) -> Dict[str, Any]:
    """
    Node 2: Circuit Designer
    Deterministic calculation of Flyback parameters with DeepRare Traceable Reasoning.
    Handles literature search, parameter extraction, and feedback loops.
    """
    print("\n" + "="*50)
    print("[circuit_designer_node] START")
    print("="*50)

    specs = state.get("specifications")
    if not specs:
        return {"error_log": ["Missing specifications in Circuit Designer step."]}
        
    if specs.get("is_chitchat"):
        print("Skipping Design: Intent = Chitchat")
        return {}

    retrieved_context = state.get("retrieved_knowledge_context") or ""
    retrieved_refs = state.get("retrieved_knowledge_references") or []
    if not retrieved_context:
        query = (
            f"flyback design constraints Vin {specs.get('input_voltage_min')}-{specs.get('input_voltage_max')} "
            f"Vout {specs.get('output_voltage')} Iout {specs.get('output_current')} "
            f"soft-switching duty-cycle snubber ripple"
        )
        rag_bundle = retrieve_flyback_context(query, top_k=6)
        retrieved_context = rag_bundle.get("context_text", "")
        retrieved_refs = rag_bundle.get("references", [])
    
    # --- SYSTEM 2 REASONING START ---
    print("[System 2] Generating Reasoning Chain...")
    
    current_itr = state.get("iteration", 0)
    verification = state.get("verification")
    current_design = state.get("theoretical_design")
    
    # Get mutable references or init new lists
    reasoning_trace: List[ReasoningTraceItem] = state.get("reasoning_trace", [])
    literature_references: List[EvidenceSource] = state.get("literature_references", [])
    design_overrides = state.get("design_overrides", {})
    collected_logs = [] # Strings for UI log display
    input_domain = resolve_power_stage_input(specs)
    collected_logs.append("[PLAN] Start flyback synthesis pipeline: constraints -> topology equations -> parameter solve.")
    collected_logs.append("[EXECUTION] Reading user constraints and previous iteration feedback.")
    collected_logs.append(
        f"[DATA] Vin={specs.get('input_voltage_min')}-{specs.get('input_voltage_max')}Vac, "
        f"Vout={specs.get('output_voltage')}V, Iout={specs.get('output_current')}A, "
        f"eta_target={specs.get('efficiency_target')}"
    )
    collected_logs.append(
        f"[SEMANTICS] Power-stage bus interpretation: "
        f"{input_domain['dc_bus_min']:.1f}-{input_domain['dc_bus_max']:.1f}Vdc "
        f"({'offline AC rectified bus' if input_domain['is_offline_ac'] else 'direct DC input'})."
    )

    # Lifelong memory retrieval (episodic CBR): reuse stable regions before solving from scratch.
    memory_engine = get_memory_engine()
    strategy_bundle_current = verification.get("strategy_bundle") if isinstance(verification, dict) else {}
    mem_query = (
        f"flyback Vin {specs.get('input_voltage_min')}-{specs.get('input_voltage_max')} "
        f"Vout {specs.get('output_voltage')} Iout {specs.get('output_current')} "
        f"eff {specs.get('efficiency_target')}"
    )
    successful_hits = memory_engine.search(
        ("episodes", "flyback", "successful_designs"),
        query=mem_query,
        limit=3,
        store=store,
    )
    failed_hits = memory_engine.search(
        ("episodes", "flyback", "failed_or_review"),
        query=mem_query,
        limit=2,
        store=store,
    )
    lesson_hits = memory_engine.search(
        ("lessons", "flyback", "iteration_playbooks"),
        query=mem_query,
        limit=4,
        store=store,
    )
    skill_hits = memory_engine.search(
        ("skills", "flyback", "design_patterns"),
        query=(
            mem_query
            + f" dominant_loss {strategy_bundle_current.get('dominant_loss') or ''}"
            + f" primary_action {strategy_bundle_current.get('primary_action') or ''}"
        ),
        limit=4,
        store=store,
    )
    lesson_guidance = summarize_learning_hits(lesson_hits)
    skill_guidance = summarize_skill_hits(skill_hits)

    if successful_hits:
        collected_logs.append(f"[MEMORY] Retrieved {len(successful_hits)} successful historical episodes.")
        top_payload = (successful_hits[0] or {}).get("payload") or {}
        prev_design = top_payload.get("theoretical_design") or {}
        if isinstance(prev_design, dict):
            if not design_overrides.get("switching_frequency") and prev_design.get("switching_frequency"):
                design_overrides["switching_frequency"] = prev_design.get("switching_frequency")
            if not design_overrides.get("reflected_output_voltage") and prev_design.get("reflected_output_voltage"):
                design_overrides["reflected_output_voltage"] = prev_design.get("reflected_output_voltage")
            if not design_overrides.get("ripple_factor") and prev_design.get("ripple_factor") is not None:
                try:
                    ripple_hint = float(prev_design.get("ripple_factor"))
                    if 0.1 <= ripple_hint <= 0.8:
                        design_overrides["ripple_factor"] = ripple_hint
                except Exception:
                    pass
        collected_logs.append(
            f"[MEMORY] Applied memory-informed overrides: fsw={design_overrides.get('switching_frequency')}, "
            f"vor={design_overrides.get('reflected_output_voltage')}, k_rf={design_overrides.get('ripple_factor')}"
        )

    if failed_hits:
        collected_logs.append(f"[MEMORY] Retrieved {len(failed_hits)} failed/review episodes to avoid known dead-ends.")
    if lesson_hits:
        collected_logs.append(f"[LEARNING] Retrieved {len(lesson_hits)} structured iteration playbooks.")
        if lesson_guidance.get("focus"):
            collected_logs.append(f"[LEARNING] Focus from similar iterations: {lesson_guidance.get('focus')[:3]}")
        if lesson_guidance.get("avoid"):
            collected_logs.append(f"[LEARNING] Avoid-patterns: {lesson_guidance.get('avoid')[:2]}")
        lesson_overrides = lesson_guidance.get("suggested_overrides") if isinstance(lesson_guidance.get("suggested_overrides"), dict) else {}
        if lesson_overrides:
            if not design_overrides.get("switching_frequency") and lesson_overrides.get("switching_frequency_scale"):
                base_fsw = float((current_design or {}).get("switching_frequency") or 65000.0)
                design_overrides["switching_frequency"] = max(30000.0, base_fsw * float(lesson_overrides.get("switching_frequency_scale")))
            if not design_overrides.get("reflected_output_voltage") and lesson_overrides.get("reflected_output_voltage_scale"):
                base_vor = float((current_design or {}).get("reflected_output_voltage") or 80.0)
                design_overrides["reflected_output_voltage"] = max(
                    float(specs.get("output_voltage") or 12.0) + 2.0,
                    base_vor * float(lesson_overrides.get("reflected_output_voltage_scale")),
                )
            if not design_overrides.get("ripple_factor") and lesson_overrides.get("ripple_factor_scale"):
                base_k = float(design_overrides.get("ripple_factor") or 0.4)
                design_overrides["ripple_factor"] = max(
                    0.15,
                    min(0.75, base_k * float(lesson_overrides.get("ripple_factor_scale"))),
                )
            collected_logs.append(
                f"[LEARNING] Applied playbook-informed overrides: fsw={design_overrides.get('switching_frequency')}, "
                f"vor={design_overrides.get('reflected_output_voltage')}, k_rf={design_overrides.get('ripple_factor')}"
            )

    if skill_hits:
        collected_logs.append(f"[SKILL] Retrieved {len(skill_hits)} reusable design/debug skill cards.")
        if skill_guidance.get("objectives"):
            collected_logs.append(f"[SKILL] Objectives matched: {skill_guidance.get('objectives')[:3]}")
        if skill_guidance.get("focus"):
            collected_logs.append(f"[SKILL] Focus cues: {skill_guidance.get('focus')[:3]}")
        if skill_guidance.get("avoid"):
            collected_logs.append(f"[SKILL] Avoid-patterns: {skill_guidance.get('avoid')[:2]}")

        skill_warm_start = skill_guidance.get("warm_start_hints") if isinstance(skill_guidance.get("warm_start_hints"), dict) else {}
        if skill_warm_start:
            if not design_overrides.get("switching_frequency") and skill_warm_start.get("switching_frequency"):
                design_overrides["switching_frequency"] = max(30000.0, float(skill_warm_start.get("switching_frequency")))
            if not design_overrides.get("reflected_output_voltage") and skill_warm_start.get("reflected_output_voltage"):
                design_overrides["reflected_output_voltage"] = max(
                    float(specs.get("output_voltage") or 12.0) + 2.0,
                    float(skill_warm_start.get("reflected_output_voltage")),
                )
            if not design_overrides.get("ripple_factor") and skill_warm_start.get("ripple_factor") is not None:
                design_overrides["ripple_factor"] = max(
                    0.15,
                    min(0.75, float(skill_warm_start.get("ripple_factor"))),
                )
            collected_logs.append(
                f"[SKILL] Applied warm-start hints: fsw={design_overrides.get('switching_frequency')}, "
                f"vor={design_overrides.get('reflected_output_voltage')}, k_rf={design_overrides.get('ripple_factor')}"
            )

        skill_overrides = skill_guidance.get("suggested_overrides") if isinstance(skill_guidance.get("suggested_overrides"), dict) else {}
        if skill_overrides:
            if not design_overrides.get("switching_frequency") and skill_overrides.get("switching_frequency_scale"):
                base_fsw = float((current_design or {}).get("switching_frequency") or skill_warm_start.get("switching_frequency") or 65000.0)
                design_overrides["switching_frequency"] = max(30000.0, base_fsw * float(skill_overrides.get("switching_frequency_scale")))
            if not design_overrides.get("reflected_output_voltage") and skill_overrides.get("reflected_output_voltage_scale"):
                base_vor = float((current_design or {}).get("reflected_output_voltage") or skill_warm_start.get("reflected_output_voltage") or 80.0)
                design_overrides["reflected_output_voltage"] = max(
                    float(specs.get("output_voltage") or 12.0) + 2.0,
                    base_vor * float(skill_overrides.get("reflected_output_voltage_scale")),
                )
            if not design_overrides.get("ripple_factor") and skill_overrides.get("ripple_factor_scale"):
                base_k = float(design_overrides.get("ripple_factor") or skill_warm_start.get("ripple_factor") or 0.4)
                design_overrides["ripple_factor"] = max(
                    0.15,
                    min(0.75, base_k * float(skill_overrides.get("ripple_factor_scale"))),
                )
            collected_logs.append(
                f"[SKILL] Applied skill-informed overrides: fsw={design_overrides.get('switching_frequency')}, "
                f"vor={design_overrides.get('reflected_output_voltage')}, k_rf={design_overrides.get('ripple_factor')}"
            )

    node_research = collect_node_research(
        "designer",
        (
            f"flyback design papers blogs forums app notes for Vin {specs.get('input_voltage_min')}-{specs.get('input_voltage_max')} "
            f"Vout {specs.get('output_voltage')} Iout {specs.get('output_current')} switching frequency duty snubber"
        ),
        max_results=8,
    )
    collected_logs.extend(node_research.get("logs", []))
    collected_logs.append("[SEARCH] Screening app notes and forum hints for duty-cycle/snubber best practices.")
    ext_refs = node_research.get("references", [])
    if isinstance(ext_refs, list) and ext_refs:
        for ref in ext_refs:
            if isinstance(ref, dict):
                literature_references.append(ref)
    
    # 1. Feedback Processing
    feedback_msg = None
    if verification and verification.get("status") == "FAIL":
        feedback_msg = verification.get("correction_strategy")
        strategy_bundle = verification.get("strategy_bundle") if isinstance(verification.get("strategy_bundle"), dict) else {}
        print(f"Feedback Received: {feedback_msg}")
        collected_logs.append(f"[CORRECTION] Feedback strategy received from validator: {feedback_msg}")
        if strategy_bundle:
            collected_logs.append(f"[CORRECTION] Structured strategy bundle: {json.dumps(strategy_bundle, ensure_ascii=False)}")
            if strategy_bundle.get("root_causes"):
                collected_logs.append(f"[LEARNING] Root causes carried into new iteration: {strategy_bundle.get('root_causes')[:3]}")
            if strategy_bundle.get("next_iteration_focus"):
                collected_logs.append(f"[LEARNING] Next-iteration focus: {strategy_bundle.get('next_iteration_focus')[:4]}")
            if strategy_bundle.get("do_not_repeat"):
                collected_logs.append(f"[LEARNING] Do-not-repeat list: {strategy_bundle.get('do_not_repeat')[:2]}")
            rec = strategy_bundle.get("recommended_overrides") if isinstance(strategy_bundle.get("recommended_overrides"), dict) else {}
            if rec:
                curr_fsw = float((current_design or {}).get("switching_frequency") or design_overrides.get("switching_frequency") or 65000.0)
                curr_vor = float((current_design or {}).get("reflected_output_voltage") or design_overrides.get("reflected_output_voltage") or 80.0)
                curr_k = float(design_overrides.get("ripple_factor") or 0.4)

                fsw_scale = float(rec.get("switching_frequency_scale") or 1.0)
                vor_scale = float(rec.get("reflected_output_voltage_scale") or 1.0)
                k_scale = float(rec.get("ripple_factor_scale") or 1.0)

                if abs(fsw_scale - 1.0) > 1e-6:
                    design_overrides["switching_frequency"] = max(30000.0, curr_fsw * fsw_scale)
                if abs(vor_scale - 1.0) > 1e-6:
                    design_overrides["reflected_output_voltage"] = max(float(specs.get("output_voltage") or 12.0) + 2.0, curr_vor * vor_scale)
                if abs(k_scale - 1.0) > 1e-6:
                    design_overrides["ripple_factor"] = max(0.15, min(0.75, curr_k * k_scale))
                collected_logs.append(
                    f"[CORRECTION] Applied deterministic overrides: fsw={design_overrides.get('switching_frequency')}, "
                    f"vor={design_overrides.get('reflected_output_voltage')}, k_rf={design_overrides.get('ripple_factor')}"
                )
        
        reasoning_trace.append({
            "step": f"Iteration {current_itr} Planning",
            "agent": "Designer",
            "action": "Review Feedback",
            "evidence": "Verification Report",
            "confidence": 1.0,
            "verification_status": "Pending"
        })
    
    # 2. [DEEPRARE FEATURE]: Knowledge Retrieval (Literature Search)
    # Perform search only on first iteration or if specifically requested by feedback
    if current_itr == 0 and not literature_references:
        print("[Knowledge Agent] Searching for Design Methodology Literature...")
        collected_logs.append("[STRATEGY] Consulting Academic Literature (ArXiv/IEEE) for optimal methodology.")
        
        try:
            query = f"flyback converter design paper SMPS transformer snubber efficiency {specs.get('output_voltage')}V {specs.get('output_current')}A"
            collected_logs.append(f"[SEARCH] Searching for: {query}")
            
            search_results = perform_search(query, max_results=5)
            
            for result in search_results:
                # Assuming perform_search returns strings. We need to parse or struct them.
                # If result is just text, we wrap it.
                # Extract URL if possible
                link_match = re.search(r"(http[s]?://[^\s]+)", result)
                url = link_match.group(1) if link_match else "N/A"
                title = result.split('\n')[0] if '\n' in result else result[:50]

                result_text = str(result).lower()
                title_text = str(title).lower()
                if "withdrawn" in result_text or "withdrawn" in title_text:
                    continue
                relevance_keys = ["flyback", "converter", "smps", "power", "transformer", "snubber", "switching"]
                if not any(k in result_text for k in relevance_keys):
                    continue
                
                evidence_item: EvidenceSource = {
                    "source_type": "Paper",
                    "title": title,
                    "url": url,
                    "insight": result[:200] + "..." # Truncate for summary
                }
                literature_references.append(evidence_item)
                print(f"Found Literature: {title}")
                collected_logs.append(f"[DATA] Found Literature: {title}")

            if not literature_references:
                collected_logs.append("[WARNING] No high-relevance literature found; using handbook-based formula defaults.")
            else:
                collected_logs.append(f"[RESULT] Literature pool assembled: {len(literature_references)} references.")

            state["literature_references"] = literature_references

            # 3. Analyze Literature for Parameters
            if literature_references:
                print("[System 2] Analyzing Literature for Design Insights...")
                papers_text = "\n\n".join([f"Source: {ref['title']}\n{ref['insight']}" for ref in literature_references])
                
                analysis_prompt = f"""
                Analyze the following literature snippets for a {specs.get('output_voltage')}V {specs.get('output_current')}A Flyback converter.
                Extract key design values if explicitly mentioned or recommended.
                Look for:
                - Switching Frequency [FSW]
                - Reflected Output Voltage [VOR] (or turns ratio hints)
                - Current Ripple Factor [K_RF]
                
                Literature:
                {papers_text}
                
                Output ONLY valid JSON with keys: 'switching_frequency', 'reflected_output_voltage', 'ripple_factor'. 
                Use null if not found.
                """
                
                analysis_logs = get_agent_reasoning(
                    agent_role="Senior Power Electronics Researcher",
                    task=analysis_prompt,
                    context_data={"specs": specs},
                    history_trace=[]
                )
                
                # Heuristic parsing of JSON from logs (assuming the LLM returns JSON in a code block or plain text)
                # This is a simplification; ideally get_agent_reasoning returns structured data or we parse the last message.
                # For now, we'll scan logs for JSON-like structures or keywords.
                
                # Actually, let's just parse the last log message if it looks like JSON
                if analysis_logs:
                    last_log = analysis_logs[-1]
                    # Try to find JSON block
                    json_match = re.search(r"\{.*\}", last_log, re.DOTALL)
                    if json_match:
                        try:
                            extracted_params = json.loads(json_match.group(0))
                            
                            # Validate and Apply Overrides
                            if extracted_params.get("reflected_output_voltage"):
                                vor = float(extracted_params["reflected_output_voltage"])
                                if vor > specs.get("output_voltage", 0):
                                    design_overrides["reflected_output_voltage"] = vor
                                    reasoning_trace.append({
                                        "step": "Parameter Selection",
                                        "agent": "Designer",
                                        "action": f"Set VOR = {vor}V",
                                        "evidence": "Literature Analysis",
                                        "citation": "Extracted from papers",
                                        "confidence": 0.8
                                    })
                            
                            if extracted_params.get("switching_frequency"):
                                fsw = float(extracted_params["switching_frequency"])
                                design_overrides["switching_frequency"] = fsw
                                reasoning_trace.append({
                                    "step": "Parameter Selection",
                                    "agent": "Designer",
                                    "action": f"Set FSW = {fsw}Hz",
                                    "evidence": "Literature Analysis",
                                    "citation": "Extracted from papers",
                                    "confidence": 0.8
                                })
                                
                        except json.JSONDecodeError:
                            print("Failed to parse JSON from Literature Analysis")
                            
        except Exception as e:
            print(f"Literature Search Failed: {e}")
            collected_logs.append(f"[WARNING] Literature Search Logic Failed: {e}")

    # 4. Math Calculation
    
    try:
        print("\nCalculating Theoretical Parameters...")
        collected_logs.append("[EXECUTION] Running deterministic flyback equations (Lm, turns ratio, Dmax, snubber seed).")
        new_design_params = calculate_flyback_params(
            spec=specs,
            current_design=current_design,
            feedback=feedback_msg,
            overrides=design_overrides
        )
        collected_logs.append(
            f"[RESULT] Preliminary solve: fsw={new_design_params.get('switching_frequency')}Hz, "
            f"Dmax={new_design_params.get('max_duty_cycle'):.3f}, Vor={new_design_params.get('reflected_output_voltage')}V"
        )
        
        # Log the calculation step
        reasoning_trace.append({
            "step": "Mathematical Calculation",
            "agent": "Designer",
            "action": "Computed Lm, Np, Ns, stress values",
            "evidence": "Standard Flyback Formulas",
            "confidence": 1.0,
            "verification_status": "Pending"
        })
        
        print("\nCalculated Parameters:")
        print(json.dumps(new_design_params, indent=2))
        
        # Fall through to original return

    except Exception as e:
        return {"error_log": [f"Math Calc Failed: {str(e)}"]}

    # Prepare return messages
    base_msgs = [f"Design Iteration {current_itr + 1} complete."]

    design_check = check_design_equations(specs, new_design_params)
    for item in design_check.get("checks", []):
        expected_val = item.get("expected", item.get("required", "-"))
        actual_val = item.get("actual", "-")
        if isinstance(expected_val, (int, float)):
            expected_str = f"{expected_val:.4g}"
        else:
            expected_str = str(expected_val)
        if isinstance(actual_val, (int, float)):
            actual_str = f"{actual_val:.4g}"
        else:
            actual_str = str(actual_val)
        collected_logs.append(
            f"[FORMULA] {item.get('name')}: expected={expected_str}, actual={actual_str}, pass={item.get('pass')}"
        )
    collected_logs.append("[DECISION] Designer node completed. Passing theoretical design to selector for BOM screening.")
    for warning in design_check.get("warnings", []):
        collected_logs.append(f"[FORMULA-WARN] {warning}")
    for fatal in design_check.get("fatal", []):
        collected_logs.append(f"[FORMULA-FAIL] {fatal}")
    
    # Update logs
    new_logs = state.get("reasoning_logs", {}) or {}
    plan = build_execution_plan({**state, "theoretical_design": new_design_params})
    plan_summary = summarize_execution_plan(plan)
    if plan_summary:
        collected_logs.append(f"[PLAN] {plan_summary}")
    new_logs["designer"] = collected_logs
    
    return {
        "theoretical_design": new_design_params,
        "iteration": current_itr + 1,
        "messages": base_msgs,
        "retrieved_knowledge_context": retrieved_context,
        "retrieved_knowledge_references": retrieved_refs,
        "formula_checks": state.get("formula_checks", {}) | {"designer": design_check},
        "node_verification": state.get("node_verification", {}) | {"designer": {"status": "PASS" if not design_check.get("fatal") else "FAIL", "fatal": design_check.get("fatal", []), "warnings": design_check.get("warnings", [])}},
        "reasoning_logs": new_logs,
        "reasoning_trace": reasoning_trace,
        "literature_references": literature_references,
        "design_overrides": design_overrides,
        "execution_plan": plan,
        "planning_summary": plan_summary,
        "learning_context": {
            "lesson_query": mem_query,
            "lesson_hits": lesson_hits,
            "lesson_guidance": lesson_guidance,
            "skill_hits": skill_hits,
            "skill_guidance": skill_guidance,
            "current_strategy_bundle": verification.get("strategy_bundle") if isinstance(verification, dict) else {},
        },
        "memory_context": {
            **(state.get("memory_context") or {}),
            "designer": {
                "query": mem_query,
                "successful_hits": successful_hits,
                "failed_hits": failed_hits,
                "lesson_hits": lesson_hits,
                "skill_hits": skill_hits,
            },
        },
    }
