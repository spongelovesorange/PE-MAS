from typing import Dict, Any, List
import os

from ..state import PowerSupplyState
from ..skills_manager import SkillManager
from ..formula_guardrails import check_bom_margins
from ..lifelong_memory import index_skills_for_dynamic_binding, select_dynamic_skills
from ..planning import build_execution_plan, summarize_execution_plan


def _build_runtime_focus(state: PowerSupplyState, user_query: str) -> str:
    specs = state.get("specifications") or {}
    verification = state.get("verification") or {}
    corr = state.get("correction_review") or {}
    return (
        f"{user_query} | "
        f"status={verification.get('status')} correction={corr.get('status')} | "
        f"Vin={specs.get('input_voltage_min')}-{specs.get('input_voltage_max')} "
        f"Vout={specs.get('output_voltage')} Iout={specs.get('output_current')}"
    )


def _execute_skill_tools(skill: Any, context: Dict[str, Any], output_data: Dict[str, Any], reasoning_logs: List[str]) -> str:
    report_content = ""
    if not skill.tools_module:
        return report_content

    user_query = context["user_input"] or ""

    if hasattr(skill.tools_module, "review_design"):
        try:
            peer_pack = skill.tools_module.review_design(context)
            output_data["peer_review_findings"] = peer_pack
            reasoning_logs.append(f"[TOOL_OUTPUT] {skill.id}: design peer review completed")
        except Exception as e:
            reasoning_logs.append(f"[TOOL_ERROR] {skill.id}: design peer review failed: {e}")

    if hasattr(skill.tools_module, "grade_evidence"):
        try:
            ev_pack = skill.tools_module.grade_evidence(context)
            output_data["evidence_grade"] = ev_pack
            reasoning_logs.append(f"[TOOL_OUTPUT] {skill.id}: evidence grading completed")
        except Exception as e:
            reasoning_logs.append(f"[TOOL_ERROR] {skill.id}: evidence grading failed: {e}")

    if hasattr(skill.tools_module, "build_citation_pack"):
        try:
            citation_pack = skill.tools_module.build_citation_pack(context)
            output_data["citation_pack"] = citation_pack
            reasoning_logs.append(f"[TOOL_OUTPUT] {skill.id}: citation audit completed")
        except Exception as e:
            reasoning_logs.append(f"[TOOL_ERROR] {skill.id}: citation audit failed: {e}")

    if hasattr(skill.tools_module, "check_consistency"):
        try:
            consistency_pack = skill.tools_module.check_consistency(context)
            output_data["simulation_consistency"] = consistency_pack
            reasoning_logs.append(f"[TOOL_OUTPUT] {skill.id}: simulation consistency check completed")
        except Exception as e:
            reasoning_logs.append(f"[TOOL_ERROR] {skill.id}: simulation consistency check failed: {e}")

    if hasattr(skill.tools_module, "plan_parameter_sensitivity"):
        try:
            sens_pack = skill.tools_module.plan_parameter_sensitivity(context)
            output_data["param_sensitivity_plan"] = sens_pack
            reasoning_logs.append(f"[TOOL_OUTPUT] {skill.id}: parameter sensitivity planning completed")
        except Exception as e:
            reasoning_logs.append(f"[TOOL_ERROR] {skill.id}: parameter sensitivity planning failed: {e}")

    if hasattr(skill.tools_module, "generate_final_report"):
        try:
            report_pack = skill.tools_module.generate_final_report(context)
            output_data["final_report"] = report_pack
            report_content = str(report_pack.get("report_markdown") or report_content)
            reasoning_logs.append(f"[TOOL_OUTPUT] {skill.id}: final_report generated")

            if report_content and hasattr(skill.tools_module, "write_report_artifact"):
                save_res = skill.tools_module.write_report_artifact(report_content, "design_report.md")
                output_data["report_artifact"] = save_res
                reasoning_logs.append(f"[TOOL_OUTPUT] {skill.id}: report artifact saved")
        except Exception as e:
            reasoning_logs.append(f"[TOOL_ERROR] {skill.id}: final_report generation failed: {e}")

    if hasattr(skill.tools_module, "research_web"):
        try:
            web_result = skill.tools_module.research_web(user_query, max_results=5)
            output_data["web_research"] = web_result
            reasoning_logs.append(f"[TOOL_OUTPUT] {skill.id}: web_research completed")
        except Exception as e:
            reasoning_logs.append(f"[TOOL_ERROR] {skill.id}: research_web failed: {e}")

    if hasattr(skill.tools_module, "fetch_datasheet_text") and "datasheet" in str(user_query).lower():
        try:
            ds_result = skill.tools_module.fetch_datasheet_text("http://example.com/mosfet.pdf")
            output_data["datasheet"] = ds_result
            reasoning_logs.append(f"[TOOL_OUTPUT] {skill.id}: datasheet fetched")
        except Exception as e:
            reasoning_logs.append(f"[TOOL_ERROR] {skill.id}: datasheet fetch failed: {e}")
    return report_content


def skill_executor_node(
    state: PowerSupplyState,
    config: Dict[str, Any] = None,
    *,
    store: Any = None,
) -> Dict[str, Any]:
    """
    Executes the currently active skill.
    """
    active_skill_id = state.get("active_skill")
    if not active_skill_id:
        return {"error_log": ["No active skill found in state."]}

    # core/flyback_mas/nodes -> core/skills
    skills_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "skills",
    )
    skill_manager = SkillManager(skills_dir)
    skill = skill_manager.get_skill(active_skill_id)
    if not skill:
        return {"error_log": [f"Skill {active_skill_id} not found."]}

    print(f"\n[Skill Executor] Running Skill: {skill.get_info()['name']}")

    messages_state = state.get("messages", []) or []
    last_user_text = ""
    if messages_state:
        maybe_last = messages_state[-1]
        last_user_text = str(getattr(maybe_last, "content", maybe_last))

    context = {
        "user_input": state.get("user_input") or last_user_text,
        "specifications": state.get("specifications"),
        "request_profile": state.get("request_profile"),
        "theoretical_design": state.get("theoretical_design"),
        "bom": state.get("bom"),
        "simulation_results": state.get("simulation_results"),
        "verification": state.get("verification"),
        "correction_review": state.get("correction_review"),
        "formula_checks": state.get("formula_checks"),
        "node_verification": state.get("node_verification"),
        "literature_references": state.get("literature_references"),
        "retrieved_knowledge_references": state.get("retrieved_knowledge_references"),
        "reasoning_trace": state.get("reasoning_trace"),
        "config": state.get("config"),
        "skill_history": state.get("skill_state", {}).get("skill_history", []),
    }

    print("[Skill Agent] Reasoning...")
    reasoning_logs = [
        "[PLAN] Use dynamic tool binding: retrieve top-k relevant skills from long-term memory namespace.",
        "[EXECUTION] Running selected skills and merging artifacts into state.",
    ]

    output_data: Dict[str, Any] = {}
    formula_logs: list[str] = []
    report_content = ""

    _ = skill.prompt.format(context=str(context))
    focus = _build_runtime_focus(state, context["user_input"] or "")
    indexed = index_skills_for_dynamic_binding(skill_manager, store=store)
    skill_catalog = skill_manager.catalog_snapshot()
    available_ids = [s.get("id") for s in skill_manager.list_skills() if s.get("id")]
    selected_ids = select_dynamic_skills(focus, available_ids, top_k=5, store=store)
    manifest_ranked = skill_manager.recommend_skills(focus, active_skill_id=active_skill_id, limit=5)
    manifest_selected = [str(item.get("id")) for item in manifest_ranked if str(item.get("id") or "").strip()]
    if active_skill_id not in selected_ids:
        selected_ids = [active_skill_id] + selected_ids
    for sid in manifest_selected:
        if sid and sid not in selected_ids:
            selected_ids.append(sid)
    selected_ids = selected_ids[:5]
    reasoning_logs.append(f"[MEMORY] Skill metadata indexed into skills/executable_tools: {indexed}")
    reasoning_logs.append(f"[MEMORY] Dynamic selected skills(top-k): {selected_ids}")
    for item in manifest_ranked[:3]:
        reasoning_logs.append(
            f"[PLAN] Skill candidate {item.get('id')}: score={item.get('score')} reason={item.get('reason')}"
        )

    for sid in selected_ids:
        sk = skill_manager.get_skill(sid)
        if not sk:
            continue
        report_content = _execute_skill_tools(sk, context, output_data, reasoning_logs) or report_content

    specs = state.get("specifications") or {}
    design = state.get("theoretical_design") or {}
    bom = state.get("bom") or {}
    if specs and design and bom:
        bom_check = check_bom_margins(specs, design, bom)
        for check_item in bom_check.get("checks", []):
            formula_logs.append(
                f"[FORMULA] {check_item.get('name')}: required={check_item.get('required', '-')}, "
                f"actual={check_item.get('actual', '-')}, pass={check_item.get('pass')}"
            )
        for warn in bom_check.get("warnings", []):
            formula_logs.append(f"[FORMULA-WARN] {warn}")
    else:
        bom_check = {"checks": [], "warnings": ["Missing specs/design/bom for formula check"]}

    new_skill_state = state.get("skill_state", {}) or {}
    new_skill_state["active_skill_id"] = active_skill_id
    new_skill_state["skill_output"] = output_data
    history = new_skill_state.get("skill_history", [])
    history.append(f"Executed dynamic skill bundle: {selected_ids}")
    new_skill_state["skill_history"] = history

    current_logs = state.get("reasoning_logs", {}) or {}
    current_logs[f"skill_{active_skill_id}"] = reasoning_logs + formula_logs
    plan = build_execution_plan(state)
    plan_summary = summarize_execution_plan(plan)

    if report_content.strip():
        messages_out = [f"Skill '{active_skill_id}' executed successfully. Final report generated."]
    else:
        messages_out = [f"Skill '{active_skill_id}' executed successfully."]

    return {
        "skill_state": new_skill_state,
        "report_content": report_content or state.get("report_content", ""),
        "peer_review_findings": output_data.get("peer_review_findings") or state.get("peer_review_findings"),
        "evidence_grade": output_data.get("evidence_grade") or state.get("evidence_grade"),
        "citation_pack": output_data.get("citation_pack") or state.get("citation_pack"),
        "simulation_consistency": output_data.get("simulation_consistency") or state.get("simulation_consistency"),
        "param_sensitivity_plan": output_data.get("param_sensitivity_plan") or state.get("param_sensitivity_plan"),
        "messages": messages_out,
        "reasoning_logs": current_logs,
        "formula_checks": state.get("formula_checks", {}) | {f"skill_{active_skill_id}": bom_check},
        "execution_plan": plan,
        "planning_summary": plan_summary,
        "skill_catalog": skill_catalog,
        "skill_recommendations": manifest_ranked,
        "node_verification": state.get("node_verification", {}) | {
            "skill_executor": {
                "status": "PASS" if not bom_check.get("warnings") else "WARN",
                "warnings": bom_check.get("warnings", []),
            }
        },
    }
