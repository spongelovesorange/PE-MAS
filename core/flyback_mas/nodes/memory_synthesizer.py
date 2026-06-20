from typing import Any, Dict

from ..lifelong_memory import (
    build_episode_summary,
    build_iteration_playbook,
    build_skill_card_from_state,
    build_semantic_rule_from_state,
    get_memory_engine,
    mark_state_memory_usage,
    summarize_memory_hits,
    summarize_learning_hits,
    summarize_skill_hits,
)
from ..planning import build_execution_plan, summarize_execution_plan
from ..state import PowerSupplyState


def memory_synthesizer_node(
    state: PowerSupplyState,
    config: Dict[str, Any] = None,
    *,
    store: Any = None,
) -> Dict[str, Any]:
    """
    Consolidates one run into long-term memory.
    Writes both:
    - episodic memory: concrete design trajectory snapshot
    - semantic memory: abstract rule distilled from success/failure
    """
    engine = get_memory_engine()

    verification = state.get("verification") or {}
    status = str(verification.get("status") or "UNKNOWN")

    episode_payload = {
        "summary": build_episode_summary(state),
        "status": status,
        "specifications": state.get("specifications") or {},
        "theoretical_design": state.get("theoretical_design") or {},
        "simulation_results": state.get("simulation_results") or {},
        "verification": verification,
        "correction_review": state.get("correction_review") or {},
        "bom_excerpt": {
            "mosfet": (state.get("bom") or {}).get("mosfet"),
            "diode": (state.get("bom") or {}).get("diode"),
            "controller": (state.get("bom") or {}).get("controller"),
        },
    }

    episode_ns = (
        "episodes",
        "flyback",
        "successful_designs" if status == "PASS" else "failed_or_review",
    )
    ep_res = engine.put(episode_ns, episode_payload, kind="episode", store=store)

    playbook_payload = state.get("iteration_learning") or build_iteration_playbook(state)
    playbook_res = engine.put(
        ("lessons", "flyback", "iteration_playbooks"),
        playbook_payload,
        kind="iteration_playbook",
        store=store,
    )

    skill_card_payload = build_skill_card_from_state({
        **state,
        "iteration_learning": playbook_payload,
    })
    skill_res: Dict[str, Any] = {}
    if skill_card_payload:
        skill_res = engine.put(
            ("skills", "flyback", "design_patterns"),
            skill_card_payload,
            kind="skill_card",
            store=store,
        )

    sem_rule = build_semantic_rule_from_state(state)
    sem_res: Dict[str, Any] = {}
    if sem_rule:
        sem_payload = {
            "summary": sem_rule,
            "status": status,
            "tags": ["flyback", "heuristic", "validator"],
            "quality_score": 0.7 if status == "PASS" else 0.55,
        }
        sem_res = engine.put(("semantic", "flyback", "rules"), sem_payload, kind="semantic_rule", store=store)

    lookback_query = episode_payload["summary"]
    recent_episodes = engine.search(episode_ns, query=lookback_query, limit=3, store=store)
    recent_lessons = engine.search(("lessons", "flyback", "iteration_playbooks"), query=lookback_query, limit=3, store=store)
    recent_skills = engine.search(("skills", "flyback", "design_patterns"), query=lookback_query, limit=3, store=store)
    usage_update = mark_state_memory_usage(
        state,
        engine=engine,
        success=status == "PASS",
        failed=status in {"FAIL", "NEEDS_HUMAN_REVIEW", "TOPOLOGY_CHANGE_NEEDED"},
    )
    memory_stats = engine.stats()
    plan = build_execution_plan(state)
    plan_summary = summarize_execution_plan(plan)

    logs = state.get("reasoning_logs", {}) or {}
    logs["memory_synthesizer"] = [
        "[PLAN] Distill episodic and semantic memory from the latest run.",
        f"[RESULT] Episodic memory stored via {ep_res.get('backend')} in {ep_res.get('namespace')}.",
        f"[RESULT] Iteration playbook stored via {playbook_res.get('backend')} in {playbook_res.get('namespace')}.",
        f"[RESULT] Skill card stored: {'yes' if skill_res else 'no'}.",
        f"[RESULT] Semantic memory stored: {'yes' if sem_res else 'no'}.",
        f"[MEMORY] Updated usage counters for {usage_update.get('count')} retrieved memories.",
        f"[DATA] Final verification status: {status}.",
        f"[MEMORY] Total stored memories: {memory_stats.get('total_rows')}.",
    ]
    if plan_summary:
        logs["memory_synthesizer"].append(f"[PLAN] {plan_summary}")

    return {
        "reasoning_logs": logs,
        "messages": ["Memory synthesizer stored lifelong memory artifacts."],
        "execution_plan": plan,
        "planning_summary": plan_summary,
        "memory_insights": {
            "status": status,
            "episodic_namespace": ep_res.get("namespace"),
            "playbook_namespace": playbook_res.get("namespace"),
            "semantic_written": bool(sem_res),
            "usage_update": usage_update,
            "lookback": summarize_memory_hits(recent_episodes),
            "lesson_lookback": summarize_learning_hits(recent_lessons),
            "skill_lookback": summarize_skill_hits(recent_skills),
            "stats": memory_stats,
        },
        "iteration_learning": playbook_payload,
        "memory_writeback": {
            "episode": ep_res,
            "playbook": playbook_res,
            "skill_card": skill_res,
            "semantic": sem_res,
            "usage_update": usage_update,
            "status": status,
        },
    }
