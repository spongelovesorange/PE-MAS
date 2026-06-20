from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9_\u4e00-\u9fff]+", (text or "").lower()) if t]


def _payload_tags(payload: Dict[str, Any]) -> List[str]:
    tags = payload.get("tags")
    if isinstance(tags, list):
        return [str(x).strip().lower() for x in tags if str(x).strip()]
    return []


def _payload_quality(payload: Dict[str, Any]) -> float:
    try:
        return max(0.0, min(1.0, float(payload.get("quality_score", 0.55))))
    except Exception:
        return 0.55


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        m = re.search(r"(-?\d+(?:\.\d+)?)", str(value or ""))
        return float(m.group(1)) if m else default
    except Exception:
        return default


def _power_band_from_specs(specs: Dict[str, Any]) -> str:
    p_out = _safe_float(specs.get("output_voltage"), 0.0) * _safe_float(specs.get("output_current"), 0.0)
    if p_out < 15.0:
        return "low_power"
    if p_out < 45.0:
        return "mid_power"
    return "high_power"


def _vin_class_from_specs(specs: Dict[str, Any]) -> str:
    vin_min = _safe_float(specs.get("input_voltage_min"), 0.0)
    vin_max = _safe_float(specs.get("input_voltage_max"), 0.0)
    if vin_max >= 240.0 and vin_min <= 100.0:
        return "offline_wide_range"
    if vin_max >= 180.0:
        return "offline_mains"
    return "dc_or_narrow_input"


def _vout_class_from_specs(specs: Dict[str, Any]) -> str:
    vout = _safe_float(specs.get("output_voltage"), 0.0)
    if vout <= 5.5:
        return "low_voltage"
    if vout <= 15.0:
        return "medium_voltage"
    return "high_voltage"


def _fingerprint(namespace: str, kind: str, text_body: str, payload: Dict[str, Any]) -> str:
    semantic_keys = {
        "summary": payload.get("summary"),
        "status": payload.get("status"),
        "skill_id": payload.get("skill_id"),
        "specifications": payload.get("specifications"),
        "verification": payload.get("verification"),
        "kind": kind,
        "namespace": namespace,
        "text_body": text_body,
    }
    raw = json.dumps(semantic_keys, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


@dataclass
class MemoryHit:
    memory_id: int
    namespace: str
    payload: Dict[str, Any]
    score: float


class LifelongMemoryEngine:
    """
    Pragmatic lifelong memory:
    - LangGraph store when provided.
    - SQLite fallback with dedupe, quality-aware ranking, and cross-session persistence.
    """

    def __init__(self, sqlite_path: Optional[str] = None):
        runtime_dir = Path(os.getenv("PE_MAS_RUNTIME_DIR", ".pe_mas_runtime"))
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = sqlite_path or str(runtime_dir / "lifelong_memory.sqlite")
        self._conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._setup()

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        cur = self._conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        columns = {str(row[1]) for row in cur.fetchall()}
        if column not in columns:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            self._conn.commit()

    def _setup(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL,
                kind TEXT NOT NULL,
                text_body TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                usage_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._ensure_column("memories", "fingerprint", "TEXT")
        self._ensure_column("memories", "tags_json", "TEXT DEFAULT '[]'")
        self._ensure_column("memories", "quality_score", "REAL DEFAULT 0.55")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_namespace ON memories(namespace)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_updated_at ON memories(updated_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_fingerprint ON memories(namespace, fingerprint)")
        self._conn.commit()

    def _sqlite_put(self, namespace: str, kind: str, text_body: str, payload: Dict[str, Any]) -> int:
        now = time.time()
        fp = _fingerprint(namespace, kind, text_body, payload)
        tags_json = json.dumps(_payload_tags(payload), ensure_ascii=False)
        quality = _payload_quality(payload)
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id FROM memories WHERE namespace = ? AND fingerprint = ?",
            (namespace, fp),
        )
        row = cur.fetchone()
        if row:
            cur.execute(
                """
                UPDATE memories
                SET payload_json = ?,
                    text_body = ?,
                    kind = ?,
                    tags_json = ?,
                    quality_score = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(payload, ensure_ascii=False),
                    text_body,
                    kind,
                    tags_json,
                    quality,
                    now,
                    int(row["id"]),
                ),
            )
            self._conn.commit()
            return int(row["id"])

        cur.execute(
            """
            INSERT INTO memories(
                namespace, kind, text_body, payload_json, success_count, usage_count,
                failure_count, created_at, updated_at, fingerprint, tags_json, quality_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                namespace,
                kind,
                text_body,
                json.dumps(payload, ensure_ascii=False),
                0,
                0,
                0,
                now,
                now,
                fp,
                tags_json,
                quality,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def _sqlite_search(self, namespace: str, query: str, limit: int = 5) -> List[MemoryHit]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM memories WHERE namespace = ?", (namespace,))
        rows = cur.fetchall()
        q_tokens = set(_tokenize(query))
        now = time.time()

        scored: List[MemoryHit] = []
        for row in rows:
            text_body = row["text_body"] or ""
            body_tokens = set(_tokenize(text_body))
            overlap = len(q_tokens & body_tokens)
            semantic_match = overlap / max(1.0, float(len(q_tokens) or 1))
            usage = float(row["usage_count"] or 0)
            success = float(row["success_count"] or 0)
            failure = float(row["failure_count"] or 0)
            quality = float(row["quality_score"] or 0.55)
            age_hours = max(0.0, (now - float(row["updated_at"] or now)) / 3600.0)
            tags = []
            try:
                tags = json.loads(row["tags_json"] or "[]")
            except Exception:
                tags = []
            tag_overlap = len(set(tags) & q_tokens) / max(1.0, float(len(q_tokens) or 1))

            score = (
                0.45 * semantic_match
                + 0.20 * tag_overlap
                + 0.15 * min(1.0, usage / 20.0)
                + 0.18 * min(1.0, success / 10.0)
                + 0.12 * quality
                - 0.12 * min(1.0, failure / 6.0)
                - 0.04 * min(1.0, age_hours / 24.0 / 45.0)
            )
            if score <= 0:
                continue

            try:
                payload = json.loads(row["payload_json"] or "{}")
            except Exception:
                payload = {"raw": row["payload_json"]}

            scored.append(
                MemoryHit(
                    memory_id=int(row["id"]),
                    namespace=str(row["namespace"]),
                    payload=payload,
                    score=float(score),
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: max(1, limit)]

    def _sqlite_mark_usage(self, memory_id: int, success: bool = False, failed: bool = False) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE memories
            SET usage_count = usage_count + 1,
                success_count = success_count + ?,
                failure_count = failure_count + ?,
                updated_at = ?
            WHERE id = ?
            """,
            (1 if success else 0, 1 if failed else 0, time.time(), memory_id),
        )
        self._conn.commit()

    def put(self, namespace: Tuple[str, ...], payload: Dict[str, Any], *, kind: str = "generic", store: Any = None) -> Dict[str, Any]:
        namespace_key = "/".join(namespace)
        text_body = _safe_text(payload.get("summary") or payload)

        if store is not None:
            key = _fingerprint(namespace_key, kind, text_body, payload)
            store.put(namespace, key, payload)
            return {"backend": "langgraph_store", "namespace": namespace_key, "key": key}

        row_id = self._sqlite_put(namespace_key, kind, text_body, payload)
        return {"backend": "sqlite", "namespace": namespace_key, "row_id": row_id}

    def search(self, namespace: Tuple[str, ...], query: str, *, limit: int = 5, store: Any = None) -> List[Dict[str, Any]]:
        namespace_key = "/".join(namespace)

        if store is not None:
            results = store.search(namespace, query=query, limit=limit)
            out: List[Dict[str, Any]] = []
            for idx, item in enumerate(results):
                val = getattr(item, "value", None) or getattr(item, "dict", lambda: {})()
                out.append({
                    "memory_id": idx,
                    "score": float(getattr(item, "score", 0.0) or 0.0),
                    "payload": val if isinstance(val, dict) else {"value": _safe_text(val)},
                    "backend": "langgraph_store",
                })
            return out

        hits = self._sqlite_search(namespace_key, query=query, limit=limit)
        return [
            {
                "memory_id": h.memory_id,
                "score": h.score,
                "payload": h.payload,
                "backend": "sqlite",
            }
            for h in hits
        ]

    def mark_usage(self, memory_id: int, *, success: bool = False, failed: bool = False) -> None:
        self._sqlite_mark_usage(memory_id, success=success, failed=failed)

    def stats(self) -> Dict[str, Any]:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM memories")
        total = int((cur.fetchone() or [0])[0])
        cur.execute("SELECT namespace, COUNT(*) AS c FROM memories GROUP BY namespace ORDER BY c DESC")
        namespaces = [{"namespace": str(row["namespace"]), "count": int(row["c"])} for row in cur.fetchall()]
        return {
            "backend": "sqlite",
            "total_rows": total,
            "top_namespaces": namespaces[:10],
        }

    def prune(self, *, max_rows_per_namespace: int = 400, min_score: float = 0.02) -> Dict[str, Any]:
        cur = self._conn.cursor()
        cur.execute("SELECT DISTINCT namespace FROM memories")
        namespaces = [str(r[0]) for r in cur.fetchall()]

        now = time.time()
        deleted = 0
        kept = 0

        for ns in namespaces:
            cur.execute(
                """
                SELECT id, usage_count, success_count, failure_count, updated_at, quality_score
                FROM memories WHERE namespace = ?
                """,
                (ns,),
            )
            rows = cur.fetchall()
            scored_rows: List[Tuple[float, int]] = []
            for row in rows:
                usage = float(row["usage_count"] or 0)
                success = float(row["success_count"] or 0)
                failure = float(row["failure_count"] or 0)
                quality = float(row["quality_score"] or 0.55)
                age_days = max(0.0, (now - float(row["updated_at"] or now)) / 86400.0)
                score = (
                    0.35 * min(1.0, usage / 25.0)
                    + 0.35 * min(1.0, success / 12.0)
                    + 0.20 * quality
                    - 0.20 * min(1.0, failure / 10.0)
                    - 0.10 * min(1.0, age_days / 120.0)
                )
                scored_rows.append((score, int(row["id"])))

            scored_rows.sort(key=lambda x: x[0], reverse=True)
            keep_ids = {rid for _, rid in scored_rows[:max_rows_per_namespace]}
            for idx, (score, rid) in enumerate(scored_rows):
                should_drop = (idx >= max_rows_per_namespace) or (score < min_score)
                if should_drop and rid not in keep_ids:
                    cur.execute("DELETE FROM memories WHERE id = ?", (rid,))
                    deleted += 1
                else:
                    kept += 1

        self._conn.commit()
        return {
            "namespaces": namespaces,
            "kept": kept,
            "deleted": deleted,
            "max_rows_per_namespace": max_rows_per_namespace,
            "min_score": min_score,
        }


_ENGINE: Optional[LifelongMemoryEngine] = None


def get_memory_engine() -> LifelongMemoryEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = LifelongMemoryEngine()
    return _ENGINE


def build_episode_summary(state: Dict[str, Any]) -> str:
    specs = state.get("specifications") or {}
    design = state.get("theoretical_design") or {}
    sim = state.get("simulation_results") or {}
    verification = state.get("verification") or {}

    return (
        f"Flyback design episode | "
        f"Vin={specs.get('input_voltage_min')}-{specs.get('input_voltage_max')}Vac, "
        f"Vout={specs.get('output_voltage')}V, Iout={specs.get('output_current')}A, "
        f"fsw={design.get('switching_frequency')}Hz, Dmax={design.get('max_duty_cycle')}, "
        f"eff={sim.get('efficiency_measured')}, ripple={sim.get('v_out_ripple_measured')}, "
        f"status={verification.get('status')}"
    )


def build_skill_card_from_state(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    specs = state.get("specifications") or {}
    if not specs:
        return None

    design = state.get("theoretical_design") or {}
    sim = state.get("simulation_results") or {}
    verification = state.get("verification") or {}
    correction = state.get("correction_review") or {}
    learning = state.get("iteration_learning") or build_iteration_playbook(state)
    strategy_bundle = verification.get("strategy_bundle") if isinstance(verification.get("strategy_bundle"), dict) else {}

    status = str(verification.get("status") or "UNKNOWN").upper()
    primary_action = str(strategy_bundle.get("primary_action") or "").strip()
    dominant_loss = str(strategy_bundle.get("dominant_loss") or "").strip().lower()
    component_actions = [
        str(x).strip()
        for x in (
            learning.get("recommended_component_actions")
            or strategy_bundle.get("recommended_component_actions")
            or []
        )
        if str(x).strip()
    ]
    focus_items = [
        str(x).strip()
        for x in (
            learning.get("next_iteration_focus")
            or strategy_bundle.get("next_iteration_focus")
            or []
        )
        if str(x).strip()
    ]
    anti_patterns = [
        str(x).strip()
        for x in (
            learning.get("do_not_repeat")
            or strategy_bundle.get("do_not_repeat")
            or []
        )
        if str(x).strip()
    ]
    overrides = learning.get("recommended_overrides") if isinstance(learning.get("recommended_overrides"), dict) else {}
    root_causes = [
        str(x).strip()
        for x in (
            learning.get("root_causes")
            or strategy_bundle.get("root_causes")
            or correction.get("mismatches")
            or []
        )
        if str(x).strip()
    ]
    fail_axes = [str(x).strip().lower() for x in (strategy_bundle.get("failed_axes") or []) if str(x).strip()]

    meaningful = any([
        primary_action,
        dominant_loss,
        component_actions,
        focus_items,
        anti_patterns,
        overrides,
        root_causes,
        status == "PASS",
    ])
    if not meaningful:
        return None

    power_band = _power_band_from_specs(specs)
    vin_class = _vin_class_from_specs(specs)
    vout_class = _vout_class_from_specs(specs)
    application_type = str(specs.get("application_type") or "generic").strip() or "generic"

    if status == "PASS":
        skill_type = "warm_start_recipe"
    elif dominant_loss or primary_action:
        skill_type = "debug_strategy"
    else:
        skill_type = "iteration_heuristic"

    if status == "PASS":
        objective = "warm_start_successful_region"
    elif dominant_loss:
        objective = f"mitigate_{dominant_loss}"
    elif primary_action:
        objective = primary_action.lower()
    else:
        objective = "stabilize_design_iteration"

    trigger_signals = list(dict.fromkeys([
        *fail_axes,
        *root_causes[:4],
        *focus_items[:3],
    ]))

    warm_start_hints = {}
    for key in ["switching_frequency", "reflected_output_voltage", "ripple_factor", "max_duty_cycle"]:
        if design.get(key) is not None:
            warm_start_hints[key] = design.get(key)

    expected_effects: List[str] = []
    if status == "PASS":
        expected_effects.append("Reuse this parameter region as a warm-start for nearby specifications.")
    if "efficiency" in fail_axes:
        expected_effects.append("Primary goal is to recover efficiency before broad exploration.")
    if "ripple" in fail_axes:
        expected_effects.append("Any change should preserve or improve ripple compliance.")
    if "stress" in fail_axes:
        expected_effects.append("Reduce stress before increasing switching aggressiveness.")
    if dominant_loss:
        expected_effects.append(f"Target the dominant loss bucket: {dominant_loss}.")

    summary = (
        f"Skill card | {objective} | {power_band} {vin_class} {vout_class} | "
        f"Vout={specs.get('output_voltage')}V Iout={specs.get('output_current')}A | "
        f"status={status} | action={primary_action or 'n/a'} | loss={dominant_loss or 'n/a'}"
    )

    quality_score = 0.84 if status == "PASS" else 0.74
    if not component_actions and not overrides and status != "PASS":
        quality_score = 0.66

    return {
        "summary": summary,
        "skill_type": skill_type,
        "objective": objective,
        "status": status,
        "primary_action": primary_action,
        "dominant_loss": dominant_loss,
        "trigger_signals": trigger_signals,
        "applicability_region": {
            "power_band": power_band,
            "vin_class": vin_class,
            "vout_class": vout_class,
            "application_type": application_type,
            "output_power_w": round(_safe_float(specs.get("output_voltage"), 0.0) * _safe_float(specs.get("output_current"), 0.0), 3),
            "isolation": bool(specs.get("isolation", True)),
        },
        "spec_signature": {
            "vin_min": specs.get("input_voltage_min"),
            "vin_max": specs.get("input_voltage_max"),
            "vout": specs.get("output_voltage"),
            "iout": specs.get("output_current"),
            "eff_target": specs.get("efficiency_target"),
            "ripple_target": specs.get("max_ripple_voltage"),
        },
        "action_template": {
            "recommended_overrides": overrides,
            "recommended_component_actions": component_actions,
            "next_iteration_focus": focus_items,
        },
        "anti_patterns": anti_patterns,
        "warm_start_hints": warm_start_hints,
        "expected_effects": expected_effects,
        "outcome_snapshot": {
            "efficiency_measured": sim.get("efficiency_measured"),
            "ripple_measured": sim.get("v_out_ripple_measured"),
            "vds_spike_max": sim.get("v_ds_spike_max"),
        },
        "transfer_note": (
            f"Best reused for {application_type} flyback designs in {power_band} / {vin_class} region. "
            f"Use when symptoms resemble: {', '.join(trigger_signals[:3]) or 'nearby spec drift'}."
        ),
        "tags": [
            "flyback",
            "skill_card",
            skill_type,
            objective,
            power_band,
            vin_class,
            vout_class,
            application_type.lower(),
            status.lower(),
            *(fail_axes[:3]),
            dominant_loss,
            primary_action.lower() if primary_action else "",
        ],
        "quality_score": quality_score,
    }


def build_semantic_rule_from_state(state: Dict[str, Any]) -> Optional[str]:
    specs = state.get("specifications") or {}
    sim = state.get("simulation_results") or {}
    verification = state.get("verification") or {}
    corr = state.get("correction_review") or {}

    status = str(verification.get("status") or "")
    eff = sim.get("efficiency_measured")
    target = specs.get("efficiency_target")

    if status == "PASS" and eff is not None and target is not None:
        if float(eff) < float(target):
            return (
                "Even when validator returns PASS, efficiency can be below requested target; "
                "correction review should gate final sign-off for strict applications."
            )
        return (
            f"For Vin {specs.get('input_voltage_min')}-{specs.get('input_voltage_max')}Vac and "
            f"{specs.get('output_voltage')}V output class, this parameter region converged successfully."
        )

    if status in {"FAIL", "NEEDS_HUMAN_REVIEW"}:
        mismatch = "; ".join((corr.get("mismatches") or [])[:3])
        return f"Known failed region: {mismatch or 'insufficient margin or non-convergence'}"

    return None


def summarize_memory_hits(hits: List[Dict[str, Any]], *, limit: int = 3) -> Dict[str, Any]:
    preview = []
    for hit in hits[: max(1, limit)]:
        payload = hit.get("payload") or {}
        preview.append(
            {
                "score": round(float(hit.get("score") or 0.0), 3),
                "summary": str(payload.get("summary") or payload.get("status") or "memory"),
            }
        )
    return {
        "count": len(hits),
        "preview": preview,
    }


def _merge_override_hint(merged: Dict[str, float], key: str, value: Any) -> None:
    try:
        val = float(value)
    except Exception:
        return
    if key not in merged:
        merged[key] = val
    else:
        merged[key] = (merged[key] + val) / 2.0


def build_iteration_playbook(state: Dict[str, Any]) -> Dict[str, Any]:
    specs = state.get("specifications") or {}
    sim = state.get("simulation_results") or {}
    verification = state.get("verification") or {}
    correction = state.get("correction_review") or {}
    strategy_bundle = verification.get("strategy_bundle") if isinstance(verification.get("strategy_bundle"), dict) else {}
    sensitivity = state.get("param_sensitivity_plan") or {}

    failed_items = [str(x) for x in (verification.get("failed_items") or []) if str(x).strip()]
    mismatches = [str(x) for x in (correction.get("mismatches") or []) if str(x).strip()]
    recommendations = [str(x) for x in (correction.get("recommendations") or []) if str(x).strip()]
    top_actions = [str(x) for x in (sensitivity.get("top_actions") or []) if str(x).strip()]

    root_causes: List[str] = []
    dominant_loss = str(strategy_bundle.get("dominant_loss") or "").strip()
    failed_axes = [str(x) for x in (strategy_bundle.get("failed_axes") or []) if str(x).strip()]
    if "efficiency" in failed_axes:
        root_causes.append("Efficiency bottleneck remains unresolved in this operating region.")
    if "ripple" in failed_axes:
        root_causes.append("Output ripple remains above the requested target.")
    if "stress" in failed_axes:
        root_causes.append("Device stress margin is too close to the rating limit.")
    if dominant_loss:
        root_causes.append(f"Dominant modeled loss bucket: {dominant_loss}.")

    next_iteration_focus = []
    for item in [
        *(failed_items[:3]),
        *(mismatches[:3]),
        *(top_actions[:3]),
    ]:
        if item not in next_iteration_focus:
            next_iteration_focus.append(item)

    recommended_overrides = strategy_bundle.get("recommended_overrides") if isinstance(strategy_bundle.get("recommended_overrides"), dict) else {}
    component_actions = [
        str(x) for x in (strategy_bundle.get("recommended_component_actions") or []) if str(x).strip()
    ]
    do_not_repeat = []
    primary_action = str(strategy_bundle.get("primary_action") or "").strip()
    last_action = str(strategy_bundle.get("last_action") or "").strip()
    if primary_action and primary_action == last_action and str(verification.get("status") or "").upper() != "PASS":
        do_not_repeat.append(f"Do not repeat '{primary_action}' blindly without changing the loss bottleneck.")
    if dominant_loss == "diode_conduction":
        do_not_repeat.append("Do not keep the same secondary rectifier assumptions if diode conduction dominates.")
    if dominant_loss == "transformer_core":
        do_not_repeat.append("Do not push switching frequency upward before reducing core-loss pressure.")

    summary = (
        f"Iteration playbook | status={verification.get('status')} | "
        f"Vin={specs.get('input_voltage_min')}-{specs.get('input_voltage_max')}Vac | "
        f"Vout={specs.get('output_voltage')}V {specs.get('output_current')}A | "
        f"action={primary_action or 'N/A'} | dominant_loss={dominant_loss or 'n/a'}"
    )

    tags = [
        "flyback",
        str(verification.get("status") or "unknown").lower(),
        *(str(x).lower() for x in failed_axes),
        primary_action.lower() if primary_action else "",
        dominant_loss.lower() if dominant_loss else "",
    ]
    tags = [tag for tag in tags if tag]

    quality_score = 0.8 if str(verification.get("status") or "").upper() == "PASS" else 0.68

    return {
        "summary": summary,
        "status": verification.get("status"),
        "spec_signature": {
            "vin_min": specs.get("input_voltage_min"),
            "vin_max": specs.get("input_voltage_max"),
            "vout": specs.get("output_voltage"),
            "iout": specs.get("output_current"),
            "eff_target": specs.get("efficiency_target"),
        },
        "observed_metrics": {
            "efficiency_measured": sim.get("efficiency_measured"),
            "ripple_measured": sim.get("v_out_ripple_measured"),
            "vds_spike_max": sim.get("v_ds_spike_max"),
        },
        "root_causes": root_causes,
        "failed_items": failed_items[:6],
        "next_iteration_focus": next_iteration_focus,
        "recommended_overrides": recommended_overrides,
        "recommended_component_actions": component_actions,
        "do_not_repeat": do_not_repeat,
        "recommendations": recommendations[:6],
        "sensitivity_actions": top_actions[:4],
        "tags": tags,
        "quality_score": quality_score,
    }


def summarize_learning_hits(hits: List[Dict[str, Any]], *, limit: int = 3) -> Dict[str, Any]:
    merged_overrides: Dict[str, float] = {}
    focus: List[str] = []
    component_actions: List[str] = []
    avoid: List[str] = []
    preview: List[Dict[str, Any]] = []

    for hit in hits[: max(1, limit)]:
        payload = hit.get("payload") or {}
        preview.append(
            {
                "score": round(float(hit.get("score") or 0.0), 3),
                "summary": str(payload.get("summary") or ""),
                "status": str(payload.get("status") or ""),
            }
        )
        overrides = payload.get("recommended_overrides") if isinstance(payload.get("recommended_overrides"), dict) else {}
        for key, value in overrides.items():
            _merge_override_hint(merged_overrides, str(key), value)
        for item in payload.get("next_iteration_focus") or []:
            text = str(item).strip()
            if text and text not in focus:
                focus.append(text)
        for item in payload.get("recommended_component_actions") or []:
            text = str(item).strip()
            if text and text not in component_actions:
                component_actions.append(text)
        for item in payload.get("do_not_repeat") or []:
            text = str(item).strip()
            if text and text not in avoid:
                avoid.append(text)

    return {
        "count": len(hits),
        "preview": preview,
        "suggested_overrides": merged_overrides,
        "focus": focus[:5],
        "component_actions": component_actions[:5],
        "avoid": avoid[:5],
    }


def summarize_skill_hits(hits: List[Dict[str, Any]], *, limit: int = 3) -> Dict[str, Any]:
    merged_overrides: Dict[str, float] = {}
    merged_warm_start: Dict[str, float] = {}
    component_actions: List[str] = []
    focus: List[str] = []
    avoid: List[str] = []
    objectives: List[str] = []
    preview: List[Dict[str, Any]] = []

    for hit in hits[: max(1, limit)]:
        payload = hit.get("payload") or {}
        preview.append(
            {
                "score": round(float(hit.get("score") or 0.0), 3),
                "objective": str(payload.get("objective") or ""),
                "skill_type": str(payload.get("skill_type") or ""),
                "summary": str(payload.get("summary") or ""),
            }
        )

        action_template = payload.get("action_template") if isinstance(payload.get("action_template"), dict) else {}
        for key, value in (action_template.get("recommended_overrides") or {}).items():
            _merge_override_hint(merged_overrides, str(key), value)
        for key, value in (payload.get("warm_start_hints") or {}).items():
            _merge_override_hint(merged_warm_start, str(key), value)

        for item in action_template.get("recommended_component_actions") or []:
            text = str(item).strip()
            if text and text not in component_actions:
                component_actions.append(text)
        for item in action_template.get("next_iteration_focus") or []:
            text = str(item).strip()
            if text and text not in focus:
                focus.append(text)
        for item in payload.get("trigger_signals") or []:
            text = str(item).strip()
            if text and text not in focus:
                focus.append(text)
        for item in payload.get("anti_patterns") or []:
            text = str(item).strip()
            if text and text not in avoid:
                avoid.append(text)
        objective = str(payload.get("objective") or "").strip()
        if objective and objective not in objectives:
            objectives.append(objective)

    return {
        "count": len(hits),
        "preview": preview,
        "suggested_overrides": merged_overrides,
        "warm_start_hints": merged_warm_start,
        "component_actions": component_actions[:6],
        "focus": focus[:6],
        "avoid": avoid[:6],
        "objectives": objectives[:4],
    }


def _collect_sqlite_memory_ids(value: Any, out: set[int]) -> None:
    if isinstance(value, dict):
        backend = str(value.get("backend") or "").strip().lower()
        memory_id = value.get("memory_id")
        if backend == "sqlite" and isinstance(memory_id, int):
            out.add(int(memory_id))
            return
        for child in value.values():
            _collect_sqlite_memory_ids(child, out)
        return
    if isinstance(value, list):
        for child in value:
            _collect_sqlite_memory_ids(child, out)


def mark_state_memory_usage(
    state: Dict[str, Any],
    *,
    success: bool = False,
    failed: bool = False,
    engine: Optional[LifelongMemoryEngine] = None,
) -> Dict[str, Any]:
    """
    Update usage/success/failure counters for memory hits that influenced the run.
    """
    eng = engine or get_memory_engine()
    memory_ids: set[int] = set()
    _collect_sqlite_memory_ids((state or {}).get("memory_context") or {}, memory_ids)

    for memory_id in sorted(memory_ids):
        eng.mark_usage(memory_id, success=success, failed=failed)

    return {
        "count": len(memory_ids),
        "memory_ids": sorted(memory_ids),
        "success": bool(success),
        "failed": bool(failed),
    }


def index_skills_for_dynamic_binding(skill_manager: Any, store: Any = None) -> int:
    engine = get_memory_engine()
    count = 0
    for s in skill_manager.list_skills():
        payload = {
            "summary": f"Skill {s.get('id')}: {s.get('name')} | {s.get('description')}",
            "skill_id": s.get("id"),
            "name": s.get("name"),
            "description": s.get("description"),
            "version": s.get("version"),
            "tags": s.get("capabilities") or [],
            "quality_score": 0.8 if s.get("health") == "ok" else 0.45,
        }
        engine.put(("skills", "executable_tools"), payload, kind="skill_meta", store=store)
        count += 1
    return count


def select_dynamic_skills(
    query: str,
    available_skill_ids: List[str],
    *,
    top_k: int = 5,
    store: Any = None,
) -> List[str]:
    engine = get_memory_engine()
    hits = engine.search(("skills", "executable_tools"), query=query, limit=max(top_k * 2, 6), store=store)
    ranked: List[str] = []
    seen = set()
    for h in hits:
        skill_id = str(((h.get("payload") or {}).get("skill_id") or "")).strip()
        if not skill_id or skill_id in seen:
            continue
        if skill_id in available_skill_ids:
            ranked.append(skill_id)
            seen.add(skill_id)
        if len(ranked) >= top_k:
            break

    if ranked:
        return ranked

    q_tokens = set(_tokenize(query))
    scored: List[Tuple[int, str]] = []
    for sid in available_skill_ids:
        score = len(q_tokens & set(_tokenize(sid.replace("_", " "))))
        scored.append((score, sid))
    scored.sort(key=lambda x: x[0], reverse=True)
    fallback = [sid for score, sid in scored if score > 0][:top_k]
    if fallback:
        return fallback
    return available_skill_ids[:top_k]
