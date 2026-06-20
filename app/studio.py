import asyncio
import contextlib
import hashlib
import json
import math
import os
import re
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from core.flyback_mas.graph import app as graph_app
from core.flyback_mas.knowledge_guardrails import normalize_source_label, retrieve_flyback_context
from core.flyback_mas.skills_manager import SkillManager
from core.flyback_mas.lifelong_memory import (
    build_episode_summary,
    build_iteration_playbook,
    build_semantic_rule_from_state,
    get_memory_engine,
    mark_state_memory_usage,
)
from core.requirements_agent import RequirementAnalysisAgent
from core.requirements_agent.plecs_registry import PlecsModelRegistry
from core.requirements_agent.schemas import RequirementAnalysisRequest
from core.requirements_agent.topology_service import TopologyKnowledgeService
from core.studio_contract import StudioRequirementsGateService
from core.runtime import env_flag, quiet_stdio

try:
    from core.skills.web_research.tools import research_web as mcp_research_web
except Exception:
    mcp_research_web = None

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="PE-MAS Industrial Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_TTL_SEC = int(os.getenv("PE_MAS_SESSION_TTL_SEC", "7200") or "7200")
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_LOCK = Lock()

runtime_dir_value = os.getenv("PE_MAS_RUNTIME_DIR")
RUNTIME_DIR = Path(runtime_dir_value).expanduser() if runtime_dir_value else BASE_DIR / ".pe_mas_runtime"
if not RUNTIME_DIR.is_absolute():
    RUNTIME_DIR = (BASE_DIR / RUNTIME_DIR).resolve()
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR = BASE_DIR / "frontend"
SESSIONS_FILE = RUNTIME_DIR / "sessions.json"
EVENT_LOG_FILE = RUNTIME_DIR / "events.jsonl"
METRICS_FILE = RUNTIME_DIR / "metrics.json"
RUNTIME_LOCK = Lock()
TRACE_LOG_FILE = RUNTIME_DIR / "workflow_trace.log"
BACKEND_STDIO_LOG_FILE = RUNTIME_DIR / "backend_stdio.log"
TRACE_STREAM = env_flag("PE_MAS_TRACE_STREAM", default=False)
VERBOSE_BACKEND_STDIO = env_flag("PE_MAS_VERBOSE_STDIO", default=False)
TOOL_POOL = ThreadPoolExecutor(max_workers=6)
TOOL_BREAKERS: Dict[str, Dict[str, Any]] = {}
TOOL_IDEMPOTENCY_CACHE: Dict[str, Any] = {}
PENDING_STREAMS: Dict[str, Dict[str, Any]] = {}
PENDING_STREAM_LOCK = Lock()
PENDING_STREAM_TTL_SEC = 600


def _trace_log(event: str, sid: str = "", **fields: Any) -> None:
    """Emit optional workflow trace lines for long-running Studio streams."""
    if not TRACE_STREAM:
        return
    payload: Dict[str, Any] = {"ts": time.strftime("%H:%M:%S"), "event": event}
    if sid:
        payload["sid"] = sid
    for key, value in fields.items():
        if value is None:
            continue
        text = str(value)
        payload[key] = text[:800]
    line = "[PE-MAS] " + json.dumps(payload, ensure_ascii=False)
    print(line, flush=True)
    try:
        with RUNTIME_LOCK:
            with TRACE_LOG_FILE.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        pass


async def _next_graph_event(event_iter: Any) -> Any:
    """Read one LangGraph event while keeping node print output out of stdout."""

    with quiet_stdio(BACKEND_STDIO_LOG_FILE, enabled=not VERBOSE_BACKEND_STDIO):
        return await event_iter.__anext__()


class ChatStartRequest(BaseModel):
    q: str
    sid: Optional[str] = None


@app.middleware("http")
async def _no_cache_for_demo(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith(".html") or path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def _resolve_report_asset(path_value: str) -> Path:
    raw = str(path_value or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="missing path")

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    else:
        path = path.resolve()

    allowed_roots = {
        (RUNTIME_DIR / "reports").resolve(),
        (BASE_DIR / ".pe_mas_runtime" / "reports").resolve(),
    }
    if os.getenv("PE_MAS_ALLOW_EXTRA_REPORT_ASSETS", "0").strip().lower() in {"1", "true", "yes", "on"}:
        allowed_roots.add(BASE_DIR.resolve())

    if not any(_path_is_relative_to(path, root) for root in allowed_roots):
        raise HTTPException(status_code=403, detail="path not allowed")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="asset not found")
    return path


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

CONTROL_COMMANDS: Dict[str, Dict[str, Any]] = {
    "ADOPT_DEFAULT_ASSUMPTIONS": {
        "aliases": [
            "adopt default assumptions",
            "accept default assumptions",
            "lock critical specs",
            "lock requirements",
            "continue with default assumptions",
            "start engineering workflow",
            "采用默认假设并继续",
            "采用默认假设",
            "锁定关键规格",
            "锁定规格",
            "继续工程流程",
        ],
    },
    "LOCK_CRITICAL_SPECS": {
        "aliases": ["lock critical specs v0.1", "lock spec v0.1", "lock specs", "锁定关键规格 v0.1", "锁定关键规格"],
    },
    "GENERATE_TOPOLOGY_CANDIDATES": {
        "aliases": ["generate topology candidates", "compare topology", "生成拓扑候选", "比较拓扑"],
    },
    "CONFIRM_QR_FLYBACK": {
        "aliases": ["confirm qr flyback", "confirm topology qr flyback", "确认拓扑 qr flyback", "确认拓扑"],
    },
    "GENERATE_CONTROLLER_CANDIDATES": {
        "aliases": ["controller candidates", "generate controller candidates", "控制器候选"],
    },
    "LOCK_CONTROLLER_UCC28740": {
        "aliases": ["lock controller ucc28740", "use ti ucc28740", "锁定主控方案", "锁定 ucc28740"],
    },
    "ENTER_POWER_STAGE_CALC": {
        "aliases": ["enter power stage calculation", "power stage calculation", "进入功率级计算"],
    },
    "RUN_VR_SCAN": {
        "aliases": ["run vr scan", "rescan reflected voltage", "重新扫描 vr", "扫描 vr"],
    },
    "GENERATE_MAGNETICS_CANDIDATES": {
        "aliases": ["generate magnetics candidates", "magnetics design", "生成磁件方案", "磁性元件设计"],
    },
    "GENERATE_DEVICE_CARDS": {
        "aliases": ["generate device cards", "device selection", "器件选型", "生成 bom 候选"],
    },
    "DESIGN_SNUBBER_CLAMP": {
        "aliases": ["design snubber clamp", "calculate rcd", "吸收钳位设计", "计算 rcd"],
    },
    "DESIGN_LOOP_COMPENSATION": {
        "aliases": ["design loop compensation", "feedback loop", "反馈环路补偿", "运行小信号模型"],
    },
    "RUN_SYSTEM_VALIDATION": {
        "aliases": ["run system validation", "run simulation validation", "prepare simulation checklist", "prepare plecs matrix", "运行仿真验证", "准备 plecs 矩阵", "全部运行"],
    },
    "GENERATE_LOCAL_FIX_OPTIONS": {
        "aliases": ["generate local fix options", "生成修正方案", "应用局部变更"],
    },
    "GENERATE_SCHEMATIC_DRAFT": {
        "aliases": ["generate schematic draft", "生成原理图草案"],
    },
    "GENERATE_PCB_CONSTRAINTS": {
        "aliases": ["generate pcb constraints", "导出布局约束", "生成 pcb 约束"],
    },
    "GENERATE_BOM_AVL": {
        "aliases": ["generate bom avl", "生成 bom avl", "生成 bom v0.1"],
    },
    "GENERATE_EVT_TEST_PLAN": {
        "aliases": ["generate evt test plan", "生成测试计划", "生成 evt 测试包"],
    },
    "RUN_FINAL_DESIGN_REVIEW": {
        "aliases": ["run final design review", "design review", "最终 design review"],
    },
    "GENERATE_RELEASE_PACKAGE": {
        "aliases": ["generate release package", "生成 release package"],
    },
    "CONTINUE_SELECTION": {
        "aliases": ["continue selection", "继续选型"],
    },
    "CONTINUE_SIMULATION": {
        "aliases": [
            "continue simulation",
            "run plecs validation matrix",
            "run plecs simulation",
            "run plecs",
            "plecs simulation",
            "继续仿真",
            "运行 plecs",
        ],
    },
    "CONTINUE_REVIEW": {
        "aliases": ["continue review", "继续审查", "continue correction"],
    },
    "GENERATE_REPORT": {
        "aliases": ["generate report", "生成报告"],
    },
    "SKIP_CORRECTION_AND_REPORT": {
        "aliases": ["skip correction and report", "skip correction", "跳过纠偏并生成报告"],
    },
    "RETRY_REQUESTED": {
        "aliases": ["continue auto-iteration", "continue auto iteration", "retry", "继续自动迭代", "继续自动"],
    },
    "ACCEPT_CURRENT_RESULT": {
        "aliases": ["accept current result", "accept", "接受当前结果", "接受结果"],
    },
    "MANUAL_ADJUSTMENTS": {
        "aliases": ["apply manual adjustments (json)", "manual adjustments", "手动调整", "手动修改"],
    },
    "STOP_SESSION": {
        "aliases": ["stop", "abort", "terminate", "cancel", "停止"],
    },
    "MODIFY_CONSTRAINTS": {
        "aliases": ["modify constraints", "change constraints", "修改约束"],
    },
    "CHANGE_COMPONENT_STRATEGY": {
        "aliases": ["change component", "change components", "return to adjust", "更换器件策略"],
    },
}


def _load_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_sessions_from_disk() -> None:
    persisted = _load_json(SESSIONS_FILE, {})
    if isinstance(persisted, dict):
        with SESSION_LOCK:
            SESSIONS.update(persisted)


def _persist_sessions_to_disk() -> None:
    with SESSION_LOCK:
        snapshot = dict(SESSIONS)
    with RUNTIME_LOCK:
        _save_json(SESSIONS_FILE, snapshot)


def _event_idempotency_key(kind: str, sid: str, payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(f"{kind}|{sid}|{raw}".encode("utf-8")).hexdigest()


def _record_runtime_event(kind: str, sid: str, payload: Dict[str, Any]) -> None:
    ts = time.time()
    event = {
        "ts": ts,
        "event": kind,
        "sid": sid,
        "payload": payload,
    }
    key = _event_idempotency_key(kind, sid, payload)
    if key in TOOL_IDEMPOTENCY_CACHE:
        return
    TOOL_IDEMPOTENCY_CACHE[key] = ts
    # Keep idempotency cache bounded.
    if len(TOOL_IDEMPOTENCY_CACHE) > 5000:
        stale_keys = sorted(TOOL_IDEMPOTENCY_CACHE.items(), key=lambda kv: kv[1])[:1000]
        for k, _ in stale_keys:
            TOOL_IDEMPOTENCY_CACHE.pop(k, None)

    with RUNTIME_LOCK:
        with EVENT_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _update_metrics(updates: Dict[str, float]) -> Dict[str, Any]:
    with RUNTIME_LOCK:
        metrics = _load_json(METRICS_FILE, {
            "run_total": 0,
            "run_success": 0,
            "run_error": 0,
            "run_timeout": 0,
            "run_retry_fallback": 0,
            "sum_iteration": 0,
        })
        for k, v in updates.items():
            metrics[k] = float(metrics.get(k, 0)) + float(v)
        _save_json(METRICS_FILE, metrics)
        return metrics


def _metrics_snapshot() -> Dict[str, Any]:
    m = _load_json(METRICS_FILE, {
        "run_total": 0,
        "run_success": 0,
        "run_error": 0,
        "run_timeout": 0,
        "run_retry_fallback": 0,
        "sum_iteration": 0,
    })
    total = max(1.0, float(m.get("run_total", 0) or 0))
    def _bounded_rate(key: str) -> float:
        return max(0.0, min(1.0, float(m.get(key, 0) or 0) / total))

    return {
        **m,
        "slo": {
            "success_rate": _bounded_rate("run_success"),
            "error_rate": _bounded_rate("run_error"),
            "timeout_rate": _bounded_rate("run_timeout"),
            "retry_fallback_rate": _bounded_rate("run_retry_fallback"),
            "avg_iterations": float(m.get("sum_iteration", 0) or 0) / total,
        },
    }


def _tool_call_with_contract(tool_name: str, func, *args, timeout_sec: float = 18.0, retries: int = 1, idempotency_key: str = "", **kwargs):
    state = TOOL_BREAKERS.setdefault(tool_name, {"fails": 0, "open_until": 0.0})
    now = time.time()
    if now < float(state.get("open_until", 0.0) or 0.0):
        raise RuntimeError(f"tool_circuit_open:{tool_name}")

    cache_key = ""
    if idempotency_key:
        cache_key = f"{tool_name}:{idempotency_key}"
        if cache_key in TOOL_IDEMPOTENCY_CACHE:
            return TOOL_IDEMPOTENCY_CACHE[cache_key]

    last_err: Optional[Exception] = None
    for attempt in range(max(1, retries + 1)):
        try:
            fut = TOOL_POOL.submit(func, *args, **kwargs)
            result = fut.result(timeout=timeout_sec)
            state["fails"] = 0
            state["open_until"] = 0.0
            if cache_key:
                TOOL_IDEMPOTENCY_CACHE[cache_key] = result
            return result
        except FuturesTimeoutError as te:
            last_err = te
            state["fails"] = int(state.get("fails", 0)) + 1
            _update_metrics({"run_timeout": 1})
        except Exception as e:
            last_err = e
            state["fails"] = int(state.get("fails", 0)) + 1
        if int(state.get("fails", 0)) >= 3:
            state["open_until"] = time.time() + 20.0
        if attempt < retries:
            time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"tool_call_failed:{tool_name}:{last_err}")


def _parse_control_command(user_query: str) -> Optional[str]:
    q = str(user_query or "").strip()
    if not q:
        return None

    if q.startswith("CMD:"):
        raw = q[len("CMD:"):].strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                cmd = str(obj.get("id") or "").strip().upper()
                if cmd in CONTROL_COMMANDS:
                    return cmd
        except Exception:
            cmd = raw.upper()
            if cmd in CONTROL_COMMANDS:
                return cmd

    q_low = q.lower()
    if q_low.startswith("/cmd "):
        cmd = q_low.replace("/cmd ", "", 1).strip().upper()
        if cmd in CONTROL_COMMANDS:
            return cmd

    for cmd, cfg in CONTROL_COMMANDS.items():
        aliases = [str(x).strip().lower() for x in (cfg.get("aliases") or []) if str(x).strip()]
        if q_low in aliases:
            return cmd
    return None


_load_sessions_from_disk()

NODE_TITLE = {
    "requirements": "Requirements",
    "designer": "Theoretical Design",
    "magnetics_advisor": "Magnetic Advisor",
    "selector": "Component Selection",
    "simulator": "PLECS Simulation",
    "validator": "Design Review",
    "correction": "Correction",
    "memory_synthesizer": "Memory Synthesizer",
    "reporter": "Report Generation",
    "skill_executor": "Skill Execution",
}

WORKFLOW_SEQUENCE = [
    "requirements",
    "designer",
    "magnetics_advisor",
    "selector",
    "simulator",
    "validator",
    "correction",
    "memory_synthesizer",
    "reporter",
]

HEARTBEAT_HINTS = {
    "requirements": [
        "Parsing user requirements and design constraints.",
        "Checking input/output specs against flyback feasibility rules.",
        "Validating efficiency and ripple targets with conservative guardrails.",
    ],
    "designer": [
        "Synthesizing topology parameters and duty-cycle window.",
        "Estimating magnetics, current stress, and reflected voltage.",
        "Cross-checking equations and margin assumptions.",
    ],
    "magnetics_advisor": [
        "Probing optional OpenMagnetics helper availability.",
        "Estimating core family, gap, turns, and winding arrangement.",
        "Passing manufacturability and magnetic loss guidance into component selection.",
    ],
    "selector": [
        "Screening component candidates with voltage/current derating.",
        "Comparing part availability, datasheet limits, and reliability margin.",
        "Assembling BOM candidates for simulation pass.",
    ],
    "simulator": [
        "Running waveform and stress estimation for key operating points.",
        "Evaluating ripple, peak stress, and estimated loss balance.",
        "Checking convergence and realism of simulation outputs.",
    ],
    "validator": [
        "Reviewing pass/fail against KPI and safety constraints.",
        "Verifying stress margins, ripple target, and correction triggers.",
        "Preparing iteration decision for next action.",
    ],
    "correction": [
        "Inspecting mismatches and proposing correction strategy.",
        "Prioritizing high-impact adjustments with low risk.",
        "Packaging updated guidance for final handoff.",
    ],
    "memory_synthesizer": [
        "Distilling this run into long-term episodic memory.",
        "Extracting reusable engineering heuristics.",
        "Persisting memory artifacts for future retrieval.",
    ],
    "reporter": [
        "Compiling engineering evidence and design rationale.",
        "Summarizing final metrics, assumptions, and limitations.",
        "Formatting report for delivery and traceability.",
    ],
}

URL_RE = re.compile(r"https?://[^\s\]\)\}\>,\"']+")
REASON_TAG_RE = re.compile(r"^\[([A-Z0-9_\- ]+)\]\s*(.*)$")

REASON_TAG_TITLES = {
    "OBSERVATION": "Observation",
    "KNOWLEDGE": "Domain Knowledge",
    "THOUGHT": "Reasoning",
    "TOOL": "Tool Use",
    "PLAN": "Plan",
    "SEARCH": "Web Search",
    "RAG": "Local RAG",
    "EVIDENCE": "Evidence",
    "DATA": "Data Found",
    "DECISION": "Decision",
    "EXECUTION": "Execution",
    "RESULT": "Result",
    "FORMULA": "Formula Check",
    "FORMULA-WARN": "Formula Warning",
    "FORMULA-FAIL": "Formula Failure",
    "WARNING": "Warning",
    "CRITIQUE": "Self Critique",
    "CORRECTION": "Correction",
    "FALLBACK": "Fallback",
    "META-COGNITION": "Meta Cognition",
}

REASON_TAG_ORDER = [
    "OBSERVATION",
    "KNOWLEDGE",
    "THOUGHT",
    "PLAN",
    "TOOL",
    "SEARCH",
    "RAG",
    "EVIDENCE",
    "DATA",
    "FORMULA",
    "FORMULA-WARN",
    "FORMULA-FAIL",
    "DECISION",
    "EXECUTION",
    "RESULT",
    "WARNING",
    "CRITIQUE",
    "CORRECTION",
    "META-COGNITION",
    "FALLBACK",
]


def _classify_user_intent(user_query: str) -> str:
    q = str(user_query or "").strip()
    low = q.lower()
    design_action_keywords = [
        "design", "redesign", "optimize", "calculate", "simulate", "iterate",
        "设计", "重设计", "优化", "计算", "仿真", "迭代", "选型", "生成方案",
    ]
    design_subject_keywords = [
        "flyback", "converter", "power supply", "mosfet", "transformer", "ripple", "efficiency",
        "反激", "电源", "变压器", "mosfet", "纹波", "效率", "电路",
    ]
    research_keywords = [
        "search", "find", "datasheet", "paper", "arxiv", "blog", "forum", "reddit", "post",
        "元件", "器件", "datasheet", "文献", "论文", "帖子", "博客", "搜索",
    ]
    qa_keywords_low = [
        "what", "why", "how", "difference", "explain", "can you", "help me", "who are you",
        "what can you do", "hello", "hi", "capability", "capabilities",
        "used for", "use case", "use cases", "application", "applications", "purpose",
    ]
    qa_keywords_raw = [
        "什么是", "是什么", "是啥", "什么事", "有什么用", "有啥用", "用途", "用处", "用来", "干嘛", "能干嘛",
        "应用", "场景", "啥时候", "什么时候", "何时", "何年", "哪年", "发明", "发现",
        "你好", "您好", "你能做什么", "能做什么", "能干什么", "可以做什么", "会做什么",
        "你能", "可以帮", "怎么", "如何", "为何", "为什么", "介绍", "解释",
    ]

    is_qa = any(k in low for k in qa_keywords_low) or any(k in q for k in qa_keywords_raw) or "?" in q or "？" in q
    has_design_action = any(k in low for k in design_action_keywords) or any(k in q for k in design_action_keywords)
    has_design_subject = any(k in low for k in design_subject_keywords) or any(k in q for k in design_subject_keywords)

    if is_qa:
        return "chat"
    if has_design_action and has_design_subject:
        return "design"
    if any(k in low for k in research_keywords):
        return "design"
    if has_design_subject and len(q) <= 24:
        return "chat"
    return "design"


def _looks_like_engineering_intake_request(user_query: str) -> bool:
    text = str(user_query or "")
    low = text.lower()
    design_terms = [
        "flyback", "qr", "valley", "ac/dc", "offline", "power supply", "converter",
        "反激", "电源", "隔离", "变压器", "开关电源",
    ]
    numeric_power_terms = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:v|vac|a|w)\b", low))
    return numeric_power_terms and any(term in low or term in text for term in design_terms)


def _extract_first_number(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _fmt_engineering_value(value: Any, suffix: str = "") -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value}{suffix}"


def _design_power_label(specs: Dict[str, Any]) -> str:
    try:
        power = float(specs.get("output_power") or 0)
    except Exception:
        power = 0.0
    if power > 0:
        return f"{power:g} W"
    try:
        vout = float(specs.get("output_voltage") or 0)
        iout = float(specs.get("output_current") or 0)
        if vout > 0 and iout > 0:
            return f"{vout * iout:g} W"
    except Exception:
        pass
    return "the requested power"


def _locked_spec_phrase(specs: Dict[str, Any]) -> str:
    vin_min = specs.get("input_voltage_min")
    vin_max = specs.get("input_voltage_max")
    vout = specs.get("output_voltage")
    iout = specs.get("output_current")
    power = specs.get("output_power")
    input_type = str(specs.get("input_type") or "DC").upper()
    input_unit = "Vac" if input_type == "AC" else "Vdc"
    parts = []
    if vin_min is not None and vin_max is not None:
        parts.append(f"{float(vin_min):g}-{float(vin_max):g}{input_unit}")
    if vout is not None and iout is not None:
        parts.append(f"{float(vout):g}V/{float(iout):g}A")
    if power is not None:
        parts.append(f"{float(power):g}W")
    return ", ".join(parts) or "the locked spec"


def _efficiency_requirement_label(specs: Dict[str, Any]) -> str:
    low_line = specs.get("efficiency_115vac_full_load")
    high_line = specs.get("efficiency_230vac_full_load")
    if low_line is not None and high_line is not None:
        return f">={float(low_line) * 100:g}% at 115 Vac; >={float(high_line) * 100:g}% at 230 Vac"
    if specs.get("efficiency_target") is not None:
        return f">={float(specs.get('efficiency_target')) * 100:g}% target"
    return "efficiency target not locked"


def _requirement_question_for_missing(item: str) -> str:
    text = str(item or "").lower()
    if "safety" in text or "hipot" in text or "pollution" in text:
        return "Which market/standard, insulation class, pollution degree, altitude, and hipot level should the design target?"
    if "standby" in text or "no-load" in text or "audible" in text:
        return "What no-load/standby limit is required, and is burst-mode audible noise constrained?"
    if "pcb" in text or "height" in text or "airflow" in text or "connector" in text:
        return "What board area, height, enclosure, airflow, connector, and mounting constraints apply?"
    if "cost" in text or "vendor" in text or "supply" in text:
        return "What BOM cost, production volume, approved vendors, and sourcing restrictions apply?"
    if "hold-up" in text or "brown" in text:
        return "What hold-up time, brown-in threshold, brown-out threshold, and restart behavior are required?"
    if "emi" in text or "emc" in text:
        return "Which conducted/radiated EMI standard, class, and margin policy should be used?"
    return "Please confirm this requirement before release-quality design work."


def _missing_input_label(item: str, index: int = 0) -> str:
    text = str(item or "").lower()
    prefix = "P0" if index == 0 else "P1"
    if "safety" in text or "hipot" in text or "pollution" in text:
        return f"{prefix} Safety approval target"
    if "standby" in text or "no-load" in text or "audible" in text:
        return f"{prefix} Standby/no-load behavior"
    if "pcb" in text or "height" in text or "airflow" in text or "connector" in text:
        return f"{prefix} Mechanical envelope"
    if "cost" in text or "vendor" in text or "supply" in text:
        return f"{prefix} Cost and sourcing"
    if "hold-up" in text or "brown" in text:
        return f"{prefix} Hold-up / brown-out"
    if "emi" in text or "emc" in text:
        return f"{prefix} EMI compliance target"
    return f"{prefix} Open requirement {index + 1}"


def _assumption_impact(label: str) -> str:
    text = str(label or "").lower()
    if "safety" in text:
        return "Affects creepage/clearance, transformer insulation, Y-cap choice, hipot, and release checklist."
    if "emi" in text:
        return "Affects input filter, noisy-loop layout, Y-cap path, and LISN pre-scan plan."
    if "environment" in text:
        return "Affects thermal derating, transformer/core size, capacitor lifetime, and airflow margin."
    if "topology" in text:
        return "Affects controller shortlist, transformer turns ratio, clamp stress, loop design, and EMI behavior."
    if "release" in text:
        return "Prevents polished schematic/BOM artifacts from being treated as release evidence."
    if "mosfet" in text:
        return "Affects primary switch footprint, VDS margin, clamp target, and sourcing alternates."
    return "Affects downstream sizing, evidence requirements, and release sign-off."


def _requirements_assumption_summary(row: Dict[str, Any]) -> str:
    label = str(row.get("label") or "").lower()
    value = str(row.get("value") or "")
    if "safety" in label:
        return f"{value}. Good enough for first-pass topology work; exact market/hipot is needed before release."
    if "emi" in label:
        return f"{value}. Use this as the pre-compliance direction; filter/layout details come later."
    if "environment" in label:
        return f"{value}. Edit only if board size, enclosure, or airflow is already known."
    if "topology" in label:
        return f"{value}. The actual topology choice is made in the next gate."
    if "release" in label:
        return "No final schematic, BOM, PCB, or manufacturing package from the intake gate."
    return value


def _requirements_release_input_summary(missing: list[str]) -> list[Dict[str, str]]:
    rows: list[Dict[str, str]] = []
    for index, item in enumerate(missing[:8]):
        text = str(item or "").lower()
        if "safety" in text or "hipot" in text or "pollution" in text or "altitude" in text:
            rows.append({
                "label": "Safety target",
                "value": "Optional now; lock market, insulation class, altitude, pollution degree, and hipot before release.",
                "status": "release-blocker",
            })
        elif "standby" in text or "no-load" in text or "audible" in text:
            rows.append({
                "label": "Standby/no-load",
                "value": "Optional now; needed later for no-load power, burst mode, and audible-noise choices.",
                "status": "later",
            })
        elif "pcb" in text or "height" in text or "airflow" in text or "connector" in text or "mechanical" in text:
            rows.append({
                "label": "Mechanical envelope",
                "value": "Optional now; default open-frame natural convection unless board size, height, enclosure, airflow, or connectors are known.",
                "status": "later",
            })
        elif "cost" in text or "vendor" in text or "supply" in text:
            rows.append({
                "label": "Cost/sourcing",
                "value": "Optional now; needed later for BOM target, volume, approved vendors, and alternate-source policy.",
                "status": "later",
            })
        elif "hold-up" in text or "brown" in text:
            rows.append({
                "label": "Hold-up/brown-out",
                "value": "Optional now; needed later for bulk capacitor sizing and restart behavior.",
                "status": "later",
            })
        elif "emi" in text or "emc" in text:
            rows.append({
                "label": "EMI scope",
                "value": "Optional now; lock conducted/radiated scope and margin policy before claiming compliance.",
                "status": "later",
            })
        else:
            rows.append({
                "label": f"Open input {index + 1}",
                "value": "Optional now; keep visible until this release input is confirmed.",
                "status": "later",
            })
    return rows


def _build_requirement_analysis_cards(
    user_query: str,
    specs: Dict[str, Any],
    assumptions: list[Dict[str, Any]],
    missing_inputs: list[str],
    topology_candidates: list[Dict[str, Any]],
) -> Dict[str, list[Dict[str, Any]]]:
    vin_min = specs.get("input_voltage_min")
    vin_max = specs.get("input_voltage_max")
    vout = specs.get("output_voltage")
    iout = specs.get("output_current")
    power_label = _design_power_label(specs)
    input_label = f"{float(vin_min):g}-{float(vin_max):g} Vac" if vin_min is not None and vin_max is not None else "input range not locked"
    output_label = f"{float(vout):g} V / {float(iout):g} A" if vout is not None and iout is not None else "output rail not locked"
    efficiency_label = _efficiency_requirement_label(specs)
    ripple_label = (
        f"{float(specs.get('max_ripple_mvpp')):g} mVp-p"
        if specs.get("max_ripple_mvpp") is not None
        else "ripple limit not locked"
    )
    ambient_label = (
        f"{float(specs.get('ambient_c_min')):g}-{float(specs.get('ambient_c_max')):g} C"
        if specs.get("ambient_c_min") is not None and specs.get("ambient_c_max") is not None
        else (f"0-{float(specs.get('ambient_c_max')):g} C" if specs.get("ambient_c_max") is not None else "ambient not locked")
    )
    topology_path = next(
        (str(row.get("value") or "") for row in assumptions if isinstance(row, dict) and row.get("label") == "Topology path"),
        "flyback topology path not locked",
    )
    emi_target = specs.get("emi_target") or "EMI target not locked"
    isolation = specs.get("isolation") or "isolation requirement not locked"

    missing_rows = [
        {
            "label": f"{'P0' if index == 0 else 'P1'} clarification {index + 1}",
            "value": item,
            "status": "open",
        }
        for index, item in enumerate(missing_inputs or [])
    ]
    clarification_rows = [
        {
            "label": f"Clarification question {index + 1}",
            "value": _requirement_question_for_missing(item),
            "status": "question",
        }
        for index, item in enumerate(missing_inputs or [])
    ]

    a1 = [
        {
            "label": "Application scenario",
            "value": f"Universal-input isolated AC/DC flyback supply converting {input_label} mains to {output_label} ({power_label}).",
            "status": "inferred",
        },
        {
            "label": "Operating environment",
            "value": f"Open-frame power PCB assumption, {ambient_label} ambient, offline mains input, reinforced-isolation direction, and {emi_target}.",
            "status": "assumed",
        },
        {
            "label": "Dominant priorities",
            "value": "Safety isolation, stable regulated output, efficiency/thermal margin, EMI pre-compliance, reliability, fault protection, cost awareness, manufacturable transformer, and orderable BOM evidence.",
            "status": "priority",
        },
        {
            "label": "Application implication",
            "value": "Because this is an offline isolated supply, transformer insulation, primary-side voltage stress, leakage clamp behavior, EMI filter/layout, and hazardous-voltage test access are design drivers.",
            "status": "risk",
        },
        {
            "label": "Assumption control",
            "value": "Only qualitative implications are inferred. No safety market, hipot level, hold-up time, enclosure, cost target, or detailed protection threshold is invented.",
            "status": "guardrail",
        },
    ]

    a2 = [
        {"label": "Converter class", "value": "Isolated AC/DC flyback power supply", "status": "extracted"},
        {"label": "Input requirement", "value": input_label, "status": "critical"},
        {"label": "Output requirement", "value": output_label, "status": "critical"},
        {"label": "Power requirement", "value": power_label, "status": "critical"},
        {"label": "Efficiency requirement", "value": efficiency_label, "status": "soft"},
        {"label": "Ripple requirement", "value": ripple_label, "status": "soft"},
        {"label": "Thermal/environment", "value": ambient_label, "status": "soft"},
        {"label": "Safety/isolation", "value": isolation, "status": "critical"},
        {"label": "EMI requirement", "value": emi_target, "status": "risk"},
        {"label": "Preferred topology/control", "value": topology_path, "status": "assumption"},
        {"label": "Requirement provenance", "value": "Numerical electrical limits come from the user request. Safety market, detailed protection thresholds, hold-up, cost, mechanical envelope, and exact test limits remain unconfirmed instead of being invented.", "status": "trace"},
    ] + missing_rows + clarification_rows

    a3 = [
        {"label": "Requirement finalization", "value": "Confirm open safety, standby/no-load, mechanical, cost/vendor, hold-up, protection, and test requirements before release-quality work.", "status": "task"},
        {"label": "Topology selection", "value": "Compare QR flyback, fixed-frequency flyback, and active-clamp flyback for efficiency, EMI, cost, thermal, complexity, and supply risk.", "status": "task"},
        {"label": "Power-stage sizing", "value": "Calculate Vbulk range, reflected voltage, duty cycle, Lp, Ipk/Irms, MOSFET VDS stack, current sense, output capacitor ripple/RMS, and clamp boundaries.", "status": "task"},
        {"label": "Magnetic design", "value": "Define core/bobbin/gap/turns/wire/insulation/pinout targets and later verify Lp, Llk, DCR, turns ratio, hipot, and temperature rise.", "status": "task"},
        {"label": "Device selection", "value": "Select MOSFET, rectifier/SR option, bridge, bulk cap, output caps, sense parts, clamp parts, optocoupler, and alternates with derating evidence.", "status": "task"},
        {"label": "Thermal design", "value": "Estimate semiconductor, transformer, clamp, bridge, capacitor, and PCB copper losses early; check worst-case ambient and airflow assumptions.", "status": "task"},
        {"label": "Control and loop design", "value": "Define TL431/opto bias and compensation, crossover target, phase/gain margin, CTR aging/corners, and load transient behavior.", "status": "task"},
        {"label": "Protection design", "value": "Define OCP, OVP, UVLO/brown-in/brown-out, short-circuit, OTP, startup, restart, and fault recovery once thresholds are confirmed.", "status": "task"},
        {"label": "Safety/compliance planning", "value": "Translate isolation and market assumptions into creepage/clearance, transformer insulation, Y-cap class, fuse/MOV/inrush path, hipot, and documentation checks.", "status": "task"},
        {"label": "EMI/layout design", "value": "Plan input filter, Y-cap path, high-di/dt loops, RCD loop, secondary current loop, isolation boundary, debug footprints, and LISN pre-scan.", "status": "task"},
        {"label": "Mechanical/package planning", "value": "Carry board area, height, connector, mounting, enclosure, airflow, isolation slot, and service-clearance constraints into topology, magnetics, and PCB decisions.", "status": "task"},
        {"label": "Verification planning", "value": "Prepare low/high-line, load, startup, transient, VDS, ripple, loop, thermal, EMI, safety, and BOM/AVL evidence gates.", "status": "task"},
    ]

    a4 = [
        {"label": "Efficiency vs thermal", "value": f"{power_label} at {efficiency_label} still leaves losses that must be assigned to MOSFET, transformer, rectifier, clamp, bridge, control, and capacitors. Thermal feasibility cannot wait until layout.", "status": "risk"},
        {"label": "Universal input vs stress", "value": f"{input_label} creates very different low-line current stress and high-line voltage stress. Both must be checked before transformer, MOSFET, and clamp values are frozen.", "status": "risk"},
        {"label": "Reinforced isolation vs size/cost", "value": "Reinforced isolation affects transformer construction, creepage/clearance, slots, optocoupler/Y-cap selection, hipot, PCB area, and vendor qualification.", "status": "risk"},
        {"label": "QR flyback vs EMI/debug risk", "value": "QR/valley switching can help switching loss, but VDS ringing, leakage energy, RCD loss, burst-mode behavior, and conducted EMI still need measurement or credible simulation.", "status": "risk"},
        {"label": "Ripple target vs output network", "value": f"{ripple_label} requires output capacitor ESR/ripple-current checks, layout control, load-transient definition, and measurement bandwidth rules.", "status": "risk"},
        {"label": "Power density vs cost/reliability", "value": "A compact offline supply pushes smaller magnetics, hotter semiconductors, tighter creepage routing, and denser EMI layout. That can conflict with low cost, long life, and manufacturable transformer construction.", "status": "tradeoff"},
        {"label": "Transient response undefined", "value": "Ripple is specified, but load-step size, allowable overshoot/undershoot, recovery time, and minimum load are not. Loop compensation and output capacitor sizing cannot be finalized without them.", "status": "open"},
        {"label": "Reliability target undefined", "value": "Lifetime, mission profile, capacitor life target, derating rules, and failure-rate target are not specified, so reliability can only be handled as a qualitative priority at this gate.", "status": "open"},
        {"label": "Standby target missing", "value": "Without no-load/standby and audible-noise limits, startup/VDD supply, TL431/opto bias, burst mode, bleeders, and X-cap discharge choices cannot be finalized.", "status": "open"},
        {"label": "Safety/EMI standards missing", "value": "CISPR 32 direction and reinforced isolation assumption are useful, but market, exact standard edition, altitude, pollution degree, hipot, and EMI margin policy remain release blockers.", "status": "open"},
    ]

    a5 = [
        {"label": "Hard constraints", "value": f"Operate from {input_label}; regulate {output_label}; deliver {power_label}; respect {isolation}; target {emi_target}; keep ripple below {ripple_label}.", "status": "hard"},
        {"label": "High-priority objectives", "value": "Maintain safety isolation, avoid MOSFET/rectifier overstress, keep thermal rise manageable, preserve loop stability, and reserve EMI/debug options.", "status": "priority"},
        {"label": "Medium-priority objectives", "value": "Optimize efficiency, cost, PCB area, transformer manufacturability, alternate sourcing, and ease of EVT debugging after hard constraints are protected.", "status": "priority"},
        {"label": "Pending requirements", "value": "; ".join(missing_inputs[:8]) if missing_inputs else "No major requirement gaps detected at intake.", "status": "open"},
        {"label": "Workflow adjustment", "value": "Resolve architecture-critical unknowns first, compare topology/controller options, then move power-stage stress and magnetics earlier than schematic generation.", "status": "plan"},
        {"label": "Early verification trigger", "value": "Run low-line current stress and high-line VDS/clamp checks before BOM freeze; keep RCD, transformer, loop, EMI, and thermal claims open until evidence exists.", "status": "plan"},
        {"label": "Recommended strategy", "value": "Proceed as a gated QR-flyback first pass only after defaults are accepted or edited; keep fixed-frequency and active-clamp paths as rollback options.", "status": "strategy"},
        {"label": "Final node output", "value": "A reviewed requirement package with extracted specs, explicit assumptions, open questions, task decomposition, feasibility risks, and downstream workflow plan.", "status": "summary"},
        {"label": "Release rule", "value": "A release package can only close when every claim links to calculation, datasheet/source, simulator/EDA output, or bench measurement.", "status": "hold"},
    ]

    return {
        "a1_application_scenario": a1,
        "a2_specifications_and_missing": a2,
        "a3_preliminary_design_tasks": a3,
        "a4_feasibility_conflicts": a4,
        "a5_refined_objectives_workflow": a5,
    }


def _source_snippet(text: str, pattern: str, fallback: str = "") -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return fallback
    start = max(0, match.start() - 32)
    end = min(len(text), match.end() + 32)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def _infer_requirement_project_summary(text: str, specs: Dict[str, Any]) -> Dict[str, str]:
    low = str(text or "").lower()
    if "flyback" in low or "反激" in text:
        design_object = "isolated flyback power supply"
    elif "dc-dc" in low or "dcdc" in low or "converter" in low:
        design_object = "power converter"
    elif "power supply" in low or "电源" in text:
        design_object = "power supply"
    else:
        design_object = "power electronics system"

    application = "offline mains power supply"
    if any(token in low for token in ["ev", "electric vehicle", "automotive", "vehicle", "车载", "汽车"]):
        application = "automotive / EV power system"
    elif any(token in low for token in ["data center", "server", "数据中心", "服务器"]):
        application = "data-center / server power system"
    elif any(token in low for token in ["grid", "inverter", "solar", "电网", "光伏", "逆变"]):
        application = "grid-connected power system"
    elif any(token in low for token in ["aerospace", "aircraft", "航空", "航天"]):
        application = "aerospace power system"
    elif any(token in low for token in ["consumer", "charger", "adapter", "消费", "充电器", "适配器"]):
        application = "consumer adapter / charger"
    elif specs.get("input_type") == "AC" or "ac/dc" in low or "offline" in low:
        application = "offline universal-input AC/DC supply"

    if specs.get("input_type") == "AC":
        conversion_type = "isolated AC/DC conversion" if specs.get("isolation") else "AC/DC conversion"
    elif specs.get("input_voltage_min") is not None or specs.get("input_voltage_nominal") is not None:
        conversion_type = "DC/DC conversion"
    else:
        conversion_type = "power conversion"

    return {
        "design_object": design_object,
        "application": application,
        "conversion_type": conversion_type,
        "main_user_goal": str(text or "").strip()[:500],
    }


def _requirement_application_analysis(summary: Dict[str, str], specs: Dict[str, Any]) -> Dict[str, Any]:
    application = summary.get("application") or "unspecified application"
    operating_environment: list[str] = []
    implications: list[str] = []
    priorities: list[str] = []
    if "offline" in application.lower() or specs.get("input_type") == "AC":
        operating_environment.extend(["offline mains input", "hazardous primary voltage", "isolated secondary output when isolation is required"])
        implications.extend([
            "Safety spacing, transformer insulation, optocoupler/Y-cap selection, fuse/MOV/inrush path, and hipot planning are design drivers.",
            "Conducted EMI and high-di/dt loop layout need to be considered before schematic freeze.",
            "High-line drain stress and low-line RMS current stress must both be reviewed.",
        ])
        priorities.extend(["safety isolation", "EMI pre-compliance", "thermal margin", "manufacturable transformer"])
    if "automotive" in application.lower() or "ev" in application.lower():
        operating_environment.extend(["vehicle electrical environment", "thermal cycling and vibration risk"])
        implications.extend([
            "Automotive suitability implies reliability, input disturbance, EMI/EMC, protection, and thermal-derating awareness.",
            "The exact automotive EMI/EMC standard is not assumed unless the user provides it.",
        ])
        priorities.extend(["reliability", "protection behavior", "EMI/EMC readiness"])
    if not operating_environment:
        operating_environment.append("application environment not fully specified")
        implications.append("Only requirement-level implications are inferred; no unsupported numerical standards are invented.")
    if specs.get("ambient_c_max") is not None:
        operating_environment.append(f"ambient up to {float(specs.get('ambient_c_max')):g} C")
    priorities.extend(["regulated output", "efficiency target", "ripple target", "verification traceability"])
    return {
        "use_case": application,
        "operating_environment": list(dict.fromkeys(operating_environment)),
        "application_driven_implications": list(dict.fromkeys(implications)),
        "design_priorities": list(dict.fromkeys(priorities)),
    }


def _requirement_explicit_specs(text: str, specs: Dict[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    derived_fields = set(specs.get("_derived_fields") or [])

    def add(name: str, value: Any, unit: str, pattern: str, requirement_type: str = "electrical") -> None:
        if value is None or value == "-":
            return
        rows.append({
            "name": name,
            "value": value,
            "unit": unit,
            "source_text": _source_snippet(text, pattern, fallback=str(value)),
            "requirement_type": requirement_type,
            "status": "confirmed",
        })

    if specs.get("input_voltage_min") is not None and specs.get("input_voltage_max") is not None:
        unit = "Vac" if specs.get("input_type") == "AC" else "V"
        add("input_voltage_range", [specs.get("input_voltage_min"), specs.get("input_voltage_max")], unit, r"\d+(?:\.\d+)?\s*(?:-|–|—|~|to|至|到)\s*\d+(?:\.\d+)?\s*v(?:ac)?")
    if specs.get("input_voltage_nominal") is not None:
        add("input_voltage_nominal", specs.get("input_voltage_nominal"), "V", r"\d+(?:\.\d+)?\s*v\s*(?:to|->|到|至)")
    if specs.get("output_voltage") is not None:
        add("output_voltage", specs.get("output_voltage"), "V", r"\d+(?:\.\d+)?\s*v\s*(?:/|,|，|\s+)\s*\d+(?:\.\d+)?\s*a|\d+(?:\.\d+)?\s*v\s*(?:output|输出|out)")
    if specs.get("output_current") is not None and "output_current" not in derived_fields:
        add("output_current", specs.get("output_current"), "A", r"\d+(?:\.\d+)?\s*v\s*(?:/|,|，|\s+)\s*\d+(?:\.\d+)?\s*a|\d+(?:\.\d+)?\s*a\b")
    if specs.get("output_power") is not None and re.search(r"\d+(?:\.\d+)?\s*k?w\b", text, re.I):
        add("rated_output_power", specs.get("output_power"), "W", r"\d+(?:\.\d+)?\s*k?w\b")
    if specs.get("peak_output_power") is not None:
        add("peak_output_power", specs.get("peak_output_power"), "W", r"(?:peak|峰值)[^.\n,;]{0,40}\d+(?:\.\d+)?\s*k?w|\d+(?:\.\d+)?\s*k?w[^.\n,;]{0,24}(?:peak|峰值)", "electrical")
    if specs.get("peak_duration_s") is not None:
        add("peak_duration", specs.get("peak_duration_s"), "s", r"\d+(?:\.\d+)?\s*s\b", "electrical")
    if specs.get("efficiency_target") is not None:
        add("efficiency_target", f">={float(specs.get('efficiency_target')) * 100:g}", "%", r"(?:≥|>=|>|at least|效率|efficiency)[^0-9]{0,30}\d+(?:\.\d+)?\s*%", "thermal")
    if specs.get("max_ripple_mvpp") is not None:
        add("output_ripple_limit", specs.get("max_ripple_mvpp"), "mVp-p", r"(?:ripple|纹波)[^0-9]{0,24}\d+(?:\.\d+)?\s*m?v", "electrical")
    if specs.get("ambient_c_max") is not None:
        ambient_value = [specs.get("ambient_c_min"), specs.get("ambient_c_max")] if specs.get("ambient_c_min") is not None else specs.get("ambient_c_max")
        add("ambient_temperature", ambient_value, "C", r"-?\d+(?:\.\d+)?\s*(?:-|–|—|~|to|至|到)?\s*-?\d*(?:\.\d+)?\s*(?:°\s*)?c", "thermal")
    if specs.get("emi_target"):
        add("emi_target", specs.get("emi_target"), "", r"cispr\s*32|en\s*55032|class\s*b|emi", "EMI")
    if specs.get("isolation"):
        add("isolation_requirement", specs.get("isolation"), "", r"reinforced|isolation|隔离|加强绝缘|\d+(?:\.\d+)?\s*kvac", "safety")
    if specs.get("feedback"):
        add("feedback_preference", specs.get("feedback"), "", r"tl431|opto|光耦", "control")
    if specs.get("topology_preference"):
        add("topology_preference", specs.get("topology_preference"), "", r"qr|valley|flyback|反激|谷底|准谐振", "other")
    return rows


def _requirement_derived_specs(specs: Dict[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    vout = specs.get("output_voltage")
    iout = specs.get("output_current")
    pout = specs.get("output_power")
    eta = specs.get("efficiency_target") or specs.get("efficiency_230vac_full_load") or specs.get("efficiency_115vac_full_load")
    derived_fields = set(specs.get("_derived_fields") or [])

    def add(name: str, value: Any, unit: str, formula: str, based_on: list[str]) -> None:
        rows.append({
            "name": name,
            "value": value,
            "unit": unit,
            "formula": formula,
            "based_on": based_on,
            "status": "derived",
        })

    try:
        if vout is not None and iout is not None and "output_current" not in derived_fields:
            derived_pout = float(vout) * float(iout)
            add("rated_output_power_from_output_rating", round(derived_pout, 3), "W", f"Pout = Vout * Iout = {float(vout):g} V * {float(iout):g} A", ["output_voltage", "output_current"])
            if pout is None:
                pout = derived_pout
        if pout is not None and vout is not None and (iout is None or float(iout) <= 0 or "output_current" in derived_fields):
            derived_iout = float(pout) / float(vout)
            add("rated_output_current", round(derived_iout, 3), "A", f"Iout = Pout / Vout = {float(pout):g} W / {float(vout):g} V", ["rated_output_power", "output_voltage"])
        if specs.get("peak_output_power") is not None and vout is not None:
            peak_i = float(specs.get("peak_output_power")) / float(vout)
            add("peak_output_current", round(peak_i, 3), "A", f"Iout,peak = Ppeak / Vout = {float(specs.get('peak_output_power')):g} W / {float(vout):g} V", ["peak_output_power", "output_voltage"])
        if pout is not None and eta is not None and float(eta) > 0:
            pin = float(pout) / float(eta)
            loss = pin - float(pout)
            add("estimated_input_power_at_target_efficiency", round(pin, 3), "W", f"Pin = Pout / eta = {float(pout):g} W / {float(eta):g}", ["rated_output_power", "efficiency_target"])
            add("estimated_loss_at_target_efficiency", round(loss, 3), "W", f"Ploss = Pout / eta - Pout = {float(pout):g} W / {float(eta):g} - {float(pout):g} W", ["rated_output_power", "efficiency_target"])
        if specs.get("max_ripple_voltage") is not None and vout is not None and float(vout) > 0:
            pct = float(specs.get("max_ripple_voltage")) / float(vout) * 100.0
            add("ripple_limit_percent_of_output", round(pct, 3), "%", f"Vripple/Vout = {float(specs.get('max_ripple_voltage')):g} V / {float(vout):g} V * 100", ["output_ripple_limit", "output_voltage"])
    except Exception:
        pass
    return rows


def _requirement_qualitative_specs(text: str, specs: Dict[str, Any]) -> list[Dict[str, Any]]:
    low = str(text or "").lower()
    candidates = [
        ("compact", ["compact", "small", "小型", "紧凑"], "compact packaging requested or implied"),
        ("cost_effective", ["cost-effective", "low cost", "cheap", "低成本", "成本"], "cost sensitivity requested or implied"),
        ("manufacturable", ["manufacturable", "production", "量产", "可制造"], "manufacturability requested"),
        ("reliable", ["reliable", "可靠", "reliability"], "reliability requested or implied"),
        ("pre_compliance", ["pre-compliance", "预一致", "预认证"], "pre-compliance target requested"),
        ("no_final_schematic_until_gated", ["do not generate final schematic", "不要生成最终原理图", "gates are locked"], "release-generation gating requested"),
    ]
    rows = []
    for name, terms, description in candidates:
        if any(term in low or term in text for term in terms):
            rows.append({
                "name": name,
                "description": description,
                "status": "requires_clarification" if name in {"compact", "cost_effective", "reliable"} else "confirmed",
            })
    if specs.get("topology_preference"):
        rows.append({
            "name": "preferred_topology_family",
            "description": ", ".join(str(x) for x in specs.get("topology_preference") or []),
            "status": "confirmed",
        })
    if specs.get("feedback"):
        rows.append({
            "name": "preferred_feedback_method",
            "description": str(specs.get("feedback")),
            "status": "confirmed",
        })
    return rows


def _missing_info_entry(item: str, priority: str, why: str, tasks: list[str], question: str) -> Dict[str, Any]:
    return {
        "item": item,
        "priority": priority,
        "why_it_matters": why,
        "affected_design_tasks": tasks,
        "clarification_question": question,
    }


def _requirement_missing_information(specs: Dict[str, Any], base_missing: list[str]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    if not specs.get("isolation"):
        rows.append(_missing_info_entry(
            "galvanic_isolation",
            "blocking",
            "Isolation requirement strongly affects topology selection, transformer or inductor choice, safety strategy, cost, size, efficiency, and verification.",
            ["topology_selection", "magnetic_design", "safety_design", "pcb_layout"],
            "Is galvanic isolation required between input and output?",
        ))
    if (not specs.get("input_voltage_min") or not specs.get("input_voltage_max")) and not specs.get("input_voltage_nominal"):
        rows.append(_missing_info_entry(
            "input_voltage_range",
            "blocking",
            "Input voltage range is required before topology, duty ratio, semiconductor stress, magnetics, and protection limits can be sized.",
            ["topology_selection", "power_stage_design", "protection_design"],
            "What is the minimum, nominal, and maximum input voltage?",
        ))
    elif specs.get("input_voltage_nominal") and (not specs.get("input_voltage_min") or not specs.get("input_voltage_max")):
        rows.append(_missing_info_entry(
            "input_voltage_range",
            "high",
            "A nominal input voltage is enough for rough discussion, but min/max input range is needed for duty-cycle, current-stress, protection, and thermal design.",
            ["topology_selection", "power_stage_design", "protection_design"],
            "What minimum and maximum input voltage range should be used around the nominal input?",
        ))
    if not specs.get("output_voltage") or not (specs.get("output_current") or specs.get("output_power")):
        rows.append(_missing_info_entry(
            "output_rating",
            "blocking",
            "Output voltage and current or power define load current, magnetics energy, rectifier stress, thermal load, and verification points.",
            ["topology_selection", "power_stage_design", "thermal_design", "verification_planning"],
            "What output voltage and rated current or output power are required?",
        ))

    if not specs.get("safety_standard"):
        rows.append(_missing_info_entry(
            "safety_approval_target",
            "high",
            "Exact safety market, insulation class, pollution degree, altitude, and hipot level control creepage/clearance, transformer insulation, Y-cap, and release documentation.",
            ["safety_design", "magnetic_design", "pcb_layout", "verification_planning"],
            "Which market/standard, insulation class, pollution degree, altitude, and hipot level should the design target?",
        ))
    if not specs.get("emi_target"):
        rows.append(_missing_info_entry(
            "emi_emc_standard",
            "high",
            "The EMI standard, class, conducted/radiated scope, and margin policy affect filter topology, PCB placement, Y-cap path, and LISN pre-scan plan.",
            ["emi_design", "pcb_layout", "verification_planning"],
            "Which EMI/EMC standard, class, conducted/radiated scope, and margin policy should be used?",
        ))
    if specs.get("ambient_c_max") is None:
        rows.append(_missing_info_entry(
            "ambient_temperature",
            "high",
            "Ambient temperature sets thermal derating, capacitor lifetime, transformer size, semiconductor package margin, and validation conditions.",
            ["thermal_design", "component_selection", "verification_planning"],
            "What ambient temperature range should the design survive?",
        ))
    if not any("mechanical" in item.lower() or "pcb" in item.lower() for item in base_missing) and not specs.get("mechanical_limit"):
        pass
    if not specs.get("mechanical_limit"):
        rows.append(_missing_info_entry(
            "mechanical_envelope_and_cooling",
            "high",
            "Board area, height, enclosure, airflow, connector, and cooling path affect topology choice, transformer size, creepage routing, and thermal feasibility.",
            ["mechanical_design", "thermal_design", "magnetic_design", "pcb_layout"],
            "What board area, height, enclosure, airflow, connector, and mounting constraints apply?",
        ))
    if specs.get("max_ripple_mvpp") is None:
        rows.append(_missing_info_entry(
            "output_ripple_limit",
            "medium",
            "Ripple limit affects output capacitor ESR/RMS current, post-filter needs, layout, and measurement method.",
            ["power_stage_design", "control_design", "verification_planning"],
            "What output ripple limit and measurement bandwidth should be used?",
        ))
    rows.append(_missing_info_entry(
        "transient_response",
        "medium",
        "Load-step size, allowed overshoot/undershoot, and recovery time affect output capacitance, compensation, and controller selection.",
        ["control_design", "output_filter_design", "verification_planning"],
        "What load transient step, overshoot/undershoot limit, and recovery time are required?",
    ))
    rows.append(_missing_info_entry(
        "protection_thresholds_and_fault_response",
        "medium",
        "OCP, OVP, UVLO/brown-in/out, short-circuit, OTP, latch/retry behavior, and restart policy affect controller, sense, thermal, and test design.",
        ["protection_design", "control_design", "verification_planning"],
        "What protection thresholds and fault response behavior are required?",
    ))
    if not specs.get("standby_power_mw_target"):
        rows.append(_missing_info_entry(
            "standby_no_load_behavior",
            "medium",
            "No-load power and burst-mode audible-noise constraints affect startup/VDD, feedback bias, bleeders, and control mode.",
            ["controller_selection", "control_design", "thermal_design"],
            "What no-load/standby limit is required, and is burst-mode audible noise constrained?",
        ))
    if not specs.get("hold_up_time_ms"):
        rows.append(_missing_info_entry(
            "hold_up_and_brownout",
            "medium",
            "Hold-up and brown-in/brown-out behavior affect bulk capacitor size, startup, UVLO, and restart validation.",
            ["input_stage_design", "protection_design", "verification_planning"],
            "What hold-up time, brown-in threshold, brown-out threshold, and restart behavior are required?",
        ))
    if not specs.get("cost_target"):
        rows.append(_missing_info_entry(
            "cost_and_sourcing",
            "low",
            "Cost, production volume, approved vendors, and second-source policy affect topology complexity, package choices, magnetics vendor, and AVL depth.",
            ["component_selection", "bom_avl", "manufacturing_planning"],
            "What BOM cost, production volume, approved vendors, and sourcing restrictions apply?",
        ))
    rows.append(_missing_info_entry(
        "lifetime_reliability_target",
        "low",
        "Lifetime and reliability targets drive capacitor life, derating rules, thermal margin, and qualification testing.",
        ["thermal_design", "component_selection", "verification_planning"],
        "What lifetime, mission profile, derating policy, or reliability target should be used?",
    ))

    priority_order = {"blocking": 0, "high": 1, "medium": 2, "low": 3}
    dedup: dict[str, Dict[str, Any]] = {}
    for row in rows:
        item = str(row.get("item") or "")
        if item and item not in dedup:
            dedup[item] = row
    return sorted(dedup.values(), key=lambda row: priority_order.get(str(row.get("priority")), 9))


def _requirement_task_decomposition() -> list[Dict[str, Any]]:
    return [
        {"task_name": "Topology Selection", "purpose": "Compare viable converter architecture paths without freezing hardware.", "required_inputs": ["Vin range", "Vout", "rated power", "peak power if any", "isolation status", "efficiency target", "cost priority"], "expected_outputs": ["candidate topologies", "selected primary path", "backup paths", "rollback criteria"], "downstream_agent": "topology_selection_agent"},
        {"task_name": "Power Stage Design", "purpose": "Translate requirements into voltage/current stress and energy targets.", "required_inputs": ["selected topology", "Vin range", "Vout/Iout", "efficiency target", "switching frequency range"], "expected_outputs": ["duty/reflected-voltage sweep", "Lp/Ipk/Irms targets", "MOSFET/rectifier stress", "capacitor ripple requirements"], "downstream_agent": "power_stage_design_agent"},
        {"task_name": "Semiconductor Selection", "purpose": "Select derated switch/rectifier devices and alternates.", "required_inputs": ["stress sweep", "thermal budget", "package constraints", "sourcing policy"], "expected_outputs": ["MOSFET/diode/SR requirements", "loss estimate", "derating table", "AVL candidates"], "downstream_agent": "component_selection_agent"},
        {"task_name": "Magnetic Design", "purpose": "Define manufacturable inductor/transformer targets.", "required_inputs": ["topology", "power level", "switching frequency", "current ripple/energy", "isolation requirements"], "expected_outputs": ["core/bobbin/gap/winding package", "Lp/Llk/DCR targets", "insulation/hipot notes"], "downstream_agent": "magnetic_design_agent"},
        {"task_name": "Thermal Design", "purpose": "Check feasibility of loss budget in the declared environment.", "required_inputs": ["estimated losses", "ambient", "cooling method", "board/enclosure constraints"], "expected_outputs": ["thermal risk", "cooling strategy", "temperature-rise targets"], "downstream_agent": "thermal_design_agent"},
        {"task_name": "Control Design", "purpose": "Define regulation and dynamic-response strategy.", "required_inputs": ["topology", "feedback method", "output ripple/transient targets", "load range"], "expected_outputs": ["loop architecture", "compensation targets", "PM/GM/CTR corner plan"], "downstream_agent": "control_design_agent"},
        {"task_name": "Protection Design", "purpose": "Define fault thresholds and recovery behavior.", "required_inputs": ["fault list", "thresholds", "controller protections", "system recovery policy"], "expected_outputs": ["OCP/OVP/UVLO/OTP strategy", "restart/latch behavior", "test cases"], "downstream_agent": "protection_design_agent"},
        {"task_name": "EMI / Filter Design", "purpose": "Plan conducted/radiated noise mitigation and layout constraints.", "required_inputs": ["EMI standard", "switching frequency", "noisy-loop geometry", "Y-cap policy"], "expected_outputs": ["filter strategy", "layout rules", "LISN pre-scan plan"], "downstream_agent": "emi_design_agent"},
        {"task_name": "Verification Planning", "purpose": "Turn requirements and assumptions into a test matrix.", "required_inputs": ["locked requirements", "risk register", "simulation scope", "bench capabilities"], "expected_outputs": ["PLECS/bench matrix", "pass/fail criteria", "evidence traceability"], "downstream_agent": "verification_agent"},
    ]


def _requirement_feasibility_conflicts(specs: Dict[str, Any], missing_info: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    power = specs.get("output_power")
    eta = specs.get("efficiency_target") or specs.get("efficiency_230vac_full_load") or specs.get("efficiency_115vac_full_load")
    loss_basis = ""
    risk_level = "medium"
    try:
        if power is not None and eta is not None and float(eta) > 0:
            loss = float(power) / float(eta) - float(power)
            loss_basis = f"Ploss = {float(power):g} W / {float(eta):g} - {float(power):g} W = {loss:.2f} W"
            if loss > 20 or (specs.get("ambient_c_max") and float(specs.get("ambient_c_max")) >= 50):
                risk_level = "high"
    except Exception:
        pass
    rows.append({
        "issue": "Efficiency vs thermal",
        "risk_level": risk_level,
        "explanation": "Even if the efficiency target is met, the remaining loss must be assigned to switch, rectifier, transformer, clamp, bridge, control, and capacitors.",
        "quantitative_basis": loss_basis or "Loss cannot be quantified until output power and efficiency target are both known.",
        "affected_requirements": ["efficiency_target", "ambient_temperature", "output_power"],
        "recommended_action": "Create an early loss budget and thermal check before schematic freeze.",
    })
    rows.append({
        "issue": "Universal input vs voltage/current stress",
        "risk_level": "high" if specs.get("input_type") == "AC" else "medium",
        "explanation": "Low-line operation drives RMS/peak current and thermal stress, while high-line operation drives MOSFET drain stress and clamp/EMI behavior.",
        "quantitative_basis": _locked_spec_phrase(specs),
        "affected_requirements": ["input_voltage_range", "output_power", "isolation_requirement"],
        "recommended_action": "Run low-line full-load and high-line stress sweeps before topology/controller freeze.",
    })
    rows.append({
        "issue": "Isolation/safety vs size/cost",
        "risk_level": "high" if specs.get("isolation") else "high",
        "explanation": "Isolation changes topology, transformer construction, creepage/clearance, optocoupler/Y-cap choices, PCB area, and vendor qualification.",
        "quantitative_basis": str(specs.get("isolation") or "isolation not confirmed"),
        "affected_requirements": ["isolation_requirement", "mechanical_envelope", "cost_target"],
        "recommended_action": "Keep isolation assumptions explicit and require safety target confirmation before release.",
    })
    rows.append({
        "issue": "EMI readiness vs missing standard/margin",
        "risk_level": "high" if any(row.get("item") == "emi_emc_standard" for row in missing_info) else "medium",
        "explanation": "Pre-compliance target affects filter design, switching edge control, Y-cap path, PCB layout, and LISN/radiated test plan.",
        "quantitative_basis": str(specs.get("emi_target") or "specific EMI/EMC limit not provided"),
        "affected_requirements": ["emi_target", "layout", "switching_frequency"],
        "recommended_action": "Treat EMI as an early design constraint, not a late filter add-on.",
    })
    rows.append({
        "issue": "Protection request vs undefined thresholds",
        "risk_level": "medium",
        "explanation": "Protection functions cannot be finalized without thresholds and recovery behavior.",
        "quantitative_basis": "No protection thresholds parsed from the user request.",
        "affected_requirements": ["OCP", "OVP", "UVLO", "OTP", "short_circuit"],
        "recommended_action": "Carry protection thresholds as pending requirements into controller and verification planning.",
    })
    return rows


def _requirement_refined_objectives(specs: Dict[str, Any], missing_info: list[Dict[str, Any]]) -> Dict[str, list[str]]:
    hard_constraints: list[str] = []
    if specs.get("input_voltage_min") is not None and specs.get("input_voltage_max") is not None:
        hard_constraints.append(f"Input range: {float(specs.get('input_voltage_min')):g}-{float(specs.get('input_voltage_max')):g} {'Vac' if specs.get('input_type') == 'AC' else 'V'}")
    if specs.get("output_voltage") is not None:
        if specs.get("output_current") is not None:
            hard_constraints.append(f"Output: {float(specs.get('output_voltage')):g} V / {float(specs.get('output_current')):g} A")
        else:
            hard_constraints.append(f"Output voltage: {float(specs.get('output_voltage')):g} V")
    if specs.get("output_power") is not None:
        hard_constraints.append(f"Rated output power: {float(specs.get('output_power')):g} W")
    if specs.get("isolation"):
        hard_constraints.append(f"Isolation: {specs.get('isolation')}")
    if specs.get("max_ripple_mvpp") is not None:
        hard_constraints.append(f"Output ripple: <{float(specs.get('max_ripple_mvpp')):g} mVp-p")
    if specs.get("ambient_c_max") is not None:
        hard_constraints.append(f"Ambient: up to {float(specs.get('ambient_c_max')):g} C")
    if specs.get("efficiency_target") is not None:
        hard_constraints.append(f"Efficiency target: >={float(specs.get('efficiency_target')) * 100:g}%")
    pending = [
        f"{row.get('item')}: {row.get('clarification_question')}"
        for row in missing_info
        if row.get("priority") in {"blocking", "high", "medium"}
    ]
    return {
        "hard_constraints": hard_constraints,
        "high_priority_objectives": ["thermal feasibility", "safe voltage/current derating", "stable regulated output", "EMI-aware layout/filter path", "manufacturable magnetics", "fault-protection strategy"],
        "medium_priority_objectives": ["cost effectiveness", "compact packaging", "alternate sourcing", "debug/test access", "low audible noise where applicable"],
        "pending_requirements": pending[:12],
    }


def _requirement_handoff_package(summary: Dict[str, str], specs: Dict[str, Any], derived: list[Dict[str, Any]], missing_info: list[Dict[str, Any]], conflicts: list[Dict[str, Any]]) -> Dict[str, Any]:
    derived_by_name = {row.get("name"): row for row in derived if isinstance(row, dict)}
    thermal_loss = derived_by_name.get("estimated_loss_at_target_efficiency", {}).get("value")
    return {
        "for_topology_selection_agent": {
            "conversion": summary.get("conversion_type"),
            "vin_range": [specs.get("input_voltage_min"), specs.get("input_voltage_max")],
            "vout": specs.get("output_voltage"),
            "rated_power": specs.get("output_power"),
            "efficiency_target": specs.get("efficiency_target"),
            "isolation_requirement": specs.get("isolation") or "unknown",
            "application": summary.get("application"),
            "major_constraints": [row.get("item") for row in missing_info if row.get("priority") in {"blocking", "high"}],
            "blocking_questions": [row.get("clarification_question") for row in missing_info if row.get("priority") == "blocking"],
        },
        "for_thermal_design_agent": {
            "rated_output_power": specs.get("output_power"),
            "efficiency_target": specs.get("efficiency_target"),
            "estimated_loss_at_target_efficiency_w": thermal_loss,
            "ambient_temperature_max_c": specs.get("ambient_c_max"),
            "cooling_condition": specs.get("cooling_method") or "unknown / default open-frame assumption",
            "thermal_risk": next((row.get("risk_level") for row in conflicts if row.get("issue") == "Efficiency vs thermal"), "medium"),
            "early_analysis_required": True,
        },
        "for_control_design_agent": {
            "feedback_preference": specs.get("feedback") or "unknown",
            "output_voltage": specs.get("output_voltage"),
            "ripple_limit_mvpp": specs.get("max_ripple_mvpp"),
            "transient_response": "not specified",
            "loop_evidence_required": ["Bode plot", "phase margin", "gain margin", "load transient", "opto CTR corners"],
        },
        "for_protection_design_agent": {
            "required_protection": "pending definition",
            "missing_thresholds": ["input undervoltage", "input overvoltage", "output overvoltage", "overcurrent", "short circuit", "overtemperature"],
            "fault_response_unknown": True,
        },
        "for_emi_design_agent": {
            "application": summary.get("application"),
            "emi_awareness_required": True,
            "specific_standard": specs.get("emi_target") or "not specified",
            "status": "pending clarification" if not specs.get("emi_target") else "pre-compliance target captured",
        },
        "for_verification_agent": {
            "must_verify": [
                "operation at minimum input",
                "operation at nominal input where applicable",
                "operation at maximum input",
                "regulated output voltage",
                "rated full-load output",
                "efficiency target",
                "output ripple",
                "startup and protection behavior",
                "thermal operation at max ambient",
            ],
            "cannot_finalize_until_confirmed": [row.get("item") for row in missing_info if row.get("priority") in {"blocking", "high", "medium"}],
        },
    }


def _requirement_quality_check(explicit_specs: list[Dict[str, Any]], derived_specs: list[Dict[str, Any]], missing_info: list[Dict[str, Any]]) -> Dict[str, Any]:
    blocking = [row for row in missing_info if row.get("priority") == "blocking"]
    hard_blocking_items = {"input_voltage_range", "output_rating"}
    high = [row for row in missing_info if row.get("priority") == "high"]
    has_hard_blocker = any(row.get("item") in hard_blocking_items for row in blocking)
    ready = "blocked" if has_hard_blocker else ("partial" if (blocking or high) else "ready")
    return {
        "all_explicit_specs_have_source": all(bool(row.get("source_text")) for row in explicit_specs),
        "derived_values_show_formula": all(bool(row.get("formula")) for row in derived_specs),
        "unsupported_numbers_avoided": True,
        "critical_missing_information_identified": bool(missing_info),
        "ready_for_next_stage": ready,
        "ready_reason": (
            "Critical architecture information is missing; ask clarification before meaningful topology selection."
            if has_hard_blocker else
            "Proceed to preliminary topology comparison, but release-quality design remains blocked by high-priority confirmations."
            if (blocking or high) else
            "Requirements are sufficiently grounded for the next design stage."
        ),
    }


def _build_requirement_analysis_package(user_query: str, specs: Dict[str, Any], assumptions: list[Dict[str, Any]], missing_inputs: list[str]) -> Dict[str, Any]:
    summary = _infer_requirement_project_summary(user_query, specs)
    application = _requirement_application_analysis(summary, specs)
    explicit_specs = _requirement_explicit_specs(user_query, specs)
    derived_specs = _requirement_derived_specs(specs)
    qualitative_specs = _requirement_qualitative_specs(user_query, specs)
    missing_info = _requirement_missing_information(specs, missing_inputs)
    tasks = _requirement_task_decomposition()
    conflicts = _requirement_feasibility_conflicts(specs, missing_info)
    objectives = _requirement_refined_objectives(specs, missing_info)
    workflow = [
        {"step": 1, "action": "Review parsed explicit and derived requirements", "reason": "Prevent silent unit or source drift before design work starts.", "depends_on": []},
        {"step": 2, "action": "Resolve blocking/high-priority missing inputs or accept documented assumptions", "reason": "Avoid treating assumptions as confirmed requirements.", "depends_on": ["requirement_analysis"]},
        {"step": 3, "action": "Run preliminary topology comparison", "reason": "Topology choice drives stress, magnetics, EMI, thermal, and control design.", "depends_on": ["spec_lock", "assumption_register"]},
        {"step": 4, "action": "Create power-stage stress and loss budget", "reason": "Expose low-line/high-line and thermal risks before schematic generation.", "depends_on": ["topology_selection"]},
        {"step": 5, "action": "Plan PLECS and bench validation matrix", "reason": "Simulation and EVT evidence must close claims before release.", "depends_on": ["power_stage_design", "control_design", "emi_plan"]},
    ]
    assumption_register = [
        {
            "assumption": row.get("value"),
            "reason": row.get("impact") or _assumption_impact(row.get("label")),
            "risk_if_wrong": "Downstream topology, safety, thermal, EMI, BOM, or release claims may be invalid.",
            "must_confirm": row.get("status") != "user",
        }
        for row in assumptions
        if isinstance(row, dict)
    ]
    quality = _requirement_quality_check(explicit_specs, derived_specs, missing_info)
    return {
        "agent_name": "Power_Electronics_Requirement_Analysis_Agent",
        "task_type": "requirement_analysis",
        "project_summary": summary,
        "application_analysis": application,
        "extracted_specifications": {
            "explicit_specifications": explicit_specs,
            "derived_specifications": derived_specs,
            "qualitative_specifications": qualitative_specs,
        },
        "missing_information": missing_info,
        "preliminary_design_task_decomposition": tasks,
        "feasibility_and_conflict_check": conflicts,
        "refined_design_objectives": objectives,
        "recommended_workflow": workflow,
        "assumption_register": assumption_register,
        "handoff_package": _requirement_handoff_package(summary, specs, derived_specs, missing_info, conflicts),
        "quality_check": quality,
    }


def _requirement_package_human_report(package: Dict[str, Any]) -> str:
    summary = package.get("project_summary") or {}
    quality = package.get("quality_check") or {}
    lines = [
        f"Requirement Analysis: {summary.get('design_object') or 'power electronics design'}",
        f"Application: {summary.get('application') or 'unspecified'}",
        f"Conversion: {summary.get('conversion_type') or 'unspecified'}",
        f"Next-stage readiness: {quality.get('ready_for_next_stage')} - {quality.get('ready_reason')}",
        "",
        "Explicit specifications:",
    ]
    for row in (package.get("extracted_specifications") or {}).get("explicit_specifications", [])[:12]:
        lines.append(f"- {row.get('name')}: {row.get('value')} {row.get('unit') or ''}".rstrip())
    lines.append("")
    lines.append("Derived specifications:")
    for row in (package.get("extracted_specifications") or {}).get("derived_specifications", [])[:8]:
        lines.append(f"- {row.get('name')}: {row.get('value')} {row.get('unit') or ''}; {row.get('formula')}".rstrip())
    lines.append("")
    lines.append("Top missing information:")
    for row in package.get("missing_information", [])[:8]:
        lines.append(f"- [{row.get('priority')}] {row.get('item')}: {row.get('clarification_question')}")
    return "\n".join(lines).strip()


def _build_design_intake_payload_fallback(user_query: str) -> Dict[str, Any]:
    text = str(user_query or "")
    low = text.lower()
    specs: Dict[str, Any] = {}

    conversion_phrase = re.search(
        r"(\d+(?:\.\d+)?)\s*v\s*(?:to|->|到|至)\s*(\d+(?:\.\d+)?)\s*v",
        text,
        flags=re.IGNORECASE,
    )
    if conversion_phrase:
        specs["input_voltage_nominal"] = float(conversion_phrase.group(1))
        specs["output_voltage"] = float(conversion_phrase.group(2))
        specs["input_type"] = "DC"

    vin = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:-|–|—|~|to|至|到)\s*(\d+(?:\.\d+)?)\s*v\s*(?:ac|vac|dc|vdc|交流|直流)?",
        text,
        flags=re.IGNORECASE,
    )
    if vin:
        specs["input_voltage_min"] = float(vin.group(1))
        specs["input_voltage_max"] = float(vin.group(2))
        vin_context = _source_snippet(text, r"(\d+(?:\.\d+)?)\s*(?:-|–|—|~|to|至|到)\s*(\d+(?:\.\d+)?)\s*v\s*(?:ac|vac|dc|vdc|交流|直流)?")
        is_ac_input = bool(re.search(r"\b(vac|ac/dc|offline|mains|交流)\b|离线", low + " " + vin_context.lower(), re.I))
        specs["input_type"] = "AC" if is_ac_input else specs.get("input_type", "DC")
        if specs["input_type"] == "AC":
            specs["line_frequency_hz"] = [47, 63] if re.search(r"47\s*(?:-|–|~|to|至|到)\s*63", text, re.I) else [50, 60]

    out = re.search(
        r"(\d+(?:\.\d+)?)\s*v\s*(?:/|,|，|\s+)\s*(\d+(?:\.\d+)?)\s*a\b",
        text,
        flags=re.IGNORECASE,
    )
    if out:
        specs["output_voltage"] = float(out.group(1))
        specs["output_current"] = float(out.group(2))
        specs["output_power"] = round(specs["output_voltage"] * specs["output_current"], 3)

    if specs.get("output_voltage") is None:
        out_v = re.search(r"(?:output|输出|to|->|到|至)[^0-9]{0,20}(\d+(?:\.\d+)?)\s*v\b", text, flags=re.IGNORECASE)
        if out_v:
            specs["output_voltage"] = float(out_v.group(1))

    power_match = re.search(r"(\d+(?:\.\d+)?)\s*(kw|w)\b", text, flags=re.IGNORECASE)
    if power_match:
        power = float(power_match.group(1)) * (1000.0 if power_match.group(2).lower() == "kw" else 1.0)
        specs["output_power"] = power

    peak_match = re.search(
        r"(?:peak|峰值)[^0-9]{0,30}(\d+(?:\.\d+)?)\s*(kw|w)\b|(\d+(?:\.\d+)?)\s*(kw|w)\b[^.\n,;]{0,24}(?:peak|峰值)",
        text,
        flags=re.IGNORECASE,
    )
    if peak_match:
        peak_value = peak_match.group(1) or peak_match.group(3)
        peak_unit = peak_match.group(2) or peak_match.group(4)
        specs["peak_output_power"] = float(peak_value) * (1000.0 if peak_unit.lower() == "kw" else 1.0)
        duration_match = re.search(r"(?:peak|峰值)[^.\n,;]{0,80}?(\d+(?:\.\d+)?)\s*s\b", text, flags=re.IGNORECASE)
        if not duration_match:
            duration_match = re.search(r"\bfor\s+(\d+(?:\.\d+)?)\s*s\b|持续\s*(\d+(?:\.\d+)?)\s*s", text, flags=re.IGNORECASE)
        if duration_match:
            specs["peak_duration_s"] = float(duration_match.group(1) or duration_match.group(2))

    if specs.get("output_voltage") is not None and specs.get("output_power") is not None and specs.get("output_current") is None:
        try:
            specs["output_current"] = round(float(specs["output_power"]) / float(specs["output_voltage"]), 3)
            specs.setdefault("_derived_fields", []).append("output_current")
        except Exception:
            pass

    ripple = _extract_first_number(r"(?:ripple|纹波)[^0-9]{0,20}(?:<|≤|<=|less than)?\s*(\d+(?:\.\d+)?)\s*m?v", text)
    if ripple is not None:
        specs["max_ripple_mvpp"] = ripple
        specs["max_ripple_voltage"] = ripple / 1000.0

    standby = _extract_first_number(r"(?:standby|no[-\s]?load|待机)[^0-9]{0,25}(?:<|≤|<=)?\s*(\d+(?:\.\d+)?)\s*mw", text)
    if standby is not None:
        specs["standby_power_mw_target"] = standby

    ambient = _extract_first_number(r"(?:ambient|ta|环境温度|环境)[^0-9-]{0,25}(-?\d+(?:\.\d+)?)\s*(?:°\s*)?c\b", text)
    if ambient is not None:
        specs["ambient_c_max"] = ambient
    else:
        ambient_range = re.search(
            r"(-?\d+(?:\.\d+)?)\s*(?:-|–|—|~|to|至|到)\s*(-?\d+(?:\.\d+)?)\s*(?:°\s*)?c\b[^.\n,;]{0,40}(?:ambient|ta|环境)?",
            text,
            flags=re.IGNORECASE,
        )
        ambient_single = re.search(
            r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?c\b[^.\n,;]{0,40}(?:ambient|ta|环境)",
            text,
            flags=re.IGNORECASE,
        )
        if ambient_range:
            specs["ambient_c_min"] = float(ambient_range.group(1))
            specs["ambient_c_max"] = float(ambient_range.group(2))
        elif ambient_single:
            specs["ambient_c_max"] = float(ambient_single.group(1))

    eff_matches = re.findall(r"(?:≥|>=|>|at least|效率|efficiency)[^0-9]{0,30}(\d+(?:\.\d+)?)\s*%", text, flags=re.IGNORECASE)
    if eff_matches:
        values = [float(x) / 100.0 for x in eff_matches]
        specs["efficiency_target"] = max(values)
        if len(values) >= 2:
            specs["efficiency_115vac_full_load"] = values[0]
            specs["efficiency_230vac_full_load"] = values[1]

    if re.search(r"cispr\s*32|en\s*55032|class\s*b|class b|emi", low, re.I):
        specs["emi_target"] = "CISPR 32 Class B pre-compliance" if "class b" in low or "class\u00a0b" in low else "Conducted EMI pre-compliance"
    if "reinforced" in low or "加强绝缘" in text:
        specs["isolation"] = "reinforced isolation assumption"
    elif re.search(r"\b\d+(?:\.\d+)?\s*kvac\b", low):
        specs["isolation"] = re.search(r"\b\d+(?:\.\d+)?\s*kvac\b", low).group(0)
    if "62368" in low:
        specs["safety_standard"] = "IEC/UL 62368-1 direction"

    topology_pref = []
    if "qr" in low or "quasi" in low or "谷底" in text or "准谐振" in text:
        topology_pref.append("QR / valley-switching flyback")
    if "flyback" in low or "反激" in text:
        topology_pref.append("isolated flyback")
    if "active clamp" in low or "acf" in low or "有源钳位" in text:
        topology_pref.append("active clamp flyback candidate")
    if topology_pref:
        specs["topology_preference"] = list(dict.fromkeys(topology_pref))
    sr_requested = bool("synchronous" in low or re.search(r"\bsr\b", low) or "同步整流" in text)
    if sr_requested:
        specs["secondary_rectification"] = "synchronous rectification"
    if "tl431" in low or "opto" in low or "光耦" in text:
        specs["feedback"] = "TL431 + optocoupler"

    vendors = []
    if re.search(r"\bti\b|texas instruments", low, re.I):
        vendors.append("TI")
    if re.search(r"\binfineon\b", low, re.I):
        vendors.append("Infineon")
    if re.search(r"power integrations", low, re.I):
        vendors.append("Power Integrations")
    if re.search(r"\bpi\b", low, re.I) or "innoswitch" in low:
        vendors.append("Power Integrations")
    if vendors:
        specs["vendor_preference"] = list(dict.fromkeys(vendors))

    recognized_specs = [
        {"label": "Input", "value": f"{_fmt_engineering_value(specs.get('input_voltage_min'))}-{_fmt_engineering_value(specs.get('input_voltage_max'))} Vac" if specs.get("input_voltage_min") is not None and specs.get("input_voltage_max") is not None else "Not locked", "status": "critical"},
        {"label": "Output", "value": f"{_fmt_engineering_value(specs.get('output_voltage'), ' V')} / {_fmt_engineering_value(specs.get('output_current'), ' A')}" if specs.get("output_voltage") is not None and specs.get("output_current") is not None else "Not locked", "status": "critical"},
        {"label": "Power", "value": _fmt_engineering_value(specs.get("output_power"), " W"), "status": "critical"},
        {"label": "Efficiency", "value": _efficiency_requirement_label(specs) if specs.get("efficiency_target") is not None else "Default needed", "status": "soft"},
        {"label": "Ripple", "value": _fmt_engineering_value(specs.get("max_ripple_mvpp"), " mVp-p"), "status": "soft"},
        {"label": "EMI", "value": specs.get("emi_target") or "Default needed", "status": "risk"},
        {"label": "Isolation", "value": specs.get("isolation") or "Default needed", "status": "critical"},
        {
            "label": "Ambient",
            "value": (
                f"{_fmt_engineering_value(specs.get('ambient_c_min'))}-{_fmt_engineering_value(specs.get('ambient_c_max'))} C"
                if specs.get("ambient_c_min") is not None and specs.get("ambient_c_max") is not None
                else _fmt_engineering_value(specs.get("ambient_c_max"), " C")
            ),
            "status": "soft",
        },
    ]

    power_label = _design_power_label(specs)
    feedback_label = specs.get("feedback") or "TL431/opto feedback"
    rectifier_label = "synchronous rectification" if sr_requested else "Schottky or synchronous rectification to be compared"
    primary_topology = "QR flyback" if any("QR" in str(item) for item in specs.get("topology_preference", [])) else "isolated flyback"

    assumptions = [
        {"label": "Safety direction", "value": specs.get("safety_standard") or "IEC/UL 62368-1 design direction", "status": "default", "impact": "Controls creepage, clearance, transformer insulation, Y-cap choice, and release checklist."},
        {"label": "EMI target", "value": specs.get("emi_target") or "CISPR 32 Class B pre-compliance", "status": "default" if not specs.get("emi_target") else "user", "impact": "Creates conducted EMI filter, LISN pre-scan, and layout constraints."},
        {"label": "Environment", "value": f"Open-frame PCB, Ta max {_fmt_engineering_value(specs.get('ambient_c_max') or 50, ' C')}", "status": "default" if not specs.get("ambient_c_max") else "user", "impact": "Drives thermal derating and transformer/core size margin."},
        {"label": "Topology path", "value": f"{primary_topology} + {feedback_label} + {rectifier_label}", "status": "default", "impact": "Keeps the first revision manufacturable while leaving rectifier and active-clamp choices as explicit trade-offs."},
        {"label": "Release policy", "value": "Concept artifacts only until specs, magnetics, loop, simulation, layout, and test gates are closed", "status": "policy", "impact": "Prevents a final schematic from being released from an assumption-only state."},
    ]

    missing_inputs = []
    missing_rules = [
        ("safety_standard", "Exact safety standard/market, insulation class, pollution degree, altitude, and required hipot level."),
        ("emi_target", "EMI target and margin policy, including conducted/radiated pre-compliance scope."),
        ("standby_power_mw_target", "No-load/standby power target and whether a burst-mode audible-noise constraint exists."),
        ("mechanical_limit", "PCB area, height limit, enclosure/open-frame condition, airflow, and connector constraints."),
        ("cost_target", "BOM cost target, volume, approved vendors, and supply-chain restrictions."),
        ("hold_up_time_ms", "Hold-up time and brown-in/brown-out behavior."),
    ]
    for key, label in missing_rules:
        if specs.get(key) is None:
            missing_inputs.append(label)

    topology_candidates = [
        {
            "label": "A. QR / valley-switching flyback + opto CV",
            "fit": f"Recommended first path for a {power_label}, cost-sensitive, universal AC design.",
            "tradeoff": "Good efficiency/EMI balance; rectifier choice, magnetics, loop, clamp, thermal, and EMI still need validation.",
            "status": "recommended",
        },
        {
            "label": "B. Fixed-frequency flyback + opto CV",
            "fit": "Backup path when controller/model availability or cost dominates.",
            "tradeoff": "Simpler to reason about, but switching loss, EMI, and light-load efficiency can be harder.",
            "status": "backup",
        },
        {
            "label": "C. Active clamp flyback + SR",
            "fit": "Efficiency and power-density upgrade path.",
            "tradeoff": "Higher BOM, control complexity, and debug risk; not ideal for first demo/review flow.",
            "status": "defer",
        },
    ]

    decisions = [
        {
            "decision": "Do not generate a release schematic at intake.",
            "reason": "Offline flyback release quality depends on locked specs, transformer leakage/insulation, VDS/SR stress, loop stability, EMI, thermal, and manufacturability evidence.",
            "status": "policy_hold",
        },
        {
            "decision": "Use QR flyback as the primary candidate path.",
            "reason": f"It matches the {power_label} cost/reliability target better than active clamp for a first manufacturable revision.",
            "status": "proposed",
        },
        {
            "decision": "Keep fixed-frequency flyback and active clamp as explicit backup paths.",
            "reason": "The design must be reversible if efficiency, EMI, controller availability, or loop evidence fails later.",
            "status": "proposed",
        },
    ]

    secondary_stress_label = "SR timing/spikes" if sr_requested else "secondary-rectifier stress/spikes"
    risks = [
        {"severity": "P0", "item": "Safety/EMI assumptions are not enough for release.", "status": "assumed"},
        {"severity": "P0", "item": "Transformer leakage, insulation system, and thermal rise are physical assumptions until vendor data or EVT samples exist.", "status": "unverified"},
        {"severity": "P1", "item": f"Primary VDS, {secondary_stress_label}, RCD loss, and conducted EMI are coupled and must be swept before schematic freeze.", "status": "calculation_required"},
        {"severity": "P1", "item": "TL431/opto loop stability must include CTR aging/corners before feedback network can be marked final.", "status": "simulation_required"},
    ]

    artifacts = [
        {"label": "Spec Lock table", "status": "ready"},
        {"label": "Assumption cards", "status": "ready"},
        {"label": "Topology trade-off", "status": "ready"},
        {"label": "Controller decision record", "status": "pending"},
        {"label": "Power-stage calculation notebook", "status": "pending"},
        {"label": "Magnetics winding package", "status": "pending"},
        {"label": "Loop/EMI/thermal evidence", "status": "pending"},
        {"label": "Release package", "status": "blocked"},
    ]
    requirement_analysis = _build_requirement_analysis_cards(
        text,
        specs,
        assumptions,
        missing_inputs,
        topology_candidates,
    )
    requirement_package = _build_requirement_analysis_package(text, specs, assumptions, missing_inputs)
    human_requirement_report = _requirement_package_human_report(requirement_package)

    return {
        "mode": "requirements_intake",
        "status": "waiting_spec_lock",
        "source_text": text,
        "specs": specs,
        "recognized_specs": recognized_specs,
        "assumptions": assumptions,
        "missing_inputs": missing_inputs,
        "topology_candidates": topology_candidates,
        "decisions": decisions,
        "risks": risks,
        "artifacts": artifacts,
        "requirement_analysis": requirement_analysis,
        "requirement_package": requirement_package,
        "human_readable_requirement_report": human_requirement_report,
        "machine_readable_requirement_package": requirement_package,
        "application_analysis": requirement_package.get("application_analysis", {}),
        "extracted_specifications": requirement_package.get("extracted_specifications", {}),
        "missing_information": requirement_package.get("missing_information", []),
        "preliminary_design_task_decomposition": requirement_package.get("preliminary_design_task_decomposition", []),
        "feasibility_and_conflict_check": requirement_package.get("feasibility_and_conflict_check", []),
        "refined_design_objectives": requirement_package.get("refined_design_objectives", {}),
        "recommended_workflow": requirement_package.get("recommended_workflow", []),
        "assumption_register": requirement_package.get("assumption_register", []),
        "handoff_package": requirement_package.get("handoff_package", {}),
        "quality_check": requirement_package.get("quality_check", {}),
        **requirement_analysis,
        "next_actions": [
            "Adopt default assumptions and continue to topology/controller selection.",
            "Edit assumptions manually before the design graph runs.",
            "Stop this session if this is only a Q&A review.",
        ],
    }


_REQUIREMENT_AGENT: Optional[RequirementAnalysisAgent] = None
_TOPOLOGY_SERVICE: Optional[TopologyKnowledgeService] = None
_PLECS_REGISTRY: Optional[PlecsModelRegistry] = None
_REQUIREMENTS_GATE_SERVICE = StudioRequirementsGateService()


def _get_topology_service() -> TopologyKnowledgeService:
    global _TOPOLOGY_SERVICE
    if _TOPOLOGY_SERVICE is None:
        _TOPOLOGY_SERVICE = TopologyKnowledgeService(repo_root=BASE_DIR)
    return _TOPOLOGY_SERVICE


def _get_plecs_registry() -> PlecsModelRegistry:
    global _PLECS_REGISTRY
    if _PLECS_REGISTRY is None:
        _PLECS_REGISTRY = PlecsModelRegistry(_get_topology_service(), repo_root=BASE_DIR)
    return _PLECS_REGISTRY


def _get_requirement_agent() -> RequirementAnalysisAgent:
    global _REQUIREMENT_AGENT
    if _REQUIREMENT_AGENT is None:
        _REQUIREMENT_AGENT = RequirementAnalysisAgent(_get_topology_service(), _get_plecs_registry())
    return _REQUIREMENT_AGENT


def _agent_derived_value(result: Any, name: str) -> Any:
    for row in ((result.extracted_specifications or {}).get("derived_specifications") or []):
        if isinstance(row, dict) and row.get("name") == name:
            return row.get("value")
    return None


def _specs_from_requirement_result(result: Any) -> Dict[str, Any]:
    normalized = dict(getattr(result, "normalized_specs", {}) or {})
    specs: Dict[str, Any] = {}
    for key in [
        "input_voltage_min",
        "input_voltage_max",
        "input_voltage_nominal",
        "input_type",
        "line_frequency_hz",
        "output_voltage",
        "output_current",
        "peak_output_power",
        "peak_duration_s",
        "ambient_c_min",
        "ambient_c_max",
        "cooling_condition",
        "application_hint",
        "protection_requirement",
    ]:
        if normalized.get(key) is not None:
            specs[key] = normalized[key]
    if normalized.get("rated_output_power") is not None:
        specs["output_power"] = normalized["rated_output_power"]
    if specs.get("output_current") is None:
        derived_iout = _agent_derived_value(result, "rated_output_current")
        if derived_iout is not None:
            specs["output_current"] = derived_iout
            specs.setdefault("_derived_fields", []).append("output_current")
    if specs.get("output_power") is None and specs.get("output_voltage") is not None and specs.get("output_current") is not None:
        try:
            specs["output_power"] = round(float(specs["output_voltage"]) * float(specs["output_current"]), 3)
            specs.setdefault("_derived_fields", []).append("output_power")
        except Exception:
            pass
    if normalized.get("output_ripple_mvpp") is not None:
        specs["max_ripple_mvpp"] = normalized["output_ripple_mvpp"]
        specs["max_ripple_voltage"] = normalized.get("output_ripple_vpp")
    if normalized.get("efficiency_target") is not None:
        specs["efficiency_target"] = normalized["efficiency_target"]
    if normalized.get("emi_emc_standard"):
        specs["emi_target"] = normalized["emi_emc_standard"]
    if normalized.get("isolation_requirement"):
        specs["isolation"] = normalized["isolation_requirement"]
    if normalized.get("feedback_preference"):
        specs["feedback"] = normalized["feedback_preference"]
    if normalized.get("topology_preference"):
        value = normalized["topology_preference"]
        specs["topology_preference"] = value if isinstance(value, list) else [value]
    return specs


def _requirement_value_label(value: Any, unit: str = "") -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, list):
        return "-".join(_requirement_value_label(item) for item in value if item is not None) + (f" {unit}" if unit else "")
    if value is None or value == "":
        return "Not specified"
    return f"{value}{(' ' + unit) if unit else ''}"


def _agent_recognized_specs(specs: Dict[str, Any]) -> list[Dict[str, Any]]:
    input_type = str(specs.get("input_type") or "DC").upper()
    input_unit = "Vac" if input_type == "AC" else "Vdc"
    input_value = (
        f"{float(specs['input_voltage_min']):g}-{float(specs['input_voltage_max']):g} {input_unit}"
        if specs.get("input_voltage_min") is not None and specs.get("input_voltage_max") is not None
        else (_requirement_value_label(specs.get("input_voltage_nominal"), input_unit) if specs.get("input_voltage_nominal") is not None else "Not locked")
    )
    output_value = (
        f"{float(specs['output_voltage']):g} V / {float(specs['output_current']):g} A"
        if specs.get("output_voltage") is not None and specs.get("output_current") is not None
        else (_requirement_value_label(specs.get("output_voltage"), "V") if specs.get("output_voltage") is not None else "Not locked")
    )
    ambient_value = (
        f"{float(specs['ambient_c_min']):g}-{float(specs['ambient_c_max']):g} C"
        if specs.get("ambient_c_min") is not None and specs.get("ambient_c_max") is not None
        else _requirement_value_label(specs.get("ambient_c_max"), "C")
    )
    return [
        {"label": "Input", "value": input_value, "status": "critical" if input_value != "Not locked" else "blocking"},
        {"label": "Output", "value": output_value, "status": "critical" if output_value != "Not locked" else "blocking"},
        {"label": "Power", "value": _requirement_value_label(specs.get("output_power"), "W"), "status": "critical" if specs.get("output_power") is not None else "blocking"},
        {"label": "Efficiency", "value": _efficiency_requirement_label(specs), "status": "soft"},
        {"label": "Ripple", "value": _requirement_value_label(specs.get("max_ripple_mvpp"), "mVp-p"), "status": "soft" if specs.get("max_ripple_mvpp") is not None else "missing"},
        {"label": "EMI", "value": specs.get("emi_target") or "Not specified", "status": "risk" if specs.get("emi_target") else "missing"},
        {"label": "Isolation", "value": specs.get("isolation") or "Not specified", "status": "critical" if specs.get("isolation") else "blocking"},
        {"label": "Ambient", "value": ambient_value, "status": "soft" if specs.get("ambient_c_max") is not None else "missing"},
    ]


def _agent_assumption_cards(specs: Dict[str, Any], result: Any) -> list[Dict[str, Any]]:
    topology_pref = ", ".join(str(x) for x in specs.get("topology_preference") or []) or "No final topology selected; downstream agent compares candidates."
    return [
        {
            "label": "Safety direction",
            "value": specs.get("isolation") or "Isolation not confirmed; architecture-critical missing item.",
            "status": "user" if specs.get("isolation") else "blocking",
            "impact": "Controls topology family, transformer/inductor choice, creepage/clearance, safety evidence, and final topology readiness.",
        },
        {
            "label": "EMI target",
            "value": specs.get("emi_target") or "No EMI/EMC standard fabricated; keep as a clarification item.",
            "status": "user" if specs.get("emi_target") else "missing",
            "impact": "Affects filter topology, layout constraints, pre-scan plan, and verification limits.",
        },
        {
            "label": "Environment",
            "value": f"{_requirement_value_label(specs.get('ambient_c_max'), 'C')} ambient; cooling: {specs.get('cooling_condition') or 'not specified'}.",
            "status": "user" if specs.get("ambient_c_max") is not None or specs.get("cooling_condition") else "missing",
            "impact": "Drives loss budget, derating, thermal path, and mechanical constraints.",
        },
        {
            "label": "Topology path",
            "value": f"Preference only: {topology_pref}",
            "status": "preference" if specs.get("topology_preference") else "candidate",
            "impact": "The requirement agent prepares candidates only; final topology is chosen by the next gate.",
        },
        {
            "label": "Release policy",
            "value": "No final topology, schematic, BOM, PCB, or manufacturing package from the requirement node.",
            "status": "policy",
            "impact": "Prevents assumption-only output from being treated as release evidence.",
        },
    ]


def _agent_topology_rows(result: Any) -> list[Dict[str, Any]]:
    handoff = result.handoff_package or {}
    candidates = ((handoff.get("topology_selection_agent") or {}).get("candidate_seed_topologies") or [])
    rows: list[Dict[str, Any]] = []
    for row in candidates[:5]:
        name = row.get("name") or "Candidate topology"
        plecs_status = row.get("plecs_model_status") or "unknown"
        rows.append({
            "label": name,
            "fit": "Seed candidate from requirement-level knowledge base.",
            "tradeoff": f"PLECS model status: {plecs_status}. Final topology is not selected in this node.",
            "status": "candidate",
        })
    if not rows:
        rows.append({
            "label": "Topology comparison pending",
            "fit": "No final topology is selected by the Requirement Analysis Agent.",
            "tradeoff": "Resolve architecture-critical inputs and run the topology selection agent next.",
            "status": "pending",
        })
    return rows


def _agent_risk_rows(result: Any) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for item in result.missing_information[:4]:
        priority = str(item.get("priority") or "medium")
        severity = "P0" if priority == "blocking" else ("P1" if priority == "high" else "P2")
        rows.append({
            "severity": severity,
            "item": f"Missing {str(item.get('item') or '').replace('_', ' ')}",
            "status": priority,
        })
    for issue in result.feasibility_and_conflict_check[:4]:
        risk = str(issue.get("risk_level") or "medium")
        rows.append({
            "severity": "P1" if risk == "high" else "P2",
            "item": issue.get("issue") or "Requirement risk",
            "status": risk,
        })
    return rows[:6]


def _agent_stage_rows(result: Any, specs: Dict[str, Any]) -> Dict[str, list[Dict[str, Any]]]:
    extracted = result.extracted_specifications or {}
    frontend = result.frontend_response or {}
    explicit = extracted.get("explicit_specifications") or []
    derived = extracted.get("derived_specifications") or []
    qualitative = extracted.get("qualitative_specifications") or []
    quality = result.quality_check or {}
    return {
        "stage0_input_normalization": [
            {"label": "Design object", "value": result.project_summary.get("design_object"), "status": "explicit"},
            {"label": "Application", "value": result.project_summary.get("application"), "status": "context"},
            {"label": "Conversion type", "value": result.project_summary.get("conversion_type"), "status": "context"},
            {"label": "Readiness", "value": result.readiness_status, "status": result.readiness_status},
        ],
        "stage1_application_scenario_analysis": [
            {"label": "Use case", "value": result.application_analysis.get("use_case"), "status": "context"},
            {"label": "Operating environment", "value": "; ".join(result.application_analysis.get("operating_environment") or []), "status": "context"},
            {"label": "Engineering implications", "value": "; ".join(result.application_analysis.get("application_driven_implications") or []), "status": "risk"},
            {"label": "Design priorities", "value": "; ".join(result.application_analysis.get("design_priorities") or []), "status": "priority"},
        ],
        "stage2_specification_extraction": [
            *[
                {"label": str(row.get("name") or "").replace("_", " "), "value": f"{row.get('value')} {row.get('unit') or ''}".strip(), "status": "explicit"}
                for row in explicit
            ],
            *[
                {"label": str(row.get("name") or "").replace("_", " "), "value": f"{row.get('value')} {row.get('unit') or ''}\nFormula: {row.get('formula')}".strip(), "status": "derived"}
                for row in derived
            ],
            *[
                {"label": str(row.get("name") or "").replace("_", " "), "value": row.get("description") or "", "status": row.get("status") or "qualitative"}
                for row in qualitative
            ],
        ],
        "stage3_missing_information": [
            {"label": str(row.get("item") or "").replace("_", " "), "value": f"{row.get('why_it_matters')}\nQuestion: {row.get('clarification_question')}", "status": row.get("priority") or "medium"}
            for row in result.missing_information
        ],
        "stage4_preliminary_task_decomposition": [
            {"label": row.get("task_name"), "value": f"{row.get('purpose')}\nInputs: {', '.join(row.get('required_inputs') or [])}\nOutputs: {', '.join(row.get('expected_outputs') or [])}", "status": "task"}
            for row in result.preliminary_design_task_decomposition
        ],
        "stage5_feasibility_conflict_check": [
            {"label": row.get("issue"), "value": f"{row.get('explanation')}\nBasis: {row.get('quantitative_basis')}\nAction: {row.get('recommended_action')}", "status": row.get("risk_level") or "medium"}
            for row in result.feasibility_and_conflict_check
        ],
        "stage6_refined_design_objectives": [
            {"label": "Hard constraints", "value": "; ".join(result.refined_design_objectives.get("hard_constraints") or []), "status": "critical"},
            {"label": "High-priority objectives", "value": "; ".join(result.refined_design_objectives.get("high_priority_objectives") or []), "status": "high"},
            {"label": "Medium-priority objectives", "value": "; ".join(result.refined_design_objectives.get("medium_priority_objectives") or []), "status": "medium"},
            {"label": "Pending requirements", "value": "; ".join(result.refined_design_objectives.get("pending_requirements") or []), "status": "open"},
        ],
        "stage7_quality_gate": [
            {"label": "No unsupported numbers", "value": str(bool(quality.get("no_unsupported_numbers"))), "status": "ready" if quality.get("no_unsupported_numbers") else "review"},
            {"label": "Explicit specs have source", "value": str(bool(quality.get("explicit_specs_have_source"))), "status": "ready" if quality.get("explicit_specs_have_source") else "review"},
            {"label": "Derived values have formula", "value": str(bool(quality.get("derived_values_have_formula"))), "status": "ready" if quality.get("derived_values_have_formula") else "review"},
            {"label": "Critical missing info checked", "value": str(bool(quality.get("critical_missing_info_checked"))), "status": "ready" if quality.get("critical_missing_info_checked") else "review"},
            {"label": "No final topology fabricated", "value": str(bool(quality.get("no_final_topology_fabricated"))), "status": "ready" if quality.get("no_final_topology_fabricated") else "review"},
            {"label": "Readiness", "value": result.readiness_status, "status": result.readiness_status},
        ],
        "machine_readable_package": [
            {"label": "JSON package", "value": json.dumps(result.to_api_dict(), ensure_ascii=False, indent=2)[:12000], "status": "json"}
        ],
        "a1_application_scenario": frontend.get("summary_cards") or [],
        "a2_specifications_and_missing": [],
        "a3_preliminary_design_tasks": [],
        "a4_feasibility_conflicts": [],
        "a5_refined_objectives_workflow": [],
    }


def _build_design_intake_payload(user_query: str) -> Dict[str, Any]:
    """Build the Requirement Gate payload using the v1 Requirement Analysis Agent.

    This definition intentionally supersedes the rule-only builder above
    while preserving the UI/session shape consumed by the existing cockpit.
    """
    result = _get_requirement_agent().analyze(str(user_query or ""))
    specs = _specs_from_requirement_result(result)
    recognized_specs = _agent_recognized_specs(specs)
    assumptions = _agent_assumption_cards(specs, result)
    missing_info = result.missing_information or []
    missing_inputs = [
        f"{row.get('item')}: {row.get('why_it_matters')} Question: {row.get('clarification_question')}"
        for row in missing_info
        if isinstance(row, dict)
    ]
    topology_rows = _agent_topology_rows(result)
    stage_rows = _agent_stage_rows(result, specs)
    requirement_package = result.to_api_dict()
    quality = dict(requirement_package.get("quality_check") or {})
    quality.update({
        "all_explicit_specs_have_source": quality.get("explicit_specs_have_source", False),
        "derived_values_show_formula": quality.get("derived_values_have_formula", False),
        "unsupported_numbers_avoided": quality.get("no_unsupported_numbers", False),
        "critical_missing_information_identified": quality.get("critical_missing_info_checked", False),
        "ready_for_next_stage": result.readiness_status,
        "ready_reason": "Can run preliminary topology comparison with documented assumptions." if result.readiness_status == "partial" else ("Enough requirement data for next stage." if result.readiness_status == "ready" else "Architecture-critical inputs block the next stage."),
    })
    requirement_package["quality_check"] = quality
    artifacts = [
        {"label": "Requirement analysis report", "status": "ready", "note": "Human-readable report generated."},
        {"label": "Machine-readable JSON handoff", "status": "ready", "note": "Downstream handoff package generated."},
        {"label": "Clarification queue", "status": "ready", "note": "Missing information prioritized."},
        {"label": "Topology seed database", "status": "ready", "note": "Candidate knowledge loaded offline."},
        {"label": "PLECS model registry", "status": "ready", "note": "Model coverage reported; missing models are not faked."},
        {"label": "Final topology", "status": "blocked", "note": "Not selected by this node."},
        {"label": "Schematic/BOM/PCB", "status": "blocked", "note": "Blocked until downstream gates close."},
    ]
    return {
        "mode": "requirements_intake",
        "status": "waiting_spec_lock",
        "source_text": str(user_query or ""),
        "specs": specs,
        "recognized_specs": recognized_specs,
        "assumptions": assumptions,
        "missing_inputs": missing_inputs,
        "topology_candidates": topology_rows,
        "decisions": [
            {"decision": "Do not select final topology in the Requirement Analysis Agent.", "reason": "Topology depends on isolation, EMI, thermal, cost, protection, and verification constraints.", "status": "policy_hold"},
            {"decision": "Prepare topology candidates and downstream handoff only.", "reason": "The next agent owns topology comparison and decision records.", "status": "proposed"},
            {"decision": "Use PLECS registry for honest model coverage.", "reason": "Only existing local models are marked available; planned/missing models remain explicit.", "status": "traceable"},
        ],
        "risks": _agent_risk_rows(result),
        "artifacts": artifacts,
        "requirement_analysis": stage_rows,
        "requirement_package": requirement_package,
        "human_readable_requirement_report": result.human_readable_report,
        "machine_readable_requirement_package": requirement_package,
        "application_analysis": result.application_analysis,
        "extracted_specifications": result.extracted_specifications,
        "missing_information": result.missing_information,
        "clarification_questions": result.clarification_questions,
        "preliminary_design_task_decomposition": result.preliminary_design_task_decomposition,
        "feasibility_and_conflict_check": result.feasibility_and_conflict_check,
        "refined_design_objectives": result.refined_design_objectives,
        "recommended_workflow": result.recommended_workflow,
        "assumption_register": result.assumption_register,
        "handoff_package": result.handoff_package,
        "readiness_status": result.readiness_status,
        "quality_check": quality,
        "frontend_response": result.frontend_response,
        "source_provenance": result.source_provenance,
        **stage_rows,
        "next_actions": [
            "Review missing information and assumptions.",
            "Run topology selection after accepting documented assumptions.",
            "Do not generate final schematic/BOM until downstream gates close.",
        ],
    }


def _requirements_decision_options(specs: Optional[Dict[str, Any]] = None) -> list[Dict[str, str]]:
    return [
        {
            "option": "Adopt defaults",
            "command": "ADOPT_DEFAULT_ASSUMPTIONS",
            "intent": "Use the simple first-pass defaults and move to topology comparison.",
            "risk": "Release remains blocked until later evidence gates close.",
        },
        {
            "option": "Edit release constraints",
            "command": "MANUAL_ADJUSTMENTS",
            "intent": "Use only if safety, standby, mechanical, cost, vendor, or hold-up limits are already known.",
            "risk": "Unknown items can stay TBD for topology comparison.",
        },
        {
            "option": "Compare topologies",
            "command": "GENERATE_TOPOLOGY_CANDIDATES",
            "intent": "Start QR/fixed-frequency/active-clamp trade-off from the locked envelope.",
            "risk": "Still conceptual until controller, magnetics, stress, loop, EMI, thermal, and layout evidence follow.",
        },
        {
            "option": "Stop",
            "command": "STOP_SESSION",
            "intent": "Pause without changing the design state.",
            "risk": "No new evidence is generated.",
        },
    ]


def _requirements_checkpoint_details(intake: Dict[str, Any]) -> Dict[str, Any]:
    missing = intake.get("missing_inputs") if isinstance(intake.get("missing_inputs"), list) else []
    spec_rows = intake.get("recognized_specs") if isinstance(intake.get("recognized_specs"), list) else []
    assumptions = intake.get("assumptions") if isinstance(intake.get("assumptions"), list) else []
    spec_dict = intake.get("specs") if isinstance(intake.get("specs"), dict) else {}
    locked_phrase = _locked_spec_phrase(spec_dict)
    topology_assumption = next((row for row in assumptions if isinstance(row, dict) and str(row.get("label") or "").lower() == "topology path"), {})
    requirement_analysis = intake.get("requirement_analysis") if isinstance(intake.get("requirement_analysis"), dict) else {}
    requirement_package = intake.get("requirement_package") if isinstance(intake.get("requirement_package"), dict) else {}
    package_specs = requirement_package.get("extracted_specifications") if isinstance(requirement_package.get("extracted_specifications"), dict) else {}
    package_quality = requirement_package.get("quality_check") if isinstance(requirement_package.get("quality_check"), dict) else {}
    package_missing = requirement_package.get("missing_information") if isinstance(requirement_package.get("missing_information"), list) else []
    package_tasks = requirement_package.get("preliminary_design_task_decomposition") if isinstance(requirement_package.get("preliminary_design_task_decomposition"), list) else []
    package_conflicts = requirement_package.get("feasibility_and_conflict_check") if isinstance(requirement_package.get("feasibility_and_conflict_check"), list) else []
    package_objectives = requirement_package.get("refined_design_objectives") if isinstance(requirement_package.get("refined_design_objectives"), dict) else {}
    package_workflow = requirement_package.get("recommended_workflow") if isinstance(requirement_package.get("recommended_workflow"), list) else []
    package_summary = requirement_package.get("project_summary") if isinstance(requirement_package.get("project_summary"), dict) else {}
    package_application = requirement_package.get("application_analysis") if isinstance(requirement_package.get("application_analysis"), dict) else {}
    a1_items = requirement_analysis.get("a1_application_scenario") or intake.get("a1_application_scenario") or []
    a2_items = requirement_analysis.get("a2_specifications_and_missing") or intake.get("a2_specifications_and_missing") or []
    a3_items = requirement_analysis.get("a3_preliminary_design_tasks") or intake.get("a3_preliminary_design_tasks") or []
    a4_items = requirement_analysis.get("a4_feasibility_conflicts") or intake.get("a4_feasibility_conflicts") or []
    a5_items = requirement_analysis.get("a5_refined_objectives_workflow") or intake.get("a5_refined_objectives_workflow") or []
    locked_spec_items = [
        row for row in a2_items
        if isinstance(row, dict)
        and str(row.get("status") or "").lower() not in {"open", "question"}
        and not str(row.get("label") or "").startswith(("P0", "P1", "Clarification question"))
    ]
    detailed_open_question_items = [
        {
            "label": _missing_input_label(item, index),
            "value": f"Missing: {item}\nQuestion: {_requirement_question_for_missing(item)}",
            "status": "open" if index else "critical",
        }
        for index, item in enumerate(missing[:8])
    ]
    package_open_question_items = [
        {
            "label": str(row.get("item") or f"Missing input {index + 1}").replace("_", " "),
            "value": f"{row.get('why_it_matters')}\nQuestion: {row.get('clarification_question')}",
            "status": str(row.get("priority") or "medium"),
        }
        for index, row in enumerate(package_missing[:8])
        if isinstance(row, dict)
    ]
    if package_open_question_items:
        detailed_open_question_items = package_open_question_items
    open_question_items = package_open_question_items[:5] if package_open_question_items else _requirements_release_input_summary(missing)
    explicit_stage_rows = [
        {
            "label": str(row.get("name") or "explicit spec").replace("_", " "),
            "value": f"{row.get('value')} {row.get('unit') or ''}".strip(),
            "status": "explicit",
        }
        for row in (package_specs.get("explicit_specifications") or [])[:10]
        if isinstance(row, dict)
    ]
    derived_stage_rows = [
        {
            "label": str(row.get("name") or "derived spec").replace("_", " "),
            "value": f"{row.get('value')} {row.get('unit') or ''}".strip() + (f"\nFormula: {row.get('formula')}" if row.get("formula") else ""),
            "status": "derived",
        }
        for row in (package_specs.get("derived_specifications") or [])[:8]
        if isinstance(row, dict)
    ]
    qualitative_stage_rows = [
        {
            "label": str(row.get("name") or "qualitative spec").replace("_", " "),
            "value": row.get("description") or "",
            "status": row.get("status") or "qualitative",
        }
        for row in (package_specs.get("qualitative_specifications") or [])[:6]
        if isinstance(row, dict)
    ]
    stage0_rows = [
        {"label": "Design object", "value": package_summary.get("design_object") or "power electronics design", "status": "explicit"},
        {"label": "Application", "value": package_summary.get("application") or "unspecified", "status": "context"},
        {"label": "Conversion type", "value": package_summary.get("conversion_type") or "unspecified", "status": "context"},
        {"label": "Readiness", "value": f"{package_quality.get('ready_for_next_stage') or 'partial'}: {package_quality.get('ready_reason') or 'Requirement gate requires review.'}", "status": package_quality.get("ready_for_next_stage") or "partial"},
    ]
    stage1_rows = [
        {"label": "Use case", "value": package_application.get("use_case") or "application not fully specified", "status": "context"},
        {"label": "Operating environment", "value": "; ".join(package_application.get("operating_environment") or []), "status": "context"},
        {"label": "Engineering implications", "value": "; ".join(package_application.get("application_driven_implications") or []), "status": "risk"},
        {"label": "Design priorities", "value": "; ".join(package_application.get("design_priorities") or []), "status": "priority"},
    ]
    stage2_rows = (explicit_stage_rows + derived_stage_rows + qualitative_stage_rows) or locked_spec_items
    stage4_rows = [
        {
            "label": row.get("task_name") or "Task",
            "value": f"{row.get('purpose')}\nInputs: {', '.join(row.get('required_inputs') or [])}\nOutputs: {', '.join(row.get('expected_outputs') or [])}",
            "status": "task",
        }
        for row in package_tasks[:9]
        if isinstance(row, dict)
    ]
    stage5_rows = [
        {
            "label": row.get("issue") or "Requirement risk",
            "value": f"{row.get('explanation')}\nBasis: {row.get('quantitative_basis')}\nAction: {row.get('recommended_action')}",
            "status": row.get("risk_level") or "medium",
        }
        for row in package_conflicts[:8]
        if isinstance(row, dict)
    ]
    stage6_rows = [
        {"label": "Hard constraints", "value": "; ".join(package_objectives.get("hard_constraints") or []), "status": "critical"},
        {"label": "High-priority objectives", "value": "; ".join(package_objectives.get("high_priority_objectives") or []), "status": "high"},
        {"label": "Medium-priority objectives", "value": "; ".join(package_objectives.get("medium_priority_objectives") or []), "status": "medium"},
        {"label": "Pending requirements", "value": "; ".join(package_objectives.get("pending_requirements") or []), "status": "open"},
    ]
    stage7_rows = [
        {"label": "Source grounding", "value": "Every explicit spec has source text." if package_quality.get("all_explicit_specs_have_source") else "Some explicit specs need source review.", "status": "ready" if package_quality.get("all_explicit_specs_have_source") else "review"},
        {"label": "Derived formulas", "value": "Every derived value includes formula." if package_quality.get("derived_values_show_formula") else "Some derived values are missing formulas.", "status": "ready" if package_quality.get("derived_values_show_formula") else "review"},
        {"label": "Assumption control", "value": "Unsupported numbers are avoided; assumptions stay assumptions.", "status": "ready" if package_quality.get("unsupported_numbers_avoided") else "review"},
        {"label": "Downstream readiness", "value": package_quality.get("ready_reason") or "Partial readiness until assumptions are accepted.", "status": package_quality.get("ready_for_next_stage") or "partial"},
    ]
    workflow_rows = [
        {
            "label": f"Step {row.get('step')}",
            "value": f"{row.get('action')} Reason: {row.get('reason')}",
            "status": "next",
        }
        for row in package_workflow[:6]
        if isinstance(row, dict)
    ]
    gate_verdict = (
        f"{locked_phrase} is enough to compare topology/controller paths. "
        "It is not enough to release a schematic, BOM, transformer drawing, or manufacturing package."
    )
    design_envelope_summary = [
        {
            "label": "Application",
            "value": f"{package_summary.get('design_object') or 'power electronics design'}, {locked_phrase}.",
            "status": "locked",
        },
    ] + locked_spec_items[:10]
    default_assumption_labels = {"Safety direction", "EMI target", "Environment", "Topology path", "Release policy"}
    assumption_cards_review = [
        {
            "label": row.get("label"),
            "value": _requirements_assumption_summary(row),
            "status": row.get("status"),
        }
        for row in assumptions
        if isinstance(row, dict) and row.get("label") in default_assumption_labels
    ][:5]
    engineering_impact_review = [
        {"label": row.get("label"), "value": row.get("value"), "status": row.get("status")}
        for row in a4_items
        if isinstance(row, dict) and row.get("label")
    ][:6]
    next_decision_review = [
        {"label": "Recommended", "value": "Adopt defaults and move to topology comparison.", "status": "next"},
        {"label": "Answer now only if", "value": "You already know exact safety market, board limits, standby/hold-up, cost, or vendor constraints.", "status": "option"},
        {"label": "Still blocked", "value": "No final schematic, BOM, PCB, transformer release, or manufacturing package from this node.", "status": "blocked"},
    ]
    return {
        "a1_application_scenario": a1_items,
        "a2_specifications_and_missing": a2_items,
        "a3_preliminary_design_tasks": a3_items,
        "a4_feasibility_conflicts": a4_items,
        "a5_refined_objectives_workflow": a5_items,
        "requirement_package": requirement_package,
        "stage0_input_normalization": stage0_rows,
        "stage1_application_scenario_analysis": stage1_rows,
        "stage2_specification_extraction": stage2_rows,
        "stage3_missing_information": open_question_items,
        "stage4_preliminary_task_decomposition": stage4_rows,
        "stage5_feasibility_conflict_check": stage5_rows,
        "stage6_refined_design_objectives": stage6_rows,
        "stage7_quality_gate": stage7_rows,
        "machine_readable_package": [
            {"label": "JSON package", "value": json.dumps(requirement_package, ensure_ascii=False, indent=2)[:12000], "status": "json"}
        ] if requirement_package else [],
        "design_brief": a1_items,
        "locked_specs_review": locked_spec_items,
        "open_questions_review": detailed_open_question_items,
        "engineering_work_plan": stage4_rows or a3_items,
        "feasibility_risk_review": stage5_rows or a4_items,
        "design_envelope_summary": design_envelope_summary,
        "assumption_cards_review": assumption_cards_review,
        "missing_inputs_review": open_question_items,
        "engineering_impact_review": engineering_impact_review,
        "next_decision_review": next_decision_review if not workflow_rows else workflow_rows[:3],
        "node_summary": [
            {"label": "Parsed envelope", "value": locked_phrase, "status": "ready"},
            {"label": "Current gate", "value": "Requirements are captured; schematic/BOM generation is intentionally blocked.", "status": "hold"},
            {"label": "Can proceed", "value": "Topology and controller comparison can start after defaults are accepted or edited.", "status": "next"},
            {"label": "User decision", "value": "Adopt defaults unless you already know release constraints that should override them.", "status": "decision"},
        ],
        "gate_decision_workflow": [
            {"label": "Gate verdict", "value": gate_verdict, "status": "hold"},
            {"label": "Allowed next work", "value": "Topology candidates and controller shortlist may proceed after assumptions are accepted or edited.", "status": "next"},
            {"label": "Still blocked", "value": "Final schematic, final BOM, transformer release, PCB release, safety/EMI claims, and manufacturing package.", "status": "blocked"},
        ] + a5_items,
        "gate_verdict": [
            {"label": "Verdict", "value": gate_verdict, "status": "hold"},
            {"label": "Proceed allowed", "value": "Topology candidates and controller shortlist only.", "status": "next"},
            {"label": "Proceed blocked", "value": "Final schematic, final BOM, PCB release, transformer release, safety/EMI claims.", "status": "blocked"},
        ],
        "locked_envelope": [
            {"label": row.get("label"), "value": row.get("value"), "status": row.get("status")}
            for row in spec_rows
            if isinstance(row, dict) and row.get("value") and row.get("value") != "Default needed"
        ],
        "assumptions_to_accept": [
            {"label": row.get("label"), "value": row.get("value"), "status": row.get("status")}
            for row in assumptions
            if isinstance(row, dict) and row.get("label") in {"Safety direction", "EMI target", "Environment", "Topology path", "Release policy"}
        ],
        "open_inputs": [
            {"label": f"Input {index + 1}", "value": item, "status": "open"}
            for index, item in enumerate(missing[:8])
        ],
        "not_generated_here": [
            {"label": "Power stage", "value": "No reflected voltage, Lp, Ipk/Irms, VDS stack, current sense, or output capacitor sizing yet.", "status": "blocked"},
            {"label": "Magnetics", "value": "No core/bobbin/gap/winding/insulation package or leakage estimate yet.", "status": "blocked"},
            {"label": "Loop/simulation", "value": "No TL431/opto compensation, Bode plot, transient, thermal, or EMI evidence yet.", "status": "blocked"},
            {"label": "BOM/layout", "value": "No orderable AVL, transformer drawing, schematic values, ERC, or PCB constraints yet.", "status": "blocked"},
        ],
        "next_actions": [
            {"label": "Adopt defaults", "value": f"Use the current assumption set, including topology path: {topology_assumption.get('value') or 'current topology assumption'}.", "status": "next"},
            {"label": "Edit first", "value": "Use this if you know safety market/hipot, standby/no-load, mechanical, vendor/cost, or hold-up constraints.", "status": "option"},
            {"label": "Topology candidates", "value": "Compare topology paths from this envelope; still no controller, transformer, BOM, or schematic freeze.", "status": "option"},
        ],
        "gate_question": "Is the design envelope explicit enough to start topology work without generating a fake release schematic?",
        "gate_checks": [
            {"label": "Recognized specs", "value": f"{len(spec_rows)} extracted spec rows: input, output, power, efficiency, ripple, EMI, isolation, ambient."},
            {"label": "Missing constraints", "value": f"{len(missing)} still worth clarifying before release-quality work."},
            {"label": "Default assumptions", "value": f"{len(assumptions)} assumption cards created; each can be accepted, edited, locked, or reverted."},
            {"label": "Release block", "value": "Final schematic/BOM/release package is blocked at intake by policy."},
        ],
        "editable_fields": [
            {"label": "Safety", "value": "Market, standard, reinforced/basic isolation, pollution degree, altitude, hipot target."},
            {"label": "EMI", "value": "CISPR/EN class, conducted/radiated scope, pre-compliance margin policy."},
            {"label": "Environment", "value": "Open-frame/enclosure, airflow, board area, height, connector constraints, Ta max."},
            {"label": "Performance", "value": "Ripple, transient, hold-up, brown-in/out, standby, efficiency targets."},
            {"label": "Business", "value": "BOM target, volume, approved vendors, supply-chain restrictions."},
        ],
        "evidence_required": [
            {"label": "User decision", "value": "Critical specs must be accepted or edited before topology/controller selection."},
            {"label": "Assumption audit", "value": "Every default is traceable and remains non-release evidence until downstream gates close it."},
            {"label": "Open-input register", "value": "Unanswered safety, EMI, mechanical, cost, and hold-up items must remain visible."},
        ],
        "artifact_outputs": [
            {"label": "Spec Lock table", "value": "Critical, soft, and risk rows for the design review sheet."},
            {"label": "Assumption cards", "value": "Safety, EMI, environment, topology, and release policy."},
            {"label": "Topology seed", "value": "QR flyback primary candidate plus fixed-frequency and active-clamp backups."},
            {"label": "Risk register update", "value": "Safety/EMI, transformer leakage/thermal, VDS/SR/RCD/EMI coupling, loop stability."},
        ],
        "agent_trace": [
            {"label": "Requirement Agent", "value": "Parsed natural-language request into normalized project spec."},
            {"label": "Standards Agent", "value": "Created safety and EMI assumption cards instead of pretending standards are finalized."},
            {"label": "Topology Agent", "value": "Seeded QR, fixed-frequency, and active-clamp candidates without selecting final hardware."},
            {"label": "Critic policy", "value": "Blocked release schematic at intake; downstream evidence gates are mandatory."},
        ],
        "reviewer_output": [
            {"label": "Gate verdict", "value": "Ready to proceed only as a gated engineering scaffold. Not release-ready, not a schematic, not a BOM."}
        ],
    }


def _requirements_analysis_sections(checkpoint_details: Dict[str, Any]) -> list[Dict[str, Any]]:
    node_summary = checkpoint_details.get("node_summary") if isinstance(checkpoint_details.get("node_summary"), list) else []
    stage0 = checkpoint_details.get("stage0_input_normalization") if isinstance(checkpoint_details.get("stage0_input_normalization"), list) else []
    stage2 = checkpoint_details.get("stage2_specification_extraction") if isinstance(checkpoint_details.get("stage2_specification_extraction"), list) else []
    open_inputs = checkpoint_details.get("stage3_missing_information") if isinstance(checkpoint_details.get("stage3_missing_information"), list) else []
    conflicts = checkpoint_details.get("stage5_feasibility_conflict_check") if isinstance(checkpoint_details.get("stage5_feasibility_conflict_check"), list) else []
    blocked_outputs = checkpoint_details.get("not_generated_here") if isinstance(checkpoint_details.get("not_generated_here"), list) else []
    sections: list[Dict[str, Any]] = []
    if node_summary:
        sections.append({"title": "Intake Result", "items": node_summary[:4]})
    if stage0:
        sections.append({"title": "Requirement Package", "items": stage0[:4]})
    if stage2:
        sections.append({"title": "Spec Extraction + Derived Values", "items": stage2[:6]})
    if open_inputs:
        sections.append({"title": "Missing Information by Priority", "items": open_inputs[:5]})
    if conflicts:
        sections.append({"title": "Feasibility / Conflict Check", "items": conflicts[:4]})
    if blocked_outputs:
        sections.append({"title": "Still Blocked", "items": blocked_outputs[:3]})
    return sections


def _requirements_agent_review(intake: Dict[str, Any]) -> Dict[str, Any]:
    specs = intake.get("specs") if isinstance(intake.get("specs"), dict) else {}
    recognized = intake.get("recognized_specs") if isinstance(intake.get("recognized_specs"), list) else []
    assumptions = intake.get("assumptions") if isinstance(intake.get("assumptions"), list) else []
    missing = intake.get("missing_inputs") if isinstance(intake.get("missing_inputs"), list) else []
    source_text = str(intake.get("source_text") or "")[:2500]

    recognized_lines = "\n".join(
        f"- {row.get('label')}: {row.get('value')} ({row.get('status')})"
        for row in recognized[:10]
        if isinstance(row, dict)
    )
    assumption_lines = "\n".join(
        f"- {row.get('label')}: {row.get('value')} [{row.get('status')}]"
        for row in assumptions[:10]
        if isinstance(row, dict)
    )
    missing_lines = "\n".join(f"- {item}" for item in missing[:10])
    review_query = (
        "Act as the Power Electronics Requirement Analysis Agent and independent critic for the first gate of a power-electronics MAS. "
        "Transform the request into requirement-level findings only: application implications, explicit specs, derived specs, missing information by priority, task decomposition, feasibility conflicts, refined objectives, and downstream handoff readiness. "
        "Do not choose final topology, schematic values, MOSFETs, magnetic cores, capacitors, final EMI filters, or BOM. "
        "Do not invent unsupported numerical standards, mechanical dimensions, protection thresholds, lifetime targets, or cost targets. "
        "Return concise English with labels: Parsed intent, Explicit specs, Derived values, Missing/blocking items, Feasibility conflicts, Downstream readiness.\n\n"
        f"User request:\n{source_text}\n\n"
        f"Extracted specs:\n{recognized_lines}\n\n"
        f"Assumption cards:\n{assumption_lines}\n\n"
        f"Missing confirmations:\n{missing_lines}\n\n"
        f"Normalized numeric specs: {json.dumps(specs, ensure_ascii=False, default=str)[:1600]}"
    )

    trace: list[str] = [
        "Rule parser normalized the request before any LLM call.",
        "Gate policy blocked schematic/BOM generation at intake.",
    ]
    search_items: list[Dict[str, Any]] = []
    rag_context = ""

    try:
        rag_bundle = retrieve_flyback_context(review_query, top_k=4)
        rag_context = str(rag_bundle.get("context_text") or "").strip()
        refs = rag_bundle.get("references") or []
        trace.append(f"Local RAG retrieved {len(refs)} source(s) for intake review.")
        seen_refs: set[str] = set()
        for ref in refs[:4]:
            if not isinstance(ref, dict):
                continue
            title = normalize_source_label(ref.get("title") or ref.get("source") or ref.get("doc") or "RAG source")
            url = str(ref.get("url") or ref.get("link") or ref.get("path") or "")
            key = f"{title}|{url}"
            if key in seen_refs:
                continue
            seen_refs.add(key)
            search_items.append(
                {
                    "channel": "local_rag",
                    "source": "Internal Knowledge Base",
                    "title": str(title),
                    "url": url,
                    "snippet": "Retrieved for requirements gate review.",
                    "status": "retrieved",
                }
            )
    except Exception as e:
        trace.append(f"Local RAG failed: {e}")

    web_enabled = str(os.getenv("PE_MAS_REQUIREMENTS_WEB_REVIEW", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if web_enabled and mcp_research_web:
        try:
            web_pack = _tool_call_with_contract(
                "requirements_mcp_research",
                lambda: mcp_research_web(
                    "offline flyback requirements specification gate UCC28740 QR flyback CISPR 32 safety isolation design guide",
                    max_results=2,
                ),
                timeout_sec=float(os.getenv("PE_MAS_REQUIREMENTS_WEB_TIMEOUT_SEC", "8") or "8"),
                retries=0,
                idempotency_key=hashlib.sha1(source_text.encode("utf-8")).hexdigest(),
            )
            rows = web_pack.get("results") or [] if isinstance(web_pack, dict) else []
            trace.append(f"MCP web research returned {len(rows)} result(s).")
            for row in rows[:2]:
                if not isinstance(row, dict):
                    continue
                search_items.append(
                    {
                        "channel": "mcp_web",
                        "source": row.get("source") or "Web research",
                        "title": str(row.get("title") or row.get("url") or "Web source"),
                        "url": str(row.get("url") or row.get("link") or ""),
                        "snippet": str(row.get("snippet") or row.get("summary") or "")[:280],
                        "status": "retrieved",
                    }
                )
        except Exception as e:
            trace.append(f"MCP web research unavailable: {e}")
    else:
        trace.append("MCP web research skipped for intake unless PE_MAS_REQUIREMENTS_WEB_REVIEW=1.")

    llm_answer = _try_llm_qa_answer(review_query, rag_context=rag_context, history=[])
    if llm_answer:
        status = "llm_reviewed"
        trace.append("LLM reviewer completed intake review.")
        reviewer_output = llm_answer[:2400]
    else:
        status = "deterministic_fallback"
        trace.append("LLM reviewer unavailable or rate-limited; deterministic intake reviewer used.")
        locked_phrase = _locked_spec_phrase(specs)
        power_label = _design_power_label(specs)
        feedback = specs.get("feedback") or "TL431/opto feedback"
        rectifier = specs.get("secondary_rectification") or "rectifier choice not locked"
        reviewer_output = (
            f"Parsed intent: offline isolated AC/DC flyback design, {locked_phrase}.\n"
            "Locked critical specs: input range, output voltage/current/power, isolation direction, EMI target, and release-policy hold.\n"
            f"Assumptions: IEC/UL 62368-1 direction, CISPR 32 Class B pre-compliance, open-frame 50 C thermal target, QR flyback + {feedback}, {rectifier}, 800 V MOSFET evaluation margin for a {power_label} universal-input path.\n"
            "Missing confirmations: exact safety market/hipot, mechanical envelope/airflow, BOM cost/vendor restrictions, hold-up and brown-in/out behavior.\n"
            "Release block: no schematic/BOM/release package can be marked final from this gate.\n"
            "Next user decision: accept defaults, edit assumptions, or proceed only to topology candidates."
        )

    return {
        "status": status,
        "review_text": reviewer_output,
        "trace": trace,
        "search_items": search_items[:8],
    }


FRAMEWORK_STAGE_SEQUENCE = [
    ("ADOPT_DEFAULT_ASSUMPTIONS", "spec"),
    ("LOCK_CRITICAL_SPECS", "spec"),
    ("GENERATE_TOPOLOGY_CANDIDATES", "topology"),
    ("CONFIRM_QR_FLYBACK", "topology"),
    ("GENERATE_CONTROLLER_CANDIDATES", "controller"),
    ("LOCK_CONTROLLER_UCC28740", "controller"),
    ("ENTER_POWER_STAGE_CALC", "power_stage"),
    ("RUN_VR_SCAN", "power_stage"),
    ("GENERATE_MAGNETICS_CANDIDATES", "magnetics"),
    ("GENERATE_DEVICE_CARDS", "devices"),
    ("DESIGN_SNUBBER_CLAMP", "snubber"),
    ("DESIGN_LOOP_COMPENSATION", "loop"),
    ("RUN_SYSTEM_VALIDATION", "simulation"),
    ("GENERATE_LOCAL_FIX_OPTIONS", "simulation"),
    ("GENERATE_SCHEMATIC_DRAFT", "schematic"),
    ("GENERATE_PCB_CONSTRAINTS", "pcb"),
    ("GENERATE_BOM_AVL", "bom"),
    ("GENERATE_EVT_TEST_PLAN", "test"),
    ("RUN_FINAL_DESIGN_REVIEW", "release"),
    ("GENERATE_RELEASE_PACKAGE", "release"),
]
FRAMEWORK_COMMANDS = {cmd for cmd, _ in FRAMEWORK_STAGE_SEQUENCE}
FRAMEWORK_STAGE_ORDER = [
    "spec", "topology", "controller", "power_stage", "magnetics", "devices", "snubber", "loop",
    "simulation", "schematic", "pcb", "bom", "test", "release",
]


def _framework_stage_status(current_stage: str, reached_stages: list[str], release_blocked: bool = True) -> list[Dict[str, Any]]:
    rows = []
    reached = set(reached_stages or [])
    current_index = FRAMEWORK_STAGE_ORDER.index(current_stage) if current_stage in FRAMEWORK_STAGE_ORDER else 0
    labels = {
        "spec": ("Spec", "Critical specs locked"),
        "topology": ("Topology", "Candidates compared"),
        "controller": ("Controller", "Decision record"),
        "power_stage": ("Power Stage", "VR/Lp/stress sweep"),
        "magnetics": ("Magnetics", "Winding package"),
        "devices": ("Devices", "Derating + alternates"),
        "snubber": ("Clamp", "RCD/TVS trade-off"),
        "loop": ("Loop", "PM/GM + CTR corners"),
        "simulation": ("PLECS Simulation", "PLECS validation matrix"),
        "schematic": ("Schematic", "ERC + debug hooks"),
        "pcb": ("PCB Rules", "Loops + isolation"),
        "bom": ("BOM/AVL", "Orderable evidence"),
        "test": ("EVT Plan", "Bench matrix"),
        "release": ("Release", "Risk sign-off"),
    }
    for index, key in enumerate(FRAMEWORK_STAGE_ORDER):
        status = "pending"
        if key in reached or index < current_index:
            status = "scaffold"
        if key == current_stage:
            status = "gate"
        if key == "release" and release_blocked:
            status = "blocked" if key not in reached else status
        label, gate = labels[key]
        rows.append({"key": key, "label": label, "gate": gate, "status": status})
    return rows


def _framework_next_actions(command_id: str) -> tuple[list[str], list[str]]:
    mapping = {
        "ADOPT_DEFAULT_ASSUMPTIONS": (
            ["Lock critical specs v0.1", "Edit assumptions manually", "Generate topology candidates"],
            ["LOCK_CRITICAL_SPECS", "MANUAL_ADJUSTMENTS", "GENERATE_TOPOLOGY_CANDIDATES"],
        ),
        "LOCK_CRITICAL_SPECS": (
            ["Generate topology candidates", "Edit assumptions manually", "Stop this session"],
            ["GENERATE_TOPOLOGY_CANDIDATES", "MANUAL_ADJUSTMENTS", "STOP_SESSION"],
        ),
        "GENERATE_TOPOLOGY_CANDIDATES": (
            ["Confirm topology: QR Flyback + SSR", "Edit assumptions manually", "Generate controller candidates"],
            ["CONFIRM_QR_FLYBACK", "MANUAL_ADJUSTMENTS", "GENERATE_CONTROLLER_CANDIDATES"],
        ),
        "CONFIRM_QR_FLYBACK": (
            ["Generate controller candidates", "Generate topology candidates", "Stop this session"],
            ["GENERATE_CONTROLLER_CANDIDATES", "GENERATE_TOPOLOGY_CANDIDATES", "STOP_SESSION"],
        ),
        "GENERATE_CONTROLLER_CANDIDATES": (
            ["Lock TI UCC28740 route v0.1", "Generate topology candidates", "Stop this session"],
            ["LOCK_CONTROLLER_UCC28740", "GENERATE_TOPOLOGY_CANDIDATES", "STOP_SESSION"],
        ),
        "LOCK_CONTROLLER_UCC28740": (
            ["Enter power-stage calculation", "Generate controller candidates", "Stop this session"],
            ["ENTER_POWER_STAGE_CALC", "GENERATE_CONTROLLER_CANDIDATES", "STOP_SESSION"],
        ),
        "ENTER_POWER_STAGE_CALC": (
            ["Run VR scan 90/100/110/120 V", "Generate magnetics candidates", "Stop this session"],
            ["RUN_VR_SCAN", "GENERATE_MAGNETICS_CANDIDATES", "STOP_SESSION"],
        ),
        "RUN_VR_SCAN": (
            ["Generate magnetics candidates", "Enter power-stage calculation", "Stop this session"],
            ["GENERATE_MAGNETICS_CANDIDATES", "ENTER_POWER_STAGE_CALC", "STOP_SESSION"],
        ),
        "GENERATE_MAGNETICS_CANDIDATES": (
            ["Generate device/BOM cards", "Run VR scan again", "Stop this session"],
            ["GENERATE_DEVICE_CARDS", "RUN_VR_SCAN", "STOP_SESSION"],
        ),
        "GENERATE_DEVICE_CARDS": (
            ["Design RCD/TVS clamp", "Generate magnetics candidates", "Stop this session"],
            ["DESIGN_SNUBBER_CLAMP", "GENERATE_MAGNETICS_CANDIDATES", "STOP_SESSION"],
        ),
        "DESIGN_SNUBBER_CLAMP": (
            ["Design feedback loop compensation", "Generate device/BOM cards", "Stop this session"],
            ["DESIGN_LOOP_COMPENSATION", "GENERATE_DEVICE_CARDS", "STOP_SESSION"],
        ),
        "DESIGN_LOOP_COMPENSATION": (
            ["Run PLECS validation matrix", "Prepare simulation checklist", "Stop this session"],
            ["CONTINUE_SIMULATION", "RUN_SYSTEM_VALIDATION", "STOP_SESSION"],
        ),
        "RUN_SYSTEM_VALIDATION": (
            ["Run PLECS validation matrix", "Generate schematic draft", "Stop this session"],
            ["CONTINUE_SIMULATION", "GENERATE_SCHEMATIC_DRAFT", "STOP_SESSION"],
        ),
        "GENERATE_LOCAL_FIX_OPTIONS": (
            ["Rerun PLECS validation matrix", "Generate schematic draft", "Stop this session"],
            ["CONTINUE_SIMULATION", "GENERATE_SCHEMATIC_DRAFT", "STOP_SESSION"],
        ),
        "GENERATE_SCHEMATIC_DRAFT": (
            ["Generate PCB constraints", "Run ERC/DRC review later", "Stop this session"],
            ["GENERATE_PCB_CONSTRAINTS", "RUN_FINAL_DESIGN_REVIEW", "STOP_SESSION"],
        ),
        "GENERATE_PCB_CONSTRAINTS": (
            ["Generate BOM/AVL", "Generate schematic draft", "Stop this session"],
            ["GENERATE_BOM_AVL", "GENERATE_SCHEMATIC_DRAFT", "STOP_SESSION"],
        ),
        "GENERATE_BOM_AVL": (
            ["Generate EVT test plan", "Generate PCB constraints", "Stop this session"],
            ["GENERATE_EVT_TEST_PLAN", "GENERATE_PCB_CONSTRAINTS", "STOP_SESSION"],
        ),
        "GENERATE_EVT_TEST_PLAN": (
            ["Run final design review", "Generate BOM/AVL", "Stop this session"],
            ["RUN_FINAL_DESIGN_REVIEW", "GENERATE_BOM_AVL", "STOP_SESSION"],
        ),
        "RUN_FINAL_DESIGN_REVIEW": (
            ["Generate Release Package", "Generate EVT test plan", "Stop this session"],
            ["GENERATE_RELEASE_PACKAGE", "GENERATE_EVT_TEST_PLAN", "STOP_SESSION"],
        ),
        "GENERATE_RELEASE_PACKAGE": (
            ["Run final design review", "Edit assumptions manually", "Stop this session"],
            ["RUN_FINAL_DESIGN_REVIEW", "MANUAL_ADJUSTMENTS", "STOP_SESSION"],
        ),
    }
    return mapping.get(command_id, mapping["ADOPT_DEFAULT_ASSUMPTIONS"])


def _framework_decision_options(command_id: str, current_stage: str, base_options: list[str], base_commands: list[str]) -> list[Dict[str, str]]:
    stage_real_command = "CONTINUE_SIMULATION" if current_stage in {
        "snubber", "loop", "simulation", "schematic", "pcb", "bom", "test", "release"
    } else "CONTINUE_SELECTION"
    if stage_real_command == "CONTINUE_SIMULATION":
        stage_real_label = "Run PLECS validation matrix"
    else:
        stage_real_label = "Run real agent/tool workflow from locked spec"
    rows: list[Dict[str, str]] = []
    for idx, option in enumerate(base_options or []):
        command = (base_commands or [""])[idx] if idx < len(base_commands or []) else ""
        if command in {"STOP_SESSION", "MANUAL_ADJUSTMENTS"}:
            continue
        intent, risk = _framework_action_copy(command, current_stage)
        rows.append(
            {
                "option": str(option),
                "command": str(command),
                "intent": intent,
                "risk": risk,
            }
        )

    edit_intent, edit_risk = _framework_action_copy("MANUAL_ADJUSTMENTS", current_stage)
    rows.append(
        {
            "option": "Edit this gate inputs / assumptions",
            "command": "MANUAL_ADJUSTMENTS",
            "intent": edit_intent,
            "risk": edit_risk,
        }
    )
    real_intent, real_risk = _framework_action_copy(stage_real_command, current_stage)
    rows.append(
        {
            "option": stage_real_label,
            "command": stage_real_command,
            "intent": real_intent,
            "risk": real_risk,
        }
    )
    stop_intent, stop_risk = _framework_action_copy("STOP_SESSION", current_stage)
    rows.append(
        {
            "option": "Stop this session",
            "command": "STOP_SESSION",
            "intent": stop_intent,
            "risk": stop_risk,
        }
    )

    deduped: list[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.get("option", ""), row.get("command", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped[:6]


def _framework_action_copy(command: str, current_stage: str) -> tuple[str, str]:
    stage_label = str(current_stage or "stage").replace("_", " ")
    mapping: Dict[str, tuple[str, str]] = {
        "LOCK_CRITICAL_SPECS": (
            "Freeze the critical electrical envelope as Spec v0.1 so topology work has a stable target.",
            "This does not freeze safety, magnetics, BOM, layout, or release evidence.",
        ),
        "GENERATE_TOPOLOGY_CANDIDATES": (
            "Compare QR flyback, fixed-frequency flyback, and active clamp paths against the locked power envelope.",
            "The result is a trade-off record, not a selected controller or schematic.",
        ),
        "CONFIRM_QR_FLYBACK": (
            "Lock QR/valley flyback with secondary-side TL431/opto feedback as the first path.",
            "Fixed-frequency and active clamp remain backups if efficiency, thermal, or EMI evidence later fails.",
        ),
        "GENERATE_CONTROLLER_CANDIDATES": (
            "Create a controller shortlist with fit checks for power range, startup, standby, feedback, model availability, and sourcing.",
            "No IC is final until loop, startup, protection, standby, and layout constraints are verified.",
        ),
        "LOCK_CONTROLLER_UCC28740": (
            "Record TI UCC28740 as the primary controller route and keep Infineon/PI as documented backups.",
            "The controller route still needs startup, VDD, standby, loop, SR timing, and model evidence.",
        ),
        "ENTER_POWER_STAGE_CALC": (
            "Open the auditable calculation notebook for bulk voltage, input power, Lp, peak currents, stress, and ripple cells.",
            "The numbers are targets until standard-value iteration and tool/bench checks confirm them.",
        ),
        "RUN_VR_SCAN": (
            "Sweep reflected-voltage targets and compare MOSFET stress, SR stress, duty ratio, magnetics, clamp loss, and EMI pressure.",
            "The selected VR remains provisional until transformer leakage and drain waveforms are measured or simulated credibly.",
        ),
        "GENERATE_MAGNETICS_CANDIDATES": (
            "Move from turns math into manufacturable core, bobbin, gap, winding, insulation, leakage, and thermal requirements.",
            "Custom magnetics cannot close from heuristics; vendor data and EVT measurements remain mandatory.",
        ),
        "GENERATE_DEVICE_CARDS": (
            "Create engineering cards for the MOSFET, SR FET, bridge, bulk/output capacitors, sense parts, clamp parts, and alternates.",
            "Cards need orderable MPNs, datasheets, derating, loss, lifetime, and package evidence before BOM freeze.",
        ),
        "DESIGN_SNUBBER_CLAMP": (
            "Build the RCD/TVS clamp trade-off from VDS stack-up, leakage energy, dissipation, thermal risk, and EMI impact.",
            "RCD values stay open until transformer leakage and high-line drain waveforms are known.",
        ),
        "DESIGN_LOOP_COMPENSATION": (
            "Define the TL431/opto compensation evidence plan: plant model, CTR corners, Bode margins, and transient response.",
            "A feedback network is not final without small-signal and transient evidence across line/load/CTR corners.",
        ),
        "RUN_SYSTEM_VALIDATION": (
            "Prepare the 09 PLECS validation matrix for steady state, startup, stress, protection, ripple, thermal, loop, and EMI checks.",
            "This checklist is not a simulator run; use Run PLECS validation matrix to attach waveforms and pass/fail data.",
        ),
        "GENERATE_LOCAL_FIX_OPTIONS": (
            "List reversible fixes such as compensation tuning or footprint-compatible SR changes before touching global magnetics.",
            "These options need rerun evidence; they are not automatic proof that the design now passes.",
        ),
        "GENERATE_SCHEMATIC_DRAFT": (
            "Create a hierarchical schematic draft checklist with input, primary, transformer, secondary, feedback, safety, and debug hooks.",
            "This remains a debug-ready draft, not a release schematic or manufacturing package.",
        ),
        "GENERATE_PCB_CONSTRAINTS": (
            "Generate layout constraints for hot loops, quiet feedback, EMI separation, isolation boundary, keepouts, and test access.",
            "No routing or release package is implied until EDA rules, ERC/DRC, and layout review are attached.",
        ),
        "GENERATE_BOM_AVL": (
            "Define the BOM/AVL schema and safety-critical evidence needed for orderable parts and approved alternates.",
            "Custom transformer and mains EMI parts still require manual engineering/procurement sign-off.",
        ),
        "GENERATE_EVT_TEST_PLAN": (
            "Create an EVT/DVT bench matrix with instruments, operating points, pass criteria, screenshots, and sim-vs-measured columns.",
            "The plan does not close the gate until the bench data exists and is linked.",
        ),
        "RUN_FINAL_DESIGN_REVIEW": (
            "Audit every release claim and keep assumed, calculated, simulated, datasheet, measured, and closed states separate.",
            "Any unverified transformer, loop, EMI, thermal, or BOM claim keeps release on hold.",
        ),
        "GENERATE_RELEASE_PACKAGE": (
            "Assemble the release-package checklist and evidence index for audit.",
            "Without measured/tool evidence it must be labeled EVT/pre-release, not manufacturing-ready.",
        ),
        "CONTINUE_SELECTION": (
            "Leave scaffold mode and run the real agent/tool workflow while preserving the locked input, output, power, isolation, EMI, and ripple targets.",
            "The run may expose sourcing, model, simulation, or validation failures instead of producing a clean demo.",
        ),
        "CONTINUE_SIMULATION": (
            "Run the 09 Simulation node using the locked spec and attach PLECS/tool output where available.",
            "The result can still be HOLD or FAIL if PLECS scope, closed-loop behavior, BOM, low-line, EMI, or thermal evidence is incomplete.",
        ),
        "MANUAL_ADJUSTMENTS": (
            f"Patch the {stage_label} gate inputs, assumptions, selected path, risk status, or evidence requirements before proceeding.",
            "Use this when the default assumption or generated scaffold would not survive a PE review.",
        ),
        "STOP_SESSION": (
            "Pause the workflow without changing the design state.",
            "No new evidence is generated.",
        ),
    }
    return mapping.get(
        str(command or ""),
        (
            f"Continue the {stage_label} gate with the current locked specification and risk register.",
            "This is still scaffold evidence unless a calculation file, simulator output, source link, or bench result is attached.",
        ),
    )


def _augment_framework_checkpoint_details(
    details: Dict[str, Any],
    command_id: str,
    current_stage: str,
    title: str,
    summary: str,
    sections: list[Dict[str, Any]],
    calculations: Dict[str, Any],
    decision_options: list[Dict[str, str]],
) -> Dict[str, Any]:
    details = dict(details or {})
    stage_label = str(current_stage or "stage").replace("_", " ")
    first_section = next((row for row in sections if isinstance(row, dict) and row.get("items")), {})
    first_section_title = str(first_section.get("title") or title)
    first_section_items = first_section.get("items") if isinstance(first_section.get("items"), list) else []
    first_fact = ""
    if first_section_items:
        item = first_section_items[0]
        if isinstance(item, dict):
            first_fact = f"{item.get('label')}: {item.get('value')}"
        else:
            first_fact = str(item)

    missing_map: Dict[str, list[tuple[str, str]]] = {
        "spec": [
            ("Open constraints", "Safety market/hipot, mechanical envelope, airflow, hold-up, cost, approved vendors."),
            ("Release proof", "No schematic, BOM, layout, or safety package is proven by this gate."),
        ],
        "topology": [
            ("Quantified trade-off", "Efficiency, EMI, thermal, cost, control complexity, and supply-chain impact still need numbers."),
            ("Downstream proof", "Controller, magnetics, clamp, loop, EMI, thermal, and layout evidence are still open."),
        ],
        "controller": [
            ("Controller proof", "Startup/VDD, standby, protection thresholds, model availability, SR timing, and loop interaction."),
            ("Sourcing proof", "Datasheet/app-note/model/source evidence and rollback criteria for backup vendors."),
        ],
        "power_stage": [
            ("Calculation closure", "Low-line Vbulk,min, hold-up, Lp tolerance, RMS currents, CS resistor, output cap RMS, and stress margins."),
            ("Tool proof", "Waveforms or validated calculations across low-line/high-line/load corners."),
        ],
        "magnetics": [
            ("Vendor data", "Core/bobbin Ae/le/Ve, gap, wire, tape stack, pinout, creepage/clearance, and hipot basis."),
            ("EVT data", "Lp, Llk, DCR, turns ratio, thermal rise, insulation, and leakage measurements."),
        ],
        "devices": [
            ("Stress evidence", "VDS/SR reverse voltage, losses, thermal path, capacitor ripple/lifetime, and clamp part stress."),
            ("AVL evidence", "Orderable MPNs, alternates, lifecycle, source links, and replacement constraints."),
        ],
        "snubber": [
            ("Measured leakage", "Transformer Llk and drain spike waveform before RCD values can close."),
            ("Thermal/EMI proof", "Clamp resistor/diode temperature and conducted EMI impact."),
        ],
        "loop": [
            ("Model evidence", "Power-stage plant, TL431/opto dynamics, CTR aging, component tolerance, Bode, PM/GM."),
            ("Transient evidence", "Load/line step response, recovery, overshoot, saturation, and noise susceptibility."),
        ],
        "simulation": [
            ("PLECS output", "Project/netlist, model versions, run conditions, waveforms, losses, and pass/fail table."),
            ("Coverage", "Low-line full-load, high-line stress, startup, SR timing, loop, thermal, and EMI corners."),
        ],
        "schematic": [
            ("Values and ERC", "Part values, debug hooks, ERC output, known waivers, and trace links to calculations/BOM."),
            ("Safety review", "Creepage/clearance, Y-cap path, net classes, and isolation annotations."),
        ],
        "pcb": [
            ("EDA evidence", "Rule export, placement screenshots, loop review, DRC/ERC, and waiver list."),
            ("Safety/EMI proof", "Keepouts, isolation slot, filter separation, return paths, thermal copper, and probe access."),
        ],
        "bom": [
            ("Populated AVL", "MPNs, alternates, lifecycle, source links, cost tiers, ratings, and certification basis."),
            ("Manual items", "Transformer drawing/inspection spec and mains EMI parts need human sign-off."),
        ],
        "test": [
            ("Bench results", "The matrix needs measured values, screenshots, instrument setup, calibration, and pass/fail deltas."),
            ("Safety procedure", "Hi-pot and hazardous-voltage procedure references."),
        ],
        "release": [
            ("Evidence index", "Every release claim must link to calculation, source, tool output, or measurement."),
            ("Physical closure", "Transformer leakage/thermal, loop, EMI, safety, and bench evidence must be attached."),
        ],
    }
    why_map: Dict[str, list[tuple[str, str]]] = {
        "spec": [("Reason", "A stable spec prevents the workflow from silently drifting into defaults or generating a fake schematic.")],
        "topology": [("Reason", "Topology choice drives controller, magnetics, stress, EMI, thermal, and layout risk.")],
        "controller": [("Reason", "The controller route fixes feedback behavior, startup, protections, model availability, and sourcing risk.")],
        "power_stage": [("Reason", "Stress and energy equations expose bad designs before transformer, clamp, and BOM values look official.")],
        "magnetics": [("Reason", "The transformer is custom, safety-critical, thermally important, and cannot be audited as a generic part.")],
        "devices": [("Reason", "Power devices need derating and alternates before the BOM can be trusted.")],
        "snubber": [("Reason", "Clamp settings trade off MOSFET safety margin, heat, efficiency, and EMI.")],
        "loop": [("Reason", "TL431/opto loops can pass nominal operation while failing CTR, load, line, or aging corners.")],
        "simulation": [("Reason", "A single optimistic waveform is not enough; release needs worst-case coverage and model scope transparency.")],
        "schematic": [("Reason", "A debug-ready schematic reserves first-revision tuning hooks without pretending values are final.")],
        "pcb": [("Reason", "Offline flyback success depends heavily on loop placement, isolation, return paths, EMI, and probe access.")],
        "bom": [("Reason", "An AVL must be orderable, derated, safety-certified where needed, and backed by source evidence.")],
        "test": [("Reason", "EVT turns planned evidence into reproducible measurements that can challenge the model.")],
        "release": [("Reason", "Release review is an evidence audit, not a checklist of generated documents.")],
    }
    next_rows = [
        {
            "label": str(row.get("option") or "Next action"),
            "value": str(row.get("intent") or ""),
            "status": "next" if index == 0 else "option",
        }
        for index, row in enumerate((decision_options or [])[:3])
        if isinstance(row, dict)
    ]
    details["stage_summary"] = [
        {"label": "Summary", "value": summary, "status": "summary"},
        {"label": "Artifact", "value": f"{title} produced a reviewable {stage_label} artifact, not release evidence.", "status": "scaffold"},
        {"label": "First read", "value": first_fact or f"{first_section_title} was prepared for review.", "status": "context"},
        {"label": "Release truth", "value": "Release remains HOLD until calculated/tool/source/measured evidence closes the downstream gates.", "status": "hold"},
    ]
    details["what_happened"] = [
        {
            "label": str(section.get("title") or "Section"),
            "value": "; ".join(
                f"{item.get('label')}: {item.get('value')}" if isinstance(item, dict) else str(item)
                for item in (section.get("items") or [])[:3]
            ),
            "status": "done",
        }
        for section in sections[:4]
        if isinstance(section, dict)
    ]
    details["why_this_matters"] = [{"label": label, "value": value, "status": "why"} for label, value in why_map.get(current_stage, [])]
    details["evidence_state"] = [
        {"label": "Current evidence", "value": "Scaffold plus reviewer trace; not a simulator result, EDA run, sourced BOM, or bench measurement.", "status": "scaffold"},
        {"label": "Calculation context", "value": f"Vbulk,max {calculations.get('vbulk_max_v')} V, Pin,max {calculations.get('pin_max_w')} W where applicable.", "status": "context"},
    ]
    details["missing_evidence"] = [
        {"label": label, "value": value, "status": "open"}
        for label, value in missing_map.get(current_stage, missing_map["release"])
    ]
    details["next_actions"] = next_rows
    return details


def _framework_stage_review_details(
    command_id: str,
    current_stage: str,
    title: str,
    specs: Dict[str, Any],
    calculations: Dict[str, Any],
) -> Dict[str, Any]:
    base = {
        "gate_question": "Can this stage move forward without pretending the design is release-ready?",
        "gate_checks": [
            ("Artifact", "Does this stage produce a reviewable engineering artifact, not only prose?"),
            ("Decision", "Is the selected path reversible and documented with reasons?"),
            ("Risk", "Are assumed, calculated, simulated, measured, and closed states separated?"),
        ],
        "editable_fields": [
            ("Assumptions", "Safety, EMI, thermal, cost, standby, hold-up, supply-chain constraints"),
            ("Gate status", "Keep as scaffold/review/hold unless evidence is attached"),
        ],
        "evidence_required": [
            ("Traceability", "Source, formula, tool output, or bench measurement tied to each claim"),
            ("Release block", "Final schematic/BOM/package remains blocked without downstream evidence"),
        ],
        "artifact_outputs": [
            ("Decision record", "Why this stage can proceed or why it must remain open"),
            ("Risk update", "New risks, owner, and required evidence"),
        ],
    }
    vin_min = specs.get("input_voltage_min", 85)
    vin_max = specs.get("input_voltage_max", 265)
    vout = specs.get("output_voltage", 12)
    iout = specs.get("output_current", 5)
    power_label = _design_power_label(specs)
    locked_spec = f"{float(vin_min):g}-{float(vin_max):g}Vac, 47-63Hz, {float(vout):g}V/{float(iout):g}A, {power_label}, isolation, EMI direction"
    stage_map: Dict[str, Dict[str, Any]] = {
        "spec": {
            "gate_question": "Are the critical electrical specs and default assumptions explicit enough to start topology work?",
            "gate_checks": [
                ("Critical specs", locked_spec),
                ("Soft targets", "88/89% efficiency, <120mVp-p ripple, <150mW standby, 0-50C open-frame"),
                ("Open inputs", "Safety market, pollution degree, altitude, hipot, hold-up, size, cost, approved vendors"),
            ],
            "editable_fields": [
                ("Safety/EMI", "Standard, market, reinforced/basic insulation, CISPR class and margin policy"),
                ("Use condition", "Ambient, enclosure/open-frame, airflow, board area, height limit"),
                ("Performance", "Ripple, transient, hold-up, standby, efficiency priorities"),
            ],
            "evidence_required": [
                ("Spec lock", "User-confirmed critical specs v0.1"),
                ("Assumption cards", "Each default accepted, edited, locked, or reverted"),
            ],
            "artifact_outputs": [
                ("Spec Lock table", "Critical/soft/risk classifications"),
                ("Assumption cards", "Editable records for safety, EMI, thermal, release policy"),
            ],
        },
        "topology": {
            "gate_question": "Which topology path should be primary, and which alternatives must remain as backups?",
            "gate_checks": [
                ("QR flyback", f"Good first {power_label} path for cost, efficiency, EMI, and debug risk"),
                ("Fixed-frequency flyback", "Backup path for simpler model/controller availability"),
                ("Active clamp flyback", "Deferred unless efficiency/thermal density forces added complexity"),
            ],
            "editable_fields": [
                ("Primary topology", "QR, fixed-frequency DCM/CCM, or active clamp"),
                ("Feedback/rectification", "TL431/opto SSR, PSR, Schottky, or SR"),
                ("Priority weights", "Cost, efficiency, risk, size, EMI, schedule"),
            ],
            "evidence_required": [
                ("Trade-off table", "Efficiency, EMI, thermal, cost, control complexity, supply-chain impact"),
                ("Backup retention", "Reason fixed-frequency/ACF alternatives remain available"),
            ],
            "artifact_outputs": [
                ("Topology decision record", "Why QR first path, why ACF deferred"),
                ("Risk update", "Loop, magnetics, clamp, EMI, and SR timing risks carried forward"),
            ],
        },
        "controller": {
            "gate_question": "Is the controller path selected from evidence, not a random IC pick?",
            "gate_checks": [
                ("TI UCC28740 path", "Universal AC fit, opto CV route, external MOSFET freedom"),
                ("Backup vendors", "Infineon QR CoolSET/controller and Power Integrations high-integration paths"),
                ("Fit checklist", "Power range, VDS margin, startup/VDD, standby, SR support, models, EVM/calculator, supply"),
            ],
            "editable_fields": [
                ("Primary IC", "TI UCC28740, Infineon QR, PI InnoSwitch-class, or user-approved controller"),
                ("Power switch policy", "800V external MOSFET footprint vs integrated MOSFET"),
                ("Model/source state", "Datasheet/app note/EVM/SPICE/SIMPLIS availability and procurement risk"),
            ],
            "evidence_required": [
                ("Controller decision record", "Candidate table with pros, risks, model availability, and backups"),
                ("Datasheet fit", "Universal input, layout constraints, feedback/stability implications"),
            ],
            "artifact_outputs": [
                ("Controller CDR", "Decision, alternatives, rollback triggers"),
                ("Evidence links", "Datasheet/app note/reference design/model source references"),
            ],
        },
        "power_stage": {
            "gate_question": "Do the first-pass equations expose stress and worst-case boundaries before magnetics/BOM freeze?",
            "gate_checks": [
                ("Input/bulk", f"Vbulk,max ~= {calculations.get('vbulk_max_v')}V; Vbulk,min still needs hold-up and ripple target"),
                ("Power", f"Pin,max ~= {calculations.get('pin_max_w')}W; bulk heuristic {calculations.get('bulk_cap_heuristic_uf')}uF"),
                ("Stress cells", "VR, Dmax, Lp, Ipk/Irms, MOSFET VDS stack, SR VDS, CS resistor, output cap RMS"),
            ],
            "editable_fields": [
                ("VR scan", "90/100/110/120V reflected-voltage candidates"),
                ("MOSFET policy", "800V first revision, measured VDS peak <680V for 15% margin"),
                ("Hold-up", "Bulk ripple and brown-in/brown-out behavior"),
            ],
            "evidence_required": [
                ("Calculation notebook", "Formula, units, assumptions, standard-value iteration"),
                ("Stress comparison", "MOSFET/SR/clamp/magnetics/EMI trade-offs per VR option"),
            ],
            "artifact_outputs": [
                ("Auditable equations", "Vbulk, Pin, bulk, VR, Dmax, Lp, currents, stress"),
                ("Decision record", "Why VR target and 800V MOSFET policy are acceptable"),
            ],
        },
        "magnetics": {
            "gate_question": "Is the transformer treated as a manufacturable high-risk component, not only turns math?",
            "gate_checks": [
                ("Core family", "EER28/EF28 primary for thermal/reliability, EE25/EER25 backup, EFD30/PQ26 higher-margin option"),
                ("Winding package", "Lp tolerance, Np/Ns/Naux, gap/AL, Bmax, DCR, copper/core loss, temperature"),
                ("Manufacturing", "Winding order, insulation tape, triple-insulated wire, shield, pins, hipot, Llk target"),
            ],
            "editable_fields": [
                ("Core choice", "Reliability/temperature vs cost/size"),
                ("Insulation system", "Reinforced isolation, creepage/clearance, bobbin and tape stack"),
                ("Leakage target", "Llk <3-5% Lp assumption until measured samples exist"),
            ],
            "evidence_required": [
                ("Vendor/manufacturing data", "Core/bobbin Ae/le/Ve, gap, wire, tape, pinout"),
                ("EVT measurements", "Lp, Llk, DCR, turns ratio, hipot, thermal rise"),
            ],
            "artifact_outputs": [
                ("Winding manual", "Buildable transformer drawing and inspection spec"),
                ("Risk item", "RCD/clamp stays open until Llk is measured"),
            ],
        },
        "devices": {
            "gate_question": "Are power devices selected with derating, alternates, and footprint strategy?",
            "gate_checks": [
                ("MOSFET card", "VDS, RDS(on) at temperature, Qg, Coss/Eoss, avalanche, package Rtheta, losses, alternates"),
                ("SR FET card", "100V initial class, RDS(on), Qg, body diode, timing/reverse-current/layout risk"),
                ("Bulk/output/sense", "105C lifetime, ripple current, tolerance, package, source alternates"),
            ],
            "editable_fields": [
                ("AVL policy", "Main MPN plus pin/footprint-compatible alternate"),
                ("Package strategy", "TO-220F/TO-252 evaluation footprint, SR package compatibility"),
                ("Lifetime/derating", "Bulk capacitor life, thermal margin, safety-certified parts"),
            ],
            "evidence_required": [
                ("Orderable MPNs", "Source links, lifecycle, stock/cost tier, alternates"),
                ("Stress ratio", "Calculated/simulated stress versus rating for each power device"),
            ],
            "artifact_outputs": [
                ("Device cards", "Engineering cards for MOSFET, SR, bridge, bulk, sense, clamp"),
                ("BOM risk update", "Manual sign-off for custom or provisional items"),
            ],
        },
        "snubber": {
            "gate_question": "Is the clamp sized from VDS stack-up and leakage energy, with release blocked until Llk is measured?",
            "gate_checks": [
                ("VDS stack", "VDS_peak = Vbulk,max + VR + leakage spike + margin"),
                ("Leakage energy", "First pass Llk = 3% Lp, E_lk = 0.5*Llk*Ipk^2, Pclamp ~= E_lk*fsw"),
                ("Clamp target", "Sweep 580/620/660/680V for VDS margin, RCD loss/temp, diode/cap stress, EMI"),
            ],
            "editable_fields": [
                ("Clamp voltage", "Conservative 620-660V first-pass range"),
                ("TVS footprint", "Reserve optional TVS containment footprint"),
                ("Snubber pads", "Primary and SR D-S RC snubber debug positions"),
            ],
            "evidence_required": [
                ("Measured Llk", "Transformer sample leakage before closing RCD values"),
                ("Bench waveform", "High-line/full-load VDS with HV differential probe and documented setup"),
                ("Thermal/EMI", "RCD resistor/clamp diode temperature and conducted EMI pre-scan"),
            ],
            "artifact_outputs": [
                ("RCD trade-off table", "Clamp target vs VDS, loss, temperature, EMI"),
                ("EVT open item", "RCD values update after measured leakage"),
            ],
        },
        "loop": {
            "gate_question": "Can TL431/opto feedback be considered stable across line/load/CTR corners?",
            "gate_checks": [
                ("Feedback mode", "Confirm SSR TL431 + optocoupler, not PSR/digital isolation"),
                ("Loop target", "Crossover 1-2kHz initial, phase margin >60deg, gain margin checked"),
                ("Corners", "85/115/230Vac and 10/50/100% load, optocoupler CTR min/typ/max and aging"),
            ],
            "editable_fields": [
                ("Compensation", "TL431 R/C values and tolerance sensitivity"),
                ("Bias/current", "TL431 bias versus standby power target"),
                ("Acceptance", "Accept as v0.1 but mark needs bench/small-signal validation"),
            ],
            "evidence_required": [
                ("Small-signal model", "Power stage, TL431 compensation, optocoupler pole"),
                ("Bode/Transient", "PM/GM, crossover, load transient, causal tuning notes"),
            ],
            "artifact_outputs": [
                ("Loop report", "Bode plot, PM/GM table, CTR corner table"),
                ("Risk update", "Feedback network not final until model/tool/bench evidence exists"),
            ],
        },
        "simulation": {
            "gate_question": "Does the validation matrix cover real worst cases rather than a single optimistic waveform?",
            "gate_checks": [
                ("Operating points", "85/115/230/265Vac, no/load/full load and transient corners"),
                ("Protections", "Startup, short/OCP, brown-in/out, SR timing/reverse current"),
                ("Reports", "Ripple, VDS, CS, loss breakdown, thermal estimate, conducted EMI 150kHz-30MHz"),
            ],
            "editable_fields": [
                ("Corners", "Choose low-line/high-line/nominal/custom bus and load points"),
                ("Pass criteria", "VDS <680V, ripple <120mVp-p, transient +/-3%, PM >60deg"),
                ("Tool scope", "SPICE/SIMPLIS/PLECS model version and exclusions"),
            ],
            "evidence_required": [
                ("Waveforms", "Drain, CS, SR gate/drain, Vout ripple, startup/load transient"),
                ("Tool files", "Netlist/project, models, version, run conditions"),
            ],
            "artifact_outputs": [
                ("Validation dashboard", "Pass/warn/fail table with evidence file links"),
                ("Correction trigger", "Warnings feed local fix options, not full schematic rewrite"),
            ],
        },
        "schematic": {
            "gate_question": "Is the schematic a hierarchical debug-ready draft, not a release drawing?",
            "gate_checks": [
                ("Sheets", "Input protection, EMI, rectifier/bulk, controller, MOSFET/CS, clamp, transformer, SR/output, feedback, protection, test points"),
                ("Debug reservations", "RCD pads, gate resistor/diode, SR snubber, output pi filter, opto RC, CS filter, VDS/CS/VOUT/FB/SR gate TPs"),
                ("Safety", "HV/LV net classes, isolation boundary, X/Y cap certifications, creepage/clearance notes"),
            ],
            "editable_fields": [
                ("Debug hooks", "Add/remove first-rev tuning positions"),
                ("Sheet ownership", "Which blocks can be generated automatically versus manually reviewed"),
            ],
            "evidence_required": [
                ("ERC", "Electrical-rule check and known waivers"),
                ("Trace links", "Each value tied to calculation/BOM/model evidence"),
            ],
            "artifact_outputs": [
                ("Schematic draft", "Hierarchical draft with test/debug hooks"),
                ("ERC checklist", "Open items before layout constraints"),
            ],
        },
        "pcb": {
            "gate_question": "Are layout constraints generated before any autoroute/manufacturing package?",
            "gate_checks": [
                ("Hot loops", "Primary switching loop, RCD loop, secondary SR high-current loop"),
                ("Quiet area", "Controller small-signal ground and TL431/opto feedback kept away from switching loops"),
                ("Safety/EMI", "HV/LV keepout, isolation slot, Y-cap path, EMI filter input/output separation, thermal copper"),
            ],
            "editable_fields": [
                ("Net classes", "HV, primary power, secondary power, feedback, safety-critical nets"),
                ("Keepouts", "Creepage/clearance, slots, copper pour limits, test point spacing"),
            ],
            "evidence_required": [
                ("Layout checklist", "Placement screenshots or EDA rule export before DRC"),
                ("DRC/ERC", "Tool-verified constraints and waivers"),
            ],
            "artifact_outputs": [
                ("PCB constraint package", "KiCad/Altium rule intent and placement guide"),
                ("Layout review checklist", "Loops, return paths, isolation, thermal, EMI"),
            ],
        },
        "bom": {
            "gate_question": "Is the BOM an AVL with orderable evidence, derating, and safety certifications?",
            "gate_checks": [
                ("Columns", "Designator, value, package, tolerance, ratings, temperature, certification, MPN, alternate, lifecycle, source, risk, cost"),
                ("Safety parts", "X/Y caps, fuse, optocoupler, transformer insulation certified/inspectable"),
                ("Power derating", "MOSFET/SR/RCD/capacitor stress ratio and thermal/lifetime evidence"),
            ],
            "editable_fields": [
                ("Approved MPNs", "Main and alternate source/manufacturer parts"),
                ("Manual sign-off", "Custom transformer and mains EMI parts cannot be fake orderable parts"),
            ],
            "evidence_required": [
                ("Source links", "Datasheet/distributor/vendor evidence for every critical part"),
                ("Magnetics drawing", "Separate manufacturing and inspection spec"),
            ],
            "artifact_outputs": [
                ("BOM/AVL table", "Orderable primary + alternate MPNs"),
                ("Derating matrix", "Stress versus rating with linked calculations/simulation"),
            ],
        },
        "test": {
            "gate_question": "Can an EVT engineer reproduce the validation and compare simulation versus bench data?",
            "gate_checks": [
                ("Bench matrix", "Startup, efficiency, ripple, VDS, thermal, load transient, short, EMI pre-scan, Hi-Pot"),
                ("Instrument setup", "AC source, e-load, power analyzer, HV differential probe, LISN, thermal camera/chamber"),
                ("Screenshot templates", "VDS, CS, SR gate/drain, Vout ripple, startup, load transient"),
            ],
            "editable_fields": [
                ("Pass criteria", "Limits, margins, conditions, owner, evidence file path"),
                ("Probe notes", "Bandwidth limit, spring ground, HV diff probe safety setup"),
            ],
            "evidence_required": [
                ("Sim vs measured", "Columns for target, simulation, measured, delta, pass/fail"),
                ("Safety setup", "Hi-Pot and hazardous voltage procedure reference"),
            ],
            "artifact_outputs": [
                ("EVT/DVT plan", "Bench matrix and waveform capture templates"),
                ("Release open items", "Physical assumptions that must be measured"),
            ],
        },
        "release": {
            "gate_question": "Is every release claim backed by calculated, simulated, datasheet, or measured evidence?",
            "gate_checks": [
                ("Artifacts", "Schematic, BOM/AVL, transformer manual, calculations, simulation files, waveforms, thermal/loss, loop, EMI, PCB constraints, EVT plan, risks, CDRs"),
                ("Risk status", "assumed/calculated/simulated/datasheet-guaranteed/measured/closed separated"),
                ("No fake closure", "Transformer Llk/RCD/EMI/thermal/loop remain open until physical/tool evidence exists"),
            ],
            "editable_fields": [
                ("Risk disposition", "Keep EVT open item or close only with evidence file"),
                ("Release label", "Concept, EVT, DVT, manufacturing-ready"),
            ],
            "evidence_required": [
                ("Evidence index", "Each artifact links to source/tool/measurement"),
                ("Reviewer sign-off", "Independent risk/critic agent plus human approval"),
            ],
            "artifact_outputs": [
                ("Release package", "Pre-release package if measured evidence is missing"),
                ("Audit trail", "Versioned decisions and unresolved risks"),
            ],
        },
    }
    selected = {**base, **stage_map.get(current_stage, {})}

    def rows(values: list[tuple[str, str]]) -> list[Dict[str, Any]]:
        return [{"label": label, "value": value} for label, value in values]

    return {
        "gate_question": selected["gate_question"],
        "gate_checks": rows(selected["gate_checks"]),
        "editable_fields": rows(selected["editable_fields"]),
        "evidence_required": rows(selected["evidence_required"]),
        "artifact_outputs": rows(selected["artifact_outputs"]),
        "agent_trace": [
            {"label": "Orchestrator", "value": f"Built {title} scaffold from locked specs and current blackboard state."},
            {"label": "Critic policy", "value": "Release remains HOLD until evidence is tool-generated, sourced, or measured."},
        ],
    }


def _framework_specs_from_session(session: Dict[str, Any]) -> Dict[str, Any]:
    direct_specs = session.get("specifications") if isinstance(session.get("specifications"), dict) else {}
    if direct_specs:
        return direct_specs
    intake = session.get("requirements_intake") if isinstance(session.get("requirements_intake"), dict) else {}
    specs = intake.get("specs") if isinstance(intake.get("specs"), dict) else {}
    if specs:
        return specs
    return {
        "input_voltage_min": 85.0,
        "input_voltage_max": 265.0,
        "output_voltage": 12.0,
        "output_current": 5.0,
        "output_power": 60.0,
        "efficiency_115vac_full_load": 0.88,
        "efficiency_230vac_full_load": 0.89,
        "efficiency_target": 0.88,
        "max_ripple_mvpp": 120.0,
        "max_ripple_voltage": 0.12,
        "ambient_c_max": 50.0,
        "emi_target": "CISPR 32 Class B pre-compliance",
        "isolation": "reinforced isolation assumption",
    }


def _recognized_specs_from_specs(specs: Dict[str, Any]) -> list[Dict[str, Any]]:
    vin_min = specs.get("input_voltage_min", 85)
    vin_max = specs.get("input_voltage_max", 265)
    vout = specs.get("output_voltage", 12)
    iout = specs.get("output_current", 5)
    pout = specs.get("output_power") or (float(vout) * float(iout) if vout and iout else 60)
    ripple_mv = specs.get("max_ripple_mvpp")
    if ripple_mv is None and specs.get("max_ripple_voltage") is not None:
        ripple_mv = float(specs.get("max_ripple_voltage") or 0) * 1000
    return [
        {"label": "Input", "value": f"{vin_min:g}-{vin_max:g} Vac, 47-63 Hz", "status": "critical"},
        {"label": "Output", "value": f"{float(vout):g} V / {float(iout):g} A", "status": "critical"},
        {"label": "Power", "value": f"{float(pout):g} W continuous", "status": "critical"},
        {"label": "Efficiency", "value": f"{float(specs.get('efficiency_115vac_full_load') or specs.get('efficiency_target') or 0.88) * 100:.0f}%+ target", "status": "soft"},
        {"label": "Ripple", "value": f"<{float(ripple_mv or 120):g} mVp-p", "status": "soft"},
        {"label": "EMI", "value": specs.get("emi_target") or "CISPR 32 Class B pre-compliance", "status": "risk"},
        {"label": "Isolation", "value": specs.get("isolation") or "reinforced isolation assumption", "status": "critical"},
    ]


def _default_framework_assumptions() -> list[Dict[str, Any]]:
    return [
        {
            "label": "Safety direction",
            "value": "IEC/UL 62368-1 design direction. Sets spacing, insulation, hipot, and Y-cap review.",
            "status": "default",
        },
        {
            "label": "EMI target",
            "value": "CISPR 32 Class B pre-compliance. Creates filter, layout, and LISN pre-scan work.",
            "status": "default",
        },
        {
            "label": "Release policy",
            "value": "No final schematic before gated evidence. Keeps early outputs honest and reviewable.",
            "status": "policy",
        },
    ]


def _default_framework_topology_candidates(specs: Optional[Dict[str, Any]] = None) -> list[Dict[str, Any]]:
    power_label = _design_power_label(specs or {})
    return [
        {
            "label": "QR / valley flyback + SR",
            "tradeoff": f"Recommended first path. Balanced cost, EMI, and efficiency for {power_label}.",
            "status": "recommended",
        },
        {
            "label": "Fixed-frequency flyback",
            "tradeoff": "Backup path. Simpler, but EMI and losses can be harder.",
            "status": "backup",
        },
        {
            "label": "Active clamp flyback",
            "tradeoff": "Upgrade path. Higher efficiency with higher debug and BOM risk.",
            "status": "defer",
        },
    ]


def _framework_common_rows(session: Dict[str, Any]) -> Dict[str, Any]:
    intake = session.get("requirements_intake") if isinstance(session.get("requirements_intake"), dict) else {}
    specs = _framework_specs_from_session(session)
    return {
        "recognized_specs": intake.get("recognized_specs") or _recognized_specs_from_specs(specs),
        "assumptions": intake.get("assumptions") or _default_framework_assumptions(),
        "topology_candidates": intake.get("topology_candidates") or _default_framework_topology_candidates(specs),
        "decisions": list(intake.get("decisions") or [
            {
                "decision": "Keep release schematic blocked until gates close.",
                "reason": "Specs, magnetics, loop, EMI, safety, and layout must be traceable first.",
                "status": "policy_hold",
            },
            {
                "decision": "Prefer QR flyback for the first path.",
                "reason": f"It is a pragmatic {_design_power_label(specs)} cost/reliability candidate before active clamp complexity.",
                "status": "proposed",
            },
        ]),
        "risks": list(intake.get("risks") or [
            {"severity": "P0", "item": "Transformer leakage and thermal rise are physical assumptions until EVT or vendor data.", "status": "unverified"},
            {"severity": "P1", "item": "Loop stability and EMI remain gated evidence, not closed facts.", "status": "pending"},
        ]),
        "artifacts": list(intake.get("artifacts") or [
            {"label": "Spec Lock table", "status": "ready"},
            {"label": "Controller decision record", "status": "pending"},
            {"label": "Magnetics winding package", "status": "pending"},
            {"label": "Loop/EMI/thermal evidence", "status": "pending"},
            {"label": "Release package", "status": "blocked"},
        ]),
        "missing_inputs": intake.get("missing_inputs") or [
            "Exact safety standard/market, insulation class, pollution degree, altitude, and required hipot level.",
            "PCB area, height limit, enclosure/open-frame condition, airflow, and connector constraints.",
            "BOM cost target, volume, approved vendors, and supply-chain restrictions.",
            "Hold-up time and brown-in/brown-out behavior.",
        ],
    }


def _build_engineering_framework_stage(command_id: str, session: Dict[str, Any]) -> Dict[str, Any]:
    specs = _framework_specs_from_session(session)
    common = _framework_common_rows(session)
    pout = float(specs.get("output_power") or 60.0)
    eta_low = float(specs.get("efficiency_115vac_full_load") or specs.get("efficiency_target") or 0.88)
    vac_max = float(specs.get("input_voltage_max") or 265.0)
    vac_min = float(specs.get("input_voltage_min") or 85.0)
    vout = float(specs.get("output_voltage") or 12.0)
    iout = float(specs.get("output_current") or 5.0)
    vbulk_max = round(vac_max * math.sqrt(2), 1)
    pin_max = round(pout / max(eta_low, 0.01), 1)
    bulk_min = round(2.0 * pout)
    bulk_max = round(3.0 * pout)
    power_label = _design_power_label(specs)

    stage_by_command = dict(FRAMEWORK_STAGE_SEQUENCE)
    current_stage = stage_by_command.get(command_id, "spec")
    state = session.get("engineering_framework") if isinstance(session.get("engineering_framework"), dict) else {}
    reached_commands = list(dict.fromkeys((state.get("reached_commands") or []) + [command_id]))
    reached_stages = list(dict.fromkeys([stage_by_command.get(cmd, "spec") for cmd in reached_commands]))

    sections: list[Dict[str, Any]] = []
    decisions = list(common["decisions"])
    risks = list(common["risks"])
    artifacts = list(common["artifacts"])
    title = "Engineering Framework"
    summary = "Framework stage completed."
    node = "requirements"
    release_blocked = True

    if command_id in {"ADOPT_DEFAULT_ASSUMPTIONS", "LOCK_CRITICAL_SPECS"}:
        title = "Spec Lock v0.1"
        summary = "Critical electrical specs are locked as v0.1; soft targets remain iterative and release remains blocked."
        node = "requirements"
        sections = [
            {"title": "Spec Lock Table", "items": common["recognized_specs"]},
            {"title": "Critical Gate Policy", "items": [
                {"label": "Allowed", "value": "Topology/controller/power-stage exploration"},
                {"label": "Blocked", "value": "Release schematic, final BOM, manufacturing package"},
                {"label": "Version", "value": "Spec v0.1"},
            ]},
            {"title": "Still Open", "items": [{"label": f"Open {idx + 1}", "value": item} for idx, item in enumerate(common["missing_inputs"][:8])]},
        ]
        decisions.append({"decision": "Lock critical specs v0.1", "reason": "Vin, Vout/Iout, power, isolation assumption, EMI target, ripple, efficiency, and ambient are sufficient for first-pass topology work.", "status": "accepted_v0.1"})
        artifacts = _upsert_artifact(artifacts, "Spec Lock table", "ready")

    elif command_id == "GENERATE_TOPOLOGY_CANDIDATES":
        title = "Topology Candidate Gate"
        summary = "Generated three explicit topology paths with trade-offs; QR flyback is recommended but not final until user confirmation."
        node = "designer"
        sections = [
            {"title": "Topology Candidates", "items": [
                {"label": "A. QR/valley flyback + opto CV", "value": f"Recommended for {power_label} cost/reliability; balanced switching loss and EMI; rectifier choice, loop, clamp, magnetics, and EMI still need validation."},
                {"label": "B. Fixed-frequency DCM/CCM flyback + opto CV", "value": "Backup; lower conceptual complexity but more switch loss/EMI/light-load risk."},
                {"label": "C. Active clamp flyback + SR", "value": "Higher density/efficiency path; higher BOM/control/debug risk; defer for first revision."},
            ]},
            {"title": "Gate Rule", "items": [
                {"label": "Before controller selection", "value": "User must confirm main topology and keep backup paths traceable."},
            ]},
        ]
        decisions.append({"decision": "Recommend QR flyback + SSR as first path", "reason": f"QR keeps the first {power_label} revision lower risk than active clamp; output rectification remains a cost/efficiency/thermal trade-off unless explicitly locked.", "status": "proposed"})
        artifacts = _upsert_artifact(artifacts, "Topology trade-off", "ready")

    elif command_id == "CONFIRM_QR_FLYBACK":
        title = "Topology Lock v0.1"
        summary = "Main topology locked as QR/valley flyback with TL431/opto feedback; rectifier choice and backup paths remain traceable."
        node = "designer"
        sections = [
            {"title": "Locked Topology", "items": [
                {"label": "Primary path", "value": "QR / valley-switching isolated flyback"},
                {"label": "Feedback", "value": "Secondary-side regulation, TL431 + optocoupler"},
                {"label": "Output rectification", "value": "Schottky versus synchronous rectification remains an explicit efficiency/cost/thermal trade-off unless SR is user-locked."},
                {"label": "Backups", "value": "Fixed-frequency flyback; active clamp flyback only if efficiency/thermal targets force it"},
            ]},
        ]
        decisions.append({"decision": "Lock topology: QR Flyback + SSR", "reason": f"Best first-revision trade-off for {power_label}, cost, reliability, and debug risk.", "status": "accepted_v0.1"})

    elif command_id == "GENERATE_CONTROLLER_CANDIDATES":
        title = "Controller Candidate Gate"
        summary = "Generated controller/reference design candidates with fit, evidence needs, and risk notes."
        node = "designer"
        sections = [
            {"title": "Controller Candidates", "items": [
                {"label": "TI UCC28740 + external 800 V MOSFET + SR controller", "value": "Primary candidate; universal AC documentation, opto CV path, custom MOSFET margin; requires external power-stage and loop tuning."},
                {"label": "Infineon QR CoolSET / QR controller family", "value": f"Backup; good QR application notes and protection integration; power/device fit must be checked for {power_label}."},
                {"label": "Power Integrations InnoSwitch class", "value": "Fastest reference-design path; high integration and complete reports; BOM/ecosystem lock-in risk."},
            ]},
            {"title": "Fit Checklist", "items": [
                {"label": "Must compare", "value": "Power range, VDS margin, startup/VDD, standby, SR support, model availability, EVM/calculator, supply-chain state."},
            ]},
        ]
        decisions.append({"decision": "Controller not locked yet", "reason": "Candidate fit must be recorded before selecting a controller.", "status": "review"})
        artifacts = _upsert_artifact(artifacts, "Controller decision record", "pending")

    elif command_id == "LOCK_CONTROLLER_UCC28740":
        title = "Controller Decision Record v0.1"
        summary = "TI UCC28740 route locked as primary; Infineon and PI remain backups."
        node = "designer"
        sections = [
            {"title": "Decision Record", "items": [
                {"label": "Decision", "value": "Use TI UCC28740 route as primary controller path"},
                {"label": "Power switch", "value": "External 800 V MOSFET evaluation footprint"},
                {"label": "Feedback", "value": "TL431 + optocoupler secondary-side CV"},
                {"label": "Backups", "value": "Infineon QR CoolSET/controller; PI InnoSwitch class"},
                {"label": "Validation", "value": "Loop stability, startup/VDD, standby power, SR timing, and model availability remain gated."},
            ]},
        ]
        decisions.append({"decision": "Use TI UCC28740 as primary controller route", "reason": "Good universal-AC fit and custom MOSFET margin; preserves backup options.", "status": "accepted_v0.1"})
        artifacts = _upsert_artifact(artifacts, "Controller decision record", "ready")

    elif command_id == "ENTER_POWER_STAGE_CALC":
        title = "Power Stage Calculation Notebook v0.1"
        summary = "Created the auditable first-pass power-stage calculation structure; values are calculation targets, not release evidence."
        node = "designer"
        sections = [
            {"title": "Input / Bulk / Power", "items": [
                {"label": "Vbulk,max", "value": f"{vac_max} Vac * sqrt(2) = {vbulk_max} V before rectifier details"},
                {"label": "Pin,max", "value": f"{pout} W / {eta_low:.2f} = {pin_max} W"},
                {"label": "Bulk capacitor heuristic", "value": f"Infineon-style 2-3 uF/W range: {bulk_min}-{bulk_max} uF for {pout:g} W"},
                {"label": "Vbulk,min", "value": "Requires low-line ripple and hold-up target; keep as open calculation cell until hold-up is specified."},
            ]},
            {"title": "Required Calculation Cells", "items": [
                {"label": "Transformer", "value": "VR, Dmax, Lp, Ipk, Irms, Np/Ns/Naux, Bmax"},
                {"label": "Stress", "value": "MOSFET VDS = Vbulk,max + VR + leakage spike + margin; SR FET reverse voltage"},
                {"label": "Sense/Output", "value": "Current sense resistor, output capacitor ripple/RMS, OVP and auxiliary winding targets"},
            ]},
        ]
        artifacts = _upsert_artifact(artifacts, "Power-stage calculation notebook", "ready")
        risks.append({"severity": "P1", "item": "Vbulk,min and hold-up remain open until hold-up time/bulk ripple are specified.", "status": "calculated_open"})

    elif command_id == "RUN_VR_SCAN":
        title = "Reflected Voltage Scan"
        summary = "Scanned VR options qualitatively for MOSFET stress, SR stress, duty ratio, transformer turns, and clamp pressure."
        node = "designer"
        sections = [
            {"title": "VR Trade-off", "items": [
                {"label": "90 V", "value": "Lower MOSFET stress; higher secondary/SR stress and larger duty variation; feasible."},
                {"label": "100 V", "value": "Balanced MOSFET/SR/clamp/magnetics trade-off; recommended first-pass target."},
                {"label": "110 V", "value": "Improves low-line duty/magnetics slightly; raises primary VDS and snubber pressure; feasible."},
                {"label": "120 V", "value": "Potentially smaller magnetics; higher VDS/clamp/EMI risk; use cautiously."},
            ]},
            {"title": "VDS Rule", "items": [
                {"label": "Target", "value": "800 V MOSFET; high-line full-load measured VDS spike must keep at least 15% margin to rating."},
                {"label": "Initial clamp target", "value": "Keep VDS peak below 680 V until EVT leakage data updates RCD/TVS values."},
            ]},
        ]
        decisions.append({"decision": "Use VR=100 V as first-pass target", "reason": "Balanced stress and magnetics trade-off; easier to keep 800 V MOSFET margin.", "status": "proposed"})

    elif command_id == "GENERATE_MAGNETICS_CANDIDATES":
        title = "Magnetics Candidate Gate"
        summary = "Generated transformer core and winding-package framework; EER28-class is primary for reliability and temperature margin."
        node = "magnetics_advisor"
        sections = [
            {"title": "Core Candidates", "items": [
                {"label": "EE25/EER25", "value": "Small and low cost; higher temperature and leakage/manufacturing risk; backup only."},
                {"label": "EER28/EF28", "value": "Recommended first path; moderate size/cost with better thermal and manufacturability margin."},
                {"label": "EFD30/PQ26", "value": "Lower thermal/leakage risk and flatter option; higher cost/space impact."},
            ]},
            {"title": "Winding Package Required", "items": [
                {"label": "Electrical", "value": "Lp +/-10%, Np/Ns/Naux, target AL/gap, Bmax, copper/core loss, DCR, temperature rise"},
                {"label": "Manufacturing", "value": "Winding order, triple-insulated wire decision, shield, insulation tape, pins, hipot, turns ratio, Llk <3-5% Lp target"},
                {"label": "Risk", "value": "Transformer leakage inductance remains EVT-measured; do not close RCD values until first samples."},
            ]},
        ]
        artifacts = _upsert_artifact(artifacts, "Magnetics winding package", "pending")
        risks.append({"severity": "P0", "item": "Magnetics package is not final until vendor bobbin/core data and leakage samples exist.", "status": "unverified"})

    elif command_id == "GENERATE_DEVICE_CARDS":
        title = "Device Engineering Cards"
        summary = "Generated the required engineering-card schema for MOSFET, SR FET, bridge, bulk capacitor, sense resistor, and alternates."
        node = "selector"
        sections = [
            {"title": "Primary MOSFET Card", "items": [
                {"label": "Required fields", "value": "800 V VDS, RDS(on) at temperature, Qg, Coss/Eoss, avalanche, package Rtheta, conduction/switching loss, footprint options TO-220F/TO-252, alternates"},
            ]},
            {"title": "SR FET Card", "items": [
                {"label": "Required fields", "value": "100 V initial class, RDS(on), Qg, body diode, controller compatibility, reverse-current risk, layout notes, alternates"},
            ]},
            {"title": "Bulk Capacitor Card", "items": [
                {"label": "Required fields", "value": f"105 C, {bulk_min}-{bulk_max} uF design range, voltage/ripple/lifetime, high-temperature life calculation, supplier alternates"},
            ]},
        ]
        artifacts = _upsert_artifact(artifacts, "BOM/AVL", "pending")

    elif command_id == "DESIGN_SNUBBER_CLAMP":
        title = "RCD / TVS Clamp Gate"
        summary = "Created clamp design framework with leakage-energy basis, VDS stack-up, RCD loss, thermal risk, and EMI trade-off."
        node = "designer"
        sections = [
            {"title": "Clamp Slider Targets", "items": [
                {"label": "580 V", "value": "Very conservative VDS; highest RCD loss and thermal penalty."},
                {"label": "620 V", "value": "Conservative first EVT target; higher loss but strong margin."},
                {"label": "660 V", "value": "Balanced target if RCD temperature is high."},
                {"label": "680 V", "value": "Upper first-pass VDS target; requires waveform evidence and 15% 800 V margin."},
            ]},
            {"title": "Required Evidence", "items": [
                {"label": "Formula", "value": "Leakage energy from assumed Llk = 3% Lp, Ipk, switching frequency, RCD clamp voltage"},
                {"label": "Layout", "value": "RCD loop shortest path; TVS footprint reserved; VDS test point required"},
            ]},
        ]
        risks.append({"severity": "P1", "item": "RCD values cannot close until Llk is measured on transformer samples.", "status": "assumed"})

    elif command_id == "DESIGN_LOOP_COMPENSATION":
        title = "Feedback Loop Compensation Gate"
        summary = "Created TL431/opto loop framework with Bode targets, CTR corners, and small-signal model evidence requirements."
        node = "designer"
        sections = [
            {"title": "Loop Requirements", "items": [
                {"label": "Feedback", "value": "TL431 + optocoupler secondary-side regulation"},
                {"label": "Target crossover", "value": "1-2 kHz initial target; must stay below optocoupler pole/corner limitations"},
                {"label": "Margins", "value": "Phase margin >60 degrees, gain margin checked at 85/115/230 Vac and 10/50/100% load"},
                {"label": "Corners", "value": "Optocoupler CTR min/typ/max and aging; TL431 bias versus standby power"},
            ]},
            {"title": "Outputs", "items": [
                {"label": "Model", "value": "Power-stage small-signal model, TL431 compensation, optocoupler pole"},
                {"label": "Evidence", "value": "Bode, PM/GM, transient response, component sensitivity, and causal tuning notes"},
            ]},
        ]
        artifacts = _upsert_artifact(artifacts, "Loop/EMI/thermal evidence", "pending")

    elif command_id == "RUN_SYSTEM_VALIDATION":
        title = "PLECS Validation Matrix Gate"
        summary = "Created the 09 Simulation checklist and PLECS run scope; no simulator evidence is claimed until the PLECS run button is used."
        node = "simulator"
        transient_load = f"{max(0.1, 0.25 * iout):g} A <-> {iout:g} A"
        sections = [
            {"title": "PLECS Matrix", "items": [
                {"label": "Run location", "value": "09 PLECS Simulation node"},
                {"label": "Steady-state", "value": "85/115/230/265 Vac, full load and load corners"},
                {"label": "Startup/line/load", "value": f"Startup low/high line, {transient_load} transient, line transient 85->265 Vac, brown-in/out"},
                {"label": "Stress", "value": "Drain waveform/VDS spike, current sense waveform, SR timing/reverse current, OCP/short"},
                {"label": "Reports", "value": "Ripple, loss breakdown, thermal estimate, conducted EMI pre-check 150 kHz-30 MHz"},
            ]},
            {"title": "Dashboard Targets", "items": [
                {"label": "VDS peak", "value": "<680 V first-pass"},
                {"label": "Vout ripple", "value": "<120 mVp-p"},
                {"label": "Transient", "value": "+/-3% recovery"},
                {"label": "Phase margin", "value": ">60 degrees"},
                {"label": "Transformer rise", "value": "<60 K target until refined"},
            ]},
        ]
        risks.append({"severity": "P1", "item": "PLECS matrix is a planned artifact until the 09 Simulation run attaches waveforms, model versions, and pass/fail data.", "status": "planned"})

    elif command_id == "GENERATE_LOCAL_FIX_OPTIONS":
        title = "Local Correction Options"
        summary = "Generated local, reversible correction framework; no full schematic rewrite is allowed for first-order warnings."
        node = "correction"
        sections = [
            {"title": "Correction Options", "items": [
                {"label": "A. Compensation tune", "value": "Improve transient and phase margin; no BOM cost impact; rerun Bode and transient."},
                {"label": "B. Lower-RDS(on) SR FET", "value": "Efficiency +0.3-0.6% potential, lower heat, higher BOM cost; keep footprint-compatible."},
                {"label": "C. Lp/VR adjustment", "value": "Can improve stress/efficiency but touches magnetics globally; defer unless A/B fail."},
            ]},
            {"title": "Rule", "items": [
                {"label": "Locality", "value": "Apply A+B first; do not move magnetics unless smaller changes fail."},
            ]},
        ]

    elif command_id == "GENERATE_SCHEMATIC_DRAFT":
        title = "Hierarchical Schematic Draft Gate"
        summary = "Generated schematic block framework and required debug reservations; still not a release schematic."
        node = "designer"
        sections = [
            {"title": "Required Schematic Sheets", "items": [
                {"label": "Input", "value": "Fuse, MOV, NTC/inrush, X-cap discharge, EMI filter, bridge, bulk"},
                {"label": "Primary", "value": "Controller startup/VDD, MOSFET, current sense, RCD/TVS clamp, transformer primary/aux"},
                {"label": "Secondary", "value": "SR controller/FET, output filter, TL431/opto feedback, OVP/OCP/OTP, test points"},
                {"label": "Safety", "value": "Isolation boundary, HV/LV net classes, Y-cap, creepage/clearance annotations"},
            ]},
            {"title": "Debug Reservations", "items": [
                {"label": "Must reserve", "value": "RCD R/C parallel pads, gate resistor + diode, SR D-S snubber, output pi filter, opto RC network, CS RC filter, VDS/CS/VOUT/FB/SR gate test points"},
            ]},
        ]
        artifacts = _upsert_artifact(artifacts, "Hierarchical schematic draft", "pending")

    elif command_id == "GENERATE_PCB_CONSTRAINTS":
        title = "PCB Constraint Gate"
        summary = "Generated layout-rule framework; no automatic final routing is allowed."
        node = "designer"
        sections = [
            {"title": "Placement / Loop Constraints", "items": [
                {"label": "Primary hot loop", "value": "Bulk cap - transformer primary - MOSFET - current sense return must be shortest and tight"},
                {"label": "Clamp loop", "value": "RCD/TVS loop tight to MOSFET/primary winding"},
                {"label": "Secondary current loop", "value": "Transformer secondary - SR FET - output caps tight and low inductance"},
                {"label": "Quiet area", "value": "Controller signal ground, TL431/opto compensation, auxiliary sensing separated from hot loops"},
            ]},
            {"title": "Safety / EMI Constraints", "items": [
                {"label": "Isolation", "value": "HV/LV keepout, creepage/clearance, optional slot, Y-cap return path"},
                {"label": "EMI", "value": "Input/output EMI filter separation, CM/DM paths, thermal copper without crossing isolation"},
            ]},
        ]
        artifacts = _upsert_artifact(artifacts, "PCB constraint package", "pending")

    elif command_id == "GENERATE_BOM_AVL":
        title = "BOM / AVL Gate"
        summary = "Generated BOM/AVL schema with required alternates, derating, source evidence, and safety certifications."
        node = "selector"
        sections = [
            {"title": "BOM Columns", "items": [
                {"label": "Core fields", "value": "Designator, value, package, tolerance, voltage/current/power rating, temperature, safety certification"},
                {"label": "Sourcing", "value": "Manufacturer MPN, approved alternate, lifecycle, source link, sourcing risk, cost tier"},
                {"label": "Engineering evidence", "value": "Derating margin, selected stress/calculation link, reason for selection, replacement constraints"},
            ]},
            {"title": "Special Rules", "items": [
                {"label": "Safety parts", "value": "X/Y caps, fuse, optocoupler, transformer insulation must show certification/hipot basis"},
                {"label": "Magnetics", "value": "Separate drawing and inspection spec, not just a BOM row"},
            ]},
        ]
        artifacts = _upsert_artifact(artifacts, "BOM/AVL", "pending")

    elif command_id == "GENERATE_EVT_TEST_PLAN":
        title = "EVT / DVT Test Plan Gate"
        summary = "Generated bench validation matrix with instruments, conditions, pass criteria, and measured-vs-simulated columns."
        node = "validator"
        sections = [
            {"title": "Bench Matrix", "items": [
                {"label": "Startup", "value": "85/115/230/265 Vac, no/full load, AC source + scope, stable Vout/no abnormal events"},
                {"label": "Efficiency", "value": "85/115/230/265 Vac, power analyzer, meet 88/89% targets where specified"},
                {"label": "Ripple", "value": "Full load, bandwidth limited, spring ground probe, <120 mVp-p"},
                {"label": "VDS/CS/SR gate", "value": "HV differential probe/current sense probing notes, screenshot templates"},
                {"label": "Thermal/EMI/Safety", "value": "50 C ambient, thermal camera/chamber, LISN 150 kHz-30 MHz, Hi-Pot per target standard"},
            ]},
            {"title": "Traceability", "items": [
                {"label": "Columns", "value": "Target, simulation value, measured value, delta, pass/fail, owner, evidence file"},
            ]},
        ]
        artifacts = _upsert_artifact(artifacts, "EVT/DVT test plan", "ready")

    elif command_id == "RUN_FINAL_DESIGN_REVIEW":
        title = "Final Design Review Gate"
        summary = "Generated release review checklist and kept unverified physical assumptions open."
        node = "validator"
        sections = [
            {"title": "Review Checklist", "items": [
                {"label": "Stress", "value": "MOSFET VDS margin, SR VDS margin, RCD loss/temperature, current sense stress"},
                {"label": "Magnetics", "value": "Lp/Llk tolerance in simulation, temperature rise, insulation, hipot, leakage measured"},
                {"label": "Loop", "value": "Optocoupler CTR aging, TL431 bias/standby conflict, phase/gain margin"},
                {"label": "BOM/Layout", "value": "Bulk capacitor lifetime, EMI debug reserve, safety spacing, isolation slot, orderable AVL"},
                {"label": "Evidence type", "value": "assumed / calculated / simulated / datasheet-guaranteed / measured / closed"},
            ]},
        ]
        risks.append({"severity": "P0", "item": "Unverified physical assumption: transformer leakage inductance; keep RCD update as EVT task.", "status": "assumed"})

    elif command_id == "GENERATE_RELEASE_PACKAGE":
        title = "Release Package Gate"
        summary = "Generated release-package checklist. Package remains EVT/pre-release unless measured evidence is attached."
        node = "reporter"
        release_blocked = True
        sections = [
            {"title": "Required Release Artifacts", "items": [
                {"label": "1", "value": "Hierarchical schematic with input, EMI, primary, transformer, SR, feedback, protection, test points, isolation boundary"},
                {"label": "2", "value": "BOM/AVL with primary + alternate MPNs, derating, certification, source risk, cost tier"},
                {"label": "3", "value": "Transformer winding manual with core, bobbin, gap, turns, wire, insulation, pins, Lp/Llk/DCR/ratio/hipot inspection"},
                {"label": "4", "value": "Calculation notebook: bulk, VR, Lp, Ipk/Irms, VDS/SR stress, RCD, output cap, thermal, efficiency"},
                {"label": "5", "value": "Simulation project files, waveform report, loss/thermal, loop stability, EMI pre-compliance, PCB constraints"},
                {"label": "6", "value": "EVT/DVT test plan, risk register, and versioned decision records"},
            ]},
            {"title": "Release State", "items": [
                {"label": "Current state", "value": "Framework complete, EVT/pre-release hold"},
                {"label": "Reason", "value": "Measured transformer leakage, loop, EMI, thermal, and bench evidence are not attached in this demo path."},
            ]},
        ]
        artifacts = _upsert_artifact(artifacts, "Release package", "blocked")

    stage_status = _framework_stage_status(current_stage, reached_stages, release_blocked=release_blocked)
    calculations = {
        "vbulk_max_v": vbulk_max,
        "pin_max_w": pin_max,
        "bulk_cap_heuristic_uf": [bulk_min, bulk_max],
        "vac_min": vac_min,
        "vac_max": vac_max,
        "vout": vout,
        "iout": iout,
    }
    base_options, base_commands = _framework_next_actions(command_id)
    decision_options = _framework_decision_options(command_id, current_stage, base_options, base_commands)
    options = [row.get("option", "") for row in decision_options]
    commands = [row.get("command", "") for row in decision_options]
    checkpoint_details = _framework_stage_review_details(command_id, current_stage, title, specs, calculations)
    checkpoint_details = _augment_framework_checkpoint_details(
        checkpoint_details,
        command_id,
        current_stage,
        title,
        summary,
        sections,
        calculations,
        decision_options,
    )
    return {
        "command": command_id,
        "node": node,
        "stage": current_stage,
        "title": title,
        "summary": summary,
        "sections": sections,
        "options": options,
        "commands": commands,
        "decision_options": decision_options,
        "checkpoint_details": checkpoint_details,
        "framework": {
            "current_stage": current_stage,
            "reached_commands": reached_commands,
            "reached_stages": reached_stages,
            "stage_status": stage_status,
            "release_blocked": release_blocked,
        },
        "design_meta": {
            "mode": "engineering_framework",
            "status": "waiting_user",
            "framework_current_stage": current_stage,
            "framework_stage_status": stage_status,
            "specs": specs,
            "recognized_specs": common["recognized_specs"],
            "assumptions": common["assumptions"],
            "topology_candidates": common["topology_candidates"],
            "decisions": decisions,
            "risks": risks,
            "artifacts": artifacts,
            "calculations": calculations,
            "decision_options": decision_options,
            "checkpoint_details": checkpoint_details,
            "stage_summary": checkpoint_details.get("stage_summary") or [],
            "stage_story": {
                "title": title,
                "stage": current_stage,
                "summary": summary,
                "what_happened": checkpoint_details.get("what_happened") or [],
                "why_this_matters": checkpoint_details.get("why_this_matters") or [],
                "evidence_state": checkpoint_details.get("evidence_state") or [],
                "missing_evidence": checkpoint_details.get("missing_evidence") or [],
                "next_actions": checkpoint_details.get("next_actions") or [],
            },
        },
    }


def _upsert_artifact(rows: list[Dict[str, Any]], label: str, status: str) -> list[Dict[str, Any]]:
    out = []
    found = False
    for row in rows or []:
        if str(row.get("label") or "").lower() == label.lower():
            new_row = dict(row)
            new_row["status"] = status
            out.append(new_row)
            found = True
        else:
            out.append(row)
    if not found:
        out.append({"label": label, "status": status})
    return out


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


def _mentions_flyback(text: str) -> bool:
    low = str(text or "").lower()
    return any(term in low for term in ["flyback", "flybakc", "flayback", "flaybakc"]) or "反激" in str(text or "")


def _looks_like_capability_question(user_query: str) -> bool:
    q = str(user_query or "").strip()
    low = q.lower()
    english_markers = [
        "what can you do", "who are you", "help me", "your capabilities", "capability", "capabilities",
        "hello", "hi",
    ]
    chinese_markers = [
        "你好", "您好", "你是谁", "你能做什么", "你能干什么", "能做什么", "能干什么",
        "可以做什么", "会做什么", "怎么帮", "能帮我什么",
    ]
    return any(marker in low for marker in english_markers) or any(marker in q for marker in chinese_markers)


def _capability_answer(user_query: str) -> str:
    if _contains_chinese(user_query):
        return (
            "你好，我是 PE-MAS Studio，一个面向电力电子设计的多智能体工作台。\n\n"
            "我主要能做四类事：\n"
            "1. 解释电源拓扑和工程概念：例如 Flyback、LLC、CCM/DCM、RHPZ、Vds 应力、纹波和 EMI。\n"
            "2. 做完整电源设计流程：从需求提取、理论参数、磁件建议、器件选型、仿真到设计审查和报告。\n"
            "3. 分析和追问当前方案：比如为什么效率低、为什么 Vds 超限、怎么降低纹波、是否需要同步整流。\n"
            "4. 保持上下文对话：你可以先问“什么是 Flyback”，再问“有什么用”“什么时候出现的”“怎么设计”，我会继承上一个主题。\n\n"
            "如果你只是问概念，我会快速回答；如果你给出设计指标，比如 `Design a 24W Flyback, 85-265Vac to 12V/2A, efficiency > 88%, ripple < 100mV`，我会进入完整 MAS 设计流程。"
        )

    return (
        "I am PE-MAS Studio, a multi-agent workspace for power-electronics design.\n\n"
        "I can help with four main tasks:\n"
        "1. Explain power-converter concepts such as flyback, LLC, CCM/DCM, RHPZ, Vds stress, ripple, and EMI.\n"
        "2. Run an end-to-end design workflow: requirements, theoretical sizing, magnetics, component selection, simulation, review, and report generation.\n"
        "3. Analyze design results and tradeoffs, such as efficiency loss, voltage stress, thermal limits, ripple, and compensation risk.\n"
        "4. Continue contextual Q&A, so follow-ups like “what is it used for?” or “when did it appear?” inherit the previous topic.\n\n"
        "For a full run, give design specs such as: `Design a 24W Flyback, 85-265Vac to 12V/2A, efficiency > 88%, ripple < 100mV`."
    )


def _looks_like_basic_flyback_question(user_query: str) -> bool:
    q = str(user_query or "").strip()
    if not _mentions_flyback(q):
        return False

    low = q.lower()
    engineering_review_terms = [
        "rcd", "snubber", "clamp", "vds", "v ds", "mosfet", "leakage", "spike",
        "margin", "derating", "release", "evidence", "review", "bode", "loop",
        "emi", "thermal", "magnetics", "transformer", "bom", "layout",
        "钳位", "吸收", "漏感", "尖峰", "裕量", "降额", "证据", "发布", "评审",
        "环路", "补偿", "磁件", "变压器", "热", "安规",
    ]
    if any(term in low or term in q for term in engineering_review_terms):
        return False

    english_markers = [
        "what is", "what's", "define", "explain", "introduction", "overview", "basics",
        "how does", "how it works",
    ]
    chinese_markers = [
        "什么是", "是什么", "是啥", "介绍", "解释", "原理", "怎么工作", "如何工作",
    ]
    if any(marker in low for marker in english_markers) or any(marker in q for marker in chinese_markers):
        return True

    compact = "".join(q.split())
    return len(compact) <= 48 and ("?" in compact or "？" in compact)


def _looks_like_flyback_clamp_review_question(user_query: str) -> bool:
    q = str(user_query or "").strip()
    if not q:
        return False
    low = q.lower()
    has_flyback_context = _mentions_flyback(q) or "offline" in low or "离线" in q or "反激" in q
    if not has_flyback_context:
        return False
    strong_terms = [
        "rcd", "snubber", "clamp", "vds", "v ds", "leakage", "spike", "mosfet",
        "derating", "margin", "release", "evidence",
        "钳位", "吸收", "漏感", "尖峰", "裕量", "降额", "发布", "证据",
    ]
    review_terms = ["review", "size", "sizing", "risk", "before release", "how should", "检查", "评审", "如何", "怎么"]
    strong_hits = sum(1 for term in strong_terms if term in low or term in q)
    return strong_hits >= 2 or (strong_hits >= 1 and any(term in low or term in q for term in review_terms))


def _flyback_clamp_review_sources() -> list[dict[str, Any]]:
    return [
        {
            "channel": "design_reference",
            "source": "Texas Instruments",
            "title": "UCC28740 flyback controller datasheet",
            "url": "https://www.ti.com/lit/gpn/ucc28740",
            "snippet": "Universal AC flyback controller reference; layout guidance emphasizes short high-current/high-frequency loops and HV/LV separation.",
            "status": "reference",
        },
        {
            "channel": "design_reference",
            "source": "PSMA / Würth Elektronik seminar",
            "title": "Design Space of Flyback Transformers",
            "url": "https://www.psma.com/sites/default/files/uploads/tech-forums-magnetics/presentations/is014-designspaceofflybacktransformers.pdf",
            "snippet": "Transformer reflected voltage, leakage inductance, MOSFET stress, and snubber behavior must be reviewed together.",
            "status": "reference",
        },
        {
            "channel": "design_reference",
            "source": "Texas Instruments",
            "title": "Conducted EMI specifications for power supplies",
            "url": "https://www.ti.com/lit/pdf/slyy136",
            "snippet": "Conducted EMI pre-compliance should include the 150 kHz to 30 MHz range and both quasi-peak and average limits.",
            "status": "reference",
        },
        {
            "channel": "release_evidence",
            "source": "PE-MAS engineering rule",
            "title": "Do not close RCD clamp until transformer Llk is measured",
            "url": "",
            "snippet": "First-pass RCD values may be calculated from assumed leakage inductance, but release evidence must include measured leakage and drain waveforms.",
            "status": "gate",
        },
    ]


def _flyback_general_sources(topic: str = "concept") -> list[dict[str, Any]]:
    base = [
        {
            "channel": "local_rag",
            "source": "Internal Knowledge Base",
            "title": "Internal Flyback Design Guide",
            "url": "",
            "snippet": "Local PE-MAS knowledge base entry for flyback operating principle, use cases, and main engineering risks.",
            "status": "retrieved",
        },
        {
            "channel": "design_reference",
            "source": "Texas Instruments",
            "title": "Common Mistakes in Flyback Power Supplies and How to Fix Them",
            "url": "https://www.ti.com/lit/pdf/slup392",
            "snippet": "Practical flyback supply design seminar covering common implementation risks and fixes.",
            "status": "reference",
        },
    ]
    if topic in {"concept", "use", "history"}:
        base.append(
            {
                "channel": "design_reference",
                "source": "Plexim",
                "title": "Modeling a Current-Controlled Flyback Converter using PLECS",
                "url": "",
                "snippet": "Application note illustrating flyback energy storage and transfer behavior in a switch-mode model.",
                "status": "reference",
            }
        )
    return base


def _flyback_general_evidence_cards(user_query: str, topic: str = "concept") -> list[dict[str, Any]]:
    english = not _contains_chinese(user_query)
    if english:
        cards = [
            {
                "title": "Energy storage and transfer",
                "source": "Topology model",
                "status": "concept",
                "detail": "A flyback stores energy in magnetizing inductance during switch on-time and transfers it to the secondary during off-time.",
            },
            {
                "title": "Isolation and low-to-mid-power fit",
                "source": "Application pattern",
                "status": "reference",
                "detail": "The topology is common for isolated adapters, standby supplies, and auxiliary supplies where simplicity and cost matter.",
            },
            {
                "title": "Engineering risks",
                "source": "Design review",
                "status": "review",
                "detail": "A serious review still needs VDS stress, leakage clamp, EMI, ripple, loop stability, thermal margin, safety spacing, and magnetics evidence.",
            },
        ]
        if topic == "history":
            cards[0] = {
                "title": "Historical context",
                "source": "Engineering terminology",
                "status": "context",
                "detail": "The term flyback originates from CRT retrace/flyback circuits; the SMPS topology matured with transistorized switching supplies.",
            }
        return cards
    cards = [
        {
            "title": "储能与传能机理",
            "source": "拓扑模型",
            "status": "concept",
            "detail": "Flyback 在开关导通时把能量存入磁化电感，关断时再通过次级释放到输出。",
        },
        {
            "title": "隔离与中小功率适配",
            "source": "应用模式",
            "status": "reference",
            "detail": "该拓扑常用于隔离适配器、待机电源和辅助电源，优势是结构简单和成本低。",
        },
        {
            "title": "工程风险",
            "source": "设计评审",
            "status": "review",
            "detail": "严肃评审仍需覆盖 VDS 应力、漏感钳位、EMI、纹波、环路稳定、热裕量、安规间距和磁件证据。",
        },
    ]
    if topic == "history":
        cards[0] = {
            "title": "历史上下文",
            "source": "工程术语",
            "status": "context",
            "detail": "Flyback 一词源于 CRT 回扫/回程电路；现代 SMPS 拓扑随晶体管开关电源成熟而普及。",
        }
    return cards


def _flyback_clamp_review_evidence_cards(user_query: str) -> list[dict[str, Any]]:
    english = not _contains_chinese(user_query)
    if english:
        return [
            {
                "title": "VDS stack-up",
                "source": "Power-stage calculation",
                "status": "calculated",
                "detail": "Review VDS_peak as Vbulk,max + reflected voltage + leakage spike. For 265 Vac, Vbulk,max is roughly 375 V before tolerance and surge considerations.",
            },
            {
                "title": "Leakage-energy basis",
                "source": "Magnetics and snubber model",
                "status": "assumed until EVT",
                "detail": "Use measured Llk when available. For first pass, assume a conservative leakage fraction such as 3 percent of Lp, then compute E_lk = 0.5 * Llk * Ipk^2.",
            },
            {
                "title": "Clamp target sweep",
                "source": "Design trade-off",
                "status": "review",
                "detail": "Compare clamp targets such as 580 V, 620 V, 660 V, and 680 V for MOSFET margin, RCD loss, temperature, and EMI.",
            },
            {
                "title": "Release evidence gate",
                "source": "Design review",
                "status": "required",
                "detail": "Before release, attach HV differential-probe drain waveforms, RCD temperature, measured transformer leakage, EMI pre-scan, layout loop review, and BOM derating.",
            },
        ]
    return [
        {
            "title": "VDS 叠加关系",
            "source": "功率级计算",
            "status": "calculated",
            "detail": "按 Vbulk,max + 反射电压 + 漏感尖峰审查 VDS_peak。265 Vac 高线整流后 Vbulk,max 约 375 V，仍需考虑容差与浪涌裕量。",
        },
        {
            "title": "漏感能量依据",
            "source": "磁件与吸收模型",
            "status": "assumed until EVT",
            "detail": "有样品后必须使用实测 Llk；首版可按 Lp 的 3% 等保守假设估算，并计算 E_lk = 0.5 * Llk * Ipk^2。",
        },
        {
            "title": "钳位目标扫描",
            "source": "设计取舍",
            "status": "review",
            "detail": "比较 580 V、620 V、660 V、680 V 等目标对 MOSFET 裕量、RCD 损耗、温升和 EMI 的影响。",
        },
        {
            "title": "发布证据门",
            "source": "设计评审",
            "status": "required",
            "detail": "Release 前必须附高压差分探头 VDS 波形、RCD 温度、变压器漏感实测、EMI 预扫、layout 回路审查和 BOM 降额证据。",
        },
    ]


def _flyback_clamp_review_answer(user_query: str) -> str:
    if _contains_chinese(user_query):
        return (
            "这个问题不能按“反激是什么”来回答。对离线 Flyback，RCD/snubber 的审查应作为一个发布前 Gate：先算应力，再用样机波形和磁件实测关闭风险。\n\n"
            "## 1. 先建立 MOSFET VDS 应力叠加\n"
            "`VDS_peak = Vbulk,max + VR + Vleakage_spike`\n\n"
            "- 265 Vac 高线整流后 `Vbulk,max` 约 375 V，实际还要考虑输入容差、浪涌和探头测量条件。\n"
            "- 60 W universal AC 第一版建议预留 800 V MOSFET footprint；如果目标是 15% 以上额定值裕量，则 800 V 器件的实测 `VDS_peak` 应控制在约 680 V 以下。\n"
            "- 650 V 器件不是永远不能用，但必须有实测漏感、RCD 温升、浪涌/高线满载波形和 EMI 证据支撑，不适合作为第一版保守默认。\n\n"
            "## 2. 扫描反射电压 VR，而不是拍脑袋取值\n"
            "至少比较 90/100/110/120 V：低 VR 降低一次侧 VDS 压力但增加次级应力和占空比压力；高 VR 改善低线占空比和磁件利用率，但提高 MOSFET/RCD/EMI 风险。60 W QR Flyback 初版通常把 100 V 作为折中点，再由仿真和实测收敛。\n\n"
            "## 3. 用漏感能量设计 RCD clamp\n"
            "首版如果没有样品，可按 `Llk = 3% * Lp` 做保守估算；有首批变压器后必须改成实测值。\n\n"
            "`E_lk = 0.5 * Llk * Ipk^2`\n"
            "`P_clamp ≈ E_lk * fsw`，再按实际钳位波形和 QR 频率范围修正。\n\n"
            "RCD 审查项：钳位目标 580/620/660/680 V 扫描、R/C 耐压和功耗降额、电阻温升、二极管反向恢复、clamp loop 是否最短、是否预留 TVS clamp footprint。\n\n"
            "## 4. RC snubber 只允许基于波形调试闭环\n"
            "MOSFET D-S 或次级 SR D-S snubber 要根据实测振铃频率、幅度和 EMI 预扫调整。起点可以用小电容加接近特征阻抗的阻值，但最终要检查 snubber 损耗、器件温升和效率损失。\n\n"
            "## 5. Release 前必须看到的证据\n"
            "- 高线满载、低线满载、启动、负载瞬态下的 VDS 波形，使用高压差分探头并标注带宽限制和接地方式。\n"
            "- CS、SR gate、SR drain、Vout ripple 的关键截图。\n"
            "- 变压器 Lp/Llk/DCR/turns ratio/Hipot 实测记录。\n"
            "- RCD 电阻、MOSFET、SR FET、变压器的满载温升。\n"
            "- LISN conducted EMI 150 kHz-30 MHz 预扫，记录 quasi-peak 和 average margin。\n"
            "- PCB hot loop、RCD loop、secondary current loop、isolation barrier 的 layout review 截图。\n"
            "- BOM derating：MOSFET VDS、SR VDS、RCD 电阻功率、电容耐压、安规件认证。\n\n"
            "Release 判定原则：如果漏感尖峰还只是 assumed/calculated，而没有 measured waveform 和 transformer Llk，RCD 风险只能标为 EVT open item，不能标成 closed。"
        )

    return (
        "This should not be answered as a generic flyback definition. For an offline flyback, the RCD/snubber review is a release gate: calculate the stress first, then close the risk with measured magnetics and bench waveforms.\n\n"
        "## 1. Build the MOSFET VDS stack-up\n"
        "`VDS_peak = Vbulk,max + VR + Vleakage_spike`\n\n"
        "- At 265 Vac, the rectified bulk rail is roughly 375 V before tolerance and surge considerations.\n"
        "- For a first 60 W universal-input design, reserve an 800 V MOSFET footprint. If the design rule is at least 15% margin to rating, measured high-line/full-load `VDS_peak` should stay below about 680 V.\n"
        "- A 650 V MOSFET may be possible in a mature cost-down design, but not without measured leakage, clamp temperature, high-line waveforms, surge margin, and EMI evidence.\n\n"
        "## 2. Sweep reflected voltage, do not choose it blindly\n"
        "Compare at least 90/100/110/120 V. Lower `VR` reduces primary MOSFET stress but pushes secondary stress and duty-cycle tradeoffs. Higher `VR` can improve low-line duty cycle and transformer utilization, but raises MOSFET, RCD-loss, and EMI risk. For a conservative first QR flyback pass, `VR ≈ 100 V` is a reasonable center point to verify.\n\n"
        "## 3. Size the RCD clamp from leakage energy\n"
        "Before transformer samples exist, assume a conservative leakage fraction such as `Llk = 3% * Lp`. Once samples arrive, replace it with measured leakage.\n\n"
        "`E_lk = 0.5 * Llk * Ipk^2`\n"
        "`P_clamp ≈ E_lk * fsw`, then correct it with the actual clamp waveform and QR operating frequency range.\n\n"
        "Review the RCD by sweeping clamp targets such as 580/620/660/680 V. For each point, record MOSFET margin, RCD resistor loss and temperature, diode stress/recovery, capacitor voltage rating, EMI effect, and whether the clamp loop is physically short. Keep a TVS clamp footprint if the first hardware needs faster containment.\n\n"
        "## 4. Treat RC snubbers as measured-waveform tuning parts\n"
        "MOSFET D-S and secondary SR D-S snubbers should be tuned from measured ringing frequency, amplitude, and EMI scan data. A first estimate can start with a small capacitor and a resistor near the ringing network impedance, but the final value must pass loss, temperature, efficiency, and EMI review.\n\n"
        "## 5. Evidence required before release\n"
        "- Drain VDS waveforms at high-line full-load, low-line full-load, startup, and load transient, captured with a HV differential probe and documented probe/bandwidth setup.\n"
        "- CS, SR gate, SR drain, and Vout ripple screenshots.\n"
        "- Transformer Lp/Llk/DCR/turns-ratio/Hipot measurements from real samples.\n"
        "- Thermal data for RCD resistor, MOSFET, SR FET, transformer, and clamp diode at worst load/ambient.\n"
        "- Conducted EMI pre-scan with LISN from 150 kHz to 30 MHz, including quasi-peak and average margins.\n"
        "- Layout review screenshots for the primary hot loop, RCD loop, secondary current loop, quiet feedback area, and isolation barrier.\n"
        "- BOM derating evidence for MOSFET VDS, SR FET VDS, RCD power, capacitor voltage, and safety-certified parts.\n\n"
        "Release rule: if leakage spike risk is still only assumed or calculated, the RCD clamp is not closed. Mark it as an EVT open item until measured transformer leakage and measured drain waveforms support the selected values."
    )


def _looks_like_contextual_followup(user_query: str) -> bool:
    q = str(user_query or "").strip()
    if not q or _mentions_flyback(q):
        return False
    low = q.lower()
    markers = [
        "他", "它", "这个", "这种", "那个", "该", "上面", "刚才", "前面",
        "啥时候", "什么时候", "何时", "何年", "哪年", "发现", "发明", "被发明", "起源", "历史", "为什么叫",
        "有什么用", "有啥用", "用途", "用处", "用来", "干嘛", "能干嘛", "应用", "场景",
        "it", "that", "this", "when was", "when did", "invented", "discovered", "origin", "history",
        "used for", "use case", "use cases", "application", "applications", "purpose",
    ]
    return len(q) <= 160 and any(marker in q or marker in low for marker in markers)


def _resolve_qa_context(
    user_query: str,
    session: Optional[Dict[str, Any]] = None,
    history: Optional[list[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    q = str(user_query or "").strip()
    rows = history if history is not None else _session_history(session)
    topic = _infer_session_topic(session, rows)
    if topic == "flyback" and _looks_like_contextual_followup(q):
        prefix = "关于 Flyback（反激变换器）的追问：" if _contains_chinese(q) else "Follow-up about flyback converter: "
        return {
            "effective_query": f"{prefix}{q}",
            "topic": "flyback",
            "context_used": True,
            "note": "Resolved follow-up pronoun from the current session topic: flyback.",
        }
    return {
        "effective_query": q,
        "topic": _infer_topic_from_text(q) or topic,
        "context_used": False,
        "note": "",
    }


def _looks_like_flyback_history_question(user_query: str) -> bool:
    q = str(user_query or "").strip()
    if not _mentions_flyback(q):
        return False
    low = q.lower()
    markers = [
        "history", "origin", "invent", "invented", "discover", "discovered", "when was", "when did",
        "历史", "起源", "发明", "被发明", "发现", "啥时候", "什么时候", "何时", "何年", "哪年", "是谁提出",
    ]
    return any(marker in low or marker in q for marker in markers)


def _looks_like_flyback_use_question(user_query: str) -> bool:
    q = str(user_query or "").strip()
    if not _mentions_flyback(q):
        return False
    low = q.lower()
    markers = [
        "used for", "use case", "use cases", "application", "applications", "purpose", "where is",
        "有什么用", "有啥用", "用途", "用处", "用来", "干嘛", "能干嘛", "应用", "场景", "用在哪里",
    ]
    return any(marker in low or marker in q for marker in markers)


def _flyback_use_answer(user_query: str) -> str:
    if _contains_chinese(user_query):
        return (
            "Flyback 的主要用途是把一个输入电源转换成隔离、稳定的直流输出，尤其适合中小功率电源。\n\n"
            "常见用途：\n"
            "1. 手机/路由器/小家电适配器：例如 5V、12V、24V 输出的小功率外置电源。\n"
            "2. 辅助电源：给 MCU、驱动芯片、继电器、通信模块等控制电路供电。\n"
            "3. 多路隔离输出：一个变压器可以做多个次级绕组，给不同电压轨供电。\n"
            "4. 高低压隔离场景：比如 85-265Vac 市电输入到安全低压输出。\n\n"
            "它的优势是结构简单、成本低、容易隔离；缺点是漏感尖峰、EMI、输出纹波、效率和热设计需要认真处理。一般 5W-75W 这类范围很常见，功率再高时通常会考虑 LLC、正激、半桥等拓扑。"
        )

    return (
        "A flyback converter is mainly used to convert an input supply into an isolated, regulated DC output, especially at low-to-mid power.\n\n"
        "Common use cases:\n"
        "1. Wall adapters and chargers: small 5 V, 12 V, or 24 V supplies.\n"
        "2. Auxiliary supplies: powering MCUs, gate drivers, relays, communication modules, and control circuits.\n"
        "3. Multiple isolated outputs: one transformer can provide several secondary windings.\n"
        "4. Offline AC-DC supplies: converting 85-265 Vac mains to a safe low-voltage output.\n\n"
        "Its strengths are simplicity, low cost, and easy isolation. Its weaknesses are leakage spikes, EMI, ripple, thermal stress, and efficiency limits. It is very common around 5-75 W; at higher power, topologies like LLC, forward, half-bridge, or full-bridge often become more attractive."
    )


def _flyback_history_answer(user_query: str) -> str:
    if _contains_chinese(user_query):
        return (
            "严格说，Flyback 不是某一天被“发现”的自然现象，而是从早期电视/CRT 的行扫描回扫电路逐步发展出来的工程拓扑。\n\n"
            "时间线可以这样理解：\n"
            "1. “flyback/retrace”这个概念来自 CRT 水平扫描结束后电子束快速回到起点的回扫过程，早期电视电路在 20 世纪中期已经大量使用 flyback transformer 来产生高压。\n"
            "2. 作为隔离型开关电源拓扑，现代 Flyback converter 主要是在功率晶体管、磁性元件和 PWM 控制技术成熟后，在 1960s-1970s 的开关电源发展中普及起来。\n"
            "3. 所以如果你问“反激式开关电源什么时候出现”，更准确的回答是：它的思想源于早期电视回扫电路，现代电源拓扑形态大约在 20 世纪 60-70 年代成熟并广泛应用。\n\n"
            "这类历史问题后续我会保持上下文，不需要你重复写 Flyback。"
        )

    return (
        "A flyback converter was not “discovered” on one exact date. It is an engineering topology that evolved from early television/CRT flyback or retrace circuits.\n\n"
        "A useful timeline is:\n"
        "1. The word “flyback” comes from CRT horizontal deflection, where the electron beam rapidly returns to the start of the next scan line during retrace.\n"
        "2. Flyback transformers were widely used in television circuits in the mid-20th century to generate high voltage.\n"
        "3. The modern isolated flyback switching-power-supply topology became common as transistorized SMPS and PWM control matured, roughly through the 1960s-1970s.\n\n"
        "So the best answer is: its roots are in early CRT retrace circuits, and its modern SMPS form matured during the 1960s-1970s."
    )


def _flyback_basic_answer(user_query: str) -> str:
    if _contains_chinese(user_query):
        return (
            "Flyback（反激变换器）是一种常见的隔离型开关电源拓扑。\n\n"
            "核心工作方式：开关管导通时，输入能量先储存在变压器的磁化电感中；开关管关断时，这部分能量通过次级整流器释放到输出电容和负载。\n\n"
            "它适合适配器、待机电源和中小功率辅助电源，因为结构简单、容易做隔离、成本较低。工程设计时最关键的是匝比、占空比、MOSFET Vds 应力、漏感尖峰钳位、输出纹波、环路稳定性、热和 EMI。"
        )

    return (
        "A flyback converter is an isolated switching power-supply topology.\n\n"
        "The key idea is energy storage and transfer: while the primary switch is on, input energy is stored in the transformer's magnetizing inductance; when the switch turns off, that stored energy is delivered through the secondary rectifier to the output capacitor and load.\n\n"
        "It is widely used in adapters, standby supplies, and low-to-mid-power auxiliary supplies because it is simple, low cost, and easy to isolate. The hard engineering parts are turns ratio, duty-cycle window, MOSFET Vds stress, leakage-spike clamp design, ripple, loop stability, thermal margin, and EMI."
    )


def _chatbot_answer(user_query: str) -> str:
    q = str(user_query or "").strip()
    low = q.lower()
    if _looks_like_capability_question(q):
        return _capability_answer(q)
    if _mentions_flyback(q):
        return (
            "Flyback（反激变换器）是一种常见的隔离型开关电源拓扑。\n\n"
            "- 工作原理：开关管导通时，能量先存到变压器磁化电感；关断时再把能量释放到次级负载。\n"
            "- 典型优势：结构简单、易做隔离、适合多路输出、成本较低。\n"
            "- 典型场景：适配器、待机电源、小中功率工业辅助电源。\n"
            "- 设计关键：匝比、占空比、开关频率、漏感尖峰/钳位、环路补偿、热与EMI。\n\n"
            "如果你愿意，我可以下一步按你的指标（输入范围/输出电压电流/效率/纹波）直接给一版可追溯的Flyback设计方案。"
        )

    return (
        "This is a knowledge Q&A response (not entered design pipeline).\n\n"
        "You can ask me for example:\n"
        "- Flyback CCM/DCM boundary formula\n"
        "- Why Vds stress exceeds limits\n"
        "- How to reduce ripple and switching losses\n\n"
        "To enter design mode, type for example:\n"
        "`Design a 24W Flyback, 85-265Vac to 12V/2A, efficiency > 88%, ripple < 100mV`\n\n"
        f"Your question: {user_query}"
    )


def _try_llm_qa_answer(user_query: str, rag_context: str = "", history: Optional[list[Dict[str, Any]]] = None) -> Optional[str]:
    if str(os.getenv("PE_MAS_QA_USE_LLM", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return None

    def _call_llm() -> Optional[str]:
        from core.llm.llm import get_llm_runtime_config, openai_init

        runtime = get_llm_runtime_config(preferred_model=os.getenv("PE_MAS_QA_MODEL") or os.getenv("PE_MAS_LLM_MODEL"))
        client = openai_init(
            openai_model=runtime["model"],
            api_key=runtime["api_key"],
            api_url=runtime["api_base"],
            require_key=False,
        )
        if client is None:
            return None

        history_lines = []
        for item in (history or [])[-6:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role") or "user"
            content = str(item.get("content") or item.get("answer") or "").strip()
            if content:
                history_lines.append(f"{role}: {content[:500]}")

        system_prompt = (
            "You are PE-MAS Studio's power electronics QA expert. Answer conversational questions clearly, "
            "but keep engineering rigor. If the user writes Chinese, answer in Chinese. If the user writes English, "
            "answer in English. Preserve conversation context, and do not start a design workflow unless the user asks "
            "for concrete design parameters or redesign. Keep answers concise but useful."
        )
        context_block = ""
        if rag_context:
            context_block += f"\n\nLocal flyback knowledge context:\n{rag_context[:2500]}"
        if history_lines:
            context_block += "\n\nRecent conversation:\n" + "\n".join(history_lines)

        response = client.chat.completions.create(
            model=runtime["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_query}{context_block}"},
            ],
            temperature=0.25,
            max_tokens=700,
        )
        return (response.choices[0].message.content or "").strip() or None

    timeout_sec = float(os.getenv("PE_MAS_QA_LLM_TIMEOUT_SEC", "18") or "18")
    try:
        return _tool_call_with_contract(
            "qa_llm",
            _call_llm,
            timeout_sec=timeout_sec,
            retries=0,
            idempotency_key=hashlib.sha1((str(user_query) + str(rag_context[:800])).encode("utf-8")).hexdigest(),
        )
    except Exception:
        return None


def _compact_framework_sections(sections: list[Dict[str, Any]]) -> str:
    lines: list[str] = []
    for section in (sections or [])[:4]:
        title = str(section.get("title") or "Section")
        lines.append(f"{title}:")
        for item in (section.get("items") or [])[:6]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("name") or "-")
            value = str(item.get("value") or item.get("tradeoff") or item.get("status") or "")
            lines.append(f"- {label}: {value[:240]}")
    return "\n".join(lines)


def _framework_agent_review(command_id: str, stage_payload: Dict[str, Any]) -> Dict[str, Any]:
    specs = (stage_payload.get("design_meta") or {}).get("specs") or {}
    stage = str(stage_payload.get("stage") or "unknown")
    title = str(stage_payload.get("title") or stage)
    compact_sections = _compact_framework_sections(stage_payload.get("sections") or [])
    spec_line = (
        f"85-265Vac target={specs.get('input_voltage_min', 85)}-{specs.get('input_voltage_max', 265)}Vac, "
        f"output={specs.get('output_voltage', 12)}V/{specs.get('output_current', 5)}A, "
        f"power={specs.get('output_power', 60)}W, topology preference=QR flyback + TL431/opto + SR"
    )
    review_query = (
        "Act as an independent senior power-electronics design reviewer for this PE-MAS gated workflow stage. "
        "Do not invent measured data or claim release readiness. Review whether this stage matches a manufacturable "
        "offline flyback engineering process and identify the evidence still required before the next irreversible gate.\n\n"
        "Project compliance direction is consumer/industrial IEC/UL 62368-1 plus CISPR 32 Class B pre-compliance; "
        "do not mention unrelated medical standards unless explicitly present in the specs.\n\n"
        f"Command: {command_id}\nStage: {stage}\nTitle: {title}\nSpecs: {spec_line}\n\nStage scaffold:\n{compact_sections}\n\n"
        "Return concise English with these labels: Verdict, Evidence checked, Missing evidence, Next required tool run, Release status."
    )
    trace: list[str] = []
    search_items: list[Dict[str, Any]] = []
    urls: list[str] = []
    rag_context = ""

    try:
        rag_bundle = retrieve_flyback_context(review_query, top_k=4)
        rag_context = str(rag_bundle.get("context_text") or "").strip()
        refs = rag_bundle.get("references") or []
        trace.append(f"Local RAG retrieved {len(refs)} source(s).")
        seen_ref_keys: set[str] = set()
        for ref in refs[:4]:
            if not isinstance(ref, dict):
                continue
            title_ref = normalize_source_label(ref.get("title") or ref.get("source") or ref.get("doc") or "RAG source")
            url = str(ref.get("url") or ref.get("link") or ref.get("path") or "")
            ref_key = f"{title_ref}|{url}"
            if ref_key in seen_ref_keys:
                continue
            seen_ref_keys.add(ref_key)
            if url:
                urls.append(url)
            search_items.append(
                {
                    "channel": "local_rag",
                    "source": "Internal Knowledge Base",
                    "title": str(title_ref),
                    "url": url,
                    "snippet": "Retrieved for independent stage review.",
                    "status": "retrieved",
                }
            )
    except Exception as e:
        trace.append(f"Local RAG failed: {e}")

    web_enabled = str(os.getenv("PE_MAS_FRAMEWORK_WEB_REVIEW", "1")).strip().lower() not in {"0", "false", "no", "off"}
    if web_enabled and mcp_research_web:
        try:
            web_pack = _tool_call_with_contract(
                "framework_mcp_research",
                lambda: mcp_research_web(
                    f"{stage} offline flyback design guide RCD snubber TL431 optocoupler EMI safety evidence",
                    max_results=2,
                ),
                timeout_sec=float(os.getenv("PE_MAS_FRAMEWORK_WEB_TIMEOUT_SEC", "8") or "8"),
                retries=0,
                idempotency_key=hashlib.sha1(f"{command_id}:{stage}".encode("utf-8")).hexdigest(),
            )
            web_results = web_pack.get("results") or [] if isinstance(web_pack, dict) else []
            trace.append(f"MCP web research returned {len(web_results)} result(s).")
            for row in web_results[:2]:
                if not isinstance(row, dict):
                    continue
                url = str(row.get("url") or row.get("link") or "")
                if url:
                    urls.append(url)
                search_items.append(
                    {
                        "channel": "mcp_web",
                        "source": row.get("source") or "Web research",
                        "title": str(row.get("title") or url or "Web source"),
                        "url": url,
                        "snippet": str(row.get("snippet") or row.get("summary") or "")[:280],
                        "status": "retrieved",
                    }
                )
        except Exception as e:
            trace.append(f"MCP web research unavailable: {e}")
    else:
        trace.append("MCP web research skipped or unavailable.")

    llm_answer = _try_llm_qa_answer(review_query, rag_context=rag_context, history=[])
    if llm_answer:
        trace.append("LLM reviewer completed.")
        reviewer_status = "reviewed"
    else:
        llm_answer = (
            "Verdict: Scaffold only. This stage is useful for workflow shape, but it is not engineering evidence.\n"
            "Evidence checked: local gate structure and available RAG references.\n"
            "Missing evidence: tool-generated calculations/simulation files, orderable part data, layout artifacts, or EVT measurements for this stage.\n"
            "Next required tool run: execute the relevant calculation, RAG/source lookup, EDA, SPICE/PLECS, sourcing, or bench-validation step before marking complete.\n"
            "Release status: HOLD."
        )
        trace.append("LLM reviewer unavailable; deterministic reviewer fallback used.")
        reviewer_status = "fallback_review"

    seen_urls: set[str] = set()
    deduped_urls: list[str] = []
    for url in urls:
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped_urls.append(url)

    return {
        "status": reviewer_status,
        "review_text": llm_answer[:2400],
        "trace": trace,
        "search_items": search_items[:8],
        "urls": deduped_urls[:8],
    }


def _should_run_web_qa(query: str) -> bool:
    q = str(query or "").strip().lower()
    cues = [
        "flyback", "flybakc", "flayback", "flaybakc", "topology", "converter", "反激", "电路", "开关电源",
        "datasheet", "paper", "arxiv", "forum", "blog", "文献", "论文", "资料",
    ]
    return any(c in q for c in cues)


def _compose_professional_qa(
    user_query: str,
    session: Optional[Dict[str, Any]] = None,
    history: Optional[list[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    original_q = str(user_query or "").strip()
    context = _resolve_qa_context(original_q, session=session, history=history)
    q = str(context.get("effective_query") or original_q).strip()
    trace_lines: list[str] = []
    evidence_lines: list[str] = []
    urls: list[str] = []
    search_items: list[dict[str, Any]] = []

    trace_lines.append("[OBSERVATION] User asked a knowledge question; route to professional QA mode.")
    if context.get("context_used"):
        trace_lines.append(f"[CONTEXT] {context.get('note')}")
        trace_lines.append(f"[CONTEXT] Effective query: {q}")
    if _looks_like_capability_question(q):
        trace_lines.append("[DECISION] Capability/greeting question detected; answer with product capability overview.")
        return {
            "answer": _capability_answer(original_q or q),
            "trace": "\n".join(trace_lines),
            "urls": [],
            "search_items": [],
            "source": "deterministic_capability_qa",
            "topic": "capabilities",
            "effective_query": q,
            "context_used": bool(context.get("context_used")),
        }
    if _looks_like_flyback_use_question(q):
        trace_lines.append("[DECISION] Flyback application/use-case follow-up detected; answer through context-aware QA.")
        source_cards = _flyback_general_sources("use")
        return {
            "answer": _flyback_use_answer(original_q or q),
            "trace": "\n".join(trace_lines),
            "urls": [str(row.get("url") or "") for row in source_cards if row.get("url")],
            "search_items": source_cards,
            "evidence_cards": _flyback_general_evidence_cards(original_q or q, "use"),
            "source": "deterministic_context_qa",
            "topic": context.get("topic") or "flyback",
            "effective_query": q,
            "context_used": bool(context.get("context_used")),
        }
    if _looks_like_flyback_history_question(q):
        trace_lines.append("[DECISION] Flyback history follow-up detected; answer through deterministic QA fast path.")
        source_cards = _flyback_general_sources("history")
        return {
            "answer": _flyback_history_answer(original_q or q),
            "trace": "\n".join(trace_lines),
            "urls": [str(row.get("url") or "") for row in source_cards if row.get("url")],
            "search_items": source_cards,
            "evidence_cards": _flyback_general_evidence_cards(original_q or q, "history"),
            "source": "deterministic_context_qa",
            "topic": context.get("topic") or "flyback",
            "effective_query": q,
            "context_used": bool(context.get("context_used")),
        }
    if _looks_like_flyback_clamp_review_question(q):
        trace_lines.append("[DECISION] Flyback RCD/snubber/VDS release-review question detected; use engineering review answer path.")
        source_cards = _flyback_clamp_review_sources()
        evidence_cards = _flyback_clamp_review_evidence_cards(original_q or q)
        urls = [str(row.get("url") or "") for row in source_cards if row.get("url")]
        return {
            "answer": _flyback_clamp_review_answer(original_q or q),
            "trace": "\n".join(trace_lines),
            "urls": urls,
            "search_items": source_cards,
            "evidence_cards": evidence_cards,
            "source": "deterministic_flyback_clamp_review",
            "topic": "flyback_clamp_review",
            "effective_query": q,
            "context_used": bool(context.get("context_used")),
        }
    if _looks_like_basic_flyback_question(q):
        trace_lines.append("[DECISION] Basic flyback concept question detected; try LLM QA first, then local expert fallback.")
        rag_context = ""
        rag_refs: list[Any] = []
        source_cards = _flyback_general_sources("concept")
        try:
            rag_bundle = retrieve_flyback_context(q, top_k=3)
            rag_context = (rag_bundle.get("context_text") or "").strip()
            rag_refs = rag_bundle.get("references") or []
            trace_lines.append(f"[RAG] Retrieved {len(rag_bundle.get('references') or [])} local knowledge sources.")
        except Exception as e:
            trace_lines.append(f"[WARNING] Local RAG retrieval failed: {e}")
        for ref in rag_refs[:3]:
            if not isinstance(ref, dict):
                continue
            title = normalize_source_label(ref.get("title") or ref.get("source") or ref.get("doc") or "RAG source")
            if not any(str(row.get("title") or "") == str(title) for row in source_cards):
                source_cards.insert(
                    0,
                    {
                        "channel": "local_rag",
                        "source": "Internal Knowledge Base",
                        "title": str(title),
                        "url": str(ref.get("url") or ref.get("link") or ref.get("path") or ""),
                        "snippet": "Matched from the internal flyback knowledge base.",
                        "status": "retrieved",
                    },
                )
        llm_answer = _try_llm_qa_answer(q, rag_context=rag_context, history=history)
        if llm_answer:
            trace_lines.append("[DECISION] LLM QA response generated successfully.")
            return {
                "answer": llm_answer,
                "trace": "\n".join(trace_lines),
                "urls": [str(row.get("url") or "") for row in source_cards if row.get("url")],
                "search_items": source_cards[:6],
                "evidence_cards": _flyback_general_evidence_cards(original_q or q, "concept"),
                "source": "llm_expert_qa",
                "topic": context.get("topic") or "flyback",
                "effective_query": q,
                "context_used": bool(context.get("context_used")),
            }
        trace_lines.append("[FALLBACK] LLM QA unavailable; return deterministic expert answer.")
        return {
            "answer": _flyback_basic_answer(q),
            "trace": "\n".join(trace_lines),
            "urls": [str(row.get("url") or "") for row in source_cards if row.get("url")],
            "search_items": source_cards[:6],
            "evidence_cards": _flyback_general_evidence_cards(original_q or q, "concept"),
            "source": "expert_fallback_qa",
            "topic": context.get("topic") or "flyback",
            "effective_query": q,
            "context_used": bool(context.get("context_used")),
        }

    trace_lines.append("[PLAN] Build answer using Local RAG first, then MCP web evidence if available.")

    rag_refs = []
    rag_context = ""
    try:
        rag_bundle = retrieve_flyback_context(q, top_k=5)
        rag_context = (rag_bundle.get("context_text") or "").strip()
        rag_refs = rag_bundle.get("references") or []
        trace_lines.append(f"[RAG] Retrieved {len(rag_refs)} local knowledge sources.")
    except Exception as e:
        trace_lines.append(f"[WARNING] Local RAG retrieval failed: {e}")

    web_results = []
    web_notes = []
    if _should_run_web_qa(q) and mcp_research_web:
        try:
            web_pack = mcp_research_web(q, max_results=3) if mcp_research_web else {}
            web_results = web_pack.get("results") or []
            web_notes = web_pack.get("notes") or []
            trace_lines.append(f"[SEARCH] MCP web research returned {len(web_results)} results.")
        except Exception as e:
            trace_lines.append(f"[WARNING] MCP web research failed: {e}")
    else:
        trace_lines.append("[SEARCH] MCP web research skipped (not needed or unavailable).")

    # Build evidence list
    for ref in rag_refs[:5]:
        if isinstance(ref, dict):
            title = normalize_source_label(ref.get("title") or ref.get("source") or ref.get("doc") or "RAG source")
            score = ref.get("score")
            url = ref.get("url") or ref.get("link") or ref.get("path") or ""
            if url:
                urls.append(str(url))
            search_items.append(
                {
                    "channel": "local_rag",
                    "source": "Internal Knowledge Base",
                    "title": str(title),
                    "url": str(url),
                    "snippet": "Matched from the internal flyback knowledge base.",
                    "status": "retrieved",
                }
            )
            evidence_lines.append(f"- [RAG] {title}" + (f" | {url}" if url else ""))
        else:
            evidence_lines.append(f"- [RAG] {ref}")

    for row in web_results[:3]:
        if not isinstance(row, dict):
            continue
        title = row.get("title") or "Web result"
        url = row.get("url") or ""
        snippet = (row.get("snippet") or row.get("extracted_text") or "").strip()
        if url:
            urls.append(str(url))
        search_items.append(
            {
                "channel": "web",
                "source": "mcp_web",
                "title": str(title),
                "url": str(url),
                "snippet": snippet[:120],
                "status": "retrieved",
            }
        )
        suffix = f" | {url}" if url else ""
        if snippet:
            suffix += f" | {snippet[:120]}"
        evidence_lines.append(f"- [MCP] {title}{suffix}")

    if web_notes:
        for n in web_notes[:2]:
            trace_lines.append(f"[DATA] {n}")

    llm_answer = _try_llm_qa_answer(q, rag_context=rag_context, history=history)
    if llm_answer:
        trace_lines.append("[DECISION] LLM QA response generated with retrieved context; evidence is attached as structured metadata.")
        return {
            "answer": llm_answer,
            "trace": "\n".join(trace_lines),
            "urls": list(dict.fromkeys(urls))[:8],
            "search_items": search_items[:8],
            "evidence_cards": [],
            "source": "llm_rag_qa",
            "topic": context.get("topic") or _infer_topic_from_text(q),
            "effective_query": q,
            "context_used": bool(context.get("context_used")),
        }

    low = q.lower()
    if _mentions_flyback(q):
        core_answer = (
            "Flyback（反激）电路是一种隔离型开关电源拓扑。\n\n"
            "工作机理：\n"
            "1) 开关管导通时，输入能量存入变压器磁化电感；\n"
            "2) 开关管关断时，磁场能量通过次级整流释放到负载与输出电容。\n\n"
            "工程特点：\n"
            "- 优点：结构简单、易隔离、适合中小功率、多路输出可行；\n"
            "- 难点：漏感尖峰、EMI、环路补偿、效率与热设计平衡；\n"
            "- 典型应用：适配器、待机电源、工业辅助电源。\n\n"
            "设计时重点关注：占空比区间、匝比与反射电压、MOSFET/二极管应力裕量、钳位与纹波目标。"
        )
    else:
        core_answer = _chatbot_answer(q)

    trace_lines.append("[DECISION] Respond with QA answer and keep design workflow idle.")
    return {
        "answer": core_answer,
        "trace": "\n".join(trace_lines),
        "urls": list(dict.fromkeys(urls))[:8],
        "search_items": search_items[:8],
        "evidence_cards": [],
        "source": "rag_mcp",
        "topic": context.get("topic") or _infer_topic_from_text(q),
        "effective_query": q,
        "context_used": bool(context.get("context_used")),
    }


def _recommended_hitl_patch(values: Dict[str, Any]) -> Dict[str, Any]:
    verification = values.get("verification") or {}
    strategy_bundle = verification.get("strategy_bundle") if isinstance(verification.get("strategy_bundle"), dict) else {}
    design_overrides = strategy_bundle.get("recommended_overrides") if isinstance(strategy_bundle.get("recommended_overrides"), dict) else {}
    payload: Dict[str, Any] = {}
    if design_overrides:
        payload["design_overrides"] = design_overrides
    if values.get("specifications"):
        payload["specifications"] = values.get("specifications")
    if values.get("theoretical_design"):
        payload["theoretical_design"] = values.get("theoretical_design")
    if values.get("bom"):
        payload["bom"] = values.get("bom")
    if values.get("simulation_results"):
        payload["simulation_results"] = values.get("simulation_results")
    if values.get("verification"):
        payload["verification"] = values.get("verification")
    return payload


_BOM_CHECKPOINT_LABELS = {
    "mosfet": "MOSFET",
    "diode": "Output Rectifier",
    "transformer": "Transformer / Core",
    "controller": "Controller",
    "input_cap": "Input Bulk Capacitor",
    "output_cap": "Output Capacitor",
    "input_protection": "Input Protection",
    "emi_filter": "EMI Filter",
    "clamp_snubber": "Clamp / Snubber",
}

_BOM_AUTO_FIXABLE_KEYS = {"mosfet", "diode", "controller", "input_cap", "output_cap", "input_protection", "clamp_snubber"}
_BOM_MANUAL_SIGNOFF_KEYS = {"transformer", "emi_filter"}


def _bom_first_value(row: Any, names: list[str]) -> str:
    if not isinstance(row, dict):
        return ""
    for name in names:
        value = row.get(name)
        if value not in (None, "", "-", "N/A", "n/a"):
            return str(value)
    return ""


def _bom_selected_name(row: Any) -> str:
    return _bom_first_value(
        row,
        ["Part Number", "Mfr Part #", "Manufacturer Part Number", "part_number", "Core", "core", "title", "description"],
    )


def _bom_is_generic(selected: str, price: Any = "") -> bool:
    low = str(selected or "").strip().lower()
    if any(flag in low for flag in ["generic", "fallback", "check online", "unknown"]):
        return True
    if low in {"pq2620", "ee25", "ee20", "er28", "25v-220uf", "35v-220uf"}:
        return True
    return False


def _bom_requires_manual_signoff(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("requires_custom_design") or "manual" in str(row.get("procurement_status") or "").lower():
        return True
    for value in row.values():
        if isinstance(value, dict) and _bom_requires_manual_signoff(value):
            return True
    return False


def _bom_next_action(key: str) -> str:
    if key == "mosfet":
        return "Replace with a real orderable 650 V or higher MOSFET; verify Rds(on), Qg/Eoss, avalanche, package thermal path, and Vds margin."
    if key == "transformer":
        return "Use a real offline flyback core/transformer set with Ae/le/Ve, bobbin, gap, insulation, and creepage data."
    if key == "emi_filter":
        return "Use an offline AC input EMI path, such as a rated common-mode choke plus X/Y capacitors; reject signal-line EMI parts."
    if key in {"input_cap", "output_cap"}:
        return "Select an orderable capacitor with voltage rating, ripple current, ESR, lifetime, and temperature derating evidence."
    return "Verify the distributor source, electrical ratings, and derating policy; replace if the evidence is incomplete."


def _bom_task_fixability(key: str, status: str, manual_required: bool, reason: str) -> tuple[str, str]:
    if manual_required or key in _BOM_MANUAL_SIGNOFF_KEYS:
        return "manual_signoff", "MANUAL_ADJUSTMENTS"
    if key in _BOM_AUTO_FIXABLE_KEYS and status in {"manual", "review", "blocked"}:
        return "auto_fixable", "CHANGE_COMPONENT_STRATEGY"
    if status in {"manual", "blocked"}:
        return "manual_signoff", "MANUAL_ADJUSTMENTS"
    if "source" in reason.lower() or "traceable" in reason.lower():
        return "verify_only", "MANUAL_ADJUSTMENTS"
    return "verify_only", "MANUAL_ADJUSTMENTS"


def _bom_checkpoint_summary(tasks: list[Dict[str, Any]]) -> Dict[str, Any]:
    auto_fixable = sum(1 for task in tasks if task.get("fixability") == "auto_fixable")
    manual = sum(1 for task in tasks if task.get("fixability") == "manual_signoff")
    verify_only = sum(1 for task in tasks if task.get("fixability") == "verify_only")
    if auto_fixable:
        primary_command = "CHANGE_COMPONENT_STRATEGY"
        primary_label = "Auto-fix sourced parts"
        recommendation = "Auto-fix the auto-fixable MPN/source issues, then manually sign off any remaining magnetics or mains EMI items."
    elif manual:
        primary_command = "MANUAL_ADJUSTMENTS"
        primary_label = "Open manual sign-off"
        recommendation = "Remaining blockers require engineering/procurement sign-off, so do not rerun auto-selection. Enter approved MPNs/specs or continue only as exploratory simulation."
    elif verify_only:
        primary_command = "MANUAL_ADJUSTMENTS"
        primary_label = "Verify evidence"
        recommendation = "Verify source links and ratings before BOM freeze, or continue only as exploratory simulation."
    else:
        primary_command = "CONTINUE_SIMULATION"
        primary_label = "Continue simulation"
        recommendation = "Continue simulation."
    return {
        "total": len(tasks),
        "auto_fixable": auto_fixable,
        "manual_signoff": manual,
        "verify_only": verify_only,
        "primary_command": primary_command,
        "primary_label": primary_label,
        "recommendation": recommendation,
    }


def _bom_checkpoint_tasks(values: Dict[str, Any]) -> list[Dict[str, Any]]:
    bom = values.get("bom") or {}
    if not isinstance(bom, dict):
        return []
    summary = bom.get("selection_summary") if isinstance(bom.get("selection_summary"), dict) else {}
    policy = bom.get("selection_policy") if isinstance(bom.get("selection_policy"), dict) else {}
    candidates = bom.get("local_db_top_candidates") if isinstance(bom.get("local_db_top_candidates"), dict) else {}
    tasks: list[Dict[str, Any]] = []
    for key, label in _BOM_CHECKPOINT_LABELS.items():
        info = summary.get(key) if isinstance(summary.get(key), dict) else {}
        raw = bom.get(key) if isinstance(bom.get(key), dict) else {}
        policy_row = policy.get(key) if isinstance(policy.get(key), dict) else {}
        selected = str(info.get("selected") or _bom_selected_name(raw) or "").strip()
        if not selected:
            continue
        source = str(info.get("source") or _bom_first_value(raw, ["Product URL", "URL", "DigiKey URL", "source", "Source"]) or "").strip()
        datasheet = _bom_first_value(raw, ["Datasheet", "Datasheet URL", "datasheet", "datasheet_url"])
        price = str(info.get("price") or _bom_first_value(raw, ["Price", "Unit Price", "price"]) or "").strip()
        color = str(policy_row.get("color") or "").lower()
        generic = _bom_is_generic(selected, price)
        price_text = str(price or "").lower()
        price_placeholder = any(token in price_text for token in ["check online", "manual selection", "quote"])
        manual_required = _bom_requires_manual_signoff(raw)
        source_missing = not source and key in {"mosfet", "controller", "output_cap", "input_protection", "emi_filter", "transformer"}
        if not (generic or price_placeholder or source_missing or manual_required or color in {"red", "yellow"}):
            continue
        if manual_required:
            reason = "This item requires manual engineering/procurement sign-off before BOM freeze."
            status = "manual"
        elif generic:
            reason = "Selected part is generic or provisional, so it cannot be ordered or traced."
            status = "manual"
        elif source_missing or price_placeholder:
            reason = "Source or rating evidence is missing, so the selection is not traceable."
            status = "review"
        elif color == "red":
            reason = "Component category is marked high risk and requires manual engineering review."
            status = "manual"
        else:
            reason = "Component category is usable only after datasheet and source cross-check."
            status = "review"
        fixability, recommended_command = _bom_task_fixability(key, status, manual_required, reason)
        task_candidates = []
        for row in (candidates.get(key) if isinstance(candidates.get(key), list) else [])[:3]:
            if not isinstance(row, dict):
                continue
            task_candidates.append(
                {
                    "part": _bom_selected_name(row) or "Candidate",
                    "manufacturer": _bom_first_value(row, ["Manufacturer", "Mfr", "manufacturer"]),
                    "price": _bom_first_value(row, ["Price", "Unit Price", "price"]),
                    "source": _bom_first_value(row, ["Product URL", "URL", "DigiKey URL", "source", "Source"]),
                }
            )
        tasks.append(
            {
                "key": key,
                "label": label,
                "part": selected,
                "status": status,
                "reason": reason,
                "next_action": _bom_next_action(key),
                "fixability": fixability,
                "recommended_command": recommended_command,
                "source": source,
                "datasheet": datasheet,
                "candidates": task_candidates,
            }
        )
    priority = {"manual": 0, "blocked": 0, "review": 1, "ready": 2}
    return sorted(tasks, key=lambda row: (priority.get(str(row.get("status")), 1), str(row.get("label"))))


def _component_strategy_prompt(values: Dict[str, Any]) -> str:
    tasks = _bom_checkpoint_tasks(values)
    if not tasks:
        return (
            " Re-run component selection with strict source traceability. "
            "Do not use generic fallback parts; every critical part must have an orderable MPN and source link."
        )
    auto_tasks = [task for task in tasks if task.get("fixability") == "auto_fixable"]
    manual_tasks = [task for task in tasks if task.get("fixability") == "manual_signoff"]
    lines = [
        " Re-run component selection and specifically fix auto-fixable BOM issues:",
    ]
    for task in (auto_tasks or tasks)[:6]:
        lines.append(
            f"- {task.get('label')}: current={task.get('part')}; issue={task.get('reason')}; action={task.get('next_action')}"
        )
    if manual_tasks:
        lines.append(
            "Do not random-select replacements for custom magnetics or AC mains EMI items; keep them as manual sign-off items unless a genuinely orderable, rating-appropriate source is found."
        )
    lines.append(
        "Prefer orderable distributor-backed parts with source evidence and rating evidence; reject generic fallback and signal-line/low-power parts for offline power-stage roles."
    )
    return " " + "\n".join(lines)


def _bom_reselect_learning(values: Dict[str, Any]) -> Dict[str, Any]:
    tasks = _bom_checkpoint_tasks(values)
    focus: list[str] = []
    actions: list[str] = []
    avoid: list[str] = ["generic fallback", "unsourced critical part", "signal-line EMI part used as AC input EMI filter"]
    for task in tasks[:8]:
        label = str(task.get("label") or task.get("key") or "component")
        part = str(task.get("part") or "").strip()
        next_action = str(task.get("next_action") or "").strip()
        focus.append(f"{label}: {task.get('reason')}")
        if next_action:
            actions.append(f"{label}: {next_action}")
        if part:
            avoid.append(part)
    return {
        "next_iteration_focus": focus[:8] or ["Replace provisional BOM entries with orderable, traceable critical components."],
        "recommended_component_actions": actions[:8] or ["Use real orderable MPNs with source evidence for all critical parts."],
        "do_not_repeat": list(dict.fromkeys(avoid))[:10],
        "dominant_loss": "",
    }


def _checkpoint_payload(next_node: str, values: Dict[str, Any]) -> Dict[str, Any]:
    plan = values.get("execution_plan") or {}
    learning = values.get("iteration_learning") or {}
    verification = values.get("verification") or {}
    context_suffix = {
        "planning_summary": values.get("planning_summary") or plan.get("headline") or "",
        "learning_focus": learning.get("next_iteration_focus") or [],
        "recommended_component_actions": ((verification.get("strategy_bundle") or {}).get("recommended_component_actions") or []),
        "recommended_patch": _recommended_hitl_patch(values),
    }
    if next_node == "selector":
        return {
            "title": "Theoretical Design Confirmation",
            "question": "Theoretical parameters are calculated. Continue to component selection?",
            "options": ["Continue selection", "Modify constraints and recompute", "Stop this session"],
            "commands": ["CONTINUE_SELECTION", "MODIFY_CONSTRAINTS", "STOP_SESSION"],
            "context": {**context_suffix, "theoretical_design": values.get("theoretical_design", {})},
        }
    if next_node == "simulator":
        bom_tasks = _bom_checkpoint_tasks(values)
        return {
            "title": "BOM Confirmation",
            "question": "BOM selection completed. Resolve provisional parts before treating this as release-ready; continuing simulation is exploratory only.",
            "options": ["Auto-fix BOM and reselect parts", "Edit BOM manually", "Continue exploratory simulation", "Stop this session"],
            "commands": ["CHANGE_COMPONENT_STRATEGY", "MANUAL_ADJUSTMENTS", "CONTINUE_SIMULATION", "STOP_SESSION"],
            "context": {
                **context_suffix,
                "bom": values.get("bom", {}),
                "bom_tasks": bom_tasks,
                "bom_checkpoint_summary": _bom_checkpoint_summary(bom_tasks),
            },
        }
    if next_node == "correction":
        return {
            "title": "Post-validation Review",
            "question": "Validation completed. Run correction agent for objective consistency review?",
            "options": ["Continue review", "Skip correction and generate report", "Stop this session"],
            "commands": ["CONTINUE_REVIEW", "SKIP_CORRECTION_AND_REPORT", "STOP_SESSION"],
            "context": {**context_suffix, "verification": values.get("verification", {})},
        }
    if next_node == "reporter":
        return {
            "title": "Report Generation Confirmation",
            "question": "Generate final engineering report and finish this session?",
            "options": ["Generate report", "Return to adjust", "Stop this session"],
            "commands": ["GENERATE_REPORT", "CHANGE_COMPONENT_STRATEGY", "STOP_SESSION"],
            "context": {
                **context_suffix,
                "verification": values.get("verification", {}),
                "correction_review": values.get("correction_review", {}),
            },
        }
    return {
        "title": "Continue Confirmation",
        "question": "Continue to next step?",
        "options": ["Continue", "Stop"],
        "commands": ["CONTINUE_SELECTION", "STOP_SESSION"],
        "context": {},
    }


def _cleanup_sessions() -> None:
    now = time.time()
    changed = False
    with SESSION_LOCK:
        stale = [sid for sid, item in SESSIONS.items() if now - float(item.get("updated_at", now)) > SESSION_TTL_SEC]
        for sid in stale:
            SESSIONS.pop(sid, None)
            changed = True
    if changed:
        _persist_sessions_to_disk()


def _get_or_create_session(sid: Optional[str]) -> Dict[str, Any]:
    _cleanup_sessions()
    with SESSION_LOCK:
        if sid and sid in SESSIONS:
            SESSIONS[sid]["updated_at"] = time.time()
            SESSIONS[sid].setdefault("history", [])
            SESSIONS[sid].setdefault("last_topic", "")
            data = {"sid": sid, **SESSIONS[sid]}
            should_persist = True
        else:
            sid_new = sid or str(uuid.uuid4())
            thread_id = str(uuid.uuid4())
            SESSIONS[sid_new] = {
                "thread_id": thread_id,
                "updated_at": time.time(),
                "base_prompt": "",
                "history": [],
                "last_topic": "",
            }
            data = {"sid": sid_new, **SESSIONS[sid_new]}
            should_persist = True

    if should_persist:
        _persist_sessions_to_disk()
    return data


def _session_history(session: Optional[Dict[str, Any]], limit: int = 12) -> list[Dict[str, Any]]:
    history = (session or {}).get("history") or []
    if not isinstance(history, list):
        return []
    return [row for row in history[-limit:] if isinstance(row, dict)]


def _infer_topic_from_text(text: str) -> str:
    if _mentions_flyback(text):
        return "flyback"
    low = str(text or "").lower()
    if any(term in low for term in ["mosfet", "vds", "switch stress"]):
        return "mosfet_stress"
    if any(term in low for term in ["transformer", "magnetics", "磁芯", "变压器"]):
        return "magnetics"
    return ""


def _infer_session_topic(session: Optional[Dict[str, Any]], history: Optional[list[Dict[str, Any]]] = None) -> str:
    cached = str((session or {}).get("last_topic") or "").strip()
    if cached:
        return cached
    rows = history if history is not None else _session_history(session)
    for row in reversed(rows or []):
        topic = _infer_topic_from_text(f"{row.get('user', '')}\n{row.get('assistant', '')}")
        if topic:
            return topic
    base_topic = _infer_topic_from_text(str((session or {}).get("base_prompt") or ""))
    return base_topic


def _record_chat_turn(sid: str, user_query: str, answer: str, meta: Optional[Dict[str, Any]] = None) -> None:
    topic = str((meta or {}).get("topic") or _infer_topic_from_text(f"{user_query}\n{answer}") or "").strip()
    with SESSION_LOCK:
        item = SESSIONS.get(sid)
        if not item:
            return
        history = item.setdefault("history", [])
        if not isinstance(history, list):
            history = []
            item["history"] = history
        history.append(
            {
                "ts": time.time(),
                "user": str(user_query or "")[:2000],
                "assistant": str(answer or "")[:4000],
                "meta": meta or {},
            }
        )
        item["history"] = history[-20:]
        if topic:
            item["last_topic"] = topic
        item["updated_at"] = time.time()
    _persist_sessions_to_disk()


def _specs_to_prompt(specs: Dict[str, Any]) -> str:
    vin_min = specs.get("input_voltage_min", 85)
    vin_max = specs.get("input_voltage_max", 265)
    vout = specs.get("output_voltage", 12)
    iout = specs.get("output_current", 5)
    pout = specs.get("output_power")
    try:
        pout = float(pout) if pout is not None else float(vout) * float(iout)
    except Exception:
        pout = 60.0
    eta = specs.get("efficiency_115vac_full_load") or specs.get("efficiency_target") or 0.88
    ripple_mv = specs.get("max_ripple_mvpp")
    if ripple_mv is None:
        ripple = specs.get("max_ripple_voltage", 0.12)
        ripple_mv = float(ripple) * 1000
    return (
        f"Design a manufacturable isolated QR Flyback power supply, {vin_min}-{vin_max}Vac input, "
        f"{vout}V/{iout}A output ({pout:g}W), efficiency target >= {float(eta)*100:.0f}%, "
        f"ripple < {float(ripple_mv):g}mVp-p. Preserve these exact locked specs."
    )


def _hitl_resume_inputs(
    patch_payload: Dict[str, Any],
    session: Dict[str, Any],
    thread_id: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Build graph inputs for a manual checkpoint JSON resume.

    The requirements node intentionally re-validates the latest user message.
    Manual checkpoint resumes must therefore carry the locked specs both as
    state and in the continuation prompt, otherwise the graph can fall back to
    default/example specs such as 12V/2A.
    """

    if not isinstance(patch_payload, dict):
        raise ValueError("HITL payload must be a JSON object")

    allowed_keys = {
        "specifications",
        "theoretical_design",
        "bom",
        "simulation_results",
        "verification",
        "correction_review",
        "design_overrides",
    }
    merged_patch = {k: v for k, v in patch_payload.items() if k in allowed_keys}

    design_overrides = (
        dict(merged_patch.get("design_overrides") or {})
        if isinstance(merged_patch.get("design_overrides"), dict)
        else {}
    )
    patch_specs = (
        dict(merged_patch.get("specifications") or {})
        if isinstance(merged_patch.get("specifications"), dict)
        else {}
    )
    locked_specs = (
        patch_specs
        or (dict(design_overrides.get("locked_specs") or {}) if isinstance(design_overrides.get("locked_specs"), dict) else {})
        or _framework_specs_from_session(session)
    )

    design_overrides["requirements_gate"] = "framework_scaffold_locked"
    design_overrides["locked_specs"] = locked_specs
    merged_patch["specifications"] = locked_specs
    merged_patch["design_overrides"] = design_overrides
    merged_patch["messages"] = [(
        "user",
        f"{_specs_to_prompt(locked_specs)}\n\n"
        "CONTINUE FROM GATED ENGINEERING SCAFFOLD: User applied a HITL JSON update. "
        "Run the real agent/tool workflow now. Preserve output current, output power, "
        "isolation, EMI, ripple, and all locked specifications exactly from the locked specification. "
        "Keep release blocked unless measured/tool evidence closes the relevant gates.",
    )]
    merged_patch["config"] = {"thread_id": thread_id}
    merged_patch["thread_id"] = thread_id
    return merged_patch, locked_specs


def _safe_json_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _mk_items(pairs: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    return [{"label": k, "value": v} for k, v in pairs if v not in (None, "", [], {})]


def _evidence_closure_items(evidence_closure: Any, limit: int = 10) -> list[dict[str, Any]]:
    if not isinstance(evidence_closure, dict):
        return []
    items: list[dict[str, Any]] = []
    summary = str(evidence_closure.get("summary") or "").strip()
    if summary:
        items.append({"label": "Summary", "value": summary})
    for gate in (evidence_closure.get("required_gates") or [])[:limit]:
        if not isinstance(gate, dict):
            continue
        label = str(gate.get("label") or gate.get("key") or "Evidence gate")
        status = str(gate.get("status") or "open").upper()
        evidence = str(gate.get("evidence") or "").strip()
        missing = str(gate.get("missing") or "").strip()
        impact = str(gate.get("release_impact") or "").strip()
        value_parts = [f"status={status}"]
        if evidence:
            value_parts.append(f"evidence={evidence}")
        if missing:
            value_parts.append(f"missing={missing}")
        if impact:
            value_parts.append(f"release impact={impact}")
        items.append({"label": label, "value": " | ".join(value_parts)})
    release_pack = evidence_closure.get("release_evidence_package")
    if isinstance(release_pack, dict):
        if release_pack.get("summary"):
            items.append({"label": "CR Evidence Package", "value": release_pack.get("summary")})
        matrix = release_pack.get("validation_matrix") if isinstance(release_pack.get("validation_matrix"), dict) else {}
        matrix_summary = matrix.get("summary") if isinstance(matrix.get("summary"), dict) else {}
        if matrix_summary:
            items.append(
                {
                    "label": "PLECS Matrix",
                    "value": (
                        f"steady={matrix_summary.get('steady_state_cases')} "
                        f"runnable={matrix_summary.get('plecs_runnable_cases')} "
                        f"manual/open={matrix_summary.get('manual_required_cases')}"
                    ),
                }
            )
        bom_signoff = release_pack.get("bom_signoff") if isinstance(release_pack.get("bom_signoff"), dict) else {}
        if bom_signoff.get("manual_required"):
            items.append({"label": "Manual BOM Signoff", "value": ", ".join([str(x) for x in bom_signoff.get("manual_required") or []])})
    return items


def _evidence_closure_report_section(values: Dict[str, Any]) -> str:
    sim = values.get("simulation_results") if isinstance(values, dict) else {}
    evidence_closure = sim.get("evidence_closure") if isinstance(sim, dict) else {}
    if not isinstance(evidence_closure, dict):
        return ""
    rows = evidence_closure.get("required_gates") or []
    if not rows:
        return ""

    def _md_cell(value: Any) -> str:
        text = str(value if value not in (None, "") else "-")
        text = re.sub(r"\s+", " ", text).strip()
        return text.replace("|", "\\|")

    def _table(headers: list[str], table_rows: list[list[Any]]) -> list[str]:
        out = ["| " + " | ".join(_md_cell(h) for h in headers) + " |"]
        out.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in table_rows:
            out.append("| " + " | ".join(_md_cell(cell) for cell in row) + " |")
        return out

    release_ready = bool(evidence_closure.get("release_ready"))
    release_pack = evidence_closure.get("release_evidence_package") if isinstance(evidence_closure.get("release_evidence_package"), dict) else {}
    cr_ready = bool(release_pack.get("controlled_release_candidate_ready")) if release_pack else release_ready
    verdict = "RELEASE CANDIDATE" if release_ready and cr_ready else "HOLD FOR ENGINEERING EVIDENCE"

    lines = ["## Evidence Closure / Release Hold", ""]
    lines.append(f"> **Verdict:** {verdict}. This package remains controlled until the open gates below have tool, source, or measured evidence.")
    lines.append("")
    summary_rows = [
        ["Release-ready", "Yes" if release_ready else "No"],
        ["Controlled release candidate", "Yes" if cr_ready else "No"],
    ]
    if evidence_closure.get("source"):
        summary_rows.append(["Primary simulation source", evidence_closure.get("source")])
    if evidence_closure.get("summary"):
        summary_rows.append(["Evidence summary", evidence_closure.get("summary")])
    if release_pack:
        if release_pack.get("summary"):
            summary_rows.append(["CR package", release_pack.get("summary")])
    lines.extend(_table(["Field", "Value"], summary_rows))
    lines.append("")

    gate_rows: list[list[Any]] = []
    for gate in rows:
        if not isinstance(gate, dict):
            continue
        label = gate.get("label") or gate.get("key") or "Evidence gate"
        status = str(gate.get("status") or "open").upper()
        missing = gate.get("missing") or gate.get("release_impact") or "-"
        evidence = gate.get("evidence") or "-"
        gate_rows.append([label, status, evidence, missing])
    if gate_rows:
        lines.append("### Required Evidence Gates")
        lines.extend(_table(["Gate", "Status", "Current Evidence", "Missing / Next Action"], gate_rows))

    if release_pack:
        lines.append("")
        lines.append("### Controlled Release Candidate Gates")
        cr_rows: list[list[Any]] = []
        for gate in release_pack.get("gates") or []:
            if not isinstance(gate, dict):
                continue
            cr_rows.append([gate.get("label") or gate.get("key") or "Gate", str(gate.get("status") or "open").upper(), gate.get("summary") or "-"])
        if cr_rows:
            lines.extend(_table(["Gate", "Status", "Engineering Evidence"], cr_rows))

        matrix = release_pack.get("validation_matrix") if isinstance(release_pack.get("validation_matrix"), dict) else {}
        matrix_summary = matrix.get("summary") if isinstance(matrix.get("summary"), dict) else {}
        if matrix_summary:
            lines.append("")
            lines.append("### PLECS Validation Matrix")
            lines.extend(
                _table(
                    ["Matrix Item", "Value"],
                    [
                        ["Steady-state cases", matrix_summary.get("steady_state_cases")],
                        ["PLECS-runnable cases", matrix_summary.get("plecs_runnable_cases")],
                        ["Manual/open cases", matrix_summary.get("manual_required_cases")],
                        ["Line points", ", ".join(str(x) for x in matrix_summary.get("lines") or [])],
                        ["Load points", ", ".join(str(x) for x in matrix_summary.get("loads") or [])],
                    ],
                )
            )

        thermal = release_pack.get("thermal_model") if isinstance(release_pack.get("thermal_model"), dict) else {}
        loop = release_pack.get("loop_evidence") if isinstance(release_pack.get("loop_evidence"), dict) else {}
        emi = release_pack.get("emi_safety") if isinstance(release_pack.get("emi_safety"), dict) else {}
        if thermal or loop or emi:
            lines.append("")
            lines.append("### Engineering Evidence Packs")
            pack_rows: list[list[Any]] = []
            if loop:
                pack_rows.append(
                    [
                        "TL431/opto loop",
                        str(loop.get("status") or "open").upper(),
                        f"Target fc {loop.get('target_crossover_hz')} Hz; PM >= {loop.get('minimum_phase_margin_deg')} deg; GM >= {loop.get('minimum_gain_margin_db')} dB",
                    ]
                )
            if thermal:
                pack_rows.append(["Thermal model", str(thermal.get("status") or "open").upper(), f"Estimated total loss {thermal.get('total_loss_estimate_w')} W; replace assumptions with EVT data."])
            if emi:
                pack_rows.append(["EMI / safety", str(emi.get("status") or "open").upper(), f"Target {emi.get('target') or '-'}; attach filter model, layout evidence, and pre-scan data."])
            lines.extend(_table(["Pack", "Status", "Required Evidence"], pack_rows))

        bom_signoff = release_pack.get("bom_signoff") if isinstance(release_pack.get("bom_signoff"), dict) else {}
        signoff_items = bom_signoff.get("items") if isinstance(bom_signoff.get("items"), list) else []
        manual_rows = []
        for item in signoff_items:
            if not isinstance(item, dict) or not item.get("required_signoff"):
                continue
            manual_rows.append([item.get("key"), item.get("part"), str(item.get("status") or "manual_required").upper(), item.get("owner")])
        if manual_rows:
            lines.append("")
            lines.append("### Manual BOM Signoff")
            lines.extend(_table(["Item", "Current Selection", "Status", "Owner"], manual_rows[:8]))

    lines.append("")
    lines.append("**Release rule:** keep schematic, BOM, transformer, safety/EMI, and manufacturing package on hold until every required gate has calculated, sourced, tool-generated, or measured evidence.")
    return "\n".join(lines).strip()


def _append_evidence_closure_report(report: str, values: Dict[str, Any]) -> str:
    section = _evidence_closure_report_section(values)
    if not section:
        return str(report or "")
    text = str(report or "").rstrip()
    if "## Evidence Closure / Release Hold" in text:
        return text + "\n"
    return (text + "\n\n" + section + "\n").lstrip()


def _extract_urls(text: str) -> list[str]:
    if not text:
        return []
    urls = URL_RE.findall(str(text))
    out: list[str] = []
    seen = set()
    for u in urls:
        u = u.rstrip(".);,")
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _extract_search_progress_items(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("[SEARCH-HIT]"):
            continue
        payload = line[len("[SEARCH-HIT]"):].strip()
        try:
            item = json.loads(payload)
        except Exception:
            continue
        if not isinstance(item, dict):
            continue
        if item.get("source"):
            item["source"] = normalize_source_label(item.get("source"))
        if item.get("channel") in {"local_rag", "component_rag"} and item.get("title"):
            item["title"] = normalize_source_label(item.get("title"))
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalize_reasoning_lines(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        lines = [str(x).strip() for x in raw if str(x).strip()]
    else:
        lines = [ln.strip() for ln in str(raw).splitlines() if ln.strip()]
    return lines


def _reasoning_label(line: str) -> tuple[str, str]:
    text = str(line or "").strip()
    if text.startswith("[") and "]" in text:
        tag = text[1:text.index("]")].strip().upper()
        rest = text[text.index("]") + 1 :].strip()
        label_map = {
            "PLAN": "Plan",
            "STRATEGY": "Strategy",
            "THOUGHT": "Thought",
            "SEARCH": "Search",
            "RAG": "RAG",
            "DATA": "Evidence",
            "OBSERVATION": "Observation",
            "RESULT": "Result",
            "FORMULA": "Formula",
            "FORMULA-WARN": "Formula Warn",
            "FORMULA-FAIL": "Formula Fail",
            "DECISION": "Decision",
            "CORRECTION": "Correction",
            "AI-INFERENCE": "Inference",
            "EXECUTION": "Execution",
            "WARNING": "Warning",
            "ERROR": "Error",
            "FILTER": "Filter",
            "META-COGNITION": "Meta",
            "TOOL_OUTPUT": "Tool",
            "TOOL_ERROR": "Tool Error",
            "FALLBACK": "Fallback",
        }
        return label_map.get(tag, tag.title()), (rest or text)
    return "Trace", text


def _reasoning_items(raw: Any, max_items: int = 80) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in _normalize_reasoning_lines(raw)[:max_items]:
        label, value = _reasoning_label(line)
        items.append({"label": label, "value": value})
    return items


def _collect_node_reasoning(node_name: str, merged: Dict[str, Any]) -> list[str]:
    logs = merged.get("reasoning_logs", {}) or {}
    if not isinstance(logs, dict):
        return []

    if node_name == "skill_executor":
        skill_lines: list[str] = []
        for k, v in logs.items():
            if str(k).startswith("skill_") or k == "skill_executor":
                skill_lines.extend(_normalize_reasoning_lines(v))
        return skill_lines

    return _normalize_reasoning_lines(logs.get(node_name))


def _collect_evidence_items(node_name: str, merged: Dict[str, Any], max_items: int = 12) -> list[dict[str, Any]]:
    refs = merged.get("retrieved_knowledge_references") or []
    lit = merged.get("literature_references") or []
    skill_out = ((merged.get("skill_state") or {}).get("skill_output") or {}) if isinstance(merged.get("skill_state"), dict) else {}
    items: list[dict[str, Any]] = []

    def _reason_for_node(kind: str) -> str:
        reason_map = {
            "requirements": "Used to anchor requirement extraction with domain constraints.",
            "designer": "Used to derive/justify theoretical parameter choices and equations.",
            "selector": "Used to support component candidate screening and availability checks.",
            "validator": "Used to justify pass/fail decision and corrective strategy.",
            "skill_executor": "Used as external evidence for research tasks.",
        }
        return f"[{kind}] {reason_map.get(node_name, 'Used as supporting evidence for this node.')}"

    dedup: dict[str, dict[str, Any]] = {}

    def _merge_item(label: str, title: str, url: str = "", score: Any = None, note: str = "") -> None:
        key = f"{label}|{title}|{url}".strip().lower()
        existing = dedup.get(key)
        score_val = None
        try:
            if score not in (None, ""):
                score_val = float(score)
        except Exception:
            score_val = None

        if existing:
            existing["count"] = int(existing.get("count", 1)) + 1
            old_score = existing.get("score")
            if score_val is not None and (old_score is None or score_val > old_score):
                existing["score"] = score_val
            if note and note not in str(existing.get("note", "")):
                existing["note"] = (str(existing.get("note", "")) + " " + note).strip()
            return

        dedup[key] = {
            "label": label,
            "title": title,
            "url": url,
            "score": score_val,
            "note": note,
            "count": 1,
        }

    if node_name == "selector":
        bom = merged.get("bom") or {}
        selection_summary = bom.get("selection_summary") if isinstance(bom, dict) else {}
        if isinstance(selection_summary, dict):
            for key in [
                "mosfet",
                "diode",
                "transformer",
                "input_cap",
                "output_cap",
                "controller",
                "input_protection",
                "emi_filter",
                "clamp_snubber",
            ]:
                info = selection_summary.get(key)
                if not isinstance(info, dict):
                    continue
                selected = str(info.get("selected") or "").strip()
                source = str(info.get("source") or "").strip()
                if not selected or selected.lower() in {"unknown", "-"}:
                    continue
                _merge_item(
                    "Selected Component",
                    f"{key}: {selected}",
                    source,
                    score=0.99,
                    note="Chosen for BOM with local DigiKey evidence.",
                )

    if node_name in {"requirements", "designer", "validator"} and isinstance(refs, list):
        for ref in refs[:max_items]:
            if isinstance(ref, dict):
                title = ref.get("title") or ref.get("source") or ref.get("doc") or "RAG Source"
                url = ref.get("url") or ref.get("link") or ref.get("path") or ""
                score = ref.get("score")
                _merge_item("RAG", str(title), str(url), score=score, note=_reason_for_node("RAG"))
            else:
                _merge_item("RAG", str(ref), "", note=_reason_for_node("RAG"))

    if node_name in {"designer", "selector"} and isinstance(lit, list):
        lit_cap = 4 if node_name == "selector" else max_items
        for ref in lit[:lit_cap]:
            if isinstance(ref, dict):
                title = ref.get("title") or "Literature"
                url = ref.get("url") or ""
                if node_name == "selector" and url and "digikey.com" not in str(url).lower() and "datasheet" not in str(url).lower():
                    continue
                insight = str(ref.get("insight") or "").strip()
                note = _reason_for_node("Literature")
                if insight:
                    note += f" Snippet: {insight}"
                _merge_item("Literature", str(title), str(url), note=note)
            else:
                _merge_item("Literature", str(ref), "", note=_reason_for_node("Literature"))

    if node_name == "skill_executor" and isinstance(skill_out, dict):
        web_result = skill_out.get("web_research")
        if isinstance(web_result, dict):
            for row in (web_result.get("results") or [])[:max_items]:
                if not isinstance(row, dict):
                    continue
                title = row.get("title") or "Web Result"
                url = row.get("url") or ""
                snippet = row.get("snippet") or row.get("extracted_text") or ""
                note = _reason_for_node("Web")
                if snippet:
                    note += f" Snippet: {str(snippet)}"
                _merge_item("Web", str(title), str(url), note=note)

    values = list(dedup.values())
    values.sort(key=lambda x: (-(x.get("score") if x.get("score") is not None else -1), -int(x.get("count", 1))))
    for row in values[:max_items]:
        score = row.get("score")
        score_txt = f"score={score:.4g}" if isinstance(score, float) else "score=n/a"
        repeats = int(row.get("count", 1))
        repeat_txt = f" · hits={repeats}" if repeats > 1 else ""
        value = f"{row.get('title')}\n{score_txt}{repeat_txt}"
        if row.get("url"):
            value += f"\n{row.get('url')}"
        if row.get("note"):
            value += f"\n{row.get('note')}"
        items.append({"label": row.get("label", "Evidence"), "value": value})

    return items


def _failure_review_payload(values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    verification = values.get("verification") or {}
    status = str(verification.get("status", "")).upper()
    if status not in {"FAIL", "NEEDS_HUMAN_REVIEW", "TOPOLOGY_CHANGE_NEEDED"}:
        return None

    sim = values.get("simulation_results") or {}
    specs = values.get("specifications") or {}
    formula_checks = values.get("formula_checks", {}) or {}
    failed_items = verification.get("failed_items") or []
    strategy = verification.get("correction_strategy") or ""
    strategy_bundle = verification.get("strategy_bundle") if isinstance(verification.get("strategy_bundle"), dict) else {}
    learning = values.get("iteration_learning") or {}

    causes: list[str] = []
    for item in failed_items[:6]:
        causes.append(str(item))

    eff = sim.get("efficiency_measured")
    eff_t = specs.get("efficiency_target")
    if isinstance(eff, (int, float)) and isinstance(eff_t, (int, float)) and eff < eff_t:
        causes.append(f"Efficiency target unmet: measured={eff:.2%}, target={eff_t:.2%}")

    ripple = sim.get("v_out_ripple_measured") or sim.get("ripple_voltage")
    ripple_t = specs.get("max_ripple_voltage")
    if isinstance(ripple, (int, float)) and isinstance(ripple_t, (int, float)) and ripple > ripple_t:
        causes.append(f"Ripple target unmet: measured={ripple:.4g}V, target={ripple_t:.4g}V")

    for node_name, pack in formula_checks.items():
        if not isinstance(pack, dict):
            continue
        for fatal in (pack.get("fatal") or [])[:2]:
            causes.append(f"Formula fatal [{node_name}]: {fatal}")
        for warning in (pack.get("warnings") or [])[:1]:
            causes.append(f"Formula warning [{node_name}]: {warning}")

    if not causes:
        causes = ["Validator flagged this round as not fully reliable."]

    lines = [
        "Outcome: This round is blocked and should not be accepted for release.",
        f"Recommended next action: {strategy or 'continue auto-iteration'}",
        "Top blockers:",
    ]
    lines.extend([f"- {c}" for c in causes[:8]])
    if strategy:
        lines.append(f"Suggested next strategy: {strategy}")
    if strategy_bundle.get("recommended_component_actions"):
        lines.append(
            "Recommended selector actions: "
            + ", ".join([str(x) for x in (strategy_bundle.get("recommended_component_actions") or [])[:4]])
        )
    if learning.get("do_not_repeat"):
        lines.append(
            "Do not repeat: "
            + " | ".join([str(x) for x in (learning.get("do_not_repeat") or [])[:3]])
        )

    context = {
        "verification": verification,
        "simulation_results": sim,
        "specifications": specs,
        "formula_checks": formula_checks,
        "probable_causes": causes[:8],
        "planning_summary": values.get("planning_summary") or "",
        "iteration_learning": learning,
        "recommended_patch": _recommended_hitl_patch(values),
    }
    return {
        "title": "Iteration Review & Decision",
        "question": "\n".join(lines),
        "options": [
            "Continue auto-iteration",
            "Apply manual adjustments (JSON)",
            "Accept failed result anyway",
            "Stop this session",
        ],
        "commands": [
            "RETRY_REQUESTED",
            "MANUAL_ADJUSTMENTS",
            "ACCEPT_CURRENT_RESULT",
            "STOP_SESSION",
        ],
        "context": context,
    }


def _extract_state(event_raw: Any) -> Dict[str, Any]:
    if isinstance(event_raw, dict):
        return event_raw
    if isinstance(event_raw, tuple) and len(event_raw) >= 2 and isinstance(event_raw[-1], dict):
        return event_raw[-1]
    return {}


def _next_workflow_step(completed_steps: set[str]) -> Optional[str]:
    for step in WORKFLOW_SEQUENCE:
        if step not in completed_steps:
            return step
    return None


def _heartbeat_hint(step_key: str, elapsed_sec: int) -> tuple[str, str]:
    hints = HEARTBEAT_HINTS.get(step_key) or []
    if not hints:
        return ("", "")
    idx = min(len(hints) - 1, max(0, elapsed_sec // 4))
    return (f"{step_key}:{idx}", str(hints[idx]))


def _parse_reasoning_text(node_name: str, thought_text: str) -> Dict[str, Any]:
    lines = [str(x).strip() for x in str(thought_text or "").splitlines() if str(x).strip()]

    if node_name == "selector":
        filtered: list[str] = []
        seen_noise: set[str] = set()
        for line in lines:
            low = line.lower()
            # Collapse repeated runtime noise and implementation-level details.
            if "chrome mcp snapshot disabled by default for stability" in low:
                key = "mcp_disabled"
                if key in seen_noise:
                    continue
                seen_noise.add(key)
            if "set pe_mas_enable_chrome_mcp=1" in low:
                key = "mcp_enable_hint"
                if key in seen_noise:
                    continue
                seen_noise.add(key)
            if low.startswith("[detail] local db candidate counts"):
                continue
            if low.startswith("- mosfet: candidates=") or low.startswith("- diode: candidates="):
                continue
            if low.startswith("- transformer: candidates=") or low.startswith("- input_cap: candidates="):
                continue
            if low.startswith("- output_cap: candidates=") or low.startswith("- controller: candidates="):
                continue
            if low.startswith("- input_protection: candidates=") or low.startswith("- emi_filter: candidates="):
                continue
            if low.startswith("- clamp_snubber: candidates="):
                continue
            filtered.append(line)
        lines = filtered
    buckets: Dict[str, list[str]] = {k: [] for k in REASON_TAG_ORDER}
    buckets["OTHER"] = []

    for line in lines:
        m = REASON_TAG_RE.match(line)
        if not m:
            buckets["OTHER"].append(line)
            continue
        tag = m.group(1).strip().upper()
        payload = m.group(2).strip()
        if tag in buckets:
            buckets[tag].append(payload or "-")
        else:
            buckets["OTHER"].append(line)

    sections: list[Dict[str, Any]] = []
    for tag in REASON_TAG_ORDER:
        items = buckets.get(tag, [])
        if not items:
            continue
        sections.append(
            {
                "tag": tag,
                "title": REASON_TAG_TITLES.get(tag, tag.title()),
                "items": items,
            }
        )

    if buckets.get("OTHER"):
        sections.append({"tag": "OTHER", "title": "Other Notes", "items": buckets["OTHER"]})

    md_lines = [f"### {NODE_TITLE.get(node_name, node_name)}"]
    for sec in sections:
        md_lines.append(f"- {sec['title']}")
        for item in sec.get("items", []):
            md_lines.append(f"  - {item}")

    return {
        "node": node_name,
        "title": NODE_TITLE.get(node_name, node_name),
        "sections": sections,
        "text": "\n".join(md_lines),
    }


def _thought_keypoints(parsed_trace: Dict[str, Any], max_items: int = 6) -> list[str]:
    out: list[str] = []
    sections = parsed_trace.get("sections") if isinstance(parsed_trace, dict) else []
    if not isinstance(sections, list):
        return out

    for sec in sections:
        if len(out) >= max_items:
            break
        if not isinstance(sec, dict):
            continue
        sec_title = str(sec.get("title") or "Key Info").strip()
        items = sec.get("items") or []
        if not isinstance(items, list):
            continue
        for item in items:
            if len(out) >= max_items:
                break
            text = str(item or "").strip()
            if not text:
                continue
            # Keep each item compact for smooth typewriter rendering.
            compact = text.replace("\n", " ").strip()
            if len(compact) > 180:
                compact = compact[:177] + "..."
            out.append(f"[{sec_title}] {compact}")
    return out


def _node_brief(node_name: str, merged: Dict[str, Any], node_state: Dict[str, Any]) -> Dict[str, Any]:
    specs = merged.get("specifications") or {}
    design = merged.get("theoretical_design") or {}
    bom = merged.get("bom") or {}
    sim = merged.get("simulation_results") or {}
    veri = merged.get("verification") or {}
    node_errors = node_state.get("error_log") or []
    if not node_errors and node_name == "requirements":
        req_verify = ((merged.get("node_verification") or {}).get("requirements") or {}) if isinstance(merged.get("node_verification"), dict) else {}
        req_fatal = req_verify.get("fatal") or []
        if req_verify.get("status") == "FAIL" and req_fatal:
            node_errors = [str(x) for x in req_fatal]

    if node_errors:
        return {
            "title": NODE_TITLE.get(node_name, node_name),
            "summary": f"{NODE_TITLE.get(node_name, node_name)} encountered an error.",
            "sections": [
                {
                    "title": "Errors",
                    "items": _mk_items([("Issue", str(msg)) for msg in node_errors]),
                }
            ],
            "content": _safe_json_str({"error_log": node_errors, "node_state": node_state}),
        }

    if node_name == "requirements":
        has_specs = any(
            specs.get(k) not in (None, "", [])
            for k in [
                "input_voltage_min",
                "input_voltage_max",
                "output_voltage",
                "output_current",
                "efficiency_target",
                "max_ripple_voltage",
            ]
        )
        spec_items = _mk_items([
            ("Input", f"{specs.get('input_voltage_min', '-')}-{specs.get('input_voltage_max', '-')} Vac"),
            ("Output", f"{specs.get('output_voltage', '-')} V / {specs.get('output_current', '-')} A"),
            ("Efficiency Target", specs.get("efficiency_target")),
            ("Ripple Target", specs.get("max_ripple_voltage")),
            ("Application", specs.get("application_type")),
        ]) if has_specs else [{"label": "Status", "value": "No new extraction payload. Reusing previous checkpoint values."}]

        return {
            "title": NODE_TITLE[node_name],
            "summary": "Requirements extracted and parameters structured.",
            "sections": [
                {
                    "title": "Specifications",
                    "items": spec_items,
                },
                {
                    "title": "Execution Plan",
                    "items": _mk_items([
                        ("Plan Summary", merged.get("planning_summary")),
                        ("Next Step", ((merged.get("execution_plan") or {}).get("next_step") if isinstance(merged.get("execution_plan"), dict) else None)),
                        ("Next Action", ((merged.get("execution_plan") or {}).get("next_action") if isinstance(merged.get("execution_plan"), dict) else None)),
                    ]),
                },
            ],
            "content": _safe_json_str(specs),
        }

    if node_name == "designer":
        return {
            "title": NODE_TITLE[node_name],
            "summary": "Completed key theoretical parameter calculations for the flyback.",
            "sections": [
                {
                    "title": "Parameters",
                    "items": _mk_items([
                        ("Switching Frequency (Hz)", design.get("switching_frequency")),
                        ("Primary Inductance (H)", design.get("primary_inductance")),
                        ("Peak Current (A)", design.get("primary_peak_current")),
                        ("Turns Ratio (Np/Ns)", design.get("turns_ratio")),
                        ("Reflected Output Voltage (V)", design.get("reflected_output_voltage")),
                    ]),
                }
            ],
            "content": _safe_json_str(design),
        }

    if node_name == "magnetics_advisor":
        magnetic = merged.get("magnetic_design", {}) if isinstance(merged.get("magnetic_design"), dict) else {}
        probe = magnetic.get("probe", {}) if isinstance(magnetic.get("probe"), dict) else {}
        turns = magnetic.get("turns", {}) if isinstance(magnetic.get("turns"), dict) else {}
        losses = magnetic.get("loss_estimate_w", {}) if isinstance(magnetic.get("loss_estimate_w"), dict) else {}
        manufacturability = magnetic.get("manufacturability", []) if isinstance(magnetic.get("manufacturability"), list) else []
        winding = magnetic.get("winding_arrangement", []) if isinstance(magnetic.get("winding_arrangement"), list) else []
        return {
            "title": NODE_TITLE[node_name],
            "summary": "Optional magnetic design advisor evaluated core family, turns, loss estimate, and manufacturability.",
            "sections": [
                {
                    "title": "Advisor Status",
                    "items": _mk_items([
                        ("Advisor Status", magnetic.get("status")),
                        ("Engine", magnetic.get("engine")),
                        ("Package Available", probe.get("available")),
                        ("Package Version", probe.get("package_version")),
                    ]),
                },
                {
                    "title": "Magnetic Recommendation",
                    "items": _mk_items([
                        ("Core Family", magnetic.get("core_family")),
                        ("Core Material", magnetic.get("core_material")),
                        ("Window Utilization (%)", magnetic.get("window_utilization_pct")),
                        ("Gap (mm)", magnetic.get("gap_mm")),
                        ("Primary Turns", turns.get("primary")),
                        ("Secondary Turns", turns.get("secondary")),
                        ("Auxiliary Turns", turns.get("auxiliary")),
                    ]),
                },
                {
                    "title": "Loss Estimate",
                    "items": _mk_items([
                        ("Copper Loss (W)", losses.get("copper")),
                        ("Core Loss (W)", losses.get("core")),
                        ("Winding DC Loss (W)", losses.get("winding_dc")),
                        ("Winding AC Loss (W)", losses.get("winding_ac")),
                        ("Total Loss (W)", losses.get("total")),
                    ]),
                },
                {
                    "title": "Winding Arrangement",
                    "items": [{"label": "Recommendation", "value": row} for row in winding[:4]],
                },
                {
                    "title": "Manufacturability",
                    "items": [{"label": "Note", "value": row} for row in manufacturability[:4]],
                },
            ],
            "content": _safe_json_str(magnetic),
        }

    if node_name == "selector":
        mosfet = bom.get("mosfet", {}) if isinstance(bom.get("mosfet"), dict) else {}
        transformer = bom.get("transformer", {}) if isinstance(bom.get("transformer"), dict) else {}
        diode = bom.get("diode", {}) if isinstance(bom.get("diode"), dict) else {}
        controller = bom.get("controller", {}) if isinstance(bom.get("controller"), dict) else {}
        selection_summary = bom.get("selection_summary", {}) if isinstance(bom.get("selection_summary"), dict) else {}
        selection_policy = bom.get("selection_policy", {}) if isinstance(bom.get("selection_policy"), dict) else {}
        local_db_top = bom.get("local_db_top_candidates", {}) if isinstance(bom.get("local_db_top_candidates"), dict) else {}

        detail_items = []
        missing_items = []
        source_missing_keys = []
        low_conf_policy_items = []
        candidate_stats = []

        def _fmt_bom_line(selected: str, price: Any, source: Any) -> str:
            parts = [selected]
            price_text = str(price or "").strip()
            normalized_price = price_text if price_text and price_text not in {"-", "N/A", "n/a"} else "N/A"
            parts.append(f"price={normalized_price}")
            return " | ".join(parts)

        def _pretty_key(key: str) -> str:
            return key.replace("_", " ").title()

        def _policy_compact(p: Dict[str, Any]) -> str:
            color = str(p.get("color") or "-").strip().lower()
            try:
                score_val = float(p.get("score", 0.0))
                score_text = f"{score_val * 100:.1f}%"
            except Exception:
                score_text = str(p.get("score", "-"))
            strategy = str(p.get("strategy") or "").strip()
            if color == "green":
                label = "Ready"
            elif color == "yellow":
                label = "Verify"
            elif color == "red":
                label = "Manual"
            else:
                label = color.title() if color else "Unknown"
            if strategy:
                strategy = strategy.replace("semi-auto:", "").strip()
            return f"{label} | confidence={score_text}" + (f" | {strategy}" if strategy else "")

        def _looks_generic_part(selected: str, price: Any) -> bool:
            low = str(selected or "").strip().lower()
            price_text = str(price or "").strip().lower()
            return any(flag in low for flag in ["generic", "fallback", "check online", "unknown"]) or "check online" in price_text

        component_actions: Dict[str, Dict[str, Any]] = {}

        def _push_component_action(label: str, severity: str, message: str) -> None:
            priority = {"ready": 0, "review": 1, "manual": 2}
            bucket = component_actions.setdefault(label, {"severity": "ready", "messages": []})
            current = str(bucket.get("severity") or "ready")
            if priority.get(severity, 0) > priority.get(current, 0):
                bucket["severity"] = severity
            if message and message not in bucket["messages"]:
                bucket["messages"].append(message)

        for k in [
            "mosfet",
            "diode",
            "transformer",
            "input_cap",
            "output_cap",
            "controller",
            "input_protection",
            "emi_filter",
            "clamp_snubber",
        ]:
            info = selection_summary.get(k, {}) if isinstance(selection_summary.get(k), dict) else {}
            selected = str(info.get("selected") or "-").strip()
            if selected in {"-", "Unknown", "unknown", "N/A", "n/a", ""}:
                missing_items.append((_pretty_key(k), "No confident auto-selected part; keep manual review or fallback path."))
                _push_component_action(_pretty_key(k), "manual", "No confident part is selected yet. Pick a real manufacturer part before freezing the BOM.")
                continue
            detail_items.append((k, _fmt_bom_line(selected, info.get("price"), info.get("source"))))
            source_text = str(info.get("source") or "").strip()
            if not source_text or source_text in {"-", "N/A", "n/a"}:
                source_missing_keys.append((_pretty_key(k), "Missing source link"))
                _push_component_action(_pretty_key(k), "review", "Add a distributor or datasheet source link so the selection can be traced.")
            if _looks_generic_part(selected, info.get("price")):
                _push_component_action(_pretty_key(k), "review", "Current part name still looks generic or provisional. Replace it with a real orderable MPN.")

        policy_items = []
        policy_color_counts: Dict[str, int] = {"green": 0, "yellow": 0, "red": 0, "other": 0}
        for k in [
            "mosfet",
            "diode",
            "transformer",
            "input_cap",
            "output_cap",
            "controller",
            "input_protection",
            "emi_filter",
            "clamp_snubber",
        ]:
            p = selection_policy.get(k, {}) if isinstance(selection_policy.get(k), dict) else {}
            if p:
                policy_items.append((_pretty_key(k), _policy_compact(p)))
                try:
                    score_val = float(p.get("score", 0.0))
                except Exception:
                    score_val = 0.0
                color_val = str(p.get("color") or "").lower()
                if color_val in policy_color_counts:
                    policy_color_counts[color_val] += 1
                else:
                    policy_color_counts["other"] += 1
                if color_val != "green" or score_val < 0.97:
                    low_conf_policy_items.append((_pretty_key(k), _policy_compact(p)))
                if color_val == "red":
                    _push_component_action(_pretty_key(k), "manual", "Policy marked this item as manual decision required. Replace it or approve it explicitly.")
                elif color_val == "yellow":
                    _push_component_action(_pretty_key(k), "review", "Policy marked this item as usable with verification. Check the datasheet and source link before BOM freeze.")
                elif color_val == "green" and score_val < 0.97:
                    _push_component_action(_pretty_key(k), "review", "Selection is acceptable, but confidence is not very high. A quick sanity check is still recommended.")

        candidate_count_items = []
        low_candidate_items = []
        for k in [
            "mosfet",
            "diode",
            "transformer",
            "input_cap",
            "output_cap",
            "controller",
            "input_protection",
            "emi_filter",
            "clamp_snubber",
        ]:
            rows = local_db_top.get(k, []) if isinstance(local_db_top.get(k), list) else []
            row_count = len(rows)
            candidate_count_items.append((_pretty_key(k), f"{row_count} candidates"))
            candidate_stats.append(row_count)
            if row_count <= 2:
                low_candidate_items.append((_pretty_key(k), f"{row_count} candidates"))
                _push_component_action(_pretty_key(k), "review", "Candidate pool is shallow. Widen search or keep a backup option in case the chosen part fails review.")

        compact_bom_payload = {
            "selection_summary": selection_summary,
            "selection_policy": selection_policy,
            "candidate_counts": {k: len(local_db_top.get(k, [])) for k in local_db_top.keys()},
            "formula_checks": (merged.get("formula_checks") or {}).get("selector", {}),
        }

        # Prefer selected summary labels first so UI stays consistent with final pick.
        def _selected_label(key: str, fallback: Any = "-") -> Any:
            info = selection_summary.get(key, {}) if isinstance(selection_summary.get(key), dict) else {}
            selected = str(info.get("selected") or "").strip()
            if selected and selected.lower() not in {"unknown", "n/a", "-"}:
                return selected
            return fallback

        coverage_summary_items = []
        if candidate_stats:
            min_c = min(candidate_stats)
            max_c = max(candidate_stats)
            avg_c = round(sum(candidate_stats) / max(len(candidate_stats), 1), 2)
            if min_c <= 1:
                pool_status = "Weak: some categories have too few candidates; manual review required."
            elif min_c == 2:
                pool_status = "Fair: most categories are usable, but low-depth categories should be double-checked."
            else:
                pool_status = "Good: candidate pool depth is generally sufficient for auto-selection."
            coverage_summary_items = _mk_items([
                ("Candidate Pool Health", pool_status),
                ("Depth Snapshot", f"min={min_c}, max={max_c}, avg={avg_c}"),
            ])

        policy_summary_items = _mk_items([
            ("Ready", policy_color_counts.get("green", 0)),
            ("Verify", policy_color_counts.get("yellow", 0)),
            ("Manual", policy_color_counts.get("red", 0)),
            ("Meaning", "Ready = safe to auto-apply. Verify = usable, but confirm datasheet/source before freeze. Manual = user decision required."),
        ])

        manual_action_count = sum(1 for row in component_actions.values() if row.get("severity") == "manual")
        review_action_count = sum(1 for row in component_actions.values() if row.get("severity") == "review")
        ready_action_count = max(0, len(detail_items) - manual_action_count - review_action_count)
        if manual_action_count > 0:
            readiness_summary = "BOM is not ready to freeze yet. Some selections still require manual decisions."
            freeze_guidance = "Resolve manual items first, then rerun selection/simulation before freezing the BOM."
        elif review_action_count > 0:
            readiness_summary = "BOM is mostly usable, but a few selections still need verification."
            freeze_guidance = "Verify yellow items and missing source links before treating this BOM as final."
        else:
            readiness_summary = "BOM is in good shape for provisional freeze."
            freeze_guidance = "Selections are traceable and auto-ready. You can proceed to final validation or procurement review."

        recommended_action_items = []
        severity_rank = {"manual": 0, "review": 1, "ready": 2}
        for label, row in sorted(component_actions.items(), key=lambda kv: severity_rank.get(str(kv[1].get("severity")), 9)):
            severity = str(row.get("severity") or "review")
            if severity == "ready":
                continue
            prefix = "Manual decision required." if severity == "manual" else "Verification recommended."
            recommended_action_items.append((label, f"{prefix} {' '.join(row.get('messages') or [])}".strip()))

        component_labels = {
            "mosfet": "MOSFET",
            "diode": "Output Rectifier",
            "transformer": "Transformer / Core",
            "input_cap": "Input Bulk Capacitor",
            "output_cap": "Output Capacitor",
            "controller": "Controller",
            "input_protection": "Input Protection",
            "emi_filter": "EMI Filter",
            "clamp_snubber": "Clamp / Snubber",
        }

        def _first_value(row: Dict[str, Any], names: list[str]) -> str:
            for name in names:
                value = row.get(name)
                if value not in (None, "", "-", "N/A", "n/a"):
                    return str(value)
            return ""

        def _candidate_label(row: Dict[str, Any]) -> str:
            return _first_value(
                row,
                [
                    "Part Number",
                    "Mfr Part #",
                    "Manufacturer Part Number",
                    "part_number",
                    "title",
                    "description",
                ],
            )

        component_cards: list[Dict[str, Any]] = []
        for key, label in component_labels.items():
            info = selection_summary.get(key, {}) if isinstance(selection_summary.get(key), dict) else {}
            raw_component = bom.get(key, {}) if isinstance(bom.get(key), dict) else {}
            policy = selection_policy.get(key, {}) if isinstance(selection_policy.get(key), dict) else {}
            candidates = local_db_top.get(key, []) if isinstance(local_db_top.get(key), list) else []
            selected = str(info.get("selected") or _candidate_label(raw_component) or "").strip()
            if not selected:
                continue
            source = str(info.get("source") or _first_value(raw_component, ["Product URL", "URL", "DigiKey URL", "source", "Source"]) or "").strip()
            datasheet = _first_value(raw_component, ["Datasheet", "Datasheet URL", "datasheet", "datasheet_url"])
            price = str(info.get("price") or _first_value(raw_component, ["Price", "Unit Price", "price"]) or "").strip()
            policy_color = str(policy.get("color") or "").lower()
            generic = _looks_generic_part(selected, price)
            status = "manual" if policy_color == "red" or generic else ("review" if policy_color == "yellow" else "ready")
            try:
                confidence = float(policy.get("score")) if policy.get("score") is not None else None
            except Exception:
                confidence = None
            action_row = component_actions.get(label) or component_actions.get(_pretty_key(key)) or {}
            component_cards.append(
                {
                    "key": key,
                    "label": label,
                    "part": selected,
                    "status": status,
                    "price": price,
                    "source": source,
                    "datasheet": datasheet,
                    "confidence": confidence,
                    "policy": policy,
                    "candidate_count": len(candidates),
                    "candidates": [
                        {
                            "part": _candidate_label(row) or f"Candidate {idx + 1}",
                            "manufacturer": _first_value(row, ["Manufacturer", "Mfr", "manufacturer"]),
                            "price": _first_value(row, ["Price", "Unit Price", "price"]),
                            "source": _first_value(row, ["Product URL", "URL", "DigiKey URL", "source", "Source"]),
                            "score": row.get("_selector_score"),
                            "reasons": row.get("_selector_reasons") or [],
                        }
                        for idx, row in enumerate(candidates[:4])
                        if isinstance(row, dict)
                    ],
                    "actions": action_row.get("messages") or [],
                }
            )

        compact_bom_payload["component_cards"] = component_cards

        sections = [
            {
                "title": "Selection Readiness",
                "items": _mk_items([
                    ("Overall", readiness_summary),
                    ("Manual Items", manual_action_count),
                    ("Verify Items", review_action_count),
                    ("Auto-Ready Items", ready_action_count),
                    ("Freeze Guidance", freeze_guidance),
                ]),
            },
            {
                "title": "Key Components",
                "items": _mk_items([
                    ("MOSFET", _selected_label("mosfet", mosfet.get("Part Number") or mosfet.get("Mfr Part #") or mosfet.get("title"))),
                    ("MOSFET Vds", mosfet.get("Vds") or mosfet.get("Drain to Source Voltage (Vdss)")),
                    ("MOSFET Price", mosfet.get("Price") or mosfet.get("price")),
                    ("Transformer", _selected_label("transformer", transformer.get("part_number") or transformer.get("core") or transformer.get("Core"))),
                    ("Turns", f"Np={transformer.get('Np', '-')}, Ns={transformer.get('Ns', '-')}"),
                    ("Diode", _selected_label("diode", diode.get("part_number") or diode.get("Part Number") or diode.get("title") or diode.get("description"))),
                    ("Controller", _selected_label("controller", controller.get("part_number") or controller.get("Part Number"))),
                ]),
            },
            {
                "title": "Detailed BOM",
                "items": _mk_items([(_pretty_key(k), v) for (k, v) in detail_items]),
            },
            {
                "title": "Needs Manual Review",
                "items": _mk_items(missing_items),
            },
            {
                "title": "What To Do Next",
                "items": _mk_items(recommended_action_items),
            },
            {
                "title": "Policy Alerts",
                "items": _mk_items(low_conf_policy_items),
            },
            {
                "title": "Policy Summary",
                "items": policy_summary_items,
            },
            {
                "title": "Candidate Pool Health",
                "items": coverage_summary_items,
            },
            {
                "title": "Low Candidate Warnings",
                "items": _mk_items(low_candidate_items),
            },
            {
                "title": "Data Gaps",
                "items": _mk_items(source_missing_keys),
            },
        ]
        # Remove empty sections to avoid noisy frontend cards.
        sections = [s for s in sections if isinstance(s, dict) and (s.get("items") or [])]

        return {
            "title": NODE_TITLE[node_name],
            "summary": "Completed BOM candidate selection.",
            "sections": sections,
            "content": _safe_json_str(compact_bom_payload),
        }

    if node_name == "simulator":
        evidence_items = _evidence_closure_items(sim.get("evidence_closure"), limit=8)
        sections = [
            {
                "title": "Simulation Results",
                "items": _mk_items([
                    ("Efficiency", sim.get("efficiency_measured")),
                    ("Estimated Efficiency (formula)", sim.get("efficiency_formula_est")),
                    ("Estimated Efficiency Raw (formula)", sim.get("efficiency_formula_raw_est")),
                    ("Formula Confidence", sim.get("efficiency_formula_confidence")),
                    ("Operation Mode", sim.get("formula_mode")),
                    ("Output Ripple (V)", sim.get("v_out_ripple_measured") or sim.get("ripple_voltage")),
                    ("Vds Peak (V)", sim.get("v_ds_spike_max")),
                    ("Converged", sim.get("is_converged")),
                    ("Data Source", sim.get("source")),
                ]),
            }
        ]
        if evidence_items:
            sections.append({"title": "Evidence Closure / Release Hold", "items": evidence_items})
        return {
            "title": NODE_TITLE[node_name],
            "summary": "Completed waveform and performance simulation.",
            "sections": sections,
            "content": _safe_json_str(sim),
        }

    if node_name == "validator":
        quality_gate = veri.get("quality_gate") if isinstance(veri.get("quality_gate"), dict) else {}
        strategy_bundle = veri.get("strategy_bundle") if isinstance(veri.get("strategy_bundle"), dict) else {}
        validator_sections = [
            {
                "title": "Review Conclusions",
                "items": _mk_items([
                    ("Status", veri.get("status")),
                    ("Failed Items", veri.get("failed_items")),
                    ("Improvement Strategy", veri.get("correction_strategy")),
                ]),
            }
        ]
        if quality_gate:
            validator_sections.append(
                {
                    "title": "Quality Gate",
                    "items": _mk_items([
                        ("Gate Status", quality_gate.get("status")),
                        ("Blockers", quality_gate.get("blockers")),
                        ("Cautions", quality_gate.get("cautions")),
                        ("Component Actions", quality_gate.get("component_actions")),
                    ]),
                }
            )
        if strategy_bundle:
            validator_sections.append(
                {
                    "title": "Evidence / Strategy Bundle",
                    "items": _mk_items([
                        ("Failed Axes", strategy_bundle.get("failed_axes")),
                        ("Root Causes", strategy_bundle.get("root_causes")),
                        ("Next Focus", strategy_bundle.get("next_iteration_focus")),
                    ]),
                }
            )
        return {
            "title": NODE_TITLE[node_name],
            "summary": "Completed rule-based and intelligent review; produced routing strategy.",
            "sections": validator_sections,
            "content": _safe_json_str(veri),
        }

    if node_name == "correction":
        correction = merged.get("correction_review") or {}
        return {
            "title": NODE_TITLE[node_name],
            "summary": "Completed requirement consistency and scenario safety review.",
            "sections": [
                {
                    "title": "Correction Conclusions",
                    "items": _mk_items([
                        ("Status", correction.get("status")),
                        ("Summary", correction.get("summary")),
                        ("Mismatches", correction.get("mismatches")),
                        ("Recommendations", correction.get("recommendations")),
                    ]),
                }
            ],
            "content": _safe_json_str(correction),
        }

    if node_name == "reporter":
        report = merged.get("report_content", "")
        return {
            "title": NODE_TITLE[node_name],
            "summary": "Report generated and ready for delivery.",
            "sections": [
                {
                    "title": "Report",
                    "items": _mk_items([
                        ("Length", len(report or "")),
                    ]),
                }
            ],
            "content": report or _safe_json_str(node_state),
        }

    return {
        "title": NODE_TITLE.get(node_name, node_name),
        "summary": "Node execution completed.",
        "sections": [],
        "content": _safe_json_str(node_state),
    }


def _final_payload(values: Dict[str, Any]) -> Dict[str, Any]:
    error_log = values.get("error_log") or []
    specs = values.get("specifications") or {}
    design = values.get("theoretical_design") or {}
    bom = values.get("bom") or {}
    sim = values.get("simulation_results") or {}
    veri = values.get("verification") or {}
    correction = values.get("correction_review") or {}
    report = values.get("report_content") or ""
    execution_plan = values.get("execution_plan") or {}
    memory_insights = values.get("memory_insights") or {}
    skill_recommendations = values.get("skill_recommendations") or []
    skill_catalog = values.get("skill_catalog") or []
    magnetic_design = values.get("magnetic_design") or {}

    if error_log and not any([specs, design, bom, sim, veri, correction, report]):
        message = str(error_log[0])
        return {
            "summary": f"Error: {message}",
            "report": "",
            "design_meta": {
                "mode": "error",
                "errors": error_log,
            },
        }

    # Q&A mode short-circuit: do not render design completion status
    if specs.get("is_chitchat"):
        msg_text = ""
        msgs = values.get("messages") or []
        if isinstance(msgs, list) and msgs:
            msg_text = str(msgs[-1])
        msg_text = msg_text or specs.get("response_text") or "Q&A complete."
        return {
            "summary": msg_text,
            "report": "",
            "design_meta": {
                "mode": "chatbot",
                "specs": specs,
            },
        }

    status = veri.get("status", "UNKNOWN")
    summary = f"Design workflow complete. Verification status: {status}."
    if veri.get("correction_strategy"):
        summary += f"\nRecommended strategy: {veri.get('correction_strategy')}"

    # Keep final delivery robust even when reporter node is skipped by HITL path.
    if not str(report).strip():
        report = _try_generate_skill_report(values, accepted_by_user=False) or _build_acceptance_report(values, accepted_by_user=False)
    report = _append_evidence_closure_report(report, values)

    return {
        "summary": summary,
        "report": report,
        "design_meta": {
            "specs": specs,
            "design": design,
            "simulation": sim,
            "verification": veri,
            "correction_review": correction,
            "bom": bom,
            "magnetic_design": magnetic_design,
            "plan": execution_plan,
            "planning_summary": values.get("planning_summary") or "",
            "memory": memory_insights,
            "skills": {
                "recommendations": skill_recommendations,
                "catalog": skill_catalog,
            },
        },
    }


def _build_acceptance_report(values: Dict[str, Any], accepted_by_user: bool = False) -> str:
    specs = values.get("specifications") or {}
    design = values.get("theoretical_design") or {}
    bom = values.get("bom") or {}
    sim = values.get("simulation_results") or {}
    veri = values.get("verification") or {}
    correction = values.get("correction_review") or {}
    formula_checks = values.get("formula_checks") or {}
    execution_plan = values.get("execution_plan") or {}

    status = str(veri.get("status") or "UNKNOWN")
    header = "# Final Engineering Summary"
    if accepted_by_user:
        header = "# Final Engineering Summary (User-Accepted Iteration)"

    lines: list[str] = [header, ""]

    lines.append("## Outcome")
    lines.append(f"- Verification status: {status}")
    lines.append(f"- Accepted by user: {'Yes' if accepted_by_user else 'No'}")
    if veri.get("correction_strategy"):
        lines.append(f"- Recommended strategy: {veri.get('correction_strategy')}")
    if veri.get("failed_items"):
        lines.append(f"- Key failed items: {', '.join([str(x) for x in (veri.get('failed_items') or [])[:8]])}")
    lines.append("")

    lines.append("## Requirements Snapshot")
    lines.append(f"- Input: {specs.get('input_voltage_min', '-')}-{specs.get('input_voltage_max', '-')} Vac")
    lines.append(f"- Output: {specs.get('output_voltage', '-')} V / {specs.get('output_current', '-')} A")
    lines.append(f"- Efficiency target: {specs.get('efficiency_target', '-')}")
    lines.append(f"- Ripple target: {specs.get('max_ripple_voltage', '-')} V")
    lines.append("")

    lines.append("## Design Snapshot")
    lines.append(f"- Switching frequency: {design.get('switching_frequency', '-')}")
    lines.append(f"- Primary inductance: {design.get('primary_inductance', '-')}")
    lines.append(f"- Peak current: {design.get('primary_peak_current', '-')}")
    lines.append(f"- Turns ratio (Np/Ns): {design.get('turns_ratio', '-')}")
    lines.append(f"- Reflected output voltage: {design.get('reflected_output_voltage', '-')}")
    lines.append("")

    mosfet = bom.get("mosfet") if isinstance(bom.get("mosfet"), dict) else {}
    diode = bom.get("diode") if isinstance(bom.get("diode"), dict) else {}
    transformer = bom.get("transformer") if isinstance(bom.get("transformer"), dict) else {}
    controller = bom.get("controller") if isinstance(bom.get("controller"), dict) else {}

    lines.append("## BOM Snapshot")
    lines.append(f"- MOSFET: {mosfet.get('Part Number') or mosfet.get('Mfr Part #') or mosfet.get('title') or '-'}")
    lines.append(f"- Diode: {diode.get('Part Number') or diode.get('title') or diode.get('description') or '-'}")
    lines.append(f"- Transformer/Core: {transformer.get('core') or transformer.get('Core') or '-'}")
    lines.append(f"- Controller: {controller.get('part_number') or controller.get('Part Number') or '-'}")
    lines.append("")

    lines.append("## Simulation & Validation")
    lines.append(f"- Measured efficiency: {sim.get('efficiency_measured', '-')}")
    lines.append(f"- Estimated efficiency (formula): {sim.get('efficiency_formula_est', '-')}")
    lines.append(f"- Estimated efficiency raw (formula): {sim.get('efficiency_formula_raw_est', '-')}")
    lines.append(f"- Formula confidence: {sim.get('efficiency_formula_confidence', '-')}")
    lines.append(f"- Output ripple: {sim.get('v_out_ripple_measured') or sim.get('ripple_voltage') or '-'}")
    lines.append(f"- Vds peak: {sim.get('v_ds_spike_max', '-')}")
    lines.append(f"- Simulation converged: {sim.get('is_converged', '-')}")
    lines.append("")

    evidence_section = _evidence_closure_report_section(values)
    if evidence_section:
        lines.append(evidence_section)
        lines.append("")

    if isinstance(formula_checks, dict) and formula_checks:
        lines.append("## Formula Audit")
        for node_name, pack in list(formula_checks.items())[:6]:
            if not isinstance(pack, dict):
                continue
            fatal = pack.get("fatal") or []
            warn = pack.get("warnings") or []
            lines.append(f"- {node_name}: fatal={len(fatal)}, warnings={len(warn)}")
        lines.append("")

    if correction:
        lines.append("## Correction Review")
        lines.append(f"- Status: {correction.get('status', '-')}")
        if correction.get("summary"):
            lines.append(f"- Summary: {correction.get('summary')}")
        if correction.get("recommendations"):
            rec = correction.get("recommendations")
            if isinstance(rec, list):
                lines.append(f"- Recommendations: {', '.join([str(x) for x in rec[:8]])}")
            else:
                lines.append(f"- Recommendations: {rec}")
        lines.append("")

    if execution_plan:
        lines.append("## Execution Plan")
        lines.append(f"- Summary: {values.get('planning_summary') or execution_plan.get('headline') or '-'}")
        lines.append(f"- Next step: {execution_plan.get('next_step', '-')}")
        lines.append(f"- Next action: {execution_plan.get('next_action', '-')}")
        risks = execution_plan.get("risks") or []
        if risks:
            lines.append(f"- Top risk: {str(risks[0])}")
        lines.append("")

    memory_insights = values.get("memory_insights") or {}
    if memory_insights:
        lines.append("## Lifelong Memory")
        lines.append(f"- Status: {memory_insights.get('status', '-')}")
        lookback = memory_insights.get("lookback") or {}
        lines.append(f"- Similar memory hits: {lookback.get('count', 0)}")
        stats = memory_insights.get("stats") or {}
        lines.append(f"- Total stored memories: {stats.get('total_rows', '-')}")
        lines.append("")

    lines.append("## Notes")
    if accepted_by_user and status not in {"PASS"}:
        lines.append("- This design was accepted by user decision despite unresolved validation items.")
        lines.append("- Use correction strategy and failed-items list above before production release.")
    else:
        lines.append("- This summary reflects the latest workflow state captured by PE-MAS.")
    return "\n".join(lines).strip()

def _try_generate_skill_report(values: Dict[str, Any], accepted_by_user: bool = False) -> str:
    """Best-effort report generation via final_report_writer for HITL accept/fallback paths."""
    try:
        skills_dir = str(BASE_DIR / "core" / "skills")
        skill_manager = SkillManager(skills_dir)

        references = (values.get("literature_references") or []) + (values.get("retrieved_knowledge_references") or [])

        citation_pack: Dict[str, Any] = {}
        citation_skill = skill_manager.get_skill("engineering_citation_manager")
        if citation_skill and citation_skill.tools_module and hasattr(citation_skill.tools_module, "build_citation_pack"):
            citation_pack = citation_skill.tools_module.build_citation_pack({"references": references}) or {}

        report_skill = skill_manager.get_skill("final_report_writer")
        if not (report_skill and report_skill.tools_module and hasattr(report_skill.tools_module, "generate_final_report")):
            return ""

        skill_context = {
            "specifications": values.get("specifications") or {},
            "theoretical_design": values.get("theoretical_design") or {},
            "bom": values.get("bom") or {},
            "simulation_results": values.get("simulation_results") or {},
            "verification": values.get("verification") or {},
            "correction_review": values.get("correction_review") or {},
            "formula_checks": values.get("formula_checks") or {},
            "node_verification": values.get("node_verification") or {},
            "literature_references": citation_pack.get("normalized_references") or references,
            "citation_audit": citation_pack.get("citation_audit") or {},
            "broken_links": citation_pack.get("broken_links") or [],
            "reasoning_trace": values.get("reasoning_trace") or [],
            "config": values.get("config") or {},
            "thread_id": values.get("thread_id") or "N/A",
            "execution_plan": values.get("execution_plan") or {},
            "planning_summary": values.get("planning_summary") or "",
            "memory_insights": values.get("memory_insights") or {},
            "skill_recommendations": values.get("skill_recommendations") or [],
        }
        report_pack = report_skill.tools_module.generate_final_report(skill_context)
        report = str((report_pack or {}).get("report_markdown") or "").strip()
        if not report:
            return ""

        if accepted_by_user:
            verify_status = str((values.get("verification") or {}).get("status") or "UNKNOWN").upper()
            report += (
                "\n## HITL Decision\n"
                "- User accepted current iteration at decision checkpoint.\n"
                f"- Verification status at acceptance: {verify_status}.\n"
            )

        try:
            thread_id = str(values.get("thread_id") or "N_A")
            report_dir = RUNTIME_DIR / "reports" / re.sub(r"[^A-Za-z0-9_.-]+", "_", thread_id)[:80]
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "final_report.md").write_text(report.rstrip() + "\n", encoding="utf-8")
        except Exception:
            pass

        return report + "\n"
    except Exception:
        return ""


def _final_payload_for_user_accept(values: Dict[str, Any]) -> Dict[str, Any]:
    payload = _final_payload(values)
    verification = ((payload.get("design_meta") or {}).get("verification") or {}) if isinstance(payload.get("design_meta"), dict) else {}
    status = str(verification.get("status") or "UNKNOWN")

    payload["summary"] = (
        f"Current iteration accepted by user. Verification status: {status}. "
        "A full engineering summary is attached for traceability."
    )
    if not str(payload.get("report") or "").strip():
        payload["report"] = _try_generate_skill_report(values, accepted_by_user=True) or _build_acceptance_report(values, accepted_by_user=True)
    elif "HITL Decision" not in str(payload.get("report") or ""):
        verify_status = str((values.get("verification") or {}).get("status") or "UNKNOWN").upper()
        payload["report"] = str(payload.get("report") or "").rstrip() + (
            "\n\n## HITL Decision\n"
            "- User accepted current iteration at decision checkpoint.\n"
            f"- Verification status at acceptance: {verify_status}.\n"
        )
    payload["report"] = _append_evidence_closure_report(payload.get("report") or "", values)
    return payload


def _persist_memory_on_user_accept(values: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort memory writeback for HITL accept path."""
    try:
        engine = get_memory_engine()
        verification = values.get("verification") or {}
        status = str(verification.get("status") or "UNKNOWN")

        episode_payload = {
            "summary": build_episode_summary(values),
            "status": status,
            "accepted_by_user": True,
            "specifications": values.get("specifications") or {},
            "theoretical_design": values.get("theoretical_design") or {},
            "simulation_results": values.get("simulation_results") or {},
            "verification": verification,
            "correction_review": values.get("correction_review") or {},
        }
        ep_res = engine.put(("episodes", "flyback", "accepted_by_user"), episode_payload, kind="episode")

        sem_res: Dict[str, Any] = {}
        sem_rule = build_semantic_rule_from_state(values)
        if sem_rule:
            sem_res = engine.put(
                ("semantic", "flyback", "rules"),
                {
                    "summary": sem_rule,
                    "status": status,
                    "accepted_by_user": True,
                    "tags": ["flyback", "hitl", "accepted_iteration"],
                },
                kind="semantic_rule",
            )
        usage_update = mark_state_memory_usage(
            values,
            engine=engine,
            success=str(status).upper() == "PASS",
            failed=str(status).upper() in {"FAIL", "NEEDS_HUMAN_REVIEW", "TOPOLOGY_CHANGE_NEEDED"},
        )
        return {"episode": ep_res, "semantic": sem_res, "usage_update": usage_update}
    except Exception as e:
        return {"error": str(e)}


def _persist_memory_on_failure_review(values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort memory writeback for FAIL / NEEDS_HUMAN_REVIEW checkpoints that
    stop before correction->memory_synthesizer is reached.
    """
    try:
        engine = get_memory_engine()
        verification = values.get("verification") or {}
        status = str(verification.get("status") or "UNKNOWN").upper()

        episode_payload = {
            "summary": build_episode_summary(values),
            "status": status,
            "checkpoint_writeback": True,
            "specifications": values.get("specifications") or {},
            "theoretical_design": values.get("theoretical_design") or {},
            "simulation_results": values.get("simulation_results") or {},
            "verification": verification,
            "correction_review": values.get("correction_review") or {},
            "bom_excerpt": {
                "mosfet": ((values.get("bom") or {}).get("mosfet")),
                "diode": ((values.get("bom") or {}).get("diode")),
                "controller": ((values.get("bom") or {}).get("controller")),
            },
        }
        ep_res = engine.put(
            ("episodes", "flyback", "successful_designs" if status == "PASS" else "failed_or_review"),
            episode_payload,
            kind="episode",
        )

        playbook_payload = values.get("iteration_learning") or build_iteration_playbook(values)
        playbook_res = engine.put(
            ("lessons", "flyback", "iteration_playbooks"),
            playbook_payload,
            kind="iteration_playbook",
        )

        sem_res: Dict[str, Any] = {}
        sem_rule = build_semantic_rule_from_state(values)
        if sem_rule:
            sem_res = engine.put(
                ("semantic", "flyback", "rules"),
                {
                    "summary": sem_rule,
                    "status": status,
                    "tags": ["flyback", "heuristic", "validator", "checkpoint"],
                    "quality_score": 0.7 if status == "PASS" else 0.55,
                },
                kind="semantic_rule",
            )

        usage_update = mark_state_memory_usage(
            values,
            engine=engine,
            success=status == "PASS",
            failed=status in {"FAIL", "NEEDS_HUMAN_REVIEW", "TOPOLOGY_CHANGE_NEEDED"},
        )

        return {
            "episode": ep_res,
            "playbook": playbook_res,
            "semantic": sem_res,
            "usage_update": usage_update,
            "status": status,
        }
    except Exception as e:
        return {"error": str(e)}


async def stream_workflow(user_query: str, sid: Optional[str] = None):
    session = _get_or_create_session(sid)
    sid_out = session["sid"]
    thread_id = session["thread_id"]

    config = {"configurable": {"thread_id": thread_id}}
    merged_state: Dict[str, Any] = {}
    completed_steps: set[str] = set()
    active_step_key: str = "requirements"
    active_step_started_at: float = time.time()
    last_heartbeat_at: float = 0.0

    keywords_continue = ["approve", "yes", "ok", "go", "continue", "simulate", "proceed"]
    keywords_stop = ["stop", "abort", "terminate", "cancel", "停止"]
    user_query_low = str(user_query or "").strip().lower()
    command_id = _parse_control_command(user_query)
    metric_deltas: Dict[str, float] = {"run_total": 1.0}
    _record_runtime_event("run_start", sid_out, {"query": str(user_query or "")[:500], "command": command_id or ""})
    _trace_log("run_start", sid_out, command=command_id or "", query=str(user_query or "")[:220], thread_id=thread_id)
    inputs: Any

    try:
        control_tokens = {
            "retry", "continue auto-iteration", "continue auto iteration", "继续自动迭代", "继续自动",
            "accept current result", "accept", "接受当前结果", "接受结果",
            "apply manual adjustments (json)", "manual adjustments", "手动调整", "手动修改",
        }
        intent = _classify_user_intent(user_query)
        session_topic = _infer_session_topic(session, _session_history(session))
        if intent != "chat" and session_topic and _looks_like_contextual_followup(user_query):
            intent = "chat"
        fresh_design_request = (
            intent != "chat"
            and not command_id
            and not user_query.strip().startswith("HITL_UPDATE_JSON:")
            and not any(k in user_query_low for k in keywords_continue)
            and not any(k in user_query_low for k in keywords_stop)
            and user_query_low not in control_tokens
        )
        _trace_log(
            "intent_classified",
            sid_out,
            intent=intent,
            command=command_id or "",
            fresh_design_request=fresh_design_request,
            session_topic=session_topic or "",
        )
        if (
            fresh_design_request
            and session.get("base_prompt")
            and not session.get("requirements_gate_pending")
            and str(user_query or "").strip() != str(session.get("base_prompt") or "").strip()
        ):
            new_thread = str(uuid.uuid4())
            with SESSION_LOCK:
                SESSIONS[sid_out]["thread_id"] = new_thread
                SESSIONS[sid_out]["base_prompt"] = user_query
                SESSIONS[sid_out]["requirements_gate_pending"] = False
                SESSIONS[sid_out]["requirements_gate_locked"] = False
                SESSIONS[sid_out]["requirements_intake"] = {}
                SESSIONS[sid_out]["specifications"] = {}
                SESSIONS[sid_out]["engineering_framework"] = {}
                SESSIONS[sid_out]["updated_at"] = time.time()
            _persist_sessions_to_disk()
            thread_id = new_thread
            config = {"configurable": {"thread_id": thread_id}}

        if (
            intent == "chat"
            and not user_query.strip().startswith("HITL_UPDATE_JSON:")
            and not command_id
            and not any(k in user_query_low for k in keywords_continue)
            and not any(k in user_query_low for k in keywords_stop)
        ):
            chat_status_items = [
                "Understanding the question",
                "Checking conversation context",
                "Selecting the answer path",
            ]
            for label in chat_status_items:
                yield {
                    "event": "qa_status",
                    "data": json.dumps(
                        {
                            "sid": sid_out,
                            "message": label,
                        },
                        ensure_ascii=False,
                    ),
                }
                await asyncio.sleep(0.12)
            qa = _compose_professional_qa(user_query, session=session, history=_session_history(session))
            qa_answer = qa.get("answer", _chatbot_answer(user_query))
            qa_meta = {
                "mode": "chatbot",
                "source": qa.get("source", "rag_mcp"),
                "topic": qa.get("topic") or "",
                "effective_query": qa.get("effective_query") or user_query,
                "context_used": bool(qa.get("context_used")),
                "evidence_cards": qa.get("evidence_cards") or [],
                "search_items": qa.get("search_items") or [],
                "urls": qa.get("urls") or [],
            }
            _record_chat_turn(sid_out, user_query, qa_answer, qa_meta)
            yield {
                "event": "qa_status",
                "data": json.dumps(
                    {
                        "sid": sid_out,
                        "message": "Composing response",
                    },
                    ensure_ascii=False,
                ),
            }
            await asyncio.sleep(0.08)
            for item in qa.get("search_items", []):
                if not isinstance(item, dict):
                    continue
                yield {
                    "event": "search_progress",
                    "data": json.dumps(
                        {
                            "sid": sid_out,
                            "step_key": "requirements",
                            **item,
                        },
                        ensure_ascii=False,
                    ),
                }
            for url in qa.get("urls", []):
                yield {
                    "event": "search_progress",
                    "data": json.dumps(
                        {
                            "sid": sid_out,
                            "step_key": "requirements",
                            "url": url,
                        },
                        ensure_ascii=False,
                    ),
                }
            trace = qa.get("trace", "")
            if trace:
                yield {
                    "event": "thought",
                    "data": json.dumps({"sid": sid_out, "content": trace[:12000]}, ensure_ascii=False),
                }
                yield {
                    "event": "thought_end",
                    "data": json.dumps({"sid": sid_out, "content": ""}, ensure_ascii=False),
                }
            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "sid": sid_out,
                        "summary": qa_answer,
                        "report": "",
                        "design_meta": qa_meta,
                    },
                    ensure_ascii=False,
                ),
            }
            return

        if command_id in FRAMEWORK_COMMANDS:
            stage_payload = _build_engineering_framework_stage(command_id, session)
            framework_state = stage_payload.get("framework", {})
            stage_specs = (stage_payload.get("design_meta") or {}).get("specs") or _framework_specs_from_session(session)
            with SESSION_LOCK:
                SESSIONS[sid_out]["requirements_gate_pending"] = False
                SESSIONS[sid_out]["requirements_gate_locked"] = True
                SESSIONS[sid_out]["engineering_framework"] = framework_state
                SESSIONS[sid_out]["specifications"] = stage_specs
                SESSIONS[sid_out]["updated_at"] = time.time()
            _persist_sessions_to_disk()

            node_name = stage_payload.get("node") or "requirements"
            _trace_log("framework_stage_start", sid_out, command=command_id, node=node_name, title=stage_payload.get("title") or "")
            yield {
                "event": "status",
                "data": json.dumps(
                    {
                        "sid": sid_out,
                        "step_key": node_name,
                        "message": f"Scaffolding {stage_payload.get('title')}",
                    },
                    ensure_ascii=False,
                ),
            }
            for item in [
                "Creating a gated artifact scaffold, not a release result.",
                "Recording decision rationale and open evidence gaps.",
                "Holding downstream work until the next explicit engineering action.",
                "Running independent stage review against local evidence and available web research.",
            ]:
                yield {
                    "event": "thought_keypoints",
                    "data": json.dumps(
                        {
                            "sid": sid_out,
                            "step_key": node_name,
                            "items": [item],
                        },
                        ensure_ascii=False,
                    ),
                }
                await asyncio.sleep(0.35)
            review_pack = await asyncio.to_thread(_framework_agent_review, command_id, stage_payload)
            _trace_log(
                "framework_stage_review_done",
                sid_out,
                node=node_name,
                reviewer_status=review_pack.get("status") or "",
                sources=len(review_pack.get("search_items") or []),
            )
            for item in review_pack.get("search_items", []):
                if not isinstance(item, dict):
                    continue
                yield {
                    "event": "search_progress",
                    "data": json.dumps(
                        {
                            "sid": sid_out,
                            "step_key": node_name,
                            **item,
                        },
                        ensure_ascii=False,
                    ),
                }
            stage_payload.setdefault("sections", []).append(
                {
                    "title": "Independent Agent Review",
                    "items": [
                        {"label": "Reviewer status", "value": review_pack.get("status") or "review"},
                        {"label": "Reviewer output", "value": review_pack.get("review_text") or "No reviewer output."},
                        {"label": "Trace", "value": " | ".join(review_pack.get("trace") or [])},
                    ],
                }
            )
            checkpoint_details = stage_payload.setdefault("checkpoint_details", {})
            agent_trace = list(checkpoint_details.get("agent_trace") or [])
            agent_trace.extend(
                [
                    {"label": "Local RAG", "value": next((x for x in review_pack.get("trace", []) if str(x).startswith("Local RAG")), "Local RAG not available")},
                    {"label": "MCP research", "value": next((x for x in review_pack.get("trace", []) if str(x).startswith("MCP")), "MCP research skipped or unavailable")},
                    {"label": "LLM reviewer", "value": next((x for x in review_pack.get("trace", []) if "LLM reviewer" in str(x)), "LLM reviewer fallback or unavailable")},
                ]
            )
            checkpoint_details["agent_trace"] = agent_trace
            checkpoint_details["reviewer_output"] = [
                {"label": "Independent review", "value": review_pack.get("review_text") or "No reviewer output."}
            ]
            stage_payload.setdefault("design_meta", {})["agent_review"] = review_pack
            stage_payload["design_meta"]["evidence_cards"] = review_pack.get("search_items") or []
            stage_payload["design_meta"]["urls"] = review_pack.get("urls") or []
            stage_payload["design_meta"]["checkpoint_details"] = checkpoint_details
            _trace_log("sse_node_result", sid_out, node=node_name, title=stage_payload.get("title") or "")
            yield {
                "event": "node_result",
                "data": json.dumps(
                    {
                        "sid": sid_out,
                        "node": node_name,
                        "step_key": node_name,
                        "title": stage_payload.get("title"),
                        "summary": stage_payload.get("summary"),
                        "result_status": "scaffold",
                        "result_label": "Scaffold + Review",
                        "sections": stage_payload.get("sections") or [],
                        "reasoning": [
                            {"label": "Gate type", "value": "Deterministic engineering scaffold. This does not claim LLM reasoning, SPICE simulation, EDA execution, or measured evidence."},
                            {"label": "Independent review", "value": review_pack.get("status") or "review"},
                            {"label": "Output", "value": "Artifact structure, decision record, and risk-register update."},
                            {"label": "Release policy", "value": "Outputs remain draft/pre-release until downstream evidence gates are closed by real tool or bench evidence."},
                        ],
                        "evidence": [
                            {"label": "Framework stage", "value": stage_payload.get("stage")},
                            {"label": "Evidence state", "value": "Scaffold only. Requires agent/tool execution or measured EVT data before it can be marked complete."},
                            {"label": "Reviewer trace", "value": " | ".join(review_pack.get("trace") or [])},
                            {"label": "Human-in-the-loop", "value": "Next action requires explicit user selection."},
                        ],
                        "content": json.dumps(stage_payload.get("design_meta") or {}, ensure_ascii=False),
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    ensure_ascii=False,
                ),
            }
            _trace_log("sse_checkpoint", sid_out, title=stage_payload.get("title") or "", options=len(stage_payload.get("decision_options") or []))
            yield {
                "event": "checkpoint",
                "data": json.dumps(
                    {
                        "sid": sid_out,
                        "phase": stage_payload.get("stage"),
                        "title": f"{stage_payload.get('title')} Decision",
                        "question": stage_payload.get("summary") or "Choose the next engineering action.",
                        "options": stage_payload.get("options") or [],
                        "commands": stage_payload.get("commands") or [],
                        "context": {
                            "engineering_framework": stage_payload.get("framework"),
                            "framework_stage": stage_payload.get("stage"),
                            "framework_title": stage_payload.get("title"),
                            "checkpoint_details": stage_payload.get("checkpoint_details"),
                            "decision_options": stage_payload.get("decision_options") or [],
                            "specifications": stage_payload.get("design_meta", {}).get("specs", {}),
                            "assumptions": stage_payload.get("design_meta", {}).get("assumptions", []),
                            "decisions": stage_payload.get("design_meta", {}).get("decisions", []),
                            "risks": stage_payload.get("design_meta", {}).get("risks", []),
                            "artifacts": stage_payload.get("design_meta", {}).get("artifacts", []),
                            "recommended_patch": {
                                "design_overrides": {
                                    "framework_stage": stage_payload.get("stage"),
                                    "decisions": stage_payload.get("design_meta", {}).get("decisions", []),
                                    "risks": stage_payload.get("design_meta", {}).get("risks", []),
                                }
                            },
                        },
                    },
                    ensure_ascii=False,
                ),
            }
            _trace_log("sse_done", sid_out, mode="framework_stage", node=node_name)
            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "sid": sid_out,
                        "summary": stage_payload.get("summary"),
                        "report": "",
                        "design_meta": stage_payload.get("design_meta"),
                    },
                    ensure_ascii=False,
                ),
            }
            return

        if (
            fresh_design_request
            and _looks_like_engineering_intake_request(user_query)
            and not session.get("requirements_gate_locked")
        ):
            base_prompt = str(session.get("base_prompt") or "").strip()
            intake_source = f"{base_prompt}\n\n{user_query}".strip() if session.get("requirements_gate_pending") and base_prompt else str(user_query or "")
            _trace_log("requirements_gate_start", sid_out, source_chars=len(intake_source), pending=session.get("requirements_gate_pending"))
            intake = _build_design_intake_payload(intake_source)
            _trace_log(
                "requirements_parsed",
                sid_out,
                specs=len(intake.get("recognized_specs") or []),
                assumptions=len(intake.get("assumptions") or []),
                missing=len(intake.get("missing_inputs") or []),
                power=(intake.get("specs") or {}).get("power") or "",
            )
            with SESSION_LOCK:
                SESSIONS[sid_out]["base_prompt"] = intake_source
                SESSIONS[sid_out]["requirements_gate_pending"] = True
                SESSIONS[sid_out]["requirements_gate_locked"] = False
                SESSIONS[sid_out]["requirements_intake"] = intake
                SESSIONS[sid_out]["specifications"] = intake.get("specs", {})
                SESSIONS[sid_out]["updated_at"] = time.time()
            _persist_sessions_to_disk()
            decision_options = _requirements_decision_options(intake.get("specs") or {})

            yield _REQUIREMENTS_GATE_SERVICE.status_event(sid_out, "Building requirements clarification gate")
            yield _REQUIREMENTS_GATE_SERVICE.thought_keypoints_event(
                sid_out,
                [
                    "Rule parser is extracting Vin, Vout/Iout, power, efficiency, ripple, EMI, isolation, thermal, and topology cues.",
                    "Release policy is applied before any schematic or BOM generation.",
                ],
            )
            yield _REQUIREMENTS_GATE_SERVICE.status_event(sid_out, "Running local RAG and independent intake reviewer")
            _trace_log("requirements_review_start", sid_out)
            review_pack = _requirements_agent_review(intake)
            _trace_log(
                "requirements_review_done",
                sid_out,
                reviewer_status=review_pack.get("status") or "",
                sources=len(review_pack.get("search_items") or []),
                trace_items=len(review_pack.get("trace") or []),
            )
            for item in review_pack.get("search_items") or []:
                if not isinstance(item, dict):
                    continue
                yield _REQUIREMENTS_GATE_SERVICE.search_progress_event(sid_out, item)
            yield _REQUIREMENTS_GATE_SERVICE.thought_keypoints_event(sid_out, list(review_pack.get("trace") or [])[:8])
            checkpoint_details = _requirements_checkpoint_details(intake)
            agent_trace = list(checkpoint_details.get("agent_trace") or [])
            for line in review_pack.get("trace") or []:
                label = "Reviewer trace"
                text_line = str(line)
                if text_line.startswith("Rule parser"):
                    label = "Rule parser"
                elif text_line.startswith("Local RAG"):
                    label = "Local RAG"
                elif text_line.startswith("MCP"):
                    label = "MCP research"
                elif "LLM reviewer" in text_line:
                    label = "LLM reviewer"
                agent_trace.append({"label": label, "value": text_line})
            checkpoint_details["agent_trace"] = agent_trace
            checkpoint_details["reviewer_output"] = [
                {"label": "Independent intake review", "value": review_pack.get("review_text") or "No reviewer output."}
            ]
            intake["agent_review"] = review_pack
            intake["evidence_cards"] = review_pack.get("search_items") or []
            requirement_sections = _requirements_analysis_sections(checkpoint_details)
            _trace_log("requirements_sections_ready", sid_out, sections=len(requirement_sections))

            _trace_log("sse_node_result", sid_out, node="requirements", sections=len(requirement_sections))
            yield _REQUIREMENTS_GATE_SERVICE.node_result_event(
                sid_out,
                intake,
                checkpoint_details,
                decision_options,
                requirement_sections,
                review_pack,
            )
            _trace_log("sse_checkpoint", sid_out, title="Requirements Gate", options=len(decision_options))
            yield _REQUIREMENTS_GATE_SERVICE.checkpoint_event(sid_out, intake, checkpoint_details, decision_options)
            _trace_log("sse_done", sid_out, mode="requirements_intake")
            yield _REQUIREMENTS_GATE_SERVICE.done_event(sid_out, intake, checkpoint_details, decision_options)
            return

        snapshot = graph_app.get_state(config)
        if snapshot:
            snap_values = getattr(snapshot, "values", {}) or {}
            if isinstance(snap_values, dict) and snap_values:
                # Seed live stream state from checkpoint to avoid empty transient UI cards.
                merged_state.update(snap_values)
        if not merged_state.get("config"):
            merged_state["config"] = {"thread_id": thread_id}
        if not merged_state.get("thread_id"):
            merged_state["thread_id"] = thread_id
        start_step_key = "requirements"
        start_message = "Starting PE-MAS collaborative design workflow"
        if snapshot and getattr(snapshot, "next", None):
            start_step_key = str(snapshot.next[0])
            start_message = f"Resuming workflow from {NODE_TITLE.get(start_step_key, start_step_key)}"
        elif command_id == "CONTINUE_SELECTION":
            start_step_key = "selector"
            start_message = f"Resuming workflow from {NODE_TITLE.get(start_step_key, start_step_key)}"
        elif command_id == "CONTINUE_SIMULATION":
            start_step_key = "simulator"
            start_message = f"Resuming workflow from {NODE_TITLE.get(start_step_key, start_step_key)}"
        elif command_id == "CONTINUE_REVIEW":
            start_step_key = "correction"
            start_message = f"Resuming workflow from {NODE_TITLE.get(start_step_key, start_step_key)}"
        elif command_id == "GENERATE_REPORT":
            start_step_key = "reporter"
            start_message = f"Resuming workflow from {NODE_TITLE.get(start_step_key, start_step_key)}"
        active_step_key = start_step_key
        active_step_started_at = time.time()

        # Intent routing: professional QA is already handled above; continue only for design/control paths.

        if snapshot and getattr(snapshot, "next", None):
            next_node = snapshot.next[0]
            if command_id == "STOP_SESSION" or any(k in user_query_low for k in keywords_stop):
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {
                            "sid": sid_out,
                            "summary": "Design session stopped per your instruction.",
                            "report": "",
                            "design_meta": {"mode": "design", "status": "stopped"},
                        },
                        ensure_ascii=False,
                    ),
                }
                return

            if user_query.strip().startswith("HITL_UPDATE_JSON:"):
                raw = user_query.strip()[len("HITL_UPDATE_JSON:"):].strip()
                try:
                    patch_payload = json.loads(raw)
                    inputs, locked_specs = _hitl_resume_inputs(patch_payload, session, thread_id)
                    with SESSION_LOCK:
                        SESSIONS[sid_out]["requirements_gate_pending"] = False
                        SESSIONS[sid_out]["requirements_gate_locked"] = True
                        SESSIONS[sid_out]["specifications"] = locked_specs
                        SESSIONS[sid_out]["updated_at"] = time.time()
                    _persist_sessions_to_disk()
                except Exception as e:
                    yield {
                        "event": "error",
                        "data": json.dumps(
                            {
                                "sid": sid_out,
                                "message": f"Invalid HITL JSON payload: {e}",
                            },
                            ensure_ascii=False,
                        ),
                    }
                    return

            if command_id == "MANUAL_ADJUSTMENTS" or user_query_low in {"apply manual adjustments (json)", "manual adjustments", "手动调整", "手动修改"}:
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {
                            "sid": sid_out,
                            "summary": "Manual adjustment mode enabled. Edit the checkpoint JSON and click 'Apply JSON + Resume'.",
                            "report": "",
                            "design_meta": {"mode": "hitl_design", "status": "awaiting_manual_patch"},
                        },
                        ensure_ascii=False,
                    ),
                }
                return

            if command_id == "MODIFY_CONSTRAINTS" or "modify constraints" in user_query_low or "change constraints" in user_query_low:
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {
                            "sid": sid_out,
                            "summary": "Please input the constraints you want to modify (e.g. efficiency >90%, or output to 5V/3A). The design will restart with the new constraints.",
                            "report": "",
                            "design_meta": {"mode": "hitl_design", "status": "awaiting_constraint"},
                        },
                        ensure_ascii=False,
                    ),
                }
                return

            if command_id == "CHANGE_COMPONENT_STRATEGY" or "change component" in user_query_low or "return to adjust" in user_query_low or "change components" in user_query_low:
                previous_values = getattr(snapshot, "values", {}) or {}
                if not isinstance(previous_values, dict):
                    previous_values = {}
                bom_tasks = _bom_checkpoint_tasks(previous_values)
                bom_summary = _bom_checkpoint_summary(bom_tasks)
                if bom_tasks and not bom_summary.get("auto_fixable") and bom_summary.get("manual_signoff"):
                    manual_lines = [
                        f"- {task.get('label')}: {task.get('part')} ({task.get('reason')})"
                        for task in bom_tasks
                        if task.get("fixability") == "manual_signoff"
                    ][:6]
                    yield {
                        "event": "done",
                        "data": json.dumps(
                            {
                                "sid": sid_out,
                                "summary": (
                                    "Auto-fix skipped: the remaining BOM blockers require manual engineering or procurement sign-off, "
                                    "so rerunning component selection would likely loop. Open the manual editor to enter approved parts/specs, "
                                    "or continue simulation only as exploratory evidence."
                                ),
                                "report": "\n".join(manual_lines),
                                "design_meta": {
                                    "mode": "hitl_design",
                                    "status": "awaiting_manual_bom_signoff",
                                    "bom_checkpoint_summary": bom_summary,
                                },
                            },
                            ensure_ascii=False,
                        ),
                    }
                    return
                new_thread = str(uuid.uuid4())
                with SESSION_LOCK:
                    SESSIONS[sid_out]["thread_id"] = new_thread
                    SESSIONS[sid_out]["updated_at"] = time.time()
                _persist_sessions_to_disk()
                thread_id = new_thread
                config = {"configurable": {"thread_id": new_thread}}
                merged_state["config"] = {"thread_id": new_thread}
                merged_state["thread_id"] = new_thread
                start_step_key = "requirements"
                start_message = "Restarting design from Requirements for BOM re-selection"
                active_step_key = start_step_key
                active_step_started_at = time.time()
                inputs = {
                    "messages": [("user", "BOM_RESELECT_REQUESTED")],
                    "specifications": previous_values.get("specifications") or {},
                    "request_profile": previous_values.get("request_profile") or {},
                    "theoretical_design": previous_values.get("theoretical_design") or {},
                    "design_overrides": previous_values.get("design_overrides") or {},
                    "hard_guardrails_prompt": previous_values.get("hard_guardrails_prompt") or "",
                    "retrieved_knowledge_context": previous_values.get("retrieved_knowledge_context") or "",
                    "retrieved_knowledge_references": previous_values.get("retrieved_knowledge_references") or [],
                    "literature_references": previous_values.get("literature_references") or [],
                    "evidence_grade": previous_values.get("evidence_grade") or {},
                    "learning_context": previous_values.get("learning_context") or {},
                    "curriculum_context": previous_values.get("curriculum_context") or {},
                    "iteration_learning": _bom_reselect_learning(previous_values),
                    "iteration": int(previous_values.get("iteration") or 0),
                    "max_iterations": int(previous_values.get("max_iterations") or 5),
                    "config": {"thread_id": new_thread},
                    "thread_id": new_thread,
                }
            elif (next_node == "correction") and (
                command_id in {"SKIP_CORRECTION_AND_REPORT", "GENERATE_REPORT"}
                or "skip correction" in user_query_low
                or "generate report" in user_query_low
                or "skip_correction_and_report" in user_query_low
                or "skip correction and report" in user_query_low
            ):
                inputs = {"messages": [("user", "SKIP_CORRECTION_AND_REPORT")]}  # consumed in correction node
            elif (next_node == "selector") and (command_id == "CONTINUE_SELECTION" or "continue selection" in user_query_low):
                inputs = None
            elif (next_node == "simulator") and (command_id == "CONTINUE_SIMULATION" or "continue simulation" in user_query_low):
                inputs = None
            elif (next_node == "correction") and (command_id == "CONTINUE_REVIEW" or "continue review" in user_query_low):
                inputs = None
            elif (next_node == "reporter") and (command_id == "GENERATE_REPORT" or "generate report" in user_query_low):
                inputs = None

            if command_id in {
                "CONTINUE_SELECTION",
                "CONTINUE_SIMULATION",
                "CONTINUE_REVIEW",
                "GENERATE_REPORT",
                "SKIP_CORRECTION_AND_REPORT",
            } or any(k in user_query_low for k in keywords_continue):
                inputs = None
            elif "inputs" not in locals() or inputs is None:
                inputs = {"messages": [("user", user_query)], "config": {"thread_id": thread_id}, "thread_id": thread_id}
        else:
            if command_id == "ADOPT_DEFAULT_ASSUMPTIONS":
                base_query = str(session.get("base_prompt") or "").strip() or str(user_query or "").strip()
                intake = session.get("requirements_intake") if isinstance(session.get("requirements_intake"), dict) else {}
                continuation_prompt = (
                    f"{base_query}\n\n"
                    "PE-MAS REQUIREMENTS GATE ACCEPTED: proceed with the default engineering assumptions, "
                    "keep the output in English, preserve decision records and risk register entries, "
                    "and continue with topology/controller/power-stage workflow. Do not mark any schematic, BOM, "
                    "magnetics, loop, EMI, safety, or release package artifact as final until its gate evidence is present."
                )
                inputs = {
                    "messages": [("user", continuation_prompt)],
                    "specifications": intake.get("specs") or _framework_specs_from_session(session),
                    "design_overrides": {
                        "requirements_gate": "accepted_defaults",
                        "assumptions": intake.get("assumptions", []),
                        "topology_candidates": intake.get("topology_candidates", []),
                        "locked_specs": intake.get("specs") or _framework_specs_from_session(session),
                    },
                    "config": {"thread_id": thread_id},
                    "thread_id": thread_id,
                }
                with SESSION_LOCK:
                    SESSIONS[sid_out]["requirements_gate_pending"] = False
                    SESSIONS[sid_out]["requirements_gate_locked"] = True
                    SESSIONS[sid_out]["base_prompt"] = base_query
                    SESSIONS[sid_out]["specifications"] = intake.get("specs") or _framework_specs_from_session(session)
                    SESSIONS[sid_out]["updated_at"] = time.time()
                _persist_sessions_to_disk()
            elif command_id in {"CONTINUE_SELECTION", "CONTINUE_SIMULATION", "CONTINUE_REVIEW", "GENERATE_REPORT"}:
                locked_specs = _framework_specs_from_session(session)
                base_query = str(session.get("base_prompt") or "").strip()
                continuation_prompt = (
                    f"{base_query}\n\n" if base_query else ""
                ) + (
                    f"{_specs_to_prompt(locked_specs)}\n\n"
                    "CONTINUE FROM GATED ENGINEERING SCAFFOLD: run the real agent/tool workflow now. "
                    "Do not re-parse the design as 12V/2A or any default fallback. Preserve output current, "
                    "output power, isolation, EMI, and ripple targets exactly from the locked specification. "
                    "Keep release blocked unless measured/tool evidence closes the relevant gates."
                )
                inputs = {
                    "messages": [("user", continuation_prompt)],
                    "specifications": locked_specs,
                    "design_overrides": {
                        "requirements_gate": "framework_scaffold_locked",
                        "locked_specs": locked_specs,
                        "framework": session.get("engineering_framework") if isinstance(session.get("engineering_framework"), dict) else {},
                    },
                    "config": {"thread_id": thread_id},
                    "thread_id": thread_id,
                }
                with SESSION_LOCK:
                    SESSIONS[sid_out]["requirements_gate_pending"] = False
                    SESSIONS[sid_out]["requirements_gate_locked"] = True
                    SESSIONS[sid_out]["specifications"] = locked_specs
                    SESSIONS[sid_out]["updated_at"] = time.time()
                _persist_sessions_to_disk()
            elif command_id == "RETRY_REQUESTED" or user_query_low in {"retry", "continue auto-iteration", "continue auto iteration", "继续自动迭代", "继续自动"}:
                latest_values = getattr(snapshot, "values", {}) if snapshot else {}
                latest_verification = (latest_values.get("verification") or {}) if isinstance(latest_values, dict) else {}
                latest_status = str(latest_verification.get("status") or "").upper()
                latest_strategy = str(latest_verification.get("correction_strategy") or "").upper()
                if latest_status == "NEEDS_HUMAN_REVIEW" or latest_strategy == "PHYSICAL_LIMIT_REACHED":
                    why = (latest_verification.get("failed_items") or [])[:3]
                    yield {
                        "event": "done",
                        "data": json.dumps(
                            {
                                "sid": sid_out,
                                "summary": (
                                    "Auto-iteration blocked: validator marked this run as physical-limit/human-review required. "
                                    "Please accept current best result or apply manual adjustments."
                                ),
                                "report": "\n".join([str(x) for x in why]) if why else "",
                                "design_meta": {
                                    "mode": "hitl_design",
                                    "status": "waiting_user",
                                    "block_reason": "PHYSICAL_LIMIT_REACHED",
                                },
                            },
                            ensure_ascii=False,
                        ),
                    }
                    return
                inputs = {"messages": [("user", "RETRY_REQUESTED")], "config": {"thread_id": thread_id}, "thread_id": thread_id}
                metric_deltas["run_retry_fallback"] = metric_deltas.get("run_retry_fallback", 0.0) + 1.0
            elif command_id == "MANUAL_ADJUSTMENTS" or user_query_low in {"apply manual adjustments (json)", "manual adjustments", "手动调整", "手动修改"}:
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {
                            "sid": sid_out,
                            "summary": "Manual adjustment mode enabled. Edit the checkpoint JSON and click 'Apply JSON + Resume'.",
                            "report": "",
                            "design_meta": {"mode": "hitl_design", "status": "awaiting_manual_patch"},
                        },
                        ensure_ascii=False,
                    ),
                }
                return
            elif command_id == "ACCEPT_CURRENT_RESULT" or user_query_low in {"accept current result", "accept", "接受当前结果", "接受结果"}:
                accepted_values = getattr(snapshot, "values", {}) if snapshot else {}
                if not isinstance(accepted_values, dict):
                    accepted_values = {}
                memory_writeback = _persist_memory_on_user_accept(accepted_values)
                payload = _final_payload_for_user_accept(accepted_values)
                accepted_report = str(payload.get("report") or "")
                yield {
                    "event": "node_result",
                    "data": json.dumps(
                        {
                            "sid": sid_out,
                            "node": "reporter",
                            "step_key": "reporter",
                            "title": NODE_TITLE.get("reporter", "reporter"),
                            "summary": "Final report packaged from user-accepted iteration.",
                            "sections": [
                                {
                                    "title": "Report",
                                    "items": [
                                        {"label": "Length", "value": len(accepted_report)},
                                        {"label": "Acceptance", "value": "Accepted by user"},
                                        {"label": "Memory Writeback", "value": "OK" if not memory_writeback.get("error") else f"WARN: {memory_writeback.get('error')}"},
                                    ],
                                }
                            ],
                            "reasoning": [],
                            "evidence": [],
                            "content": accepted_report,
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                        ensure_ascii=False,
                    ),
                }
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {
                            "sid": sid_out,
                            "summary": payload["summary"],
                            "report": payload["report"],
                            "design_meta": {
                                **(payload.get("design_meta") or {}),
                                "mode": "design",
                                "status": "accepted_by_user",
                                "memory_writeback": memory_writeback,
                            },
                        },
                        ensure_ascii=False,
                    ),
                }
                return
            elif user_query.strip().startswith("HITL_UPDATE_JSON:"):
                raw = user_query.strip()[len("HITL_UPDATE_JSON:"):].strip()
                try:
                    patch_payload = json.loads(raw)
                    inputs, locked_specs = _hitl_resume_inputs(patch_payload, session, thread_id)
                    with SESSION_LOCK:
                        SESSIONS[sid_out]["requirements_gate_pending"] = False
                        SESSIONS[sid_out]["requirements_gate_locked"] = True
                        SESSIONS[sid_out]["specifications"] = locked_specs
                        SESSIONS[sid_out]["updated_at"] = time.time()
                    _persist_sessions_to_disk()
                except Exception as e:
                    yield {
                        "event": "error",
                        "data": json.dumps(
                            {
                                "sid": sid_out,
                                "message": f"Invalid HITL JSON payload: {e}",
                            },
                            ensure_ascii=False,
                        ),
                    }
                    return
            else:
                inputs = {"messages": [("user", user_query)], "config": {"thread_id": thread_id}, "thread_id": thread_id}
            with SESSION_LOCK:
                if not command_id and not user_query.strip().startswith("HITL_UPDATE_JSON:"):
                    SESSIONS[sid_out]["base_prompt"] = user_query
                SESSIONS[sid_out]["updated_at"] = time.time()
            _persist_sessions_to_disk()

        yield {
            "event": "status",
            "data": json.dumps(
                {
                    "sid": sid_out,
                    "step_key": start_step_key,
                    "message": start_message,
                },
                ensure_ascii=False,
            ),
        }
        _trace_log("graph_stream_start", sid_out, start_step=start_step_key, command=command_id or "", thread_id=thread_id)

        event_iter = graph_app.astream(inputs, config, stream_mode="updates")
        next_event_task: Optional[asyncio.Task[Any]] = asyncio.create_task(_next_graph_event(event_iter))

        while next_event_task is not None:
            done, _ = await asyncio.wait({next_event_task}, timeout=1.0)

            if not done:
                now = time.time()
                if now - last_heartbeat_at >= 1.2:
                    elapsed = int(max(0, now - active_step_started_at))
                    title = NODE_TITLE.get(active_step_key, active_step_key)
                    hint_id, hint_text = _heartbeat_hint(active_step_key, elapsed)
                    yield {
                        "event": "heartbeat",
                        "data": json.dumps(
                            {
                                "sid": sid_out,
                                "step_key": active_step_key,
                                "elapsed_sec": elapsed,
                                "message": f"{title} running ({elapsed}s)",
                                "keypoint": hint_text,
                                "keypoint_id": hint_id,
                            },
                            ensure_ascii=False,
                        ),
                    }
                    last_heartbeat_at = now
                continue

            try:
                event_raw = next_event_task.result()
            except StopAsyncIteration:
                next_event_task = None
                break

            next_event_task = asyncio.create_task(_next_graph_event(event_iter))

            event = _extract_state(event_raw)
            if not isinstance(event, dict):
                continue

            for node_name, node_state in event.items():
                if str(node_name).startswith("__"):
                    continue
                if not isinstance(node_state, dict):
                    node_state = {"value": node_state}

                active_step_key = str(node_name)
                active_step_started_at = time.time()
                merged_state.update(node_state)
                _trace_log("graph_node_update", sid_out, node=node_name, keys=",".join(sorted(map(str, node_state.keys()))[:20]))

                yield {
                    "event": "status",
                    "data": json.dumps(
                        {
                            "sid": sid_out,
                            "step_key": node_name,
                            "message": f"{NODE_TITLE.get(node_name, node_name)} In progress",
                        },
                        ensure_ascii=False,
                    ),
                }

                status_hint = (HEARTBEAT_HINTS.get(str(node_name)) or [""])[0]
                if status_hint:
                    yield {
                        "event": "thought_keypoints",
                        "data": json.dumps(
                            {
                                "sid": sid_out,
                                "node": node_name,
                                "step_key": node_name,
                                "items": [status_hint],
                            },
                            ensure_ascii=False,
                        ),
                    }

                brief = _node_brief(node_name, merged_state, node_state)

                node_reasoning_lines = _collect_node_reasoning(node_name, merged_state)
                node_reasoning_items = _reasoning_items(node_reasoning_lines)

                evidence_items = _collect_evidence_items(node_name, merged_state)

                if node_reasoning_lines:
                    thought_text = "\n".join(node_reasoning_lines)
                    parsed_trace = _parse_reasoning_text(node_name, thought_text)
                    keypoints = _thought_keypoints(parsed_trace)

                    if keypoints:
                        yield {
                            "event": "thought_keypoints",
                            "data": json.dumps(
                                {
                                    "sid": sid_out,
                                    "node": node_name,
                                    "step_key": node_name,
                                    "items": keypoints,
                                },
                                ensure_ascii=False,
                            ),
                        }

                    for item in _extract_search_progress_items(thought_text):
                        if not isinstance(item, dict):
                            continue
                        yield {
                            "event": "search_progress",
                            "data": json.dumps(
                                {
                                    "sid": sid_out,
                                    "node": node_name,
                                    "step_key": node_name,
                                    **item,
                                },
                                ensure_ascii=False,
                            ),
                        }

                    for url in _extract_urls(thought_text):
                        yield {
                            "event": "search_progress",
                            "data": json.dumps(
                                {
                                    "sid": sid_out,
                                    "step_key": node_name,
                                    "url": url,
                                },
                                ensure_ascii=False,
                            ),
                        }

                    if thought_text.strip():
                        yield {
                            "event": "thought",
                            "data": json.dumps(
                                {
                                    "sid": sid_out,
                                    "node": node_name,
                                    "title": parsed_trace.get("title"),
                                    "content": parsed_trace.get("text", thought_text)[:12000],
                                    "sections": parsed_trace.get("sections", []),
                                },
                                ensure_ascii=False,
                            ),
                        }
                        yield {
                            "event": "thought_end",
                            "data": json.dumps({"sid": sid_out, "node": node_name, "content": ""}, ensure_ascii=False),
                        }

                _trace_log("sse_node_result", sid_out, node=node_name, title=brief["title"])
                yield {
                    "event": "node_result",
                    "data": json.dumps(
                        {
                            "sid": sid_out,
                            "node": node_name,
                            "step_key": node_name,
                            "title": brief["title"],
                            "summary": brief["summary"],
                            "sections": brief["sections"],
                            "reasoning": node_reasoning_items,
                            "evidence": evidence_items,
                            "content": brief["content"],
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                        ensure_ascii=False,
                    ),
                }

                completed_steps.add(str(node_name))
                next_step = _next_workflow_step(completed_steps)
                if next_step:
                    active_step_key = next_step
                    active_step_started_at = time.time()

        if next_event_task is not None:
            next_event_task.cancel()
            with contextlib.suppress(Exception):
                await next_event_task

        final_snapshot = graph_app.get_state(config)
        values = getattr(final_snapshot, "values", {}) if final_snapshot else {}
        if not values:
            values = merged_state

        # HITL checkpoint path
        if final_snapshot and getattr(final_snapshot, "next", None):
            next_node = final_snapshot.next[0]
            ckpt = _checkpoint_payload(next_node, values)
            _trace_log("sse_checkpoint", sid_out, title=ckpt["title"], phase=next_node, options=len(ckpt.get("options") or []))
            yield {
                "event": "checkpoint",
                "data": json.dumps(
                    {
                        "sid": sid_out,
                        "phase": next_node,
                        "title": ckpt["title"],
                        "question": ckpt["question"],
                        "options": ckpt["options"],
                        "commands": ckpt.get("commands", []),
                        "context": ckpt["context"],
                    },
                    ensure_ascii=False,
                ),
            }
            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "sid": sid_out,
                        "summary": f"Workflow paused at {next_node}. Please choose to continue or adjust at the checkpoint.",
                        "report": "",
                        "design_meta": {"mode": "hitl_design", "status": "waiting_user", "next_node": next_node},
                    },
                    ensure_ascii=False,
                ),
            }
            return

        review_payload = _failure_review_payload(values)
        if review_payload:
            failure_memory_writeback = _persist_memory_on_failure_review(values)
            yield {
                "event": "checkpoint",
                "data": json.dumps(
                    {
                        "sid": sid_out,
                        "phase": "validator_review",
                        "title": review_payload["title"],
                        "question": review_payload["question"],
                        "options": review_payload["options"],
                        "commands": review_payload.get("commands", []),
                        "context": review_payload["context"],
                    },
                    ensure_ascii=False,
                ),
            }
            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "sid": sid_out,
                        "summary": "Iteration completed with unresolved issues. Please choose how to proceed at the decision checkpoint.",
                        "report": "",
                        "design_meta": {
                            "mode": "hitl_design",
                            "status": "waiting_user",
                            "next_node": "validator_review",
                            "memory_writeback": failure_memory_writeback,
                        },
                    },
                    ensure_ascii=False,
                ),
            }
            return

        specs = values.get("specifications") or {}
        if specs.get("is_chitchat"):
            response_text = ""
            msgs = values.get("messages") or []
            if isinstance(msgs, list) and msgs:
                response_text = str(msgs[-1])
            if not response_text or response_text.strip().lower().startswith("design workflow complete"):
                qa = _compose_professional_qa(user_query, session=session, history=_session_history(session))
                response_text = qa.get("answer", _chatbot_answer(user_query))
                qa_source = qa.get("source", "rag_mcp")
            else:
                qa_source = "graph_chitchat"
            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "sid": sid_out,
                        "summary": response_text,
                        "report": "",
                        "design_meta": {"mode": "chatbot", "source": qa_source},
                    },
                    ensure_ascii=False,
                ),
            }
            return

        payload = _final_payload(values)
        metric_deltas["run_success"] = metric_deltas.get("run_success", 0.0) + 1.0
        metric_deltas["sum_iteration"] = metric_deltas.get("sum_iteration", 0.0) + float(values.get("iteration", 0) or 0)
        _update_metrics(metric_deltas)
        _record_runtime_event("run_done", sid_out, {"status": "success", "iteration": values.get("iteration", 0), "command": command_id or ""})
        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "sid": sid_out,
                    "summary": payload["summary"],
                    "report": payload["report"],
                    "design_meta": payload["design_meta"],
                },
                ensure_ascii=False,
            ),
        }

    except Exception as exc:
        metric_deltas["run_error"] = metric_deltas.get("run_error", 0.0) + 1.0
        _update_metrics(metric_deltas)
        _record_runtime_event("run_done", sid_out, {"status": "error", "error": str(exc)[:500], "command": command_id or ""})
        _trace_log("run_error", sid_out, error=repr(exc), traceback=traceback.format_exc()[-1800:])
        yield {
            "event": "error",
            "data": json.dumps({"sid": sid_out, "message": str(exc)}, ensure_ascii=False),
        }


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "runtime_dir_ready": RUNTIME_DIR.exists(),
        "sessions": len(SESSIONS),
        "metrics": _metrics_snapshot().get("slo", {}),
    }


@app.get("/api/metrics")
async def metrics() -> Dict[str, Any]:
    return _metrics_snapshot()


def _cleanup_pending_streams(now: Optional[float] = None) -> None:
    ts = now if now is not None else time.time()
    stale = [
        rid
        for rid, payload in PENDING_STREAMS.items()
        if ts - float(payload.get("created_at") or 0) > PENDING_STREAM_TTL_SEC
    ]
    for rid in stale:
        PENDING_STREAMS.pop(rid, None)


@app.post("/api/chat/start")
async def chat_start(payload: ChatStartRequest) -> Dict[str, str]:
    q = str(payload.q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="missing query")
    rid = str(uuid.uuid4())
    with PENDING_STREAM_LOCK:
        _cleanup_pending_streams()
        PENDING_STREAMS[rid] = {
            "q": q,
            "sid": str(payload.sid or "").strip() or None,
            "created_at": time.time(),
        }
    _trace_log("chat_start", str(payload.sid or "").strip(), rid=rid, query=q[:220])
    return {"rid": rid, "stream": f"/api/chat/stream?rid={rid}"}


@app.get("/api/chat/stream")
async def chat_stream(rid: str):
    with PENDING_STREAM_LOCK:
        _cleanup_pending_streams()
        payload = PENDING_STREAMS.get(str(rid or "").strip())
    if not payload:
        _trace_log("chat_stream_missing", rid=str(rid or "").strip())
        raise HTTPException(status_code=404, detail="stream id not found or expired")
    _trace_log("chat_stream_open", str(payload.get("sid") or ""), rid=rid, query=str(payload.get("q") or "")[:180])
    return EventSourceResponse(stream_workflow(payload["q"], sid=payload.get("sid")))


@app.get("/api/chat")
async def chat(q: str, sid: Optional[str] = None):
    return EventSourceResponse(stream_workflow(q, sid=sid))


@app.get("/api/report-asset")
async def report_asset(path: str = Query(..., description="Absolute or workspace-relative path to a generated report asset")):
    asset_path = _resolve_report_asset(path)
    return FileResponse(asset_path)


@app.post("/api/requirements/analyze")
async def analyze_requirements(payload: RequirementAnalysisRequest) -> Dict[str, Any]:
    try:
        result = _get_requirement_agent().analyze(payload.prompt, project_id=payload.project_id)
        return result.to_api_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/topologies")
async def list_topologies() -> Dict[str, Any]:
    return {"topologies": _get_topology_service().list_topologies()}


@app.get("/api/topologies/{name}")
async def get_topology(name: str) -> Dict[str, Any]:
    topology = _get_topology_service().get_topology(name)
    if not topology:
        raise HTTPException(status_code=404, detail="topology not found")
    return {"topology": topology}


@app.get("/api/plecs/models/status")
async def plecs_model_status() -> Dict[str, Any]:
    return {"models": _get_plecs_registry().status()}


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("PE_MAS_HOST")
    if not host:
        raise RuntimeError("PE_MAS_HOST must be configured locally before starting the server.")
    port = int(os.getenv("PORT", "8000") or "8000")
    uvicorn.run(app, host=host, port=port)
