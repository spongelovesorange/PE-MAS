from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
import json
import re
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET


HARD_GUARDRAILS_PROMPT = """
[HIGHEST PRIORITY DESIGN GUARDRAILS — NON-OVERRIDABLE]
You are a Flyback power supply engineering agent. The following principles are hard constraints and always take priority over user preferences, heuristics, or style:

1) Safety and stress margin first:
- MOSFET/diode/transformer stress must stay within conservative margin under worst-case line/load.
- Reject designs where Vds spike approaches rating without robust clamp/snubber strategy.

2) Duty-cycle and topology feasibility:
- For wide-input Flyback, keep nominal operating duty in a practical range (typically around 0.35~0.45 when possible for efficiency/robustness tradeoff).
- Avoid extreme Dmax that causes excessive RMS current, poor demagnetization margin, or control instability.

3) Soft-switching / turn-on quality:
- Prefer designs that reduce turn-on loss and dv/dt stress (valley switching / quasi-resonant / ACF where applicable).
- If hard-switching losses or ringing dominate, prioritize snubber/clamp and switching-condition improvements before cosmetic tweaks.

4) CCM/DCM mode tradeoff must be explicit:
- Explain and validate the selected conduction mode against efficiency, ripple, EMI, and control-loop complexity.
- If operation crosses boundaries, discuss boundary-condition risks and compensation implications.

5) Ripple, EMI, thermal, isolation are mandatory checks:
- Enforce ripple and EMI objectives against application KPI.
- Thermal and insulation constraints (especially medical/industrial/automotive) are pass/fail gates, not optional optimizations.

6) Physical realism:
- Flag unrealistically high efficiency for diode-rectified Flyback as suspicious and require model realism checks.
- Prefer physically plausible recommendations over optimistic idealized simulation artifacts.

7) If conflict exists:
- Follow safety/compliance/physics constraints first, then optimize efficiency/cost/size.
""".strip()


def get_hard_guardrails_prompt() -> str:
    return HARD_GUARDRAILS_PROMPT


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_docx_path() -> Path:
    source_doc = _workspace_root() / "core" / "knowledge" / "flyback" / "source_docs" / "Flyback_AI_Agent_知识库_V2.docx"
    if source_doc.exists():
        return source_doc
    return _workspace_root() / "Flyback_AI_Agent_知识库_V2.docx"


def default_markdown_path() -> Path:
    return _workspace_root() / "core" / "knowledge" / "flyback" / "source_docs" / "Flyback_AI_Agent_知识库_V2.md"


def default_flyback_kb_dir() -> Path:
    return _workspace_root() / "core" / "knowledge" / "kb" / "flyback"


def normalize_source_label(source: Any) -> str:
    text = str(source or "").strip()
    if not text:
        return "Internal Knowledge Base"

    normalized = text.replace("\\", "/")
    basename = normalized.split("/")[-1] if "/" in normalized else normalized
    low = basename.lower()

    friendly_map = {
        "flyback_ai_agent_知识库_v2.docx": "Internal Flyback Design Guide",
        "flyback_saved_index": "Indexed Flyback Reference Library",
        "pe-gpt.txt": "PE-GPT Project Notes",
        "fundamentals_of_pe.pdf": "Fundamentals of Power Electronics",
        "local_rag": "Internal Knowledge Base",
        "component_rag": "Component Reference Library",
        "component_db": "Component Reference Library",
    }
    if low in friendly_map:
        return friendly_map[low]

    if "知识库" in basename or "flyback_ai_agent" in low:
        return "Internal Flyback Design Guide"
    if "saved_index" in low:
        return "Indexed Flyback Reference Library"

    if "." in basename:
        stem = basename.rsplit(".", 1)[0]
    else:
        stem = basename
    stem = stem.replace("_", " ").strip()
    return stem or "Internal Knowledge Base"


def _extract_docx_text(docx_path: Path) -> str:
    if not docx_path.exists():
        return ""
    try:
        with zipfile.ZipFile(docx_path, "r") as zf:
            with zf.open("word/document.xml") as f:
                xml_data = f.read()
        root = ET.fromstring(xml_data)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs: List[str] = []
        for paragraph in root.findall(".//w:p", ns):
            text_runs = paragraph.findall(".//w:t", ns)
            txt = "".join(run.text or "" for run in text_runs).strip()
            if txt:
                paragraphs.append(txt)
        return "\n".join(paragraphs)
    except Exception:
        return ""


@lru_cache(maxsize=1)
def _get_docx_text_cached(docx_path_str: str) -> str:
    return _extract_docx_text(Path(docx_path_str))


@lru_cache(maxsize=1)
def _get_flyback_source_text_cached(markdown_path_str: str, docx_path_str: str) -> str:
    markdown_path = Path(markdown_path_str)
    if markdown_path.exists():
        try:
            return markdown_path.read_text(encoding="utf-8")
        except Exception:
            pass
    return _get_docx_text_cached(docx_path_str)


def _chunk_by_section(raw_text: str, max_chars: int = 1400) -> List[str]:
    if not raw_text:
        return []

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    sections: List[str] = []
    current: List[str] = []
    heading_pattern = re.compile(r"^(第[一二三四五六七八九十]+章|\d+\.\d+|\d+\.)")

    for ln in lines:
        if heading_pattern.search(ln) and current:
            sections.append("\n".join(current))
            current = [ln]
        else:
            current.append(ln)
    if current:
        sections.append("\n".join(current))

    chunks: List[str] = []
    for sec in sections:
        if len(sec) <= max_chars:
            chunks.append(sec)
            continue
        start = 0
        while start < len(sec):
            end = min(start + max_chars, len(sec))
            chunks.append(sec[start:end])
            start = end
    return chunks


def _score_chunk(query: str, chunk: str) -> float:
    q = (query or "").lower()
    c = (chunk or "").lower()
    if not q or not c:
        return 0.0
    words = [w for w in re.findall(r"[\w\u4e00-\u9fff]+", q) if len(w) > 1]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in c)
    rule_bonus_terms = ["软开关", "占空比", "duty", "ccm", "dcm", "snubber", "效率", "纹波", "emc", "isolation", "mopp"]
    bonus = sum(1 for t in rule_bonus_terms if t in q and t in c)
    return hits + 1.5 * bonus


def _retrieve_from_docx(docx_text: str, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    chunks = _chunk_by_section(docx_text)
    scored: List[Dict[str, Any]] = []
    for ch in chunks:
        s = _score_chunk(query, ch)
        if s > 0:
            scored.append({"source": normalize_source_label("Flyback_AI_Agent_知识库_V2.docx"), "score": float(s), "text": ch[:1200]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


@lru_cache(maxsize=1)
def _load_flyback_index_cached(kb_dir_str: str):
    kb_dir = Path(kb_dir_str)
    saved = kb_dir / "saved_index"
    if not saved.exists():
        return None
    try:
        from llama_index.core import StorageContext
        from llama_index.core.indices.loading import load_index_from_storage

        storage_context = StorageContext.from_defaults(persist_dir=str(saved))
        return load_index_from_storage(storage_context)
    except Exception:
        return None


def _retrieve_from_vector_index_subprocess(query: str, kb_dir: Path, top_k: int = 3) -> List[Dict[str, Any]]:
    worker = r"""
import json
import sys
from pathlib import Path

query = sys.argv[1]
kb_dir = Path(sys.argv[2])
top_k = int(sys.argv[3])
saved = kb_dir / "saved_index"
if not saved.exists():
    print("[]")
    raise SystemExit(0)

from llama_index.core import StorageContext
from llama_index.core.indices.loading import load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir=str(saved))
index = load_index_from_storage(storage_context)
retriever = index.as_retriever(similarity_top_k=top_k)
nodes = retriever.retrieve(query)
results = []
for n in nodes:
    txt = getattr(n, "text", "") or ""
    score = float(getattr(n, "score", 0.0) or 0.0)
    metadata = getattr(n, "metadata", {}) or {}
    src = metadata.get("file_name") or metadata.get("file_path") or "flyback_saved_index"
    if txt:
        results.append({"source": src, "score": score, "text": txt[:1000]})
print(json.dumps(results, ensure_ascii=False))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", worker, str(query or ""), str(kb_dir), str(max(1, int(top_k or 3)))],
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
    except Exception:
        return []

    if proc.returncode != 0:
        return []
    stdout = str(proc.stdout or "").strip()
    if not stdout:
        return []
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def _retrieve_from_vector_index(query: str, kb_dir: Path, top_k: int = 3) -> List[Dict[str, Any]]:
    return _retrieve_from_vector_index_subprocess(query, kb_dir, top_k=top_k)


def retrieve_flyback_context(query: str, top_k: int = 6) -> Dict[str, Any]:
    query_text = str(query or "").strip()
    if not query_text:
        return {"context_text": "", "references": []}

    docx_path = default_docx_path()
    markdown_path = default_markdown_path()
    kb_dir = default_flyback_kb_dir()

    source_text = _get_flyback_source_text_cached(str(markdown_path), str(docx_path))
    docx_hits = _retrieve_from_docx(source_text, query_text, top_k=max(2, top_k // 2))
    vec_hits = _retrieve_from_vector_index(query_text, kb_dir, top_k=max(2, top_k // 2))

    merged = sorted(docx_hits + vec_hits, key=lambda x: x.get("score", 0.0), reverse=True)

    unique: List[Dict[str, Any]] = []
    seen = set()
    for item in merged:
        text_key = (item.get("text") or "")[:160]
        if not text_key or text_key in seen:
            continue
        seen.add(text_key)
        unique.append(item)
        if len(unique) >= top_k:
            break

    references = [{"source": normalize_source_label(r.get("source")), "score": r.get("score", 0.0)} for r in unique]
    context_text = "\n\n".join([f"[Source] {normalize_source_label(r.get('source'))}\n{r.get('text')}" for r in unique])
    return {
        "context_text": context_text,
        "references": references,
    }
