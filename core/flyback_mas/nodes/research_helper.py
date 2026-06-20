from __future__ import annotations

from typing import Any, Dict, List
import json
import os
import re

from ..knowledge_guardrails import normalize_source_label, retrieve_flyback_context

try:
    from core.utils.component_rag_bridge import retrieve_component_rag_context
except Exception:
    retrieve_component_rag_context = None

try:
    from core.skills.web_research.tools import research_web as mcp_research_web
except Exception:
    mcp_research_web = None

local_search = None


def _parse_local_search_row(row: Any) -> Dict[str, str]:
    if isinstance(row, dict):
        return {
            "title": str(row.get("title") or row.get("Title") or row.get("name") or "Web Result"),
            "url": str(row.get("url") or row.get("href") or row.get("link") or ""),
            "snippet": str(row.get("snippet") or row.get("body") or row.get("content") or ""),
        }

    text = str(row or "")
    title = "Web Result"
    url = ""
    snippet = text[:400]

    for line in text.splitlines():
        ln = line.strip()
        if ln.lower().startswith("title:"):
            title = ln.split(":", 1)[1].strip() or title
        elif ln.lower().startswith("link:"):
            url = ln.split(":", 1)[1].strip()
        elif ln.lower().startswith("snippet:"):
            snippet = ln.split(":", 1)[1].strip() or snippet
        elif ln.lower().startswith("abstract:"):
            snippet = ln.split(":", 1)[1].strip() or snippet
        elif ln.startswith("[PAPER]"):
            title = ln

    if not url:
        m = re.search(r"https?://[^\s]+", text)
        if m:
            url = m.group(0)

    return {"title": title, "url": url, "snippet": snippet}


def collect_node_research(node_name: str, query: str, max_results: int = 6) -> Dict[str, Any]:
    q = str(query or "").strip()
    # Remove tuple-like chat wrapper and overlong punctuation noise.
    m = re.match(r"^\(\s*'[^']+'\s*,\s*'(.*)'\s*\)$", q)
    if m:
        q = m.group(1)
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) > 220:
        q = q[:220].rsplit(" ", 1)[0]
    logs: List[str] = []
    refs: List[Dict[str, Any]] = []
    trace_items: List[Dict[str, Any]] = []
    seen = set()

    if not q:
        return {"logs": [f"[SEARCH] {node_name}: empty query, skipped."], "references": []}

    logs.append(f"[SEARCH] {node_name}: query='{q[:180]}'")

    try:
        rag_bundle = retrieve_flyback_context(q, top_k=max(2, min(4, max_results)))
        rag_refs = rag_bundle.get("references") or []
        logs.append(f"[RAG] Local flyback RAG hits: {len(rag_refs)}")
        for row in rag_refs[: max(2, max_results // 2)]:
            source_label = normalize_source_label(row.get("source") or "local_rag")
            item = {
                "channel": "local_rag",
                "source": source_label,
                "title": source_label,
                "url": str(row.get("url") or row.get("path") or ""),
                "snippet": "Matched from the internal flyback knowledge base.",
                "status": "retrieved",
            }
            trace_items.append(item)
            logs.append(f"[SEARCH-HIT] {json.dumps(item, ensure_ascii=False)}")
            refs.append(
                {
                    "source_type": "AppNote",
                    "title": item["title"],
                    "url": item["url"],
                    "relevance_score": float(row.get("score") or 0.7),
                    "key_insight": item["snippet"],
                    "insight": item["snippet"],
                }
            )
    except Exception as e:
        logs.append(f"[WARNING] Local flyback RAG failed: {e}")

    if retrieve_component_rag_context and node_name in {"selector", "designer", "validator"}:
        try:
            component_bundle = retrieve_component_rag_context(q, top_k=max(3, min(6, max_results)))
            component_refs = component_bundle.get("references") or []
            logs.append(f"[RAG] Local component RAG hits: {len(component_refs)}")
            for row in component_refs[: max(2, max_results // 2)]:
                source_label = normalize_source_label(row.get("source") or "component_db")
                item = {
                    "channel": "component_rag",
                    "source": source_label,
                    "title": str(row.get("title") or source_label),
                    "url": str(row.get("url") or ""),
                    "snippet": str(row.get("key_insight") or "Matched from the internal component reference library.")[:180],
                    "status": "retrieved",
                }
                trace_items.append(item)
                logs.append(f"[SEARCH-HIT] {json.dumps(item, ensure_ascii=False)}")
                refs.append(
                    {
                        "source_type": "Datasheet",
                        "title": item["title"],
                        "url": item["url"],
                        "relevance_score": float(row.get("score") or 0.82),
                        "key_insight": item["snippet"],
                        "insight": item["snippet"],
                    }
                )
        except Exception as e:
            logs.append(f"[WARNING] Local component RAG failed: {e}")

    # 1) MCP-first
    mcp_rows: List[Dict[str, str]] = []
    if mcp_research_web:
        try:
            mcp_out = mcp_research_web(q, max_results=max_results)
            if isinstance(mcp_out, dict):
                for note in mcp_out.get("notes") or []:
                    logs.append(f"[SEARCH] MCP note: {note}")
                for row in mcp_out.get("results") or []:
                    if not isinstance(row, dict):
                        continue
                    mcp_rows.append(
                        {
                            "title": str(row.get("title") or "Web Result"),
                            "url": str(row.get("url") or ""),
                            "snippet": str(row.get("snippet") or row.get("extracted_text") or ""),
                        }
                    )
            logs.append(f"[SEARCH] MCP results: {len(mcp_rows)}")
        except Exception as e:
            logs.append(f"[WARNING] MCP search failed: {e}")

    # 2) optional local fallback if MCP result is empty
    local_rows: List[Dict[str, str]] = []
    enable_local_fallback = str(os.environ.get("PE_MAS_ENABLE_LOCAL_WEB_FALLBACK", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if not mcp_rows and enable_local_fallback:
        try:
            global local_search
            if local_search is None:
                from core.utils.web_search import perform_search as local_search  # type: ignore
            raw_rows = local_search(q, max_results=max_results)
            for row in raw_rows or []:
                local_rows.append(_parse_local_search_row(row))
            logs.append(f"[SEARCH] Fallback web results: {len(local_rows)}")
        except Exception as e:
            logs.append(f"[WARNING] Fallback search failed: {e}")

    rows = mcp_rows or local_rows
    for row in rows[:max_results]:
        title = str(row.get("title") or "Web Result").strip()
        url = str(row.get("url") or "").strip()
        snippet = str(row.get("snippet") or "").strip()
        key = (title.lower(), url.lower())
        if key in seen:
            continue
        seen.add(key)
        host = ""
        if url:
            host_match = re.search(r"https?://([^/]+)", url)
            host = host_match.group(1) if host_match else ""

        item = {
            "channel": "web",
            "source": host or "web",
            "title": title,
            "url": url,
            "snippet": snippet[:220],
            "status": "retrieved",
        }
        trace_items.append(item)
        logs.append(f"[SEARCH-HIT] {json.dumps(item, ensure_ascii=False)}")

        refs.append(
            {
                "source_type": "Web",
                "title": title,
                "url": url,
                "relevance_score": 0.8,
                "key_insight": snippet[:220],
                # Backward-compatible aliases used by existing UI/evidence code.
                "insight": snippet[:220],
            }
        )

    if not refs:
        logs.append("[WARNING] No external references collected for this node.")

    return {"logs": logs, "references": refs, "trace_items": trace_items}
