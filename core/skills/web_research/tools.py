from __future__ import annotations

from typing import Any, Dict, List
import os
import re
import subprocess
import sys

from core.utils.chrome_mcp_client import (
    MCP_AVAILABLE as CHROME_MCP_AVAILABLE,
    browser_batch,
    browser_call_tool,
    browser_session_status,
    chrome_mcp_enabled,
    close_browser_session,
    ensure_browser_session,
    list_browser_tools,
    node_supported_for_chrome_mcp,
)

MCP_AVAILABLE = CHROME_MCP_AVAILABLE


DIGIKEY_COMPONENT_KEYWORDS: Dict[str, str] = {
    "mosfet": "MOSFET transistor",
    "diode": "diode rectifier ultrafast schottky",
    "controller": "PWM controller flyback power supply IC",
    "input_protection": "fuse NTC MOV bridge rectifier",
    "transformer": "flyback transformer core ferrite",
    "input_cap": "electrolytic capacitor 400V",
    "output_cap": "low ESR capacitor",
    "emi_filter": "common mode choke X2 capacitor Y1 capacitor",
    "clamp_snubber": "snubber resistor capacitor TVS diode ultrafast diode",
}

DIGIKEY_FALLBACK_CANDIDATES: Dict[str, List[Dict[str, Any]]] = {
    "mosfet": [
        {
            "part_number": "IPD60R380C6",
            "title": "Infineon 600V N-Channel MOSFET",
            "url": "https://www.digikey.com/en/products/filter/transistors/fets-mosfets/single-fets-mosfets/278",
            "snippet": "Fallback candidate for offline/unstable search.",
            "vds": 600.0,
            "id": 6.0,
            "price": 0.0,
            "score": 0.2,
        }
    ],
    "diode": [
        {
            "part_number": "MUR460",
            "title": "Ultra-fast rectifier diode 600V",
            "url": "https://www.digikey.com/en/products/filter/diodes/rectifiers/single-diodes/280",
            "snippet": "Fallback diode candidate.",
            "vds": 600.0,
            "id": 4.0,
            "price": 0.0,
            "score": 0.2,
            "component_type": "diode",
        }
    ],
    "controller": [
        {
            "part_number": "UC3845B",
            "title": "Current-mode PWM controller",
            "url": "https://www.digikey.com/en/products/filter/pmic/power-supply-controllers-monitors/760",
            "snippet": "Fallback controller candidate.",
            "vds": 0.0,
            "id": 0.0,
            "price": 0.0,
            "score": 0.2,
            "component_type": "controller",
        }
    ],
    "input_protection": [
        {
            "part_number": "T1A250V",
            "title": "Slow-blow fuse 1A 250V",
            "url": "https://www.digikey.com/en/products/filter/fuses/139",
            "snippet": "Fallback fuse candidate.",
            "vds": 250.0,
            "id": 1.0,
            "price": 0.0,
            "score": 0.2,
            "component_type": "input_protection",
        },
        {
            "part_number": "5D-9",
            "title": "NTC inrush current limiter",
            "url": "https://www.digikey.com/en/products/filter/inrush-current-limiters-icls/151",
            "snippet": "Fallback NTC candidate.",
            "vds": 0.0,
            "id": 0.0,
            "price": 0.0,
            "score": 0.2,
            "component_type": "input_protection",
        },
    ],
    "transformer": [
        {
            "part_number": "EFD20-FLYBACK-CORE",
            "title": "Ferrite core for flyback transformer",
            "url": "https://www.digikey.com/en/products/filter/ferrite-cores/936",
            "snippet": "Fallback transformer/core candidate.",
            "vds": 0.0,
            "id": 0.0,
            "price": 0.0,
            "score": 0.2,
            "component_type": "transformer",
        }
    ],
    "input_cap": [
        {
            "part_number": "400V-68UF-ELCAP",
            "title": "Electrolytic capacitor 400V 68uF",
            "url": "https://www.digikey.com/en/products/filter/aluminum-electrolytic-capacitors/58",
            "snippet": "Fallback HV bulk capacitor candidate.",
            "vds": 400.0,
            "id": 0.0,
            "price": 0.0,
            "score": 0.2,
            "component_type": "input_cap",
        }
    ],
    "output_cap": [
        {
            "part_number": "25V-470UF-LOWESR",
            "title": "Low ESR capacitor 25V 470uF",
            "url": "https://www.digikey.com/en/products/filter/aluminum-electrolytic-capacitors/58",
            "snippet": "Fallback output capacitor candidate.",
            "vds": 25.0,
            "id": 0.0,
            "price": 0.0,
            "score": 0.2,
            "component_type": "output_cap",
        }
    ],
    "emi_filter": [
        {
            "part_number": "CMC-2X10MH",
            "title": "Common-mode choke 2x10mH",
            "url": "https://www.digikey.com/en/products/filter/common-mode-chokes/839",
            "snippet": "Fallback EMI choke candidate.",
            "vds": 0.0,
            "id": 0.0,
            "price": 0.0,
            "score": 0.2,
            "component_type": "emi_filter",
        }
    ],
    "clamp_snubber": [
        {
            "part_number": "SMBJ440A",
            "title": "TVS diode for clamp/snubber",
            "url": "https://www.digikey.com/en/products/filter/tvs-diodes/144",
            "snippet": "Fallback snubber/TVS candidate.",
            "vds": 440.0,
            "id": 0.0,
            "price": 0.0,
            "score": 0.2,
            "component_type": "clamp_snubber",
        }
    ],
}


def _extract_text_from_tool_result(result: Any) -> str:
    if result is None:
        return ""
    content = getattr(result, "content", None)
    if not content:
        return ""
    pieces: List[str] = []
    for item in content:
        text_val = getattr(item, "text", None)
        if text_val:
            pieces.append(str(text_val))
    return "\n".join(pieces)[:5000]


def _ddgs_subprocess_search(query: str, max_results: int) -> Dict[str, Any]:
    worker = r"""
import json
import sys

query = sys.argv[1]
max_results = int(sys.argv[2])

try:
    from ddgs import DDGS
except Exception:
    try:
        from duckduckgo_search import DDGS
    except Exception:
        print(json.dumps({"results": [], "error": "ddgs_unavailable"}))
        raise SystemExit(0)

rows = []
with DDGS(timeout=8) as ddgs:
    for item in ddgs.text(query, max_results=max_results):
        rows.append(
            {
                "title": item.get("title", ""),
                "url": item.get("href", ""),
                "snippet": item.get("body", ""),
            }
        )
print(json.dumps({"results": rows}, ensure_ascii=False))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", worker, str(query or ""), str(max(1, int(max_results or 5)))],
            capture_output=True,
            text=True,
            timeout=18,
            check=False,
        )
    except Exception as e:
        return {"results": [], "error": f"ddgs_subprocess_failed: {e}"}

    stdout = str(proc.stdout or "").strip()
    stderr = str(proc.stderr or "").strip()
    if proc.returncode != 0:
        return {"results": [], "error": stderr or f"ddgs_subprocess_exit_{proc.returncode}"}
    if not stdout:
        return {"results": [], "error": stderr or "ddgs_subprocess_empty"}
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except Exception:
        return {"results": [], "error": stderr or "ddgs_subprocess_parse_failed"}
    if not isinstance(payload, dict):
        return {"results": [], "error": "ddgs_subprocess_invalid_payload"}
    return payload


def _snapshot_urls_with_mcp(urls: List[str], session_key: str = "web_research") -> Dict[str, str]:
    if not MCP_AVAILABLE:
        return {}
    results: Dict[str, str] = {}
    actions: List[Dict[str, Any]] = []
    step_map: List[str] = []
    for url in urls:
        actions.append({"tool_name": "new_page", "arguments": {"url": url}})
        step_map.append(f"open::{url}")
        actions.append({"tool_name": "take_snapshot", "arguments": {"verbose": False}})
        step_map.append(f"snapshot::{url}")

    batch = browser_batch(actions, session_key=session_key, stop_on_error=False)
    if not batch.get("ok"):
        return {}

    for label, row in zip(step_map, batch.get("results") or []):
        if not label.startswith("snapshot::"):
            continue
        url = label.split("::", 1)[1]
        text = str(row.get("raw_text") or "")
        text = re.sub(r"\s+", " ", text).strip()[:3500]
        results[url] = text
    return results


def research_web(query: str, max_results: int = 5) -> Dict[str, Any]:
    query = str(query or "").strip()
    if not query:
        return {"query": query, "results": [], "notes": ["Empty query"]}

    raw_results: List[Dict[str, Any]] = []
    ddgs_payload = _ddgs_subprocess_search(query, max_results)
    raw_results = ddgs_payload.get("results") or []
    ddgs_error = str(ddgs_payload.get("error") or "").strip()
    if ddgs_error and not raw_results:
        ddgs_error = f"Search backend unavailable: {ddgs_error}"

    if not raw_results:
        severe_backend_fault = any(flag in ddgs_error.lower() for flag in ["panic", "subprocess_exit", "attempted to create a null object"])
        enable_local_fallback = str(os.getenv("PE_MAS_ENABLE_LOCAL_WEB_FALLBACK", "0")).strip().lower() in {"1", "true", "yes", "on"}
        try:
            from core.utils.web_search import perform_search as local_search

            local_rows = (
                local_search(query, max_results=max_results)
                if (local_search and not severe_backend_fault and enable_local_fallback)
                else []
            )
            parsed_rows: List[Dict[str, Any]] = []
            for row in local_rows or []:
                if isinstance(row, dict):
                    parsed_rows.append(
                        {
                            "title": row.get("title", ""),
                            "url": row.get("url", row.get("href", "")),
                            "snippet": row.get("snippet", row.get("body", "")),
                            "extracted_text": "",
                        }
                    )
                else:
                    text = str(row)
                    title, url, snippet = "Web Result", "", text[:600]
                    for ln in text.splitlines():
                        low = ln.lower().strip()
                        if low.startswith("title:"):
                            title = ln.split(":", 1)[1].strip() or title
                        elif low.startswith("link:"):
                            url = ln.split(":", 1)[1].strip()
                        elif low.startswith("snippet:"):
                            snippet = ln.split(":", 1)[1].strip() or snippet
                    parsed_rows.append({"title": title, "url": url, "snippet": snippet, "extracted_text": ""})

            if parsed_rows:
                return {
                    "query": query,
                    "results": parsed_rows[:max_results],
                    "notes": [msg for msg in [ddgs_error, "DDGS returned no rows; used local web-search fallback."] if msg],
                }
            if not enable_local_fallback:
                return {
                    "query": query,
                    "results": [],
                    "notes": [msg for msg in [ddgs_error, "Local fallback disabled by default for runtime stability (set PE_MAS_ENABLE_LOCAL_WEB_FALLBACK=1 to enable)."] if msg],
                }
            if severe_backend_fault:
                return {
                    "query": query,
                    "results": [],
                    "notes": [msg for msg in [ddgs_error, "Local fallback skipped because the primary web backend failed at process level."] if msg],
                }
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "notes": [msg for msg in [ddgs_error or f"Search backend unavailable: {e}"] if msg],
            }

    if not MCP_AVAILABLE:
        return {
            "query": query,
            "results": [{**r, "extracted_text": ""} for r in raw_results[:max_results]],
            "notes": [
                "MCP package not installed; returning text-search results only.",
                "Optional install: pip install mcp",
            ],
        }

    if not chrome_mcp_enabled():
        return {
            "query": query,
            "results": [{**r, "extracted_text": ""} for r in raw_results[:max_results]],
            "notes": [
                "Chrome MCP snapshot disabled by default for stability.",
                "Set PE_MAS_ENABLE_CHROME_MCP=1 to enable page snapshots.",
            ],
        }

    if not node_supported_for_chrome_mcp():
        return {
            "query": query,
            "results": [{**r, "extracted_text": ""} for r in raw_results[:max_results]],
            "notes": ["Chrome MCP snapshot skipped: Node.js too old (need >=20.19). Returning text-search results."],
        }

    urls = [r.get("url", "") for r in raw_results[:max_results] if r.get("url")]
    try:
        mcp_snapshots = _snapshot_urls_with_mcp(urls, session_key="web_research")
    except Exception as e:
        return {
            "query": query,
            "results": [{**r, "extracted_text": ""} for r in raw_results[:max_results]],
            "notes": [f"Chrome MCP snapshot failed: {e}. Returning text-search results without page snapshots."],
        }

    notes: List[str] = ["Chrome MCP enabled: browser snapshot extraction active."]

    enriched = []
    for r in raw_results[:max_results]:
        url = r.get("url", "")
        extracted = mcp_snapshots.get(url, "")
        enriched.append({**r, "extracted_text": extracted})

    return {
        "query": query,
        "results": enriched,
        "notes": notes,
    }


def browser_tool_catalog(session_key: str = "default") -> Dict[str, Any]:
    return list_browser_tools(session_key=session_key)


def browser_open_session(session_key: str = "default") -> Dict[str, Any]:
    return ensure_browser_session(session_key=session_key)


def browser_close(session_key: str = "default") -> Dict[str, Any]:
    return close_browser_session(session_key=session_key)


def browser_status(session_key: str = "default") -> Dict[str, Any]:
    return browser_session_status(session_key=session_key)


def browser_interact(
    action: str,
    arguments: Dict[str, Any] | None = None,
    session_key: str = "default",
) -> Dict[str, Any]:
    tool_name = str(action or "").strip()
    if not tool_name:
        return {"ok": False, "error": "missing browser action/tool name"}
    return browser_call_tool(tool_name, arguments or {}, session_key=session_key)


def browser_interact_batch(
    actions: List[Dict[str, Any]],
    session_key: str = "default",
    stop_on_error: bool = False,
) -> Dict[str, Any]:
    return browser_batch(actions or [], session_key=session_key, stop_on_error=stop_on_error)


def search_and_open_pages(
    query: str,
    *,
    max_results: int = 5,
    open_top_n: int = 3,
    session_key: str = "default",
    snapshot_verbose: bool = False,
) -> Dict[str, Any]:
    base = research_web(query, max_results=max_results)
    rows = list(base.get("results") or [])
    urls = [str(row.get("url") or "").strip() for row in rows if str(row.get("url") or "").strip()]
    if not urls:
        return {
            "ok": False,
            "query": query,
            "results": rows,
            "notes": list(base.get("notes") or []) + ["No URLs were available to open interactively."],
        }

    actions: List[Dict[str, Any]] = []
    for url in urls[: max(1, int(open_top_n or 1))]:
        actions.append({"tool_name": "new_page", "arguments": {"url": url}})
        actions.append({"tool_name": "take_snapshot", "arguments": {"verbose": bool(snapshot_verbose)}})

    batch = browser_batch(actions, session_key=session_key, stop_on_error=False)
    snapshots: List[Dict[str, Any]] = []
    batch_rows = list(batch.get("results") or [])
    for idx, url in enumerate(urls[: max(1, int(open_top_n or 1))]):
        snap_idx = idx * 2 + 1
        snap_row = batch_rows[snap_idx] if snap_idx < len(batch_rows) else {}
        snapshots.append(
            {
                "url": url,
                "snapshot": str(snap_row.get("raw_text") or "")[:4000],
                "ok": bool(snap_row.get("ok")),
            }
        )

    return {
        "ok": bool(batch.get("ok")),
        "query": query,
        "results": rows,
        "snapshots": snapshots,
        "notes": list(base.get("notes") or []),
        "browser_batch": batch,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_first_number(pattern: str, text: str) -> float:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except Exception:
        return 0.0


def _extract_part_number(title: str, snippet: str) -> str:
    # Prefer realistic MPN-like tokens and avoid generic words.
    candidates = re.findall(r"\b[A-Z][A-Z0-9\-]{4,20}\b", f"{title} {snippet}")
    deny = {
        "MOSFET", "DIGIKEY", "DIGI-KEY", "PRICE", "STOCK", "BUY", "NCHANNEL", "N-CHANNEL",
        "TRANSISTOR", "POWER", "ELECTRONICS",
    }
    for token in candidates:
        if token.upper() not in deny:
            return token
    return title.strip()[:80] or "Unknown"


def research_digikey_mosfet(
    min_vds: float,
    min_id: float,
    max_results: int = 8,
    channel: str = "N-Channel",
) -> Dict[str, Any]:
    """
    DigiKey-focused MOSFET search that prefers real distributor pages and
    returns structured candidates for MAS selector.
    """
    min_vds = max(0.0, _safe_float(min_vds))
    min_id = max(0.0, _safe_float(min_id))
    max_results = max(1, int(max_results or 8))
    channel = str(channel or "N-Channel").strip() or "N-Channel"
    strict_mode = str(os.getenv("PE_MAS_DIGIKEY_STRICT", "1")).strip().lower() in {"1", "true", "yes", "on"}

    query = (
        "site:digikey.com "
        "(\"/en/products/detail/\" OR \"/en/products/filter/\") "
        f"MOSFET {channel} {int(round(min_vds))}V {min_id:.1f}A in stock price"
    )

    base = research_web(query, max_results=max_results)
    rows = base.get("results") or []
    notes = list(base.get("notes") or [])
    candidates: List[Dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        title = str(row.get("title", ""))
        url = str(row.get("url", ""))
        snippet = str(row.get("snippet", ""))
        combined = f"{title} {snippet}"
        low = combined.lower()

        if "digikey" not in url.lower():
            continue
        if "/en/products/" not in url:
            continue

        # Prefer target channel. If channel is not explicitly present,
        # still keep record as long as it looks like MOSFET catalog entry.
        if "mosfet" not in low and "fet" not in low:
            continue
        if channel.lower().startswith("n") and ("p-channel" in low or "p channel" in low):
            continue

        vds_found = _extract_first_number(r"(\d+(?:\.\d+)?)\s*V", combined)
        id_found = _extract_first_number(r"(\d+(?:\.\d+)?)\s*A", combined)
        price_found = _extract_first_number(r"\$\s*(\d+(?:\.\d+)?)", combined)

        # Allow unknown rating in snippet/title, but heavily prefer rows with ratings.
        if vds_found > 0 and vds_found < (0.85 * min_vds):
            continue
        if id_found > 0 and id_found < (0.85 * min_id):
            continue

        part_number = _extract_part_number(title, snippet)
        score = 0.0
        if "/detail/" in url:
            score += 3.0
        if "in stock" in low:
            score += 1.5
        if "price" in low or "$" in combined:
            score += 1.0
        if vds_found >= min_vds and vds_found > 0:
            score += 2.0
        if id_found >= min_id and id_found > 0:
            score += 2.0
        if "n-channel" in low or "n channel" in low:
            score += 1.5

        candidates.append(
            {
                "part_number": part_number,
                "title": title,
                "url": url,
                "snippet": snippet,
                "vds": vds_found,
                "id": id_found,
                "price": price_found,
                "score": score,
            }
        )

    candidates.sort(key=lambda x: (-float(x.get("score", 0.0)), float(x.get("price", 999999.0))))
    top = candidates[:max_results]

    selection_mode = "live"

    if not top:
        # Fallback query for broader DigiKey MOSFET pages.
        fallback_query = (
            "site:digikey.com "
            "(\"/en/products/detail/\" OR \"/en/products/filter/transistors/fets-mosfets\") "
            "MOSFET in stock"
        )
        fb = research_web(fallback_query, max_results=max_results)
        fb_rows = fb.get("results") or []
        for row in fb_rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title", ""))
            url = str(row.get("url", ""))
            snippet = str(row.get("snippet", ""))
            if "digikey" not in url.lower() or "/en/products/" not in url:
                continue
            combined = f"{title} {snippet}"
            top.append(
                {
                    "part_number": _extract_part_number(title, snippet),
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "vds": _extract_first_number(r"(\d+(?:\.\d+)?)\s*V", combined),
                    "id": _extract_first_number(r"(\d+(?:\.\d+)?)\s*A", combined),
                    "price": _extract_first_number(r"\$\s*(\d+(?:\.\d+)?)", combined),
                    "score": 0.5,
                }
            )
            if len(top) >= max_results:
                break

    if not top:
        if strict_mode:
            selection_mode = "none"
            notes.append("DigiKey strict mode is ON; no fallback candidates allowed.")
        else:
            top = [dict(x) for x in DIGIKEY_FALLBACK_CANDIDATES.get("mosfet", [])[:max_results]]
            if top:
                selection_mode = "fallback"
                notes.append("DigiKey search unstable; used built-in MOSFET fallback candidates.")
            else:
                selection_mode = "none"
                notes.append("DigiKey-focused query returned no structured candidates.")
    else:
        notes.append(f"DigiKey-focused query produced {len(top)} candidate(s).")

    return {
        "query": query,
        "requirements": {
            "channel": channel,
            "min_vds": min_vds,
            "min_id": min_id,
        },
        "selection_mode": selection_mode,
        "results": top,
        "notes": notes,
    }


def get_supported_digikey_component_types() -> List[str]:
    return list(DIGIKEY_COMPONENT_KEYWORDS.keys())


def research_digikey_component(
    component_type: str,
    min_vds: float = 0.0,
    min_id: float = 0.0,
    max_results: int = 8,
    channel: str = "N-Channel",
) -> Dict[str, Any]:
    component_type = str(component_type or "").strip().lower()
    strict_mode = str(os.getenv("PE_MAS_DIGIKEY_STRICT", "1")).strip().lower() in {"1", "true", "yes", "on"}
    if component_type not in DIGIKEY_COMPONENT_KEYWORDS:
        return {
            "query": "",
            "requirements": {"component_type": component_type},
            "results": [],
            "notes": [
                f"Unsupported component_type: {component_type}",
                f"Supported types: {', '.join(get_supported_digikey_component_types())}",
            ],
        }

    if component_type == "mosfet":
        return research_digikey_mosfet(
            min_vds=min_vds,
            min_id=min_id,
            max_results=max_results,
            channel=channel,
        )

    max_results = max(1, int(max_results or 8))
    keyword = DIGIKEY_COMPONENT_KEYWORDS[component_type]
    query = (
        "site:digikey.com "
        "(\"/en/products/detail/\" OR \"/en/products/filter/\") "
        f"{keyword} in stock price"
    )

    notes: List[str] = []
    rows: List[Dict[str, Any]] = []

    base = research_web(query, max_results=max_results)
    rows = [r for r in (base.get("results") or []) if isinstance(r, dict)]
    notes = list(base.get("notes") or [])
    candidates: List[Dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", ""))
        url = str(row.get("url", ""))
        snippet = str(row.get("snippet", ""))
        combined = f"{title} {snippet}"
        low = combined.lower()

        if "digikey" not in url.lower() or "/en/products/" not in url:
            continue

        part_number = _extract_part_number(title, snippet)
        vds_found = _extract_first_number(r"(\d+(?:\.\d+)?)\s*V", combined)
        id_found = _extract_first_number(r"(\d+(?:\.\d+)?)\s*A", combined)
        price_found = _extract_first_number(r"\$\s*(\d+(?:\.\d+)?)", combined)

        score = 0.0
        if "/detail/" in url:
            score += 3.0
        if "in stock" in low:
            score += 1.0
        if "price" in low or "$" in combined:
            score += 1.0
        if component_type in {"diode", "controller"} and ("diode" in low or "controller" in low or "pwm" in low):
            score += 1.0

        candidates.append(
            {
                "part_number": part_number,
                "title": title,
                "url": url,
                "snippet": snippet,
                "vds": vds_found,
                "id": id_found,
                "price": price_found,
                "score": score,
                "component_type": component_type,
            }
        )

    candidates.sort(key=lambda x: (-float(x.get("score", 0.0)), float(x.get("price", 999999.0))))
    top = candidates[:max_results]
    selection_mode = "live"
    if not top:
        if strict_mode:
            selection_mode = "none"
            notes.append("DigiKey strict mode is ON; no fallback candidates allowed.")
        else:
            fallback = [dict(x) for x in DIGIKEY_FALLBACK_CANDIDATES.get(component_type, [])[:max_results]]
            if fallback:
                top = fallback
                selection_mode = "fallback"
                notes.append(f"DigiKey search unstable; used built-in fallback candidates for {component_type}.")
            else:
                selection_mode = "none"
                notes.append(f"DigiKey-focused query returned no structured candidates for {component_type}.")
    else:
        notes.append(f"DigiKey-focused query produced {len(top)} candidate(s) for {component_type}.")

    return {
        "query": query,
        "requirements": {
            "component_type": component_type,
            "channel": channel,
            "min_vds": _safe_float(min_vds),
            "min_id": _safe_float(min_id),
        },
        "selection_mode": selection_mode,
        "results": top,
        "notes": notes,
        "supported_types": get_supported_digikey_component_types(),
    }
