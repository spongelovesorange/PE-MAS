from __future__ import annotations

from typing import Any, Dict, List


def _as_row(ref: Any) -> Dict[str, str]:
    if isinstance(ref, dict):
        title = str(ref.get("title") or ref.get("source") or ref.get("doc") or "reference")
        url = str(ref.get("url") or ref.get("link") or ref.get("path") or "")
    else:
        title = str(ref)
        url = ""
    return {"title": title.strip(), "url": url.strip()}


def build_citation_pack(context: Dict[str, Any]) -> Dict[str, Any]:
    refs = context.get("references") or context.get("literature_references") or []
    rows = [_as_row(r) for r in refs[:80]]

    dedup: List[Dict[str, str]] = []
    seen = set()
    for row in rows:
        key = (row["title"].lower(), row["url"].lower())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)

    indexed = []
    broken = []
    for idx, row in enumerate(dedup, start=1):
        cid = f"R{idx:03d}"
        url = row["url"]
        if url and not (url.startswith("http://") or url.startswith("https://")):
            broken.append({"id": cid, "reason": "invalid_url", "url": url})
        indexed.append({"id": cid, "title": row["title"], "url": url})

    md_lines = ["| ID | Title | URL |", "|---|---|---|"]
    for row in indexed[:80]:
        title = str(row["title"]).replace("|", "\\|")
        url = str(row["url"]).replace("|", "\\|")
        md_lines.append(f"| {row['id']} | {title} | {url} |")

    authority_hint = {
        "high": sum(1 for r in indexed if any(k in (r.get("url") or "").lower() for k in ["ieee", "ti.com", "analog.com", "infineon", "st.com", "onsemi", "digikey", "mouser"])),
        "total": len(indexed),
    }

    return {
        "normalized_references": indexed,
        "reference_index": {row["id"]: row for row in indexed},
        "broken_links": broken,
        "bibliography_markdown": "\n".join(md_lines),
        "authority_hint": authority_hint,
        "citation_audit": {
            "input_count": len(rows),
            "unique_count": len(indexed),
            "broken_count": len(broken),
        },
    }
