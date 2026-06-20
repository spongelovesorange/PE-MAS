from __future__ import annotations

from typing import Any, Dict, List


HIGH_TRUST = (
    "ieeexplore", "analog.com", "ti.com", "infineon.com", "onsemi.com", "st.com",
    "mouser", "digikey", "arxiv", "researchgate",
)
MEDIUM_TRUST = ("wikipedia", "allaboutcircuits", "electronicdesign", "eeweb")
LOW_TRUST = ("reddit", "zhihu", "csdn", "forum")


def _grade_one(ref: Any) -> Dict[str, Any]:
    title = "reference"
    url = ""
    if isinstance(ref, dict):
        title = str(ref.get("title") or ref.get("source") or ref.get("doc") or title)
        url = str(ref.get("url") or ref.get("link") or ref.get("path") or "")
    else:
        title = str(ref)

    low = (url + " " + title).lower()
    score = 0.55

    if any(k in low for k in HIGH_TRUST):
        score = 0.9
    elif any(k in low for k in MEDIUM_TRUST):
        score = 0.72
    elif any(k in low for k in LOW_TRUST):
        score = 0.45

    if not url:
        score -= 0.1

    score = max(0.05, min(0.98, score))
    grade = "A" if score >= 0.85 else ("B" if score >= 0.7 else ("C" if score >= 0.55 else "D"))
    return {"title": title, "url": url, "score": round(score, 3), "grade": grade}


def grade_evidence(context: Dict[str, Any]) -> Dict[str, Any]:
    refs_local = context.get("retrieved_knowledge_references") or []
    refs_lit = context.get("literature_references") or []
    refs = list(refs_local) + list(refs_lit)

    graded = [_grade_one(r) for r in refs[:30]]
    if graded:
        confidence = sum(x["score"] for x in graded) / len(graded)
    else:
        confidence = 0.3

    weak = [x for x in graded if x["grade"] in {"C", "D"}]
    bias_flags: List[str] = []
    if len(weak) >= max(1, len(graded) // 2):
        bias_flags.append("Evidence base is weak or non-authoritative.")
    if not graded:
        bias_flags.append("No traceable evidence references found.")

    return {
        "aggregate_confidence": round(confidence, 3),
        "evidence_grade": "HIGH" if confidence >= 0.82 else ("MEDIUM" if confidence >= 0.62 else "LOW"),
        "graded_sources": graded,
        "bias_flags": bias_flags,
    }
