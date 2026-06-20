from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from .digikey_local_db import query_local_components, rag_lookup_components

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "knowledge", "component_db"))
READINESS_JSON_PATH = os.path.join(BASE_DIR, "mas_component_readiness.json")

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "mosfet": ["mosfet", "fet"],
    "diode_rectifier": ["rectifier diode", "diode rectifier"],
    "diode_schottky": ["schottky"],
    "tvs_diodes": ["tvs"],
    "zener_diodes": ["zener"],
    "bridge_rectifier": ["bridge rectifier"],
    "igbt_single": ["igbt"],
    "bjt_single": ["bjt", "bipolar transistor"],
    "gate_driver": ["gate driver"],
    "offline_controller": ["offline controller", "ac-dc controller"],
    "dc_dc_controller": ["dc-dc controller", "buck controller", "boost controller"],
    "controller": ["controller", "pwm"],
    "opamp": ["opamp", "op-amp", "operational amplifier"],
    "voltage_reference": ["voltage reference"],
    "isolator_digital": ["digital isolator"],
    "optocoupler_transistor": ["optocoupler", "opto"],
    "transformer_power": ["power transformer"],
    "transformer_smps": ["smps transformer", "flyback transformer"],
    "transformer": ["transformer"],
    "current_transformer": ["current transformer", "ct"],
    "ferrite_cores": ["ferrite core"],
    "inductor_power": ["power inductor", "inductor"],
    "common_mode_choke": ["common mode choke", "cm choke"],
    "emi_filter_power_line": ["power line filter"],
    "emi_filter": ["emi filter", "filter choke"],
    "x2_capacitor": ["x2 capacitor"],
    "y_capacitor": ["y capacitor"],
    "electrolytic_cap": ["electrolytic capacitor"],
    "polymer_cap": ["polymer capacitor"],
    "mlcc_cap": ["mlcc", "ceramic capacitor"],
    "film_cap": ["film capacitor"],
    "supercap": ["supercap", "super capacitor"],
    "fuse": ["fuse"],
    "fuse_holder": ["fuse holder"],
    "pptc_resettable_fuse": ["pptc", "resettable fuse"],
    "mov_varistor": ["mov", "varistor"],
    "gas_discharge_tube": ["gdt", "gas discharge"],
    "esd_suppression": ["esd"],
    "ntc_thermistor": ["ntc", "thermistor"],
    "resistor_through_hole": ["through hole resistor"],
    "resistor_smd": ["smd resistor", "chip resistor"],
    "shunt_resistor": ["shunt resistor", "current sense resistor"],
    "relay_power": ["power relay", "relay"],
    "connector_terminal_block": ["terminal block", "connector"],
    "heatsink": ["heatsink", "heat sink"],
    "thermal_pad": ["thermal pad"],
    "fan": ["fan", "blower"],
    "snubber_network": ["snubber network"],
    "clamp_network": ["clamp network"],
    "clamp_snubber": ["snubber", "clamp"],
    "input_protection": ["input protection", "surge protection"],
    "input_cap": ["input capacitor", "bulk capacitor"],
    "output_cap": ["output capacitor"],
}


def _load_readiness() -> Dict[str, Any]:
    if not os.path.exists(READINESS_JSON_PATH):
        return {}
    try:
        data = json.loads(open(READINESS_JSON_PATH, "r", encoding="utf-8").read())
        return data.get("categories", {}) if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_component_readiness_map() -> Dict[str, Any]:
    return _load_readiness()


def get_component_readiness(category: str) -> Dict[str, Any]:
    readiness = _load_readiness()
    return readiness.get(str(category or "").strip(), {}) if isinstance(readiness, dict) else {}


def _infer_categories(query: str, max_categories: int = 5) -> List[str]:
    text = str(query or "").lower()
    if not text:
        return []

    scores: List[tuple[int, str]] = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        hit = 0
        for kw in keywords:
            if kw in text:
                hit += 1
        if hit > 0:
            scores.append((hit, category))

    scores.sort(key=lambda x: (-x[0], x[1]))
    return [category for _, category in scores[:max_categories]]


def retrieve_component_rag_context(query: str, top_k: int = 8) -> Dict[str, Any]:
    """
    Retrieve local DigiKey evidence and readiness policy snippets for MAS RAG.
    """
    q = str(query or "").strip()
    if not q:
        return {"context_text": "", "references": []}

    readiness = _load_readiness()
    inferred = _infer_categories(q, max_categories=6)

    # Broad text match from SQLite.
    text_hits = rag_lookup_components(q, limit=max(4, min(top_k, 12)))

    # Explicit category hits for better coverage when query text is generic.
    category_hits: List[Dict[str, Any]] = []
    for category in inferred:
        category_hits.extend(query_local_components(category, limit=2))

    refs: List[Dict[str, Any]] = []
    seen = set()

    def add_part_ref(row: Dict[str, Any], reason: str) -> None:
        part = str(row.get("part_number") or "").strip()
        url = str(row.get("url") or "").strip()
        key = (part.lower(), url.lower())
        if not part or key in seen:
            return
        seen.add(key)

        refs.append(
            {
                "source": "digikey_local_db",
                "source_type": "ComponentDB",
                "title": part,
                "url": url,
                "score": 0.9,
                "key_insight": reason,
                "component_type": row.get("component_type"),
                "price": row.get("price"),
                "stock": row.get("stock"),
            }
        )

    for row in text_hits:
        add_part_ref(row, "Matched query in local DigiKey SQLite")
    for row in category_hits:
        add_part_ref(row, "Category-inferred candidate from local DigiKey SQLite")

    readiness_refs: List[Dict[str, Any]] = []
    for category in inferred:
        meta = readiness.get(category)
        if not isinstance(meta, dict):
            continue
        readiness_refs.append(
            {
                "source": "mas_component_readiness",
                "source_type": "ComponentPolicy",
                "title": f"{category} readiness",
                "url": "",
                "score": 0.88,
                "key_insight": (
                    f"color={meta.get('color')} score={meta.get('score')} "
                    f"rows={meta.get('rows')} strategy={meta.get('strategy')}"
                ),
                "component_type": category,
            }
        )

    references = refs[:top_k] + readiness_refs[: max(1, min(4, top_k // 2))]

    lines: List[str] = ["[DigiKey Local Component RAG]"]
    if inferred:
        lines.append("Inferred categories: " + ", ".join(inferred))

    if readiness_refs:
        lines.append("Category readiness policies:")
        for item in readiness_refs:
            lines.append(f"- {item['title']}: {item['key_insight']}")

    if refs:
        lines.append("Top component candidates:")
        for row in refs[:top_k]:
            lines.append(
                "- {part} ({ctype}) price={price} stock={stock} source={url}".format(
                    part=row.get("title") or "Unknown",
                    ctype=row.get("component_type") or "unknown",
                    price=row.get("price"),
                    stock=row.get("stock"),
                    url=row.get("url") or "https://www.digikey.com",
                )
            )

    return {
        "context_text": "\n".join(lines),
        "references": references,
        "inferred_categories": inferred,
    }
