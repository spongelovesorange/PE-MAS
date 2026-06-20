import csv
import json
import os
import re
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "knowledge", "component_db"))
MOSFET_CSV_PATH = os.path.join(BASE_DIR, "single_fets__mosfets.csv")
SQLITE_DB_PATH = os.path.join(BASE_DIR, "digikey_components.sqlite")
RAW_EXPORT_DIR = os.path.join(BASE_DIR, "raw_exports")

PART_KEY_CANDIDATES = [
    "Mfr Part #",
    "Manufacturer Part Number",
    "Part Number",
    "MPN",
]

DESC_KEY_CANDIDATES = [
    "Description",
    "Product Description",
    "Title",
]

SUPPLIER_KEY_CANDIDATES = [
    "Supplier",
    "Distributor",
]

URL_KEY_CANDIDATES = [
    "Product URL",
    "URL",
    "Link",
]

VDS_KEY_CANDIDATES = [
    "Voltage - Rated",
    "Voltage - Rated (V)",
    "Drain to Source Voltage (Vdss)",
    "Reverse Voltage",
    "Voltage - Rated",
    "Voltage Rating",
    "Voltage",
]

ID_KEY_CANDIDATES = [
    "Current Rating",
    "Current - Rated",
    "Current - Continuous Drain (Id) @ 25°C",
    "Current Rating",
    "Current",
    "Forward Current",
]

PRICE_KEY_CANDIDATES = [
    "Unit Price",
    "Unit Price (USD)",
    "Price",
    "Price (USD)",
]

STOCK_KEY_CANDIDATES = [
    "Quantity Available",
    "Stock",
    "Available",
    "In Stock",
]

PREFERENCE_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "mosfet": {
        "reduce_switching_loss_focus": ["superjunction", "low gate charge", "qg", "fast switching", "coolmos", "c7", "cfd7"],
        "reduce_rms_current": ["low rds", "milliohm", "power mosfet"],
        "prioritize_voltage_margin": ["650v", "700v", "750v", "800v"],
    },
    "diode": {
        "prefer_low_vf_diode": ["schottky", "low vf", "silicon carbide", "sic", "ultrafast"],
        "prioritize_secondary_rectifier_loss_in_selector": ["schottky", "low vf", "ultrafast", "sic"],
        "consider_sync_rectification": ["schottky", "sic"],
    },
    "transformer": {
        "revisit_clamp_and_magnetics_loss_budget": ["ferrite", "flyback", "switching", "gapped", "core"],
        "reduce_core_loss_focus": ["ferrite", "low loss", "switching", "high frequency"],
    },
    "output_cap": {
        "check_output_cap_esr_and_ripple": ["low esr", "polymer", "hybrid", "ripple current"],
    },
    "input_cap": {
        "stabilize_bus_ripple": ["electrolytic", "long life", "high ripple"],
    },
    "clamp_snubber": {
        "prioritize_voltage_margin": ["tvs", "snubber", "clamp", "transorb"],
        "revisit_clamp_and_magnetics_loss_budget": ["tvs", "snubber", "clamp", "ultrafast"],
    },
    "emi_filter": {
        "reduce_switching_noise_focus": ["common mode", "emi", "x2", "y1", "choke"],
    },
}


def _first_present(row: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in row and row.get(k) not in (None, ""):
            return row.get(k)
    return None


def _guess_component_type_from_name(file_name: str) -> str:
    n = file_name.lower()
    if "mosfet" in n or "fet" in n:
        return "mosfet"
    if "diode" in n or "rectifier" in n:
        return "diode"
    if "controller" in n or "pmic" in n or "pwm" in n:
        return "controller"
    if "transformer" in n or "ferrite" in n or "core" in n:
        return "transformer"
    if "input" in n and "cap" in n:
        return "input_cap"
    if "output" in n and "cap" in n:
        return "output_cap"
    if "cap" in n or "capacitor" in n:
        return "output_cap"
    if "fuse" in n or "varistor" in n or "ntc" in n or "bridge" in n or "protection" in n:
        return "input_protection"
    if "emi" in n or "choke" in n or "x2" in n or "y1" in n:
        return "emi_filter"
    if "snubber" in n or "tvs" in n or "clamp" in n:
        return "clamp_snubber"
    return "other"


def _normalize_row(row: Dict[str, Any], component_type: str, default_url: str) -> Dict[str, Any]:
    part_number = (_first_present(row, PART_KEY_CANDIDATES) or "").strip()
    title = (_first_present(row, DESC_KEY_CANDIDATES) or "").strip()
    supplier = (_first_present(row, SUPPLIER_KEY_CANDIDATES) or "DigiKey").strip() or "DigiKey"
    url = (_first_present(row, URL_KEY_CANDIDATES) or default_url).strip() or default_url

    # Build a best-effort URL if DK part exists.
    dk_part = str(row.get("DK Part #") or "").split(",")[0].strip()
    if (not url or url == default_url) and dk_part:
        url = f"https://www.digikey.com/en/products/result?s={dk_part}"

    return {
        "component_type": component_type,
        "part_number": part_number,
        "title": title,
        "description": title,
        "url": url,
        "supplier": supplier,
        "stock": _parse_float(_first_present(row, STOCK_KEY_CANDIDATES)),
        "price": _parse_float(_first_present(row, PRICE_KEY_CANDIDATES)),
        "vds": _parse_float(_first_present(row, VDS_KEY_CANDIDATES)),
        "id_current": _parse_float(_first_present(row, ID_KEY_CANDIDATES)),
        "source_mode": "local-digikey-csv",
        "raw_json": json.dumps(row, ensure_ascii=False),
    }


def _insert_rows(cur: sqlite3.Cursor, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    tuples = [
        (
            r.get("component_type"),
            r.get("part_number"),
            r.get("title"),
            r.get("description"),
            r.get("url"),
            r.get("supplier"),
            r.get("stock"),
            r.get("price"),
            r.get("vds"),
            r.get("id_current"),
            r.get("source_mode"),
            r.get("raw_json"),
        )
        for r in rows
    ]
    cur.executemany(
        """
        INSERT INTO components(
            component_type, part_number, title, description, url, supplier,
            stock, price, vds, id_current, source_mode, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        tuples,
    )
    return len(tuples)


def ingest_csv_file(csv_path: str, component_type: Optional[str] = None) -> int:
    if not os.path.exists(csv_path):
        return 0
    ctype = component_type or _guess_component_type_from_name(os.path.basename(csv_path))
    default_url = "https://www.digikey.com"
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = _normalize_row(row, ctype, default_url)
            if n.get("part_number"):
                rows.append(n)
    conn = _connect()
    cur = conn.cursor()
    inserted = _insert_rows(cur, rows)
    conn.commit()
    conn.close()
    return inserted


def ingest_export_directory(export_dir: Optional[str] = None) -> Dict[str, int]:
    target = export_dir or RAW_EXPORT_DIR
    os.makedirs(target, exist_ok=True)
    files = [os.path.join(target, x) for x in os.listdir(target) if x.lower().endswith(".csv")]
    summary: Dict[str, int] = {}
    for fp in files:
        ctype = _guess_component_type_from_name(os.path.basename(fp))
        inserted = ingest_csv_file(fp, component_type=ctype)
        summary[os.path.basename(fp)] = inserted
    return summary


def _parse_float(text: Any) -> Optional[float]:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip()
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _safe_json_loads(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        val = json.loads(str(raw or ""))
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}


def _raw_row_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    raw = _safe_json_loads(row.get("raw_json"))
    nested = _safe_json_loads(raw.get("Raw Row JSON"))
    return nested or raw


def _row_blob(row: Dict[str, Any]) -> str:
    raw = _raw_row_dict(row)
    parts = [
        str(row.get("part_number") or ""),
        str(row.get("title") or ""),
        str(row.get("description") or ""),
        str(raw),
    ]
    return " ".join(parts).lower()


def _normalize_actions(preference_profile: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(preference_profile, dict):
        return []
    actions: List[str] = []
    for key in ["component_actions", "preferred_actions"]:
        vals = preference_profile.get(key) or []
        if isinstance(vals, list):
            actions.extend(str(v).strip().lower() for v in vals if str(v).strip())
    dominant_loss = str(preference_profile.get("dominant_loss") or "").strip().lower()
    if dominant_loss == "diode_conduction":
        actions.extend(["prefer_low_vf_diode", "prioritize_secondary_rectifier_loss_in_selector"])
    elif dominant_loss == "transformer_core":
        actions.extend(["reduce_core_loss_focus", "revisit_clamp_and_magnetics_loss_budget"])
    elif dominant_loss in {"mosfet_switching", "switching_loss", "turn_on_loss"}:
        actions.extend(["reduce_switching_loss_focus"])
    return list(dict.fromkeys([a for a in actions if a]))


def _text_overlap_score(query_text: str, blob: str) -> float:
    tokens = [t for t in re.split(r"[^a-z0-9]+", str(query_text or "").lower()) if len(t) >= 3]
    if not tokens:
        return 0.0
    hits = sum(1 for tok in tokens if tok in blob)
    return min(1.0, hits / max(1.0, len(tokens)))


def _parse_unit_value(text: Any, unit_scale: Dict[str, float], default: Optional[float] = None) -> Optional[float]:
    if text is None:
        return default
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip()
    if not s or s == "-":
        return default
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-zA-ZµμμΩohm/]+)?", s.replace(",", ""))
    if not match:
        return default
    try:
        val = float(match.group(1))
    except Exception:
        return default
    unit = str(match.group(2) or "").strip().lower()
    for key, scale in unit_scale.items():
        if key in unit:
            return val * scale
    return val


def _first_raw_numeric(raw: Dict[str, Any], keys: List[str], unit_scale: Dict[str, float], default: Optional[float] = None) -> Optional[float]:
    for key in keys:
        if key in raw and raw.get(key) not in (None, "", "-"):
            val = _parse_unit_value(raw.get(key), unit_scale, default=None)
            if val is not None:
                return val
    return default


def _engineering_feature_score(row: Dict[str, Any], component_type: str, preference_profile: Optional[Dict[str, Any]] = None) -> tuple[float, List[str]]:
    raw = _raw_row_dict(row)
    blob = _row_blob(row)
    reasons: List[str] = []
    score = 0.0
    actions = set(_normalize_actions(preference_profile))

    if component_type == "mosfet":
        rds = _first_raw_numeric(raw, ["Rds On (Max) @ Id, Vgs", "Drain to Source On Resistance (Rds On)"], {"mohm": 1e-3, "ohm": 1.0, "ω": 1.0})
        qg = _first_raw_numeric(raw, ["Gate Charge (Qg) (Max) @ Vgs", "Qg"], {"nc": 1.0, "pc": 1e-3})
        ciss = _first_raw_numeric(raw, ["Input Capacitance (Ciss) (Max) @ Vds", "Input Capacitance"], {"pf": 1.0, "nf": 1e3})
        if rds is not None and rds > 0:
            score += min(0.5, 0.08 / max(rds, 0.004))
            if rds <= 0.08:
                reasons.append("low_rds_on")
        if qg is not None and qg > 0:
            score += min(0.35, 10.0 / qg * 0.1)
            if qg <= 25:
                reasons.append("low_gate_charge")
        if ciss is not None and ciss > 0:
            score += min(0.25, 700.0 / ciss * 0.08)
            if ciss <= 1500:
                reasons.append("manageable_ciss")
        if "reduce_switching_loss_focus" in actions and qg is not None and qg <= 20:
            score += 0.18
            reasons.append("switching_loss_bias")

    elif component_type == "diode":
        vf = _first_raw_numeric(raw, ["Voltage - Forward (Vf) (Max) @ If", "Forward Voltage (Vf) (Max) @ If"], {"kv": 1e3, "mv": 1e-3, "v": 1.0})
        trr = _first_raw_numeric(raw, ["Reverse Recovery Time (trr)", "Speed"], {"ns": 1.0, "us": 1e3})
        io = _first_raw_numeric(raw, ["Current - Average Rectified (Io)", "Average Rectified Current", "Current - Average"], {"ma": 1e-3, "a": 1.0})
        if vf is not None and vf > 0:
            score += min(0.45, 0.35 / max(vf, 0.22))
            if vf <= 0.6:
                reasons.append("low_vf")
        if trr is not None and trr > 0:
            score += min(0.25, 30.0 / trr * 0.08)
            if trr <= 35:
                reasons.append("fast_recovery")
        if io is not None and io >= 2.0:
            score += min(0.18, io / 20.0)
        if ("prefer_low_vf_diode" in actions or "prioritize_secondary_rectifier_loss_in_selector" in actions) and ("schottky" in blob or "sic" in blob):
            score += 0.28
            reasons.append("secondary_rectifier_bias")

    elif component_type in {"input_cap", "output_cap"}:
        cap_f = _first_raw_numeric(raw, ["Capacitance"], {"uf": 1e-6, "µf": 1e-6, "nf": 1e-9, "pf": 1e-12, "mf": 1e-3, "f": 1.0})
        esr = _first_raw_numeric(raw, ["ESR (Equivalent Series Resistance)", "Equivalent Series Resistance"], {"mohm": 1e-3, "ohm": 1.0})
        zohm = _first_raw_numeric(raw, ["Impedance"], {"mohm": 1e-3, "ohm": 1.0})
        ripple_hf = _first_raw_numeric(raw, ["Ripple Current @ High Frequency", "Ripple Current"], {"ma": 1e-3, "a": 1.0})
        life_hours = _first_raw_numeric(raw, ["Lifetime @ Temp."], {"hrs": 1.0, "hr": 1.0, "h": 1.0})
        if cap_f is not None and cap_f > 0:
            score += min(0.38, cap_f / 470e-6 * 0.2)
            reasons.append("capacitance_available")
        if esr is not None and esr > 0:
            score += min(0.30, 0.08 / esr)
            if esr <= 0.12:
                reasons.append("low_esr")
        elif zohm is not None and zohm > 0:
            score += min(0.22, 0.12 / zohm)
            if zohm <= 0.2:
                reasons.append("low_impedance")
        if ripple_hf is not None and ripple_hf > 0:
            score += min(0.25, ripple_hf / 2.5 * 0.12)
            reasons.append("ripple_current_margin")
        if life_hours is not None and life_hours > 0:
            score += min(0.15, life_hours / 10000.0)
        if component_type == "output_cap" and ("polymer" in blob or "low esr" in blob):
            score += 0.18
            reasons.append("output_cap_fit")
        if component_type == "input_cap" and ("electrolytic" in blob or "long life" in blob):
            score += 0.12
            reasons.append("bulk_cap_fit")

    elif component_type == "clamp_snubber":
        v_clamp = _first_raw_numeric(raw, ["Voltage - Clamping (Max) @ Ipp", "Voltage - Clamping"], {"kv": 1e3, "mv": 1e-3, "v": 1.0})
        v_standoff = _first_raw_numeric(raw, ["Voltage - Reverse Standoff (Typ)", "Reverse Standoff Voltage"], {"kv": 1e3, "mv": 1e-3, "v": 1.0})
        i_pp = _first_raw_numeric(raw, ["Current - Peak Pulse (10/1000µs)", "Peak Pulse Current"], {"ma": 1e-3, "a": 1.0})
        p_pp = _first_raw_numeric(raw, ["Power - Peak Pulse", "Peak Pulse Power"], {"kw": 1e3, "w": 1.0})
        if v_clamp is not None and v_clamp > 0:
            score += min(0.25, v_clamp / 800.0)
            reasons.append("tvs_clamp_available")
        if v_standoff is not None and v_standoff > 0:
            score += min(0.22, v_standoff / 600.0)
        if i_pp is not None and i_pp > 0:
            score += min(0.18, i_pp / 20.0)
        if p_pp is not None and p_pp > 0:
            score += min(0.18, p_pp / 300.0)
        if "revisit_clamp_and_magnetics_loss_budget" in actions and ("tvs" in blob or "transorb" in blob):
            score += 0.18
            reasons.append("clamp_bias")

    return score, reasons


def _margin_bonus(actual: Optional[float], required: float, ideal_floor: float = 1.12, ideal_ceil: float = 1.8) -> float:
    if required <= 0 or actual is None or actual <= 0:
        return 0.12
    ratio = actual / required
    if ratio < 1.0:
        return -1.0
    if ideal_floor <= ratio <= ideal_ceil:
        return 0.55
    if ratio <= 2.6:
        return 0.30
    return 0.08


def _preference_score(
    row: Dict[str, Any],
    component_type: str,
    min_vds: float,
    min_id: float,
    preference_profile: Optional[Dict[str, Any]] = None,
    text_query: str = "",
) -> tuple[float, List[str]]:
    blob = _row_blob(row)
    reasons: List[str] = []
    score = 0.0

    vds = _parse_float(row.get("vds"))
    id_current = _parse_float(row.get("id_current"))
    score += _margin_bonus(vds, float(min_vds or 0.0))
    score += _margin_bonus(id_current, float(min_id or 0.0), ideal_floor=1.08, ideal_ceil=2.0)

    price = _parse_float(row.get("price"))
    if price is None or price <= 0:
        score += 0.05
    elif price <= 0.2:
        score += 0.30
        reasons.append("low_price")
    elif price <= 1.0:
        score += 0.22
        reasons.append("reasonable_price")
    elif price <= 3.0:
        score += 0.12
    else:
        score -= 0.05

    stock = _parse_float(row.get("stock"))
    if stock is None:
        score += 0.03
    elif stock >= 10000:
        score += 0.30
        reasons.append("high_stock")
    elif stock >= 1000:
        score += 0.18
        reasons.append("good_stock")
    elif stock >= 50:
        score += 0.10
    else:
        score -= 0.08

    query_match = _text_overlap_score(text_query, blob)
    if query_match > 0:
        score += 0.28 * query_match
        reasons.append("query_match")

    actions = _normalize_actions(preference_profile)
    kw_map = PREFERENCE_KEYWORDS.get(component_type, {})
    matched_actions: Dict[str, int] = defaultdict(int)
    for action in actions:
        for kw in kw_map.get(action, []):
            if kw in blob:
                matched_actions[action] += 1
    for action, hits in matched_actions.items():
        boost = min(0.7, 0.18 + 0.10 * hits)
        score += boost
        reasons.append(f"lesson_bias:{action}")

    engineering_score, engineering_reasons = _engineering_feature_score(row, component_type, preference_profile)
    if engineering_score:
        score += engineering_score
        reasons.extend(engineering_reasons)

    avoid_terms = []
    if isinstance(preference_profile, dict):
        avoid_terms = [
            str(x).strip().lower()
            for x in (preference_profile.get("avoid_terms") or [])
            if str(x).strip()
        ]
    for term in avoid_terms:
        if term in blob:
            score -= 0.35
            reasons.append(f"avoid:{term}")

    return score, reasons


def _fallback_rows() -> List[Dict[str, Any]]:
    # Keep this local and deterministic for offline selection.
    return [
        {
            "component_type": "mosfet",
            "part_number": "IPD60R380C6",
            "title": "Infineon 600V N-Channel MOSFET",
            "description": "Fallback local HV MOSFET candidate",
            "url": "https://www.digikey.com/en/products/filter/transistors/fets-mosfets/single-fets-mosfets/278",
            "supplier": "DigiKey",
            "stock": None,
            "price": None,
            "vds": 600.0,
            "id_current": 6.0,
            "source_mode": "fallback-local",
        },
        {
            "component_type": "diode",
            "part_number": "MUR460",
            "title": "Ultra-fast rectifier diode 600V",
            "description": "Fallback local diode candidate",
            "url": "https://www.digikey.com/en/products/filter/diodes/rectifiers/single-diodes/280",
            "supplier": "DigiKey",
            "stock": None,
            "price": None,
            "vds": 600.0,
            "id_current": 4.0,
            "source_mode": "fallback-local",
        },
        {
            "component_type": "controller",
            "part_number": "UC3845B",
            "title": "Current-mode PWM controller",
            "description": "Fallback local controller candidate",
            "url": "https://www.digikey.com/en/products/filter/pmic/power-supply-controllers-monitors/760",
            "supplier": "DigiKey",
            "stock": None,
            "price": None,
            "vds": None,
            "id_current": None,
            "source_mode": "fallback-local",
        },
        {
            "component_type": "input_protection",
            "part_number": "T1A250V",
            "title": "Slow-blow fuse 1A 250V",
            "description": "Fallback local fuse candidate",
            "url": "https://www.digikey.com/en/products/filter/fuses/139",
            "supplier": "DigiKey",
            "stock": None,
            "price": None,
            "vds": 250.0,
            "id_current": 1.0,
            "source_mode": "fallback-local",
        },
        {
            "component_type": "transformer",
            "part_number": "EFD20-FLYBACK-CORE",
            "title": "Ferrite core for flyback transformer",
            "description": "Fallback local transformer/core candidate",
            "url": "https://www.digikey.com/en/products/filter/ferrite-cores/936",
            "supplier": "DigiKey",
            "stock": None,
            "price": None,
            "vds": None,
            "id_current": None,
            "source_mode": "fallback-local",
        },
        {
            "component_type": "input_cap",
            "part_number": "400V-68UF-ELCAP",
            "title": "Electrolytic capacitor 400V 68uF",
            "description": "Fallback local input capacitor candidate",
            "url": "https://www.digikey.com/en/products/filter/aluminum-electrolytic-capacitors/58",
            "supplier": "DigiKey",
            "stock": None,
            "price": None,
            "vds": 400.0,
            "id_current": None,
            "source_mode": "fallback-local",
        },
        {
            "component_type": "output_cap",
            "part_number": "25V-470UF-LOWESR",
            "title": "Low ESR capacitor 25V 470uF",
            "description": "Fallback local output capacitor candidate",
            "url": "https://www.digikey.com/en/products/filter/aluminum-electrolytic-capacitors/58",
            "supplier": "DigiKey",
            "stock": None,
            "price": None,
            "vds": 25.0,
            "id_current": None,
            "source_mode": "fallback-local",
        },
        {
            "component_type": "emi_filter",
            "part_number": "CMC-2X10MH",
            "title": "Common-mode choke 2x10mH",
            "description": "Fallback local EMI candidate",
            "url": "https://www.digikey.com/en/products/filter/common-mode-chokes/839",
            "supplier": "DigiKey",
            "stock": None,
            "price": None,
            "vds": None,
            "id_current": None,
            "source_mode": "fallback-local",
        },
        {
            "component_type": "clamp_snubber",
            "part_number": "SMBJ440A",
            "title": "TVS diode for clamp/snubber",
            "description": "Fallback local clamp/snubber candidate",
            "url": "https://www.digikey.com/en/products/filter/tvs-diodes/144",
            "supplier": "DigiKey",
            "stock": None,
            "price": None,
            "vds": 440.0,
            "id_current": None,
            "source_mode": "fallback-local",
        },
    ]


def _connect() -> sqlite3.Connection:
    os.makedirs(BASE_DIR, exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def build_local_db(force_rebuild: bool = False) -> str:
    conn = _connect()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS components (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        component_type TEXT NOT NULL,
        part_number TEXT,
        title TEXT,
        description TEXT,
        url TEXT,
        supplier TEXT,
        stock REAL,
        price REAL,
        vds REAL,
        id_current REAL,
        source_mode TEXT,
        raw_json TEXT
    )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_components_type ON components(component_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_components_vds ON components(vds)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_components_id ON components(id_current)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_components_price ON components(price)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_components_part_number ON components(part_number)")

    if force_rebuild:
        cur.execute("DELETE FROM components")

    cur.execute("SELECT COUNT(*) AS c FROM components")
    existing = int(cur.fetchone()["c"])
    if existing == 0:
        if os.path.exists(MOSFET_CSV_PATH):
            with open(MOSFET_CSV_PATH, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = []
                for r in reader:
                    n = _normalize_row(r, "mosfet", "https://www.digikey.com")
                    if n.get("part_number"):
                        rows.append(n)
                _insert_rows(cur, rows)

        # Bulk import all user-provided DigiKey export CSVs.
        os.makedirs(RAW_EXPORT_DIR, exist_ok=True)
        for fn in os.listdir(RAW_EXPORT_DIR):
            if not fn.lower().endswith(".csv"):
                continue
            fp = os.path.join(RAW_EXPORT_DIR, fn)
            ctype = _guess_component_type_from_name(fn)
            with open(fp, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = []
                for r in reader:
                    n = _normalize_row(r, ctype, "https://www.digikey.com")
                    if n.get("part_number"):
                        rows.append(n)
                _insert_rows(cur, rows)

        for r in _fallback_rows():
            cur.execute(
                """
                INSERT INTO components(
                    component_type, part_number, title, description, url, supplier,
                    stock, price, vds, id_current, source_mode, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.get("component_type"),
                    r.get("part_number"),
                    r.get("title"),
                    r.get("description"),
                    r.get("url"),
                    r.get("supplier"),
                    r.get("stock"),
                    r.get("price"),
                    r.get("vds"),
                    r.get("id_current"),
                    r.get("source_mode"),
                    json.dumps(r, ensure_ascii=False),
                ),
            )

    conn.commit()
    conn.close()
    return SQLITE_DB_PATH


def ensure_local_db() -> str:
    if not os.path.exists(SQLITE_DB_PATH):
        return build_local_db(force_rebuild=False)
    return SQLITE_DB_PATH


def query_local_components(
    component_type: str,
    min_vds: float = 0.0,
    min_id: float = 0.0,
    limit: int = 5,
    preference_profile: Optional[Dict[str, Any]] = None,
    text_query: str = "",
) -> List[Dict[str, Any]]:
    ensure_local_db()
    conn = _connect()
    cur = conn.cursor()

    # When a caller asks for a minimum voltage/current, missing ratings are not
    # acceptable for automatic selection. Allowing NULL here made low-voltage or
    # unclassified rows crowd out real offline-power candidates in the top-N set.
    sql = """
    SELECT component_type, part_number, title, description, url, supplier,
           stock, price, vds, id_current, source_mode, raw_json
    FROM components
    WHERE component_type = ?
      AND (? <= 0 OR (vds IS NOT NULL AND vds >= ?))
      AND (? <= 0 OR (id_current IS NOT NULL AND id_current >= ?))
    ORDER BY
      CASE WHEN price IS NULL THEN 1 ELSE 0 END,
      price ASC,
      CASE WHEN stock IS NULL THEN 1 ELSE 0 END,
      stock DESC
    LIMIT ?
    """
    scan_limit = max(max(1, int(limit)) * 8, 40)
    cur.execute(sql, (component_type, min_vds, min_vds, min_id, min_id, scan_limit))
    rows = [dict(x) for x in cur.fetchall()]
    conn.close()

    deduped: List[Dict[str, Any]] = []
    seen_keys = set()
    for row in rows:
        key = (
            str(row.get("part_number") or "").strip().lower(),
            str(row.get("url") or "").strip().lower(),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(row)
    rows = deduped

    if not preference_profile and not str(text_query or "").strip():
        return rows[: max(1, int(limit))]

    ranked: List[Dict[str, Any]] = []
    for row in rows:
        score, reasons = _preference_score(
            row,
            component_type=component_type,
            min_vds=min_vds,
            min_id=min_id,
            preference_profile=preference_profile,
            text_query=text_query,
        )
        row_copy = dict(row)
        row_copy["_selector_score"] = round(float(score), 4)
        row_copy["_selector_reasons"] = reasons
        ranked.append(row_copy)

    ranked.sort(
        key=lambda r: (
            float(r.get("_selector_score") or 0.0),
            -float(_parse_float(r.get("price")) or 9999.0),
            float(_parse_float(r.get("stock")) or 0.0),
        ),
        reverse=True,
    )
    return ranked[: max(1, int(limit))]


def rag_lookup_components(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    ensure_local_db()
    q = str(query or "").strip().lower()
    if not q:
        return []

    conn = _connect()
    cur = conn.cursor()
    like = f"%{q}%"
    cur.execute(
        """
        SELECT component_type, part_number, title, description, url, supplier,
               stock, price, vds, id_current, source_mode, raw_json
        FROM components
        WHERE lower(part_number) LIKE ?
           OR lower(title) LIKE ?
           OR lower(description) LIKE ?
           OR lower(component_type) LIKE ?
        LIMIT ?
        """,
        (like, like, like, like, max(1, int(limit))),
    )
    rows = [dict(x) for x in cur.fetchall()]
    conn.close()
    return rows


def list_component_types(limit: int = 256) -> List[str]:
    ensure_local_db()
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT component_type, COUNT(*) AS n
        FROM components
        GROUP BY component_type
        ORDER BY n DESC, component_type ASC
        LIMIT ?
        """,
        (max(1, int(limit)),),
    )
    rows = [str(x["component_type"]) for x in cur.fetchall() if x["component_type"]]
    conn.close()
    return rows


def query_components_by_type_prefix(type_prefix: str, limit: int = 12) -> List[Dict[str, Any]]:
    ensure_local_db()
    q = str(type_prefix or "").strip().lower()
    if not q:
        return []

    conn = _connect()
    cur = conn.cursor()
    like = f"{q}%"
    cur.execute(
        """
        SELECT component_type, part_number, title, description, url, supplier,
               stock, price, vds, id_current, source_mode, raw_json
        FROM components
        WHERE lower(component_type) LIKE ?
        ORDER BY
            CASE WHEN price IS NULL THEN 1 ELSE 0 END,
            price ASC,
            CASE WHEN stock IS NULL THEN 1 ELSE 0 END,
            stock DESC
        LIMIT ?
        """,
        (like, max(1, int(limit))),
    )
    rows = [dict(x) for x in cur.fetchall()]
    conn.close()
    return rows


def catalog_overview(max_types: int = 40, per_type: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for ctype in list_component_types(limit=max_types):
        out[ctype] = query_local_components(ctype, limit=per_type)
    return out
