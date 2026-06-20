from __future__ import annotations

import glob
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback
from typing import Any, Dict, List


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "auto"}


def _probe() -> Dict[str, Any]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    payload: Dict[str, Any] = {
        "available": False,
        "package_detected": False,
        "package_version": "",
        "python_executable": Path(sys.executable).name,
        "python_version": sys.version.split()[0],
        "platform": {
            "system": system,
            "machine": machine,
        },
        "source_build_likely": not (system == "linux" and machine in {"x86_64", "amd64"}),
        "recommended_install_mode": "conda_py311" if system == "darwin" and machine in {"arm64", "aarch64"} else "venv",
        "import_strategy": "",
        "database_loaded": False,
        "functions_detected": [],
        "diagnostics": [],
        "reason": "",
    }
    payload["diagnostics"].append(
        "macOS and non-Linux-x86_64 platforms usually need a source build because PyPI wheels are Linux-only."
    )
    if shutil.which("conda"):
        payload["diagnostics"].append("Conda is available on this machine and is preferred for the helper env on macOS.")
    module, import_note, strategy = _import_pyopenmagnetics()
    payload["import_strategy"] = strategy
    if module is not None:
        payload["package_detected"] = True
        payload["package_version"] = getattr(module, "__version__", "") or _safe_dist_version()
        payload["functions_detected"] = _detect_functions(module)
        try:
            _initialize_module(module)
            payload["database_loaded"] = True
            payload["available"] = True
            payload["reason"] = import_note or "PyOpenMagnetics import succeeded."
        except Exception as exc:
            payload["reason"] = f"{import_note or 'PyOpenMagnetics import succeeded.'}; load_databases failed: {exc}"
        return payload
    payload["reason"] = import_note or "PyOpenMagnetics not importable in helper environment."
    return payload


def _safe_dist_version() -> str:
    try:
        return importlib.metadata.version("PyOpenMagnetics")
    except Exception:
        return ""


def _detect_functions(module: Any) -> List[str]:
    wanted = [
        "load_databases",
        "design_magnetics_from_converter",
        "process_converter",
        "calculate_advised_magnetics",
        "process_inputs",
        "calculate_core_losses",
        "calculate_winding_losses",
        "get_core_material_names",
        "get_core_shape_names",
    ]
    return [name for name in wanted if callable(getattr(module, name, None))]


def _looks_usable(module: Any) -> bool:
    return any(
        callable(getattr(module, name, None))
        for name in (
            "design_magnetics_from_converter",
            "process_converter",
            "calculate_advised_magnetics",
            "process_inputs",
        )
    )


def _load_native_extension(search_roots: List[Path]) -> tuple[Any, str, str]:
    errors: List[str] = []
    patterns = [
        "PyOpenMagnetics*.so",
        "PyOpenMagnetics*.pyd",
        "pyopenmagnetics*.so",
        "pyopenmagnetics*.pyd",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches = glob.glob(str(root / pattern))
            for match in matches[:8]:
                try:
                    spec = importlib.util.spec_from_file_location("PyOpenMagnetics_runtime", match)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        if _looks_usable(module):
                            return module, f"Loaded native extension from {match}.", "native_extension"
                        errors.append(f"{match}: loaded but did not expose usable API")
                except Exception as exc:
                    errors.append(f"{match}: {exc}")
    return None, "; ".join(errors[-4:]) if errors else "No native extension candidate succeeded.", "native_extension"


def _import_pyopenmagnetics() -> tuple[Any, str, str]:
    errors: List[str] = []
    namespace_paths: List[Path] = []
    for name in ("PyOpenMagnetics", "pyopenmagnetics"):
        try:
            module = importlib.import_module(name)
            if _looks_usable(module):
                return module, f"Imported {name} via normal import.", "direct_import"
            pkg_paths = getattr(module, "__path__", None) or []
            for pkg_path in pkg_paths:
                namespace_paths.append(Path(pkg_path))
                namespace_paths.append(Path(pkg_path).parent)
            errors.append(f"{name}: imported as namespace but did not expose the expected API")
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    roots = namespace_paths + [Path(p) for p in sys.path if p and os.path.isdir(p)]
    module, note, strategy = _load_native_extension(roots)
    if module is not None:
        return module, note, strategy
    if note:
        errors.append(note)
    return None, "; ".join(errors[-6:]) if errors else "No import path succeeded.", "unavailable"


def _initialize_module(module: Any) -> None:
    if getattr(module, "_pe_mas_databases_loaded", False):
        return
    load_databases = getattr(module, "load_databases", None)
    if callable(load_databases):
        load_databases({})
    setattr(module, "_pe_mas_databases_loaded", True)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _core_family(power_w: float, energy_mj: float, fsw_hz: float) -> str:
    fsw_khz = fsw_hz / 1000.0
    if power_w >= 36 or energy_mj >= 1.8:
        return "PQ3220"
    if power_w <= 12 and energy_mj < 0.45 and fsw_khz >= 70:
        return "EE25"
    if power_w <= 20 and energy_mj < 0.8:
        return "RM10"
    return "PQ2620"


def _heuristic_advice(specs: Dict[str, Any], design: Dict[str, Any], probe: Dict[str, Any]) -> Dict[str, Any]:
    vout = _safe_float(specs.get("output_voltage"), 12.0)
    iout = _safe_float(specs.get("output_current"), 2.0)
    power_w = max(0.1, vout * iout)
    lp_h = _safe_float(design.get("primary_inductance"), 2.8e-3)
    ipk_a = _safe_float(design.get("primary_peak_current"), 0.73)
    fsw_hz = _safe_float(design.get("switching_frequency"), 65000.0)
    turns_ratio = max(1.0, _safe_float(design.get("turns_ratio"), 6.0))
    vor = _safe_float(design.get("reflected_output_voltage"), 80.0)
    energy_mj = max(0.02, 0.5 * lp_h * (ipk_a**2) * 1000.0)
    fsw_khz = fsw_hz / 1000.0

    family = _core_family(power_w, energy_mj, fsw_hz)
    ns = max(3, int(round(4.0 + vout / 6.0)))
    np = max(int(round(ns * turns_ratio)), ns + 6)
    naux = max(3, int(round(ns)))
    gap_mm = round(max(0.12, min(0.85, 0.16 + energy_mj * 0.22)), 3)
    copper_loss_w = round(max(0.08, 0.11 + 0.42 * (ipk_a**2) + 0.015 * power_w), 3)
    core_loss_w = round(max(0.1, 0.08 + 0.24 * energy_mj + 0.004 * max(0.0, fsw_khz - 50.0)), 3)
    total_loss_w = round(copper_loss_w + core_loss_w, 3)

    if family == "PQ3220":
        winding = [
            "Split primary into two halves to reduce leakage and improve coupling.",
            "Place the secondary between primary halves with reinforced insulation.",
            "Keep the auxiliary winding close-coupled to the secondary side for regulation fidelity.",
        ]
    elif family == "EE25":
        winding = [
            "Use compact layered winding with tight interleaving to control leakage.",
            "Favor triple-insulated secondary wire to simplify reinforced insulation handling.",
            "Reserve margin tape window early because bobbin area will be tight.",
        ]
    else:
        winding = [
            "Use a primary-secondary-primary sandwich winding to trade leakage against manufacturability.",
            "Keep auxiliary winding adjacent to the secondary for better reflected-voltage tracking.",
            "Control layer buildup to maintain creepage and predictable gap fringing loss.",
        ]

    manufacturability = [
        f"{family} is a practical first-pass core family for approximately {power_w:.1f} W flyback output power.",
        f"Target an effective center gap near {gap_mm:.3f} mm and confirm fringing loss after winding definition.",
        "Capture insulation system, tape stack, and bobbin creepage constraints before freezing turns.",
    ]
    if np >= 40:
        manufacturability.append("Primary turn count is relatively high; review fill factor and winding time before release.")
    if probe.get("available"):
        manufacturability.append("PyOpenMagnetics package is visible in the helper environment, but this first-pass adapter is still using the heuristic fallback until a deeper API adapter is completed.")
    else:
        manufacturability.append("PyOpenMagnetics is not currently available in the helper environment, so this advisory is heuristic-only.")

    return {
        "engine": "PyOpenMagnetics" if probe.get("available") else "heuristic_fallback",
        "integration_mode": "sidecar_json",
        "status": "available" if probe.get("available") else "heuristic",
        "core_family": family,
        "core_shape_name": family,
        "core_material": "N87/3C90-equivalent",
        "core_material_manufacturer": "heuristic_default",
        "gap_mm": gap_mm,
        "window_utilization_pct": round(min(82.0, 38.0 + power_w * 1.2), 1),
        "turns": {
            "primary": np,
            "secondary": ns,
            "auxiliary": naux,
        },
        "winding_arrangement": winding,
        "loss_estimate_w": {
            "copper": copper_loss_w,
            "core": core_loss_w,
            "total": total_loss_w,
            "winding_dc": round(copper_loss_w * 0.55, 3),
            "winding_ac": round(copper_loss_w * 0.45, 3),
        },
        "manufacturability": manufacturability,
        "design_inputs": {
            "lp_h": lp_h,
            "ipk_a": ipk_a,
            "fsw_hz": fsw_hz,
            "turns_ratio": turns_ratio,
            "reflected_output_voltage_v": vor,
            "stored_energy_mj": round(energy_mj, 3),
            "input_bus_min_v": round(_safe_float(specs.get("input_voltage_min"), 85.0) * math.sqrt(2.0), 3),
            "input_bus_max_v": round(_safe_float(specs.get("input_voltage_max"), 265.0) * math.sqrt(2.0), 3),
        },
        "notes": [
            f"Suggested first-pass family: {family}",
            f"Estimated magnetizing energy: {energy_mj:.3f} mJ",
            f"Reflected output voltage basis: {vor:.1f} V",
        ],
        "api_strategy": "heuristic_fallback",
        "next_actions": [
            "Validate gap and turns against an actual selected core geometry.",
            "Cross-check copper fill factor and creepage before prototype release.",
            "Compare magnetic loss estimate with PLECS and thermal measurements after prototype build.",
        ],
    }


def _find_callable(module: Any, candidates: List[str]) -> Any:
    for name in candidates:
        func = getattr(module, name, None)
        if callable(func):
            return func
    for attr_name in dir(module):
        attr = getattr(module, attr_name, None)
        if not attr:
            continue
        for name in candidates:
            func = getattr(attr, name, None)
            if callable(func):
                return func
    return None


def _flyback_payload(specs: Dict[str, Any], design: Dict[str, Any]) -> Dict[str, Any]:
    vac_min = _safe_float(specs.get("input_voltage_min"), 85.0)
    vac_max = _safe_float(specs.get("input_voltage_max"), 265.0)
    vbus_min = vac_min * math.sqrt(2.0)
    vbus_max = vac_max * math.sqrt(2.0)
    fsw = _safe_float(design.get("switching_frequency"), 65000.0)
    duty = _safe_float(design.get("max_duty_cycle"), 0.44)
    ts = 1.0 / max(1.0, fsw)
    ton = ts * max(0.05, min(0.9, duty))
    lp = _safe_float(design.get("primary_inductance"), 2.8e-3)
    ipk = _safe_float(design.get("primary_peak_current"), 0.73)
    turns_ratio = _safe_float(design.get("turns_ratio"), 6.0)
    vor = _safe_float(design.get("reflected_output_voltage"), 80.0)
    vout = _safe_float(specs.get("output_voltage"), 12.0)
    iout = _safe_float(specs.get("output_current"), 2.0)
    return {
        "topology": "flyback",
        "designRequirements": {
            "inputVoltageMin": vac_min,
            "inputVoltageMax": vac_max,
            "outputVoltage": vout,
            "outputCurrent": iout,
            "magnetizingInductance": lp,
            "primaryPeakCurrent": ipk,
            "switchingFrequency": fsw,
            "turnsRatio": turns_ratio,
            "reflectedOutputVoltage": vor,
            "isolation": bool(specs.get("isolation")),
        },
        "operatingPoints": [
            {
                "name": "nominal_high_line",
                "inputVoltage": vbus_max,
                "frequency": fsw,
                "dutyCycle": duty,
                "magnetizingCurrentWaveform": {
                    "time": [0.0, ton, ts],
                    "current": [0.0, ipk, 0.0],
                },
                "primaryVoltageWaveform": {
                    "time": [0.0, ton, ts],
                    "voltage": [vbus_max, -vor, -vor],
                },
            },
            {
                "name": "nominal_low_line",
                "inputVoltage": vbus_min,
                "frequency": fsw,
                "dutyCycle": min(0.92, duty * 1.12),
                "magnetizingCurrentWaveform": {
                    "time": [0.0, ton, ts],
                    "current": [0.0, ipk * 1.08, 0.0],
                },
                "primaryVoltageWaveform": {
                    "time": [0.0, ton, ts],
                    "voltage": [vbus_min, -vor, -vor],
                },
            },
        ],
    }


def _converter_payload(specs: Dict[str, Any], design: Dict[str, Any]) -> Dict[str, Any]:
    vac_min = _safe_float(specs.get("input_voltage_min"), 85.0)
    vac_max = _safe_float(specs.get("input_voltage_max"), 265.0)
    vbus_min = round(vac_min * math.sqrt(2.0) * 0.9, 3)
    vbus_max = round(vac_max * math.sqrt(2.0), 3)
    fsw = _safe_float(design.get("switching_frequency"), 65000.0)
    duty = max(0.12, min(0.85, _safe_float(design.get("max_duty_cycle"), 0.44)))
    lp = _safe_float(design.get("primary_inductance"), 2.8e-3)
    turns_ratio = max(1.0, _safe_float(design.get("turns_ratio"), 6.0))
    eff = max(0.5, min(0.98, _safe_float(specs.get("efficiency_target"), 0.88)))
    return {
        "inputVoltage": {
            "minimum": vbus_min,
            "maximum": vbus_max,
        },
        "outputVoltage": _safe_float(specs.get("output_voltage"), 12.0),
        "outputCurrent": _safe_float(specs.get("output_current"), 2.0),
        "switchingFrequency": fsw,
        "maximumDutyCycle": duty,
        "efficiency": eff,
        "desiredInductance": lp,
        "desiredTurnsRatios": [turns_ratio],
        "desiredDutyCycle": [[duty, duty]],
        "ripple": _safe_float(specs.get("max_ripple_voltage"), 0.1),
    }


def _dig(obj: Any, keys: List[str]) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if any(token in lk for token in keys):
                return v
            found = _dig(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _dig(item, keys)
            if found is not None:
                return found
    return None


def _path_get(obj: Any, path: List[Any]) -> Any:
    cur = obj
    for part in path:
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur.get(part)
        elif isinstance(cur, list) and isinstance(part, int):
            if part < 0 or part >= len(cur):
                return None
            cur = cur[part]
        else:
            return None
    return cur


def _first_of_paths(obj: Any, paths: List[List[Any]], default: Any = None) -> Any:
    for path in paths:
        found = _path_get(obj, path)
        if found not in (None, "", [], {}):
            return found
    return default


def _normalize_result(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict) and isinstance(raw.get("data"), list) and raw.get("data"):
        entry = raw["data"][0]
    else:
        entry = raw
    mas = entry.get("mas") if isinstance(entry, dict) and isinstance(entry.get("mas"), dict) else {}
    magnetic = (
        mas.get("magnetic")
        if isinstance(mas.get("magnetic"), dict)
        else entry.get("magnetic")
        if isinstance(entry, dict) and isinstance(entry.get("magnetic"), dict)
        else entry
    )
    inputs = mas.get("inputs") if isinstance(mas.get("inputs"), dict) else {}
    scoring = {}
    if isinstance(entry, dict):
        scoring = entry.get("scoringPerFilter") if isinstance(entry.get("scoringPerFilter"), dict) else entry.get("scoring", {})
    return {
        "entry": entry if isinstance(entry, dict) else {},
        "mas": mas,
        "magnetic": magnetic if isinstance(magnetic, dict) else {},
        "inputs": inputs,
        "scoring": scoring if isinstance(scoring, dict) else {},
    }


def _extract_turns(result: Any, fallback: Dict[str, Any]) -> Dict[str, int]:
    primary = _dig(result, ["primaryturn", "turnprimary", "np"])
    secondary = _dig(result, ["secondaryturn", "turnsecondary", "ns"])
    auxiliary = _dig(result, ["auxiliaryturn", "naux", "biasturn"])
    out = {
        "primary": int(round(_safe_float(primary, fallback.get("primary", 36)))),
        "secondary": int(round(_safe_float(secondary, fallback.get("secondary", 6)))),
        "auxiliary": int(round(_safe_float(auxiliary, fallback.get("auxiliary", fallback.get("secondary", 6))))),
    }
    return out


def _extract_losses(result: Any, fallback_total: float) -> Dict[str, float]:
    core = _safe_float(_dig(result, ["coreloss", "core_loss"]), max(0.1, fallback_total * 0.35))
    copper = _safe_float(_dig(result, ["copperloss", "windingloss", "dcresistanceloss"]), max(0.1, fallback_total * 0.65))
    total = _safe_float(_dig(result, ["totalloss", "loss_total"]), core + copper)
    winding_dc = _safe_float(_dig(result, ["dcloss", "dc_loss", "windingdcloss"]), copper * 0.55)
    winding_ac = _safe_float(_dig(result, ["acloss", "ac_loss", "windingacloss", "proximityloss"]), max(0.0, copper - winding_dc))
    fringing = _safe_float(_dig(result, ["fringingloss", "fringing_loss"]), 0.0)
    return {
        "copper": round(copper, 3),
        "core": round(core, 3),
        "total": round(total, 3),
        "winding_dc": round(winding_dc, 3),
        "winding_ac": round(winding_ac, 3),
        "fringing": round(fringing, 3),
    }


def _extract_notes(result: Any, keys: List[str], limit: int = 4) -> List[str]:
    found = _dig(result, keys)
    out: List[str] = []
    if isinstance(found, list):
        out = [str(x).strip() for x in found if str(x).strip()]
    elif isinstance(found, str) and found.strip():
        out = [found.strip()]
    elif isinstance(found, dict):
        out = [f"{k}: {v}" for k, v in list(found.items())[:limit] if str(v).strip()]
    return out[:limit]


def _result_paths_summary(result: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    magnetic = result["magnetic"]
    entry = result["entry"]
    family = _first_of_paths(
        magnetic,
        [
            ["core", "functionalDescription", "shape", "name"],
            ["core", "functionalDescription", "shape", "family"],
            ["functionalDescription", "shape", "name"],
            ["shape", "name"],
        ],
        _dig(magnetic, ["coreshape", "corefamily", "family", "shape"]) or fallback["core_family"],
    )
    material = _first_of_paths(
        magnetic,
        [
            ["core", "functionalDescription", "material", "name"],
            ["core", "material", "name"],
            ["material", "name"],
        ],
        _dig(magnetic, ["materialname", "material"]) or fallback.get("core_material"),
    )
    material_mfr = _first_of_paths(
        magnetic,
        [
            ["core", "functionalDescription", "material", "manufacturerInfo", "name"],
            ["core", "material", "manufacturerInfo", "name"],
            ["material", "manufacturerInfo", "name"],
        ],
        _dig(magnetic, ["manufacturermaterial", "materialmanufacturer"]),
    )
    gap = _safe_float(
        _first_of_paths(
            magnetic,
            [["gapping", "length"], ["core", "gapping", "length"], ["core", "functionalDescription", "gapping", "length"]],
            _dig(magnetic, ["gap", "gapping", "airgap"]),
        ),
        fallback["gap_mm"],
    )
    fill = _safe_float(
        _first_of_paths(
            magnetic,
            [
                ["coil", "functionalDescription", "fillingFactor"],
                ["coil", "fillingFactor"],
                ["coil", "windowUtilization"],
                ["windowUtilization"],
            ],
            _dig(magnetic, ["fillfactor", "fillingfactor", "windowutilization", "windowusage"]),
        ),
        -1.0,
    )
    if 0.0 <= fill <= 1.5:
        fill *= 100.0
    turns = _extract_turns(magnetic, fallback["turns"])
    losses = _extract_losses(magnetic, fallback["loss_estimate_w"]["total"])
    winding = _extract_notes(magnetic, ["winding", "interleav", "layer"])
    manufacturability = _extract_notes(magnetic, ["manufactur", "assembly", "recommend", "warning"])
    core_ref = _first_of_paths(
        magnetic,
        [
            ["core", "name"],
            ["core", "functionalDescription", "name"],
            ["name"],
        ],
        _dig(entry, ["reference", "partnumber", "part_number"]),
    )
    reluctance_core = _safe_float(_dig(magnetic, ["corereluctance"]), 0.0)
    reluctance_gap = _safe_float(_dig(magnetic, ["gappingreluctance"]), 0.0)
    return {
        "core_family": str(family),
        "core_shape_name": str(family),
        "core_material": str(material) if material not in (None, "") else fallback.get("core_material"),
        "core_material_manufacturer": str(material_mfr) if material_mfr not in (None, "") else fallback.get("core_material_manufacturer"),
        "core_reference": str(core_ref) if core_ref not in (None, "") else "",
        "gap_mm": round(gap, 3),
        "window_utilization_pct": round(fill, 1) if fill >= 0.0 else None,
        "turns": turns,
        "loss_estimate_w": losses,
        "winding_arrangement": winding or fallback["winding_arrangement"],
        "manufacturability": manufacturability or fallback["manufacturability"],
        "magnetic_reluctance": {
            "core": round(reluctance_core, 6) if reluctance_core else 0.0,
            "gap": round(reluctance_gap, 6) if reluctance_gap else 0.0,
        },
        "ranking_scores": result["scoring"] or {},
    }


def _real_openmagnetics_advice(specs: Dict[str, Any], design: Dict[str, Any], probe: Dict[str, Any], module: Any) -> Dict[str, Any]:
    _initialize_module(module)
    design_from_converter = _find_callable(module, ["design_magnetics_from_converter", "designMagneticsFromConverter"])
    process_converter = _find_callable(module, ["process_converter", "processConverter"])
    calculate_advised = _find_callable(module, ["calculate_advised_magnetics", "calculateAdvisedMagnetics"])
    process_inputs = _find_callable(module, ["process_inputs", "processInputs"])
    converter_payload = _converter_payload(specs, design)
    raw = None
    api_strategy = ""

    if design_from_converter:
        raw = design_from_converter("flyback", converter_payload, 3, "standard cores", True, None)
        api_strategy = "design_magnetics_from_converter"
    elif process_converter and calculate_advised:
        processed = process_converter("flyback", converter_payload, True)
        raw = calculate_advised(processed, 3, "standard cores")
        api_strategy = "process_converter+calculate_advised_magnetics"
    elif process_inputs and calculate_advised:
        payload = _flyback_payload(specs, design)
        processed = process_inputs(payload)
        raw = calculate_advised(processed, 3, "standard cores")
        api_strategy = "process_inputs+calculate_advised_magnetics"
    else:
        raise RuntimeError(
            "Expected PyOpenMagnetics design_magnetics_from_converter or process_converter/calculate_advised_magnetics API was not found."
        )

    fallback = _heuristic_advice(specs, design, probe)
    normalized = _normalize_result(raw)
    extracted = _result_paths_summary(normalized, fallback)
    manufacturability = list(extracted["manufacturability"])
    if extracted.get("window_utilization_pct") is not None:
        manufacturability = [
            f"Estimated winding window utilization is {extracted['window_utilization_pct']:.1f}%."
        ] + manufacturability

    return {
        "engine": "PyOpenMagnetics",
        "integration_mode": "sidecar_json",
        "status": "available",
        "core_family": extracted["core_family"],
        "core_shape_name": extracted.get("core_shape_name"),
        "core_material": extracted.get("core_material"),
        "core_material_manufacturer": extracted.get("core_material_manufacturer"),
        "core_reference": extracted.get("core_reference"),
        "gap_mm": extracted["gap_mm"],
        "window_utilization_pct": extracted.get("window_utilization_pct"),
        "turns": extracted["turns"],
        "winding_arrangement": extracted["winding_arrangement"] or fallback["winding_arrangement"],
        "loss_estimate_w": extracted["loss_estimate_w"],
        "manufacturability": manufacturability or fallback["manufacturability"],
        "design_inputs": fallback["design_inputs"],
        "magnetic_reluctance": extracted.get("magnetic_reluctance") or {},
        "ranking_scores": extracted.get("ranking_scores") or {},
        "notes": [
            "Real PyOpenMagnetics API invocation succeeded.",
            f"Result source module: {getattr(module, '__name__', 'PyOpenMagnetics')}",
            f"API strategy: {api_strategy}",
        ] + fallback["notes"][:2],
        "api_strategy": api_strategy,
        "next_actions": fallback["next_actions"],
        "raw_result_summary": str(type(raw).__name__),
    }


def main() -> int:
    try:
        request = json.loads(sys.stdin.read() or "{}")
    except Exception as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": f"invalid_json: {exc}"}))
        return 1

    action = str(request.get("action") or "probe").strip().lower()
    probe = _probe()

    if action == "probe":
        sys.stdout.write(json.dumps({"ok": True, "probe": probe}))
        return 0

    if action == "advise":
        specs = request.get("specifications") or {}
        design = request.get("theoretical_design") or {}
        module = None
        if probe.get("available"):
            module, _ = _import_pyopenmagnetics()
        if module is not None:
            try:
                response = _real_openmagnetics_advice(specs, design, probe, module)
            except Exception as exc:
                response = _heuristic_advice(specs, design, probe)
                response["status"] = "heuristic"
                response["notes"] = list(response.get("notes") or []) + [
                    f"PyOpenMagnetics API adapter failed and heuristic fallback was used: {exc}"
                ]
                response["adapter_error"] = traceback.format_exc(limit=4)
        else:
            response = _heuristic_advice(specs, design, probe)
        sys.stdout.write(json.dumps({"ok": True, "probe": probe, "advisor": response}))
        return 0

    sys.stdout.write(json.dumps({"ok": False, "error": f"unknown_action: {action}", "probe": probe}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
