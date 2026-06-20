import math
from typing import Dict, Tuple, Optional


def detect_mode(vin: float, vout: float, iout: float, fsw: float, lpri: float, d: float) -> Tuple[str, float]:
    pout = vout * iout
    l_crit = (vin ** 2 * d ** 2) / (2.0 * fsw * pout)
    return ("CCM" if lpri > l_crit else "DCM"), l_crit


def _ccm_currents(vin: float, vout: float, iout: float, lpri: float, fsw: float, n: float, d: float) -> Dict[str, float]:
    pout = vout * iout
    i_avg = pout / (vin * d)
    delta_i = (vin * d) / (lpri * fsw)
    i_pk = i_avg + delta_i / 2.0
    i_rms_pri = math.sqrt(d * (i_avg ** 2 + delta_i ** 2 / 12.0))
    delta_i_sec = delta_i / n
    i_sec_avg = iout / (1.0 - d)
    i_rms_sec = math.sqrt((1.0 - d) * (i_sec_avg ** 2 + delta_i_sec ** 2 / 12.0))
    return {"i_pk": i_pk, "i_rms_pri": i_rms_pri, "i_rms_sec": i_rms_sec}


def _dcm_currents(vin: float, vout: float, iout: float, lpri: float, fsw: float, n: float, d: float) -> Dict[str, float]:
    pout = vout * iout
    i_pk = math.sqrt((2.0 * pout) / (lpri * fsw * d ** 2))
    i_rms_pri = i_pk * math.sqrt(d / 3.0)
    d2 = min((i_pk * lpri * fsw) / (vout * n), 1.0 - d)
    i_pk_sec = i_pk * n
    i_rms_sec = i_pk_sec * math.sqrt(d2 / 3.0)
    return {"i_pk": i_pk, "i_rms_pri": i_rms_pri, "i_rms_sec": i_rms_sec}


def calculate_flyback_efficiency(
    vin: float,
    vout: float,
    iout: float,
    fsw: float,
    lpri: float,
    n: float,
    d: float,
    r_ds_on: float,
    t_r: float,
    t_f: float,
    v_f: float,
    q_rr: float,
    v_r: float,
    np: int,
    r_winding_pri: float,
    r_winding_sec: float,
    k_fe: float,
    alpha: float,
    beta: float,
    ve: float,
    ae: float,
    core_loss_scale: float = 1.0,
    max_core_loss_ratio_of_pout: float = 0.35,
    max_core_loss_w: Optional[float] = None,
) -> Dict:
    pout = vout * iout
    mode, l_crit = detect_mode(vin, vout, iout, fsw, lpri, d)

    cur = _ccm_currents(vin, vout, iout, lpri, fsw, n, d) if mode == "CCM" else _dcm_currents(vin, vout, iout, lpri, fsw, n, d)
    i_pk = cur["i_pk"]
    i_rms_pri = cur["i_rms_pri"]
    i_rms_sec = cur["i_rms_sec"]

    p_cond = i_rms_pri ** 2 * r_ds_on
    p_sw_on = 0.5 * vin * i_pk * t_r * fsw
    p_sw_off = 0.5 * vin * i_pk * t_f * fsw
    p_fet = p_cond + p_sw_on + p_sw_off

    p_diode_cond = v_f * iout
    p_rr = (0.5 * q_rr * v_r * fsw) if mode == "CCM" else 0.0
    p_diode = p_diode_cond + p_rr

    delta_b_pkpk = (vin * d) / (np * ae * fsw)
    b_hat = delta_b_pkpk / 2.0
    p_core_raw = k_fe * (fsw ** alpha) * (b_hat ** beta) * ve
    p_core_scaled = p_core_raw * max(0.0, float(core_loss_scale or 0.0))
    core_cap = float(max_core_loss_w) if max_core_loss_w is not None else (max(0.0, float(max_core_loss_ratio_of_pout or 0.0)) * max(pout, 1e-12))
    if core_cap > 0.0:
        p_core = min(p_core_scaled, core_cap)
    else:
        p_core = p_core_scaled
    core_loss_clamped = p_core + 1e-12 < p_core_scaled

    p_cu_pri = i_rms_pri ** 2 * r_winding_pri
    p_cu_sec = i_rms_sec ** 2 * r_winding_sec
    p_transformer = p_core + p_cu_pri + p_cu_sec

    p_total = p_fet + p_diode + p_transformer
    pin = pout + p_total
    eta = (pout / pin) * 100.0 if pin > 0 else 0.0

    p_total_raw = p_fet + p_diode + p_core_scaled + p_cu_pri + p_cu_sec
    pin_raw = pout + p_total_raw
    eta_raw = (pout / pin_raw) * 100.0 if pin_raw > 0 else 0.0

    confidence = "high"
    confidence_reasons = []
    if core_loss_clamped:
        confidence = "medium"
        confidence_reasons.append("Core-loss guardrail applied (raw core loss exceeded configured cap).")
    if abs(eta - eta_raw) > 10.0:
        confidence = "low"
        confidence_reasons.append("Guarded efficiency deviates significantly from raw Steinmetz estimate.")

    return {
        "mode": mode,
        "l_crit": l_crit,
        "efficiency": eta,
        "efficiency_raw": eta_raw,
        "pin": pin,
        "pin_raw": pin_raw,
        "pout": pout,
        "p_total": p_total,
        "p_total_raw": p_total_raw,
        "confidence": confidence,
        "confidence_reasons": confidence_reasons,
        "guardrails": {
            "core_loss_clamped": core_loss_clamped,
            "core_loss_raw_w": p_core_scaled,
            "core_loss_used_w": p_core,
            "core_loss_cap_w": core_cap,
            "core_loss_scale": float(core_loss_scale or 0.0),
        },
        "currents": {
            "i_pk": i_pk,
            "i_rms_pri": i_rms_pri,
            "i_rms_sec": i_rms_sec,
            "delta_b_pkpk_mT": delta_b_pkpk * 1e3,
            "b_hat_mT": b_hat * 1e3,
        },
        "losses": {
            "mosfet_conduction": p_cond,
            "mosfet_turn_on_switching": p_sw_on,
            "mosfet_turn_off_switching": p_sw_off,
            "diode_conduction": p_diode_cond,
            "diode_reverse_recovery": p_rr,
            "transformer_core": p_core,
            "transformer_core_raw": p_core_scaled,
            "transformer_copper_primary": p_cu_pri,
            "transformer_copper_secondary": p_cu_sec,
        },
    }
