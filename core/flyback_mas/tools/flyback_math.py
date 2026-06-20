import math
from typing import Dict, Any, Tuple
from .input_semantics import resolve_power_stage_input

def calculate_flyback_params(
    spec: Dict[str, Any], 
    current_design: Dict[str, Any] = None, 
    feedback: str = None,
    overrides: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Calculates theoretical Flyback design parameters based on specs and feedback.
    
    Args:
        spec: Design specifications (Vin range, Vout, Iout, etc.)
        current_design: Previous design parameters (if any)
        feedback: Feedback string from validation (e.g. "DECREASE_FSW", "INCREASE_INDUCTANCE")
        overrides: Dictionary of specific parameter overrides (e.g., from literature analysis)
        
    Returns:
        Dict of theoretical design parameters.
    """
    
    # 1. Initialize variables from Spec or Previous Design
    vin_min_ac = spec.get('input_voltage_min')
    vin_max_ac = spec.get('input_voltage_max')

    # Handle missing/None values (e.g., fixed input like "12V")
    if vin_min_ac is None and vin_max_ac is None:
        vin_min_ac = 85.0 # Universal default
        vin_max_ac = 265.0
    elif vin_min_ac is None:
        vin_min_ac = float(vin_max_ac)
    elif vin_max_ac is None:
        vin_max_ac = float(vin_min_ac)
    
    vout = spec['output_voltage']
    iout = spec['output_current']
    eff_est = spec.get('efficiency_target', 0.85)
    
    input_domain = resolve_power_stage_input(spec)
    vin_dc_min = float(input_domain["dc_bus_min"])
    vin_dc_max = float(input_domain["dc_bus_max"])
    
    p_out = vout * iout
    p_in = p_out / eff_est
    
    # Default Design Parameters
    f_sw = 65000.0 # 65 kHz default
    
    # Smart Default for Reflected Voltage (Vor)
    # Ideally Vor ~ Vin_min for D ~ 0.5 to minimize stress
    # If High Voltage (Mains), Vor ~ 80-120V is typical.
    # If Low Voltage (e.g. 12V), Vor ~ 12-24V is typical.
    if vin_dc_min < 60:
        # [SOTA FIX]: For DC-DC Flyback, Vor determines Duty Cycle.
        # D = Vor / (Vor + Vin).
        # We want D ~ 0.3 - 0.5.
        # If D=0.5 -> Vor = Vin.
        # So Vor should be roughly equal to Vin_min.
        vor = max(vin_dc_min, 12.0) 
        # Vin=12, Vor=12 -> D=0.5. Perfect.
    else:
        vor = 80.0       # Reflected voltage default for Mains
        
    k_rf = 0.4     # Ripple factor default (0.4 = 40% ripple)
    
    # Apply previous design first so that learned/nudged overrides can supersede it.
    # This avoids the bug where current_design silently wipes out retry adjustments.
    if current_design:
        f_sw = current_design.get('switching_frequency', f_sw)
        vor = current_design.get('reflected_output_voltage', vor)
        k_rf = current_design.get('ripple_factor', k_rf)
        
        # [SOTA FIX]: Sanity Check for Bad Initial Design (e.g. Iter 0 Hallucination)
        if vor < 2.0:
            # If Vor got stuck at < 2V (like 0.7V), reset it to Vin or at least 15V
            vor = max(vin_dc_min, 15.0)

    # Learned/literature overrides must take precedence over stale previous-design values.
    if overrides:
        if 'switching_frequency' in overrides:
            f_sw = float(overrides['switching_frequency'])
        if 'reflected_output_voltage' in overrides:
            vor = float(overrides['reflected_output_voltage'])
        if 'ripple_factor' in overrides:
            k_rf = float(overrides['ripple_factor'])

    if feedback:
        if "DECREASE_FSW" in feedback:
            f_sw *= 0.8 # Decrease by 20%
        elif "INCREASE_FSW" in feedback:
            f_sw *= 1.2
        
        if "INCREASE_INDUCTANCE" in feedback:
            k_rf *= 0.8 # Lower ripple factor -> Higher Inductance
            
        if "REDUCE_TURNS_RATIO" in feedback:
             # Reduce reflected voltage -> Lower Np/Ns
            # BUT clamp it to prevent infinite descent into absurdity
            vor_new = vor * 0.9
            if vor_new < (vout + 1.0): # Vor must be at least Vout + Margin
                 # Cannot reduce turns ratio further without breaking physics
                 # Maybe try increasing Fsw or Inductance instead?
                 # For now, just clamp.
                 vor = max(vor_new, vout + 2.0)
            else:
                 vor = vor_new
            
        if "ADD_REALISTIC_PARASITICS" in feedback:
             eff_est = 0.75 # Lower estimated efficiency to account for real losses
    
    # 2. Main Calculations
    
    # Transfer function in CCM: Vout/Vin = D / (1-D) * (Ns/Np)
    # But usually we define D_max at Vin_min
    # D_max = Vor / (Vor + Vin_dc_min)
    d_max = vor / (vor + vin_dc_min)
    
    # Primary Inductance
    # Energy balance? Or Ripple constraint?
    # I_in_avg = P_in / Vin_dc_min
    # I_L_avg (primary side during on-time in CCM approximation/equivalent)
    # For Flyback, I_primary_avg_during_on = I_in_avg / D_max
    # Delta_I_L = k_rf * I_primary_avg_during_on
    # L_p = (Vin_dc_min * D_max) / (f_sw * Delta_I_L)
    
    i_in_avg = p_in / vin_dc_min
    i_pri_avg_on = i_in_avg / d_max
    delta_i = k_rf * i_pri_avg_on
    
    l_p = (vin_dc_min * d_max) / (f_sw * delta_i)
    
    # Peak Current
    i_pk = i_pri_avg_on + (delta_i / 2)
    
    # Turns Ratio
    # Vor = (Np/Ns) * (Vout + Vf)
    # Let's assume Diode drop Vf = 0.7 (or 1.0 for margin)
    vf = 1.0
    turns_ratio = vor / (vout + vf)
    
    # Snubber (RCD) Estimation
    # Leakage Inductance L_lk approx 2% of L_p
    l_lk = 0.02 * l_p
    # Power in leakage: 0.5 * L_lk * I_pk^2 * f_sw
    p_lk = 0.5 * l_lk * (i_pk**2) * f_sw
    # V_snubber usually clamp at 1.5 * Vor to 2.0 * Vor
    v_snip = 2.0 * vor 
    # R_snubber = (V_snip^2) / (P_lk * (V_snip / (V_snip - Vor))) ? 
    # Simplified: Dissipate leakage energy.
    # V_clamp = V_snip. R = V_clamp^2 / P_res. 
    # Actually standard design: R_sn = (V_sn^2) / (0.5 * Llk * Ipk^2 * fsw * (Vsn/(Vsn-Vor)))
    v_sn = 1.5 * vor # Clamp voltage above Vor
    if v_sn <= vor: float("inf") # prevent div by zero
    
    p_sn = p_lk * (v_sn / (v_sn - vor))
    r_sn = (v_sn ** 2) / p_sn
    
    # C_sn usually allow some voltage ripple on snubber, e.g. 5-10%
    # dV_sn = 0.1 * V_sn
    # C_sn = V_sn / (dV_sn * R_sn * f_sw) 
    c_sn = v_sn / (0.1 * v_sn * r_sn * f_sw)
    
    
    return {
        "topology": "Flyback",
        "switching_frequency": f_sw,
        "primary_inductance": l_p,
        "primary_peak_current": i_pk,
        "turns_ratio": turns_ratio,
        "max_duty_cycle": d_max,
        "ripple_factor": k_rf,
        "magnetizing_current_ripple": delta_i,
        "snubber_r": r_sn,
        "snubber_c": c_sn,
        "reflected_output_voltage": vor
    }
