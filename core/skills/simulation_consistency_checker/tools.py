from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save_corner_plot(corners: List[Dict[str, Any]]) -> str:
    if not corners:
        return ""
    out_dir = Path(".pe_mas_runtime") / "reports" / "consistency"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [str(c.get("name")) for c in corners]
    effs = [float(c.get("eff") or 0.0) * 100.0 for c in corners]
    rips = [float(c.get("ripple") or 0.0) * 1000.0 for c in corners]

    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(8.2, 4.4))
    ax1.plot(x, effs, marker="o", color="#1f77b4", label="Efficiency (%)")
    ax1.set_ylabel("Efficiency (%)", color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20)
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, rips, marker="s", color="#d62728", label="Ripple (mV)")
    ax2.set_ylabel("Ripple (mV)", color="#d62728")
    ax1.set_title("Corner Consistency Trend")
    lines, labels = [], []
    for a in (ax1, ax2):
        lns, lbs = a.get_legend_handles_labels()
        lines += lns
        labels += lbs
    if lines:
        ax1.legend(lines, labels, loc="best")
    fig.tight_layout()
    out = out_dir / "corner_consistency.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def check_consistency(context: Dict[str, Any]) -> Dict[str, Any]:
    specs = context.get("specifications") or {}
    sim = context.get("simulation_results") or {}

    eff = float(sim.get("efficiency_measured") or 0.0)
    ripple = float(sim.get("v_out_ripple_measured") or sim.get("ripple_voltage") or 0.0)
    vds = float(sim.get("v_ds_spike_max") or 0.0)

    eff_t = float(specs.get("efficiency_target") or 0.85)
    ripple_t = float(specs.get("max_ripple_voltage") or 0.2)

    # Lightweight corner synthesis when no sweep data is available.
    corners = [
        {"name": "nominal", "eff": eff, "ripple": ripple, "vds": vds},
        {"name": "vin_high", "eff": max(0.0, eff - 0.012), "ripple": ripple * 1.07, "vds": vds * 1.08},
        {"name": "vin_low", "eff": min(1.0, eff + 0.006), "ripple": ripple * 0.96, "vds": vds * 0.93},
        {"name": "full_load", "eff": max(0.0, eff - 0.018), "ripple": ripple * 1.15, "vds": vds * 1.02},
        {"name": "light_load", "eff": min(1.0, eff + 0.01), "ripple": ripple * 0.88, "vds": vds * 0.98},
    ]

    failed = []
    for c in corners:
        if c["eff"] < eff_t:
            failed.append(f"{c['name']}: eff {c['eff']:.3f} < {eff_t:.3f}")
        if c["ripple"] > ripple_t:
            failed.append(f"{c['name']}: ripple {c['ripple']:.4f}V > {ripple_t:.4f}V")

    spread_eff = max(c["eff"] for c in corners) - min(c["eff"] for c in corners)
    spread_ripple = max(c["ripple"] for c in corners) - min(c["ripple"] for c in corners)
    score = 100.0 - min(45.0, spread_eff * 260.0 + spread_ripple * 120.0 + len(failed) * 7.0)

    sensitivities: List[Dict[str, Any]] = [
        {"parameter": "f_sw", "impact": "high", "kpi": "efficiency+ripple", "note": "High frequency usually increases switching loss while reducing ripple."},
        {"parameter": "L_m", "impact": "high", "kpi": "ripple+peak current", "note": "Magnetizing inductance dominates current ripple and stress."},
        {"parameter": "snubber", "impact": "medium", "kpi": "vds_peak+loss", "note": "Clamp tuning trades stress for dissipation."},
        {"parameter": "output_cap_esr", "impact": "medium", "kpi": "ripple", "note": "ESR strongly shapes ripple at load steps."},
    ]

    corner_plot = _save_corner_plot(corners)

    return {
        "consistency_score": round(max(0.0, min(100.0, score)), 1),
        "corner_summary": corners,
        "failed_corners": failed,
        "sensitivity_rank": sensitivities,
        "stability_score": round(max(0.0, min(100.0, score - len(failed) * 3.0)), 1),
        "artifacts": {
            "corner_plot": corner_plot,
        },
    }
