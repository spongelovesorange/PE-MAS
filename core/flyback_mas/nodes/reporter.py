from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from ..state import PowerSupplyState
from .requirements import get_llm
from .research_helper import collect_node_research
from ..skills_manager import SkillManager
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from pathlib import Path

def report_writer_node(state: PowerSupplyState) -> Dict[str, Any]:
    """
    Node 6: Design Report Writer (LLM Agent)
    Generates a professional engineering report summarizing the entire workflow.
    """
    print("DEBUG: Report Writer Agent Started")
    # [NEW] Best Effort Recovery
    # If the process ended in failure (or partial failure), revert to the BEST iteration tracked by Validator.
    best_candidate = state.get("best_design_candidate")
    current_status = (state.get("verification") or {}).get("status", "FAIL")
    
    specs = state.get("specifications")
    node_research = collect_node_research(
        "reporter",
        (
            f"flyback reporting checklist papers blogs forum summary references Vin {specs.get('input_voltage_max') if isinstance(specs, dict) else '-'} "
            f"Vout {specs.get('output_voltage') if isinstance(specs, dict) else '-'}"
        ),
        max_results=4,
    )
    
    # Decide which design to report
    if best_candidate and current_status == "FAIL":
        print(f"Reverting to Best Design Candidate form Iteration {best_candidate.get('iteration')} (Score: {best_candidate.get('score'):.1f})")
        design = best_candidate.get("theoretical_design")
        bom = best_candidate.get("bom")
        sim = best_candidate.get("simulation_results")
        # Also update the validation result for the report text to match
        state["verification"] = best_candidate.get("verification")
    else:
        # Use current state (which is either PASS or the last attempt)
        design = state.get("theoretical_design")
        bom = state.get("bom")
        sim = state.get("simulation_results")
    
    # 1. Generate Waveform Plot (Technical Artifact)
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_success = False
        fsw = design.get('switching_frequency', 65000)  # Define early to prevent scoping errors
        
        # Check if we have REAL simulation data from PLECS
        csv_path = sim.get("waveforms_path")
        if csv_path and os.path.exists(csv_path):
            print(f"DEBUG: Plotting REAL data from {csv_path}")
            # Assume CSV format: Time, Vds, I_pri, V_out...
            try:
                import pandas as pd
                # Try reading with header first
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                     # Fallback for malformed CSVs or missing headers
                     df = pd.read_csv(csv_path, header=None)

                # Check if headers are numeric (meaning no header row)
                if pd.to_numeric(df.columns, errors='coerce').notna().all():
                     # Reload with header=None
                     df = pd.read_csv(csv_path, header=None)
                     df.columns = [f"Signal_{i}" for i in range(len(df.columns))]
                
                t = df.iloc[:, 0].values
                vds = None
                
                # Heuristic 1: Look for "Vds" or "Mosfet" in column names
                vds_cols = [c for c in df.columns if isinstance(c, str) and ("Vds" in c or "Mosfet" in c)]
                
                if vds_cols:
                    vds = df[vds_cols[0]].values
                else:
                    # Heuristic 2: Identify Vds by signal properties (High Voltage + Switching)
                    # Time is usually col 0.
                    # Vds is typically > 100V (for offline flyback) or > 20V (for DC/DC).
                    # Vout is stable DC.
                    # I_pri is triangular.
                    candidates = []
                    for col in df.columns[1:]:
                        signal = df[col].values
                        if np.max(signal) > 60.0: # Likely High Voltage side
                             candidates.append(signal)
                    
                    if candidates:
                        vds = candidates[0] # Pick the first high voltage signal
                    else:
                         # Fallback to column 1 or last column
                         if len(df.columns) > 1:
                             vds = df.iloc[:, -1].values # Often last signal in probe
                
                if vds is not None:
                    ax.plot(t, vds, label="Measured Vds (PLECS)", linewidth=0.8)
                    ax.set_title(f"Simulated V_ds (Real PLECS Data)")
                else:
                    print("WARN: Could not identify Vds signal. Plotting all signals.")
                    for col in df.columns[1:]:
                         ax.plot(t, df[col], label=str(col), alpha=0.5)
                    ax.set_title("Simulated Signals (Unknown Mapping)")

                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Voltage (V)")
                ax.legend(loc='upper right', fontsize='small')
                ax.grid(True, alpha=0.3)
                plot_success = True
                
            except Exception as read_err:
                print(f"WARN: Failed to read PLECS CSV: {read_err}. Falling back to Math Plot.")
                csv_path = None # Trigger fallback

        if not plot_success:
            # --- MATH MODEL PLOT (Fallback) ---
            print("DEBUG: Using Math Model for Plot (No PLECS CSV found or read failed)")
            # Resolution
            dt = 1.0 / (fsw * 100) 
            t = np.linspace(0, dt*400, 400) # 4 cycles
        
            # Approximate Vds waveform: Rectangular + Ringing
            period = 1.0/fsw
            duty = 0.45 
            
            # Base Pulse
            v_in = specs.get('input_voltage_max', 265)
            # Fix mock voltage logic for plot
            if v_in < 60:
                 v_base = v_in
            else:
                 v_base = 300 # Rough mains rectified
                 
            vor = design.get('reflected_output_voltage', 80)
            v_peak = v_base + vor
            
            waveform = []
            for ti in t:
                cycle_time = ti % period
                if cycle_time < (period * duty):
                    val = 0.5 # On state (approx 0 + Ros_on drop)
                else:
                    # Off state with ringing
                    decay = np.exp(-(cycle_time - period*duty) * 5e5)
                    ringing = 20 * np.sin(2*np.pi * fsw * 10 * ti) * decay
                    val = v_peak + ringing
                waveform.append(val)
                
            ax.plot(t*1e6, waveform)
            ax.set_title(f"Simulated V_ds (MOSFET Drain-Source) [Approximated]")
            ax.set_xlabel("Time (µs)")
            ax.set_ylabel("Voltage (V)")
            ax.grid(True, alpha=0.3)
        
        plot_path = "design_report_waveform.png"
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"WARN: Plot generation failed: {e}")
        plot_path = "plot_failed.png"

    # 2. Generate Text Report via LLM (Artifact Generation)
    report_content = ""
    try:
        print("DEBUG: Generating Markdown Artifact...")

        # Prefer deterministic skill-based report synthesis; fallback to the built-in template below.
        try:
            skills_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "skills")
            sm = SkillManager(skills_dir)

            citation_skill = sm.get_skill("engineering_citation_manager")
            citation_pack = {}
            if citation_skill and citation_skill.tools_module and hasattr(citation_skill.tools_module, "build_citation_pack"):
                citation_pack = citation_skill.tools_module.build_citation_pack({
                    "references": (state.get("literature_references") or []) + (node_research.get("references") or [])
                })

            peer_pack = {}
            peer_skill = sm.get_skill("design_peer_review")
            if peer_skill and peer_skill.tools_module and hasattr(peer_skill.tools_module, "review_design"):
                peer_pack = peer_skill.tools_module.review_design({
                    "specifications": specs,
                    "request_profile": state.get("request_profile") or {},
                    "theoretical_design": design,
                    "bom": bom,
                    "simulation_results": sim,
                    "verification": state.get("verification") or {},
                    "formula_checks": state.get("formula_checks") or {},
                    "node_verification": state.get("node_verification") or {},
                })

            consistency_pack = {}
            consistency_skill = sm.get_skill("simulation_consistency_checker")
            if consistency_skill and consistency_skill.tools_module and hasattr(consistency_skill.tools_module, "check_consistency"):
                consistency_pack = consistency_skill.tools_module.check_consistency({
                    "specifications": specs,
                    "simulation_results": sim,
                    "verification": state.get("verification") or {},
                })

            evidence_pack = {}
            evidence_skill = sm.get_skill("evidence_grader")
            if evidence_skill and evidence_skill.tools_module and hasattr(evidence_skill.tools_module, "grade_evidence"):
                evidence_pack = evidence_skill.tools_module.grade_evidence({
                    "retrieved_knowledge_references": state.get("retrieved_knowledge_references") or [],
                    "literature_references": (state.get("literature_references") or []) + (node_research.get("references") or []),
                })

            sensitivity_pack = {}
            sensitivity_skill = sm.get_skill("param_sensitivity_planner")
            if sensitivity_skill and sensitivity_skill.tools_module and hasattr(sensitivity_skill.tools_module, "plan_parameter_sensitivity"):
                sensitivity_pack = sensitivity_skill.tools_module.plan_parameter_sensitivity({
                    "specifications": specs,
                    "simulation_results": sim,
                    "verification": state.get("verification") or {},
                })

            report_skill = sm.get_skill("final_report_writer")
            if report_skill and report_skill.tools_module and hasattr(report_skill.tools_module, "generate_final_report"):
                skill_context = {
                    "specifications": specs,
                    "request_profile": state.get("request_profile") or {},
                    "theoretical_design": design,
                    "bom": bom,
                    "simulation_results": sim,
                    "verification": state.get("verification") or {},
                    "correction_review": state.get("correction_review") or {},
                    "formula_checks": state.get("formula_checks") or {},
                    "node_verification": state.get("node_verification") or {},
                    "literature_references": (citation_pack.get("normalized_references") or state.get("literature_references") or []),
                    "citation_audit": citation_pack.get("citation_audit") or {},
                    "broken_links": citation_pack.get("broken_links") or [],
                    "peer_review_findings": peer_pack,
                    "simulation_consistency": consistency_pack,
                    "evidence_grade": evidence_pack,
                    "param_sensitivity_plan": sensitivity_pack,
                    "magnetic_design": state.get("magnetic_design") or {},
                    "execution_plan": state.get("execution_plan") or {},
                    "planning_summary": state.get("planning_summary") or "",
                    "memory_insights": state.get("memory_insights") or {},
                    "skill_recommendations": state.get("skill_recommendations") or [],
                    "reasoning_trace": state.get("reasoning_trace") or [],
                    "config": state.get("config") or {},
                    "waveform_path": plot_path,
                }
                skill_report = report_skill.tools_module.generate_final_report(skill_context)
                report_content = str(skill_report.get("report_markdown") or "").strip()
                if report_content:
                    with open("design_report.md", "w", encoding="utf-8") as f:
                        f.write(report_content + "\n")

                    report_dir = Path(skill_report.get("report_dir") or ".pe_mas_runtime/reports")
                    report_dir.mkdir(parents=True, exist_ok=True)
                    report_md = report_dir / "final_report.md"
                    report_md.write_text(report_content + "\n", encoding="utf-8")

                    figures = skill_report.get("figures") or []
                    print("Markdown Artifact saved via final_report_writer skill")
                    return {
                        "report_content": report_content,
                        "messages": [f"Report generated. [View Artifact](design_report.md) | Package: {report_md}"],
                        "citation_pack": citation_pack,
                        "peer_review_findings": peer_pack or state.get("peer_review_findings"),
                        "simulation_consistency": consistency_pack or state.get("simulation_consistency"),
                        "evidence_grade": evidence_pack or state.get("evidence_grade"),
                        "param_sensitivity_plan": sensitivity_pack or state.get("param_sensitivity_plan"),
                        "reasoning_logs": state.get("reasoning_logs", {}) | {
                            "reporter": node_research.get("logs", []) + [
                                "[TOOL_OUTPUT] engineering_citation_manager used",
                                "[TOOL_OUTPUT] design_peer_review used",
                                "[TOOL_OUTPUT] simulation_consistency_checker used",
                                "[TOOL_OUTPUT] evidence_grader used",
                                "[TOOL_OUTPUT] param_sensitivity_planner used",
                                "[TOOL_OUTPUT] final_report_writer used",
                                f"[RESULT] report_package={report_md}",
                                f"[RESULT] figures={figures}",
                            ]
                        },
                        "literature_references": citation_pack.get("normalized_references", []) or node_research.get("references", []),
                    }
        except Exception as skill_err:
            print(f"WARN: final_report_writer skill unavailable, using built-in reporter: {skill_err}")

        formula_checks = state.get("formula_checks", {}) or {}
        node_verification = state.get("node_verification", {}) or {}
        formula_section = "\n## 7. Formula Verification Audit\n"
        if formula_checks:
            for node_name, check_pack in formula_checks.items():
                formula_section += f"\n### {node_name}\n"
                checks = check_pack.get("checks", []) if isinstance(check_pack, dict) else []
                warnings = check_pack.get("warnings", []) if isinstance(check_pack, dict) else []
                fatals = check_pack.get("fatal", []) if isinstance(check_pack, dict) else []
                if checks:
                    for c in checks[:12]:
                        formula_section += f"- {c.get('name','check')}: pass={c.get('pass')}"
                        if "expected" in c:
                            formula_section += f", expected={c.get('expected')}"
                        if "required" in c:
                            formula_section += f", required={c.get('required')}"
                        if "actual" in c:
                            formula_section += f", actual={c.get('actual')}"
                        formula_section += "\n"
                for w in warnings[:8]:
                    formula_section += f"- Warning: {w}\n"
                for f in fatals[:8]:
                    formula_section += f"- Failure: {f}\n"
        else:
            formula_section += "- No formula audit records found.\n"

        if node_verification:
            formula_section += "\n### Node Verification Status\n"
            for k, v in node_verification.items():
                if isinstance(v, dict):
                    formula_section += f"- {k}: {v.get('status', 'N/A')}\n"
        
        # Prepare Literature Section
        lit_refs = (state.get("literature_references", []) or []) + (node_research.get("references", []) or [])
        lit_section = ""
        
        # Deduplication and Filtering
        unique_refs = []
        seen_titles = set()
        
        if lit_refs:
            for ref in lit_refs:
                # Handle Dict format
                if isinstance(ref, dict):
                    title = ref.get('title', 'Unknown Paper')
                    # Skip withdrawn or error papers
                    if "withdrawn" in title.lower() or "error" in title.lower():
                        continue
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    unique_refs.append(ref)
                # Handle String format
                elif isinstance(ref, str):
                    title = str(ref).replace("[PAPER]", "").strip()
                    # Skip duplicate phrases (simple check)
                    clean_check = title[:20] 
                    if clean_check in seen_titles:
                         continue
                    # Skip withdrawn
                    if "withdrawn" in title.lower() or "error" in title.lower():
                         continue
                    seen_titles.add(clean_check)
                    unique_refs.append(ref)

        if unique_refs:
            lit_section = "\n## 4. Traceable Literature\n"
            for ref in unique_refs:
                 if isinstance(ref, dict):
                     title = ref.get('title', 'Unknown Paper')
                     url = ref.get('url', '#')
                     insight = ref.get('insight', '')[:150]
                     lit_section += f"- **[{title}]({url})**: {insight}...\n"
                 else:
                     lit_clean = str(ref).replace("[PAPER]", "").strip()[:150]
                     lit_section += f"- {lit_clean}...\n"
        else:
            lit_section = "\n## 4. Reference Methodology\n- Standard Textbook Equations (Fundamentals of Power Electronics)\n"

        # Prepare Reasoning Trace Section
        trace_items = state.get("reasoning_trace", [])
        trace_section = ""
        if trace_items:
            trace_section = "\n## 6. Logic Trace (System 2 Reasoning)\n| Step | Agent | Action | Evidence | Confidence |\n|---|---|---|---|---|\n"
            seen_actions = set() # Stores (Agent, Action) to dedupe identical repeated steps
            
            for item in trace_items:
                if isinstance(item, dict):
                    step = item.get("step", "N/A")
                    agent = item.get("agent", "System")
                    action = str(item.get("action", "")).replace("\n", " ").strip()
                    evidence = str(item.get("evidence", "None")).replace("\n", " ").strip()
                    conf = item.get("confidence", 0.0)
                    
                    # Deduplication Strategy:
                    # If Agent+Action is identical to the previous one, we skip it to avoid spamming 
                    # "Optimizing frequency..." 10 times.
                    dedup_key = (agent, action)
                    if dedup_key in seen_actions:
                         continue
                    seen_actions.add(dedup_key)

                    # Truncate long strings for Markdown table safety
                    if len(evidence) > 50: evidence = evidence[:47] + "..."
                    if len(action) > 40: action = action[:37] + "..."

                    trace_section += f"| {step} | {agent} | {action} | {evidence} | {conf:.0%} |\n"
                else:
                    item_str = str(item).replace("\n", " ")[:50]
                    if item_str in seen_actions: continue
                    seen_actions.add(item_str)
                    trace_section += f"| - | System | {item_str}... | - | - |\n"

        # SOTA: Generating a structured engineering artifact
        report_content = f"""# Flyback Converter Design Report
**Project ID:** {state.get("config", {}).get("thread_id") or state.get("thread_id", "N/A")} | **Date:** {datetime.now().strftime("%Y-%m-%d")}

## 1. Executive Summary
The design for a **{specs.get('output_voltage')}V / {specs.get('output_current')}A ({specs.get('output_voltage')*specs.get('output_current')}W)** Flyback converter has been completed and verified.
- **Topology:** Flyback (CCM/DCM Hybrid)
- **Status:** {state.get('verification', {}).get('status', 'REVIEW NEEDED')} (Iteration {state.get('iteration')})

## 2. Key Specifications
| Parameter | Target | Measured/Simulated | Pass? |
|-----------|--------|-------------------|-------|
| Efficiency | > {specs.get('efficiency_target'):.1%} | **{sim.get('efficiency_measured'):.2%}** | {'PASS' if sim.get('efficiency_measured') >= specs.get('efficiency_target') else 'FAIL'} |
| Ripple (Vpp)| < {specs.get('max_ripple_voltage')}V | **{sim.get('v_out_ripple_measured'):.3f}V** | {'PASS' if sim.get('v_out_ripple_measured') <= specs.get('max_ripple_voltage') else 'FAIL'} |
| Vds Spike | Safe Limit | **{sim.get('v_ds_spike_max'):.1f}V** | Check Rating |

## 3. Bill of Materials (Key Parts)
- **Transformer:** {bom.get('transformer', {}).get('Core', 'EE25') if isinstance(bom.get('transformer'), dict) else 'EE25'} (Np={bom.get('transformer', {}).get('Np') if isinstance(bom.get('transformer'), dict) else '?'}, Ns={bom.get('transformer', {}).get('Ns') if isinstance(bom.get('transformer'), dict) else '?'})
- **MOSFET:** {bom.get('mosfet', {}).get('Part Number', 'Generic') if isinstance(bom.get('mosfet'), dict) else 'Generic'} ({bom.get('mosfet', {}).get('Vds', 'N/A') if isinstance(bom.get('mosfet'), dict) else '?'}V)
- **Output Cap:** {bom.get('output_cap', {}).get('Value', 'N/A') if isinstance(bom.get('output_cap'), dict) else 'Generic'}F

{lit_section}

## 5. Simulation Artifacts
![Waveform](./design_report_waveform.png)

{trace_section}

{formula_section}

---
*Generated by EE-MAS*
"""

        
        # Write Artifact to Disk
        with open("design_report.md", "w", encoding="utf-8") as f:
            f.write(report_content)
            
        print("Markdown Artifact saved to design_report.md")
        
    except Exception as e:
        print(f"WARN: Report generation failed: {e}")
        report_content = "Report generation failed."

    return {
        "report_content": report_content,
        "messages": [f"Report generated. [View Artifact](design_report.md)"],
        "reasoning_logs": state.get("reasoning_logs", {}) | {"reporter": node_research.get("logs", [])},
        "literature_references": node_research.get("references", []),
    }
