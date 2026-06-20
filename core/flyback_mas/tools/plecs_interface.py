import sys
import xmlrpc.client
import logging
import os
import shutil
import time
import subprocess
import numpy as np
import re
from pathlib import Path


def _plecs_rpc_url() -> str:
    return str(os.getenv("PLECS_RPC_URL") or os.getenv("PE_MAS_PLECS_RPC_URL") or "").strip()


def run_plecs_simulation(params: dict, model_name_original: str = "Flyback") -> dict:
    """
    Robust Simulation: "Bake" parameters into a temporary PLECS file, run it, and read CSV.
    """
    project_root = Path(__file__).resolve().parents[3]
    model_dir = project_root / "core" / "simulation" / "flyback"
    runtime_dir_value = os.getenv("PE_MAS_RUNTIME_DIR")
    runtime_dir = Path(runtime_dir_value).expanduser() if runtime_dir_value else project_root / ".pe_mas_runtime"
    if not runtime_dir.is_absolute():
        runtime_dir = (project_root / runtime_dir).resolve()
    simulation_dir = runtime_dir / "plecs" / "flyback"
    simulation_dir.mkdir(parents=True, exist_ok=True)
    original_model = model_dir / "Flyback_effi.plecs"
    
    run_model_name = "Flyback_Auto_Run"
    run_model_file = simulation_dir / f"{run_model_name}.plecs"
    
    # MacOS Sandboxing/Permission Fix: 
    # Try using the same directory as the model file to avoid permission issues
    csv_file = simulation_dir / "flyback_effi.csv"
    waveforms_file = simulation_dir / "flyback_waveforms.csv"
    waveform_extras_file = simulation_dir / "flyback_waveforms_extras.csv"
    
    if csv_file.exists():
        try: os.remove(csv_file)
        except: pass
    if waveforms_file.exists():
        try: os.remove(waveforms_file)
        except: pass
    if waveform_extras_file.exists():
        try: os.remove(waveform_extras_file)
        except: pass

    # 1. Clean Parameter Values
    p = {}
    for k, v in params.items():
        if isinstance(v, (int, float)):
            p[k] = v
        elif isinstance(v, str):
            try: p[k] = float(v)
            except: pass 

    explicit_ron = 'Ron' in p
    explicit_ro = 'Ro' in p
    explicit_vref = 'Vref' in p
    defaults = {
        'Kp': 0.01, 'Ki': 10, 'PI_Upper': 0.65, 'PI_Lower': 0, 'Tstop': 0.1, 
        'Vin': 265.0, 'fs':100000.0, 'Ro': 10, 'Vref': 20.0,
        'Rsn': 1000.0, 'Csn': 1e-9,  # Include Snubber params to prevent 'undefined' errors
        'Ron': 1.2,                  # MOSFET on-resistance used by MOSFET1
        'Rdiode': 0.04,              # Diode dynamic resistance (used by D1 Ron)
        'Lp': 600e-6, 'Co': 470e-6,   # Default Inductance and Capacitance
        # Steinmetz / Core Loss Parameters (Missing 'k' caused crash)
        'k': 2.46, 'afa': 1.45, 'beta': 2.7, 'Ae': 25e-6, 'Ve': 2e-6,
        # Flyback_effi.plecs uses a raw Steinmetz-like expression that is not
        # material-calibrated. Keep the scale explicit so closed-loop runs do
        # not silently report uncalibrated 10s-of-watts core loss.
        'core_loss_scale': 0.016,
    }
    for k,v in defaults.items():
        if k not in p: p[k] = v
    if not explicit_vref and p.get('Vout') not in (None, 0):
        p['Vref'] = float(p['Vout'])
    if not explicit_ro:
        try:
            vout_val = float(p.get('Vout') or p.get('Vref') or 0.0)
            pout_val = float(p.get('Pout') or 0.0)
            iout_val = float(p.get('Iout') or 0.0)
            if vout_val > 0 and pout_val > 0:
                p['Ro'] = (vout_val * vout_val) / pout_val
            elif vout_val > 0 and iout_val > 0:
                p['Ro'] = vout_val / iout_val
        except Exception:
            pass
    if not explicit_ron:
        for alias in ('ron_ohm', 'Rds_on', 'Rds', 'RdsOn', 'mosfet_ron'):
            if alias in p and p.get(alias):
                p['Ron'] = p[alias]
                break
    if 'Ts' not in p and 'fs' in p:
        p['Ts'] = 1.0 / float(p['fs'])
        
    # CRITICAL FIX: The PLECS Model uses 'n' for Turns Ratio, but our design uses 'Np'/'Ns'
    # We must calculate 'n' if it's missing to avoid "n undefined" error.
    # Also, the efficiency calculation block requires 'N_pri' explicitly.
    if 'n' not in p:
        if 'Np' in p and 'Ns' in p:
            p['n'] = float(p['Np']) / float(p['Ns'])
        else:
            p['n'] = 1.6 # Default fallback
            
    if 'N_pri' not in p:
        p['N_pri'] = p.get('Np', 16) 
             
    # CRITICAL FIX 3: Inject Parasitics
    # The original InitializationCommands calculated Rp and Rs. Since we wipe it, we must provide them.
    if 'Rp' not in p:
        p['Rp'] = 10e-3 # Default 10mOhm
    if 'Rs' not in p:
        # Rs = Rp / n^2
        p['Rs'] = p['Rp'] / (p['n']**2)

    # CRITICAL FIX 2: Sanitize Control Limits.
    # Flyback cannot operate at D=1.0 (switch permanently ON), it needs to demagnetize.
    # Keep a conservative default close to the model's original closed-loop setting.
    if p.get('PI_Upper', 0) > 0.95:
        print(f"DEBUG: Clamping PI_Upper from {p.get('PI_Upper')} to 0.65 to prevent saturation.", file=sys.stderr)
        p['PI_Upper'] = 0.65

    # Build Init String
    init_str = "; ".join([f"{k}={v}" for k,v in p.items()]) + ";"
    print(f"DEBUG: Injecting Init String: {init_str}", file=sys.stderr)

    # 2. Create Baked Model
    try:
        with open(original_model, 'r', encoding='utf-8') as f:
            content = f.read()        
        # --- DEBUG LOGGING START ---
        debug_log_path = simulation_dir / "plecs_bake_debug.log"
        with open(debug_log_path, "w") as dbg:
            dbg.write("--- ORIGINAL CONTENT SNIPPET ---\n")
            # Find snippet around Thermal or C2M
            idx = content.find("C2M0080120D")
            if idx != -1:
                dbg.write(content[max(0, idx-100):min(len(content), idx+100)])
            else:
                dbg.write("C2M0080120D NOT FOUND in original file!\n")
            dbg.write("\n------------------------------\n")
        # --- DEBUG LOGGING END ---
        # Rename
        # Handle variations in spacing for Name property
        if 'Name          "Flyback_effi"' in content:
            content = content.replace('Name          "Flyback_effi"', f'Name          "{run_model_name}"')
        else:
             # Regex fallback for Name
             content = re.sub(r'Name\s+"Flyback_effi"', f'Name          "{run_model_name}"', content)
        
        # FULL FORCE REPLACEMENT FOR THERMAL FILE
        # We need to replace ANY reference to C2M0080120D with the absolute path
        # The file system has C2M0080120D.xml
        # PLECS usually references it as file:C2M0080120D (no extension) or with extension
        
        # Absolute path to xml
        thermal_file_name = "C2M0080120D.xml"
        thermal_abs_path = str((model_dir / thermal_file_name))
        # Ensure forward slashes for PLECS compatibility
        thermal_abs_path = thermal_abs_path.replace("\\", "/")
        
        # Target strings to replace
        targets = [
            "file:C2M0080120D.xml", 
            "file:C2M0080120D",
            "C2M0080120D.xml",
            "C2M0080120D"
        ]
        
        # We replace the most specific ones first
        # But we must be careful not to double replace.
        # Strategy: Replace "file:C2M0080120D" -> "file:ABS_PATH"
        # If it's just "C2M0080120D" inside a Value string, matches "Value "C2M0080120D"" -> "Value "file:ABS_PATH""
        
        # 1. Replace "file:C2M0080120D" (with or without xml)
        # Note: If we replace "C2M0080120D" first, we might break "file:C2M0080120D".
        # So we handle "file:..." first.
        
        new_val = f"file:{thermal_abs_path}"
        
        if "file:C2M0080120D" in content:
            print(f"DEBUG: Replacing file:C2M0080120D with {new_val}", file=sys.stderr)
            content = content.replace("file:C2M0080120D", new_val)
            
        # 2. Handle cases where it might just be the filename in quotes (common in some libs)
        # Be careful not to replace it if it's already fixed
        # Regex for Value "C2M0080120D"
        content = re.sub(r'Value\s+"C2M0080120D"', f'Value         "{new_val}"', content)
        
        # 3. Generic fallback: If the filename exists anywhere else in a parameter value
        # Pattern: " ... C2M0080120D ... "
        # We won't do global replace generally to avoid hitting comments etc, but for this specific file it's low risk.
        
        # --- DEBUG LOGGING START ---
        with open(debug_log_path, "a") as dbg:
            dbg.write("\n--- AFTER REPLACEMENTS ---\n")
            # Find snippet around Thermal or C2M
            idx = content.find("C2M0080120D")
            if idx != -1:
                dbg.write(content[max(0, idx-100):min(len(content), idx+100)])
                # Also check if it has file: prefix and absolute path
                if thermal_abs_path in content:
                    dbg.write("\n\nSUCCESS CONFIRMATION: Absolute path found in content.\n")
                else:
                    dbg.write("\n\nWARNING: Absolute path NOT found in expected locations!\n")
            else:
                dbg.write("C2M0080120D NOT FOUND in modified content (Which is weird unless renamed)!\n")
        # --- DEBUG LOGGING END ---

        # FIX: MaxStep must be smaller than switching period
        # Default was 1e-4, which is 100us. fs=65kHz -> 15us period. 
        # We need step size around 100ns (1e-7) to capture pulses.
        # FIX 2: Also set Refine to ensure outputs are dense enough
        if 'MaxStep       "1e-4"' in content:
             content = content.replace('MaxStep       "1e-4"', 'MaxStep       "1e-7"')
             content = content.replace('Refine        "1"', 'Refine        "5"')
             print("DEBUG: Fixed MaxStep to 1e-7 and Refine to 5", file=sys.stderr)
        
        # Inject Init Params
        # We must include \n to ensure subsequent lines (which might be comments) are separated
        # And we use a regex that greedily consumes the entire InitializationCommands block if possible, 
        # OR we just ensure our injection ends with a newline and comment char if needed.
        # But safest is to replace the first line and assume the rest are comments or don't matter as we override them.
        # Wait, if we replace "Tstop = ...\n..." with "init_str", we lose the \n.
        # If "line2" starts with "%", it becomes "init_str%...".
        # This merges init_str with the comment! If init_str ends with ";", it becomes ";%...".
        # Which is fine: "; % comment".
        # BUT if "line2" starts with a command, e.g. "P = 15", then it becomes ";P = 15".
        # Which executes P=15.
        
        # In our case, the file has:
        # InitializationCommands "Tstop... \n ... "
        # " ... "
        
        # We want to force ignore of the old hardcoded values.
        # So we should try to comment out the REST of the block if we can, 
        # or just hope the edits I made to the file (adding %) are enough.
        # Since I verified the file has % for P, Ts, Ro, they should be ignored.
        
        # But maybe the error "'P' undefined" comes from somewhere else?
        # Simulation Parameters? No.
        # Masks? No.
        
        # Let's verify exactly what init_str looks like. 
        # If init_str was injected without \n at the end, and the next line in file is:
        # "%%%%%%%%%%%%%%%%\n% ..."
        # Then we get "init_str%%%%%%%%%%%%%%%%\n% ...".
        # This looks like "init_str" followed by comment.
        
        # Correct Regex for multiline quoted property strings in PLECS
        # Matches: Property "string" "string2" ...
        # We need to eat until the next Property Name (which is unquoted, Capitalized).
        # Actually, properties are usually on separate lines.
        # So we can match until we see a property name at start of line?
        # A safer bet: match until 'InitialState' specifically for this file.
        
        # We construct the new property string clearly.
        new_prop_block = f'InitializationCommands "{init_str}\\n"'
        
        # Use DOTALL to match across lines.
        # This regex matches 'InitializationCommands' followed by anything until 'InitialState'
        # Be careful to escape backslashes if needed, though raw string r'' helps.
        # FIX: The lookahead (?=\s+InitialState) might fail if there's no space or different formatting.
        # We'll use a safer regex that captures the parameter value.
        # Matches: InitializationCommands "..." (quoted string handling escaped quotes)
        # But PLECS strings can be multiline.
        # Simpler: Just replace the whole block if we can identify start/end.
        # Or just append the new init string to the end of the existing commands?
        # No, we want to override.
        
        # New Strategy: Append our init string at the VERY END of the InitializationCommands string.
        # This requires finding the closing quote of that parameter.
        # But parsing is hard.
        # Let's try the replacement with a more forgiving regex.
        # We assume InitializationCommands is followed by a quoted string.
        # We replace the content inside the quotes? Or the whole attribute?
        
        # Original regex which works for standard files:
        # content = re.sub(r'InitializationCommands\s+.*?(?=\s+InitialState)', new_prop_block, content, flags=re.DOTALL)
        
        # Robust Fix: Search for 'InitializationCommands' and then 'InitialState' and replace everything in between.
        if "InitializationCommands" in content and "InitialState" in content:
            start_idx = content.find("InitializationCommands")
            end_idx = content.find("InitialState", start_idx)
            if start_idx != -1 and end_idx != -1:
                 # Check if there is a closing brace component in between? No, they are attributes of Plecs { ... }
                 # Use substring replacement
                 content = content[:start_idx] + new_prop_block + "\n  " + content[end_idx:]
            else:
                 print("DEBUG: Could not locate InitializationCommands block robustly.", file=sys.stderr)

        # Make the magnetic-loss calibration explicit in the baked model.
        # Older source models use the raw expression directly.
        raw_core_loss_expr = "k*fs^afa*(Vin*u(1)/N_pri/Ae/fs)^beta*Ve"
        scaled_core_loss_expr = "core_loss_scale*k*fs^afa*(Vin*u(1)/N_pri/Ae/fs)^beta*Ve"
        if raw_core_loss_expr in content and scaled_core_loss_expr not in content:
            content = content.replace(raw_core_loss_expr, scaled_core_loss_expr)
            print("DEBUG: Applied explicit core_loss_scale to PLECS core-loss function.", file=sys.stderr)
        
        # Update CSV Path using more generic matching
        clean_csv_path = str(csv_file).replace("\\", "/")
        # Use regex to find and replace the CSV filename regardless of spacing
        if 'flyback_effi.csv' in content:
            # Replace the model's relative CSV output path with the runtime artifact path.
            # We match 'Value\s+"flyback_effi.csv"' allowing any amount of whitespace
            content = re.sub(r'Value\s+"flyback_effi\.csv"', f'Value         "{clean_csv_path}"', content)
            print(f"DEBUG: Successfully baked CSV path to: {clean_csv_path}", file=sys.stderr)
        else:
            print("DEBUG: Could not find 'flyback_effi.csv' reference to update.", file=sys.stderr)
            
        # Update Waveforms CSV Path
        clean_waves_path = str(waveforms_file).replace("\\", "/")
        if 'flyback_waveforms.csv' in content:
            content = re.sub(r'Value\s+"flyback_waveforms\.csv"', f'Value         "{clean_waves_path}"', content)
            print(f"DEBUG: Successfully baked Waveforms CSV path to: {clean_waves_path}", file=sys.stderr)
        else:
            # Fallback: If not found (maybe first run before mod applied?), we can't update.
            # But since I modified the source file, it should be there.
            print("DEBUG: Could not find 'flyback_waveforms.csv' reference to update.", file=sys.stderr)

        export_vbus_waveform = str(params.get("export_vbus_waveform", "1")).strip().lower() not in {"0", "false", "no"}
        if export_vbus_waveform and 'Component     "V_dc"\n        Path          ""\n        Signals       {"Source voltage"}' not in content:
            content = content.replace(
'''      Probe {
        Component     "R1"
        Path          ""
        Signals       {"Resistor voltage"}
      }
    }
    Component {
      Type          ToFile''',
'''      Probe {
        Component     "R1"
        Path          ""
        Signals       {"Resistor voltage"}
      }
      Probe {
        Component     "V_dc"
        Path          ""
        Signals       {"Source voltage"}
      }
    }
    Component {
      Type          ToFile''',
                1,
            )
            content = re.sub(
                r'(Name\s+"Demux".*?Variable\s+"Width"\s+Value\s+)"4"',
                r'\1"5"',
                content,
                count=1,
                flags=re.DOTALL,
            )
            print("DEBUG: Added Vbus source-voltage probe to main waveform export.", file=sys.stderr)

        # Add an auxiliary waveform export in the baked model. The source model
        # already exports ILm/IQ/ID/Vo. This export adds the closed-loop gate
        # signal, a drain-source voltmeter, the DC bus source probe, and an
        # engineering clamp-envelope signal. The current model has no physical
        # clamp network, so the clamp channel is intentionally a labelled
        # envelope signal, not a hardware clamp-node probe.
        enhanced_default = os.getenv("PE_MAS_PLECS_ENHANCED_WAVEFORMS", "0")
        export_extra_waveforms = str(params.get("export_enhanced_waveforms", enhanced_default)).strip().lower() not in {"0", "false", "no"}
        if export_extra_waveforms and "WaveformExtras_ToFile" not in content:
            clean_extras_path = str(waveform_extras_file).replace("\\", "/")
            extra_components = f'''
    Component {{
      Type          Voltmeter
      Name          "VDS_probe"
      Show          off
      Position      [380, 310]
      Direction     up
      Flipped       off
      LabelPosition west
    }}
    Component {{
      Type          SignalDemux
      Name          "Vbus_Demux"
      Show          off
      Position      [625, 100]
      Direction     right
      Flipped       off
      LabelPosition south
      Parameter {{
        Variable      "Width"
        Value         "2"
        Show          off
      }}
    }}
    Component {{
      Type          Constant
      Name          "Vclamp_envelope"
      Show          off
      Position      [700, 385]
      Direction     right
      Flipped       off
      LabelPosition south
      Frame         [-10, -10; 10, 10]
      Parameter {{
        Variable      "Value"
        Value         "Vin+(Vref+Vf)*n+50"
        Show          off
      }}
      Parameter {{
        Variable      "DataType"
        Value         "10"
        Show          off
      }}
    }}
    Component {{
      Type          SignalMux
      Name          "WaveformExtras_Mux"
      Show          off
      Position      [760, 315]
      Direction     right
      Flipped       off
      LabelPosition south
      Parameter {{
        Variable      "Width"
        Value         "4"
        Show          off
      }}
    }}
    Component {{
      Type          ToFile
      Name          "WaveformExtras_ToFile"
      Show          on
      Position      [835, 315]
      Direction     right
      Flipped       off
      LabelPosition south
      Parameter {{
        Variable      "Filename"
        Value         "{clean_extras_path}"
        Show          off
        Evaluate      off
      }}
      Parameter {{
        Variable      "FileType"
        Value         "1"
        Show          off
      }}
      Parameter {{
        Variable      "WriteSignalNames"
        Value         "1"
        Show          off
      }}
      Parameter {{
        Variable      "SampleTime"
        Value         "-1"
        Show          off
      }}
    }}
'''
            content = content.replace('    Component {\n      Type          PulseGenerator', extra_components + '    Component {\n      Type          PulseGenerator', 1)

            content = content.replace(
'''    Connection {
      Type          Wire
      SrcComponent  "MOSFET1"
      SrcTerminal   2
      Points        [275, 255; 170, 255]
      DstComponent  "V_dc"
      DstTerminal   2
    }''',
'''    Connection {
      Type          Wire
      SrcComponent  "MOSFET1"
      SrcTerminal   2
      Points        [275, 255; 170, 255]
      Branch {
        DstComponent  "V_dc"
        DstTerminal   2
      }
      Branch {
        DstComponent  "VDS_probe"
        DstTerminal   2
      }
    }''',
                1,
            )
            content = content.replace(
'''      Branch {
        DstComponent  "MOSFET1"
        DstTerminal   1
      }
    }
    Connection {
      Type          Wire
      SrcComponent  "D1"''',
'''      Branch {
        DstComponent  "MOSFET1"
        DstTerminal   1
      }
      Branch {
        DstComponent  "VDS_probe"
        DstTerminal   1
      }
    }
    Connection {
      Type          Wire
      SrcComponent  "D1"''',
                1,
            )
            content = content.replace(
'''    Connection {
      Type          Signal
      SrcComponent  "Manual Switch"
      SrcTerminal   1
      DstComponent  "Goto1"
      DstTerminal   1
    }''',
'''    Connection {
      Type          Signal
      SrcComponent  "Manual Switch"
      SrcTerminal   1
      Branch {
        DstComponent  "Goto1"
        DstTerminal   1
      }
      Branch {
        DstComponent  "WaveformExtras_Mux"
        DstTerminal   1
      }
    }''',
                1,
            )
            content = content.replace(
'''    Connection {
      Type          Signal
      SrcComponent  "Probe1"
      SrcTerminal   1
      Points        [605, 140]
      DstComponent  "In/Out Voltages"
      DstTerminal   1
    }''',
'''    Connection {
      Type          Signal
      SrcComponent  "Probe1"
      SrcTerminal   1
      Points        [605, 140]
      Branch {
        DstComponent  "In/Out Voltages"
        DstTerminal   1
      }
      Branch {
        DstComponent  "Vbus_Demux"
        DstTerminal   1
      }
    }''',
                1,
            )
            extra_connections = '''
    Connection {
      Type          Signal
      SrcComponent  "VDS_probe"
      SrcTerminal   3
      DstComponent  "WaveformExtras_Mux"
      DstTerminal   2
    }
    Connection {
      Type          Signal
      SrcComponent  "Vbus_Demux"
      SrcTerminal   2
      DstComponent  "WaveformExtras_Mux"
      DstTerminal   3
    }
    Connection {
      Type          Signal
      SrcComponent  "Vclamp_envelope"
      SrcTerminal   1
      DstComponent  "WaveformExtras_Mux"
      DstTerminal   4
    }
    Connection {
      Type          Signal
      SrcComponent  "WaveformExtras_Mux"
      SrcTerminal   1
      DstComponent  "WaveformExtras_ToFile"
      DstTerminal   1
    }
'''
            content = content.replace('    Annotation {\n      Name          "<html><body>', extra_connections + '    Annotation {\n      Name          "<html><body>', 1)
            print(f"DEBUG: Added enhanced waveform export to: {clean_extras_path}", file=sys.stderr)

        # FIX THERMAL PATH for C2M0080120D or others
        # Use absolute path to ensure PLECS can find the thermal description
        # (thermal_abs_path is defined at start of function)
        
        # Ensure the file exists before checking regex
        if not os.path.exists(thermal_abs_path):
             print(f"DEBUG: Thermal file {thermal_abs_path} missing. Checking if it can be found nearby...", file=sys.stderr)
             # Try to find it in kb/component_db or similar if needed, but for now just warn.
        
        # This block is now covered by the global replacement above.
        # But we keep the generic logic just in case there are other files.
        # However, specifically ensure C2M0080120D is not processed again incorrectly.
        
        # Regex to find ANY remaining "file:..." relative path and fix it
        # But we skip C2M0080120D because we just fixed it.
        # Wait, if we replaced it with "file:/path/...", the regex below will see "file:/path/...".
        # The regex looks for `Value "file:([^"]+)"`.
        # `replace_thermal` checks if `file_ref` has NO slashes.
        # If we replaced with absolute path, it HAS slashes. So it won't be touched.
        # Perfect.
        
        def replace_thermal(match):
            # match.0 is 'Value "file:C2M0080120D.xml"' or similar
            # match.1 is 'C2M0080120D.xml'
            file_ref = match.group(1)
            
            # Additional safety: If it's ALREADY absolute, skip
            if file_ref.startswith("/") or file_ref.startswith("C:") or ":" in file_ref and len(file_ref) > 5:
                # Likely absolute path (C:... or /User...)
                return match.group(0)
                
            # Only proceed if relative (no slashes or simple filename)
            if "/" not in file_ref and "\\" not in file_ref:
                # If specifically our file (found via regex matching remaining occurrences)
                if "C2M0080120D" in file_ref:
                    # We already tried global replace, so this shouldn't trigger unless weird spacing
                    new_val_str = f'file:{thermal_abs_path}'
                    # Ensure it has .xml if missing? The abs path has .xml
                    try:
                        # Return the full string with absolute path
                        # We use match.group(0) to get full string, then replace the relative part
                        return match.group(0).replace(file_ref, thermal_abs_path)
                    except:
                        pass
                else: 
                     # Generic upgrade for other relative paths if any
                     # But don't break things unless we know where they are.
                     pass
            return match.group(0)

        # Replace thermal references
        content = re.sub(r'Value\s+"file:([^"]+)"', replace_thermal, content)
        
        # ALSO: Replace plain "C2M0080120D" in 'Value' context
        # Some components use Value "C2M0080120D" without file: prefix?
        # Pattern: Value "C2M0080120D"
        content = re.sub(r'Value\s+"C2M0080120D"', f'Value         "file:{thermal_abs_path}"', content) 

        # Replace the problematic relative path if hardcoded in file
        if 'file:core/simulation/flyback/C2M0080120D' in content:
            # We must verify thermal_val is defined. It was defined earlier but let's be safe.
            if 'thermal_val' not in locals():
                thermal_val = f"file:{thermal_abs_path}"
            content = content.replace('file:core/simulation/flyback/C2M0080120D', thermal_val)
            print(f"DEBUG: Fixed Hardcoded Relative Thermal Path to: {thermal_val}", file=sys.stderr)
        # Also handle if it was just specific filename but maybe wrapped differently
        # But safest is absolute path replacement if we find the pattern.
        
        # Control-mode selection.
        # SwitchState=2 is the model's original PI/PWM closed-loop path.
        # SwitchState=1 selects the standalone Pulse Generator open-loop path.
        control_mode = str(
            params.get("control_mode")
            or params.get("ControlMode")
            or params.get("TargetLoops")
            or p.get("control_mode")
            or p.get("ControlMode")
            or p.get("TargetLoops")
            or "closed_loop"
        ).strip().lower().replace("-", "_")
        closed_loop = control_mode in {"closed", "closed_loop", "pi", "pwm", "feedback"}
        
        # FIX 5: Use OPEN LOOP to debug physics, BUT calculate correct Duty Cycle.
        # D = Vout / (Vin/n + Vout)
        try:
            v_ref = float(p.get('Vref', 12))
            v_in = float(p.get('Vin', 265))
            n_ratio = float(p.get('n', 2.0))
            
            # Theoretical CCM Duty Cycle
            d_open = v_ref / ((v_in / n_ratio) + v_ref)
            
            # Add small margin for losses?
            d_open = d_open * 1.05
            
            # Safety Clamp
            if d_open > 0.45: d_open = 0.45
            if d_open < 0.05: d_open = 0.05
            
            print(f"DEBUG: Calculated Open Loop Duty Cycle: {d_open:.3f} (Target Vout={v_ref}V)", file=sys.stderr)
        except:
            d_open = 0.2 # Safe default

        switch_val = "2" if closed_loop else "1"
        if closed_loop:
            print("DEBUG: Preserving CLOSED LOOP control path (SwitchState=2).", file=sys.stderr)
        else:
            print(f"DEBUG: Using OPEN LOOP control path (SwitchState=1) with D={d_open:.3f}", file=sys.stderr)
        
        # Keep the PWM carrier and pulse generator frequency tied to fs.
        content = re.sub(r'(Variable\s+"f"\s+Value\s+)"[^"]*"', f'\\1"fs"', content)
        if not closed_loop:
            content = re.sub(r'(Variable\s+"DutyCycle"\s+Value\s+)"[^"]*"', f'\\1"{d_open:.4f}"', content) 
        
        # Force Switch State
        content = re.sub(r'(Variable\s+"SwitchState"\s+Value\s+)"[^"]*"', f'\\1"{switch_val}"', content)
        
        # Set Initial Current to small non-zero to jumpstart? No, Vdc is source.

        with open(run_model_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
    except Exception as e:
        print(f"DEBUG: Failed to bake model: {e}", file=sys.stderr)
        return {}


    # 3. Exec Simulation
    try:
        rpc_url = _plecs_rpc_url()
        if not rpc_url:
            raise RuntimeError("PLECS RPC URL is not configured. Set PLECS_RPC_URL locally.")
        server = xmlrpc.client.Server(rpc_url)

        # Check connection - plecs.ping might not exist, use listMethods or just try
        connected = False
        try:
            server.system.listMethods()
            connected = True
        except Exception as e:
            # if we get a response (even error like method not found), it is running.
            # only start if connection refused.
            if "refused" in str(e).lower():
                connected = False
            else:
                connected = True

        if not connected:
            print("DEBUG: Launching PLECS...", file=sys.stderr)
            subprocess.call(["open", "/Applications/PLECS 5.0.app"])
            time.sleep(15)
            server = xmlrpc.client.Server(rpc_url)
        
        try: server.plecs.close(run_model_name)
        except: pass
        
        if os.path.exists(waveforms_file):
            try: os.remove(waveforms_file)
            except: pass
        if os.path.exists(waveform_extras_file):
            try: os.remove(waveform_extras_file)
            except: pass
        run_model_path = str(run_model_file)
        original_model_path = str(original_model)

        print(f"DEBUG: Loading model {run_model_path}...", file=sys.stderr)
        server.plecs.load(run_model_path)
        
        # CLEAR console to avoid reading old errors
        try: server.plecs.clearConsole()
        except: pass

        if os.path.exists(csv_file):
            try: os.remove(csv_file)
            except: pass
            
        # We rely on the baked file for parameters, so we do NOT pass ModelVars 
        # to avoid whatever XML-RPC serialization issues were causing failures.
        opts = {
            # 'ModelVars': p,  <-- REMOVED
            'StartTime': 0.0,
            'StopTime': float(p.get('Tstop', 0.1)),
            'TimeOut': 20
        }
        
        print(f"DEBUG: Simulating {run_model_name} (Baked Params)...", file=sys.stderr)
        print(f"DEBUG: CSV Path: {csv_file}", file=sys.stderr)

        start_time = time.time()
        
        # Ensure we catch ANY exception during simulate call
        res = {}
        try:
            res = server.plecs.simulate(run_model_name, opts)
        except Exception as sim_err:
             print(f"DEBUG: Simulation Call Error: {sim_err}", file=sys.stderr)

        # Retrieve Console Output (if supported) to catch silent warnings/errors
        console_log = ""
        try:
            console_log = server.plecs.getConsoleOutput()
            if console_log and len(console_log.strip()) > 0:
                print(f"DEBUG: PLECS Console Output:\n{console_log}", file=sys.stderr)
        except:
            pass

        dur = time.time() - start_time
        print(f"DEBUG: Simulation returned: {res} in {dur:.2f}s", file=sys.stderr)
        
        # 4. Check CSV
        found = False
        target_csv = csv_file
        
        # Check primary CSV
        for i in range(20): 
            if os.path.exists(target_csv) and os.path.getsize(target_csv) > 100:
                found = True
                break
            time.sleep(0.5)
            
        # FALLBACK: If baked simulation failed, try running ORIGINAL model with ModelVars
        if not found and (not res.get('Time') or len(res.get('Time')) == 0):
             print(f"⚠️  DEBUG: Simulation failed to produce valid CSV ({target_csv}). Attempting Direct Simulation...", file=sys.stderr)
             if console_log:
                 print(f"⚠️  Last Console Log: {console_log}", file=sys.stderr)
             
             try:
                 # Load original model
                 server.plecs.load(original_model_path)
                 direct_model_name = "Flyback_effi"
                 
                 # Prepare opts with ModelVars
                 opts_direct = {
                     'ModelVars': p,
                     'StartTime': 0.0,
                     'StopTime': float(p.get('Tstop', 0.1)),
                     'TimeOut': 20
                 }
                 
                 # Run
                 res = server.plecs.simulate(direct_model_name, opts_direct)
                 print(f"DEBUG: Direct Simulation returned: {res}", file=sys.stderr)
                 
                 # ------------------------------------------------------------------
                 # FINAL FALLBACK: Extract data from SCOPE if CSV failed.
                 # The user says "I see waveforms", so Scopes are working.
                 # We can use xml-rpc to get scope data directly.
                 # ------------------------------------------------------------------
                 if not found:
                     print("DEBUG: CSV Extraction Failed. Attempting to read Scope 'Key Waveforms'...", file=sys.stderr)
                     print(f"DEBUG: Skipping Scope Extraction (Method Unknown). Trying to fix CSV path blindly...", file=sys.stderr)
                     pass

                 # Check if we can find WHERE it wrote.
                 # Let's try to find ANY csv in that folder or temp
                 fallback_csv = os.path.join(simulation_dir, "plecs_result.csv")
                 temp_fallback = simulation_dir / "plecs_result.csv"
                 
                 found_csv = None
                 runtime_waveform_fallback = simulation_dir / "pe_gpt_flyback.csv"
                 if runtime_waveform_fallback.exists() and runtime_waveform_fallback.stat().st_size > 100:
                      found_csv = str(runtime_waveform_fallback)
                 
                 if not found_csv:
                     for chk_path in [fallback_csv, temp_fallback, csv_file]:
                         if os.path.exists(chk_path) and os.path.getsize(chk_path) > 100:
                             found_csv = chk_path
                             break
                 
                 if found_csv:
                     found = True
                     target_csv = found_csv
                     print(f"DEBUG: Found Fallback CSV at {target_csv}", file=sys.stderr)
                 else:
                     # FINAL RESORT: Hardcoded Dummy Data based on Simulation Success
                     # If the user sees waveforms, the efficiency is likely > 0.
                     # We can't let the Agent stop here.
                     # We infer "Simulation Success" from the fact that direct simulation didn't throw exception.
                     print("⚠️ DEBUG: Simulation ran but CSV extraction failed; no heuristic metrics returned.", file=sys.stderr)
                     return {
                        "is_converged": False,
                        "raw_data": {
                            "Note": "Simulation ran but metric extraction failed.",
                            "control_mode": control_mode,
                        }
                    }
             except Exception as fallback_err:
                 print(f"DEBUG: Direct Simulation Failed: {fallback_err}", file=sys.stderr)

        if found:
            print(f"DEBUG: Reading CSV {target_csv}...", file=sys.stderr)
            try:
                data = np.genfromtxt(target_csv, delimiter=',')
                print(f"DEBUG: CSV Shape: {data.shape}", file=sys.stderr)
                
                # FIX: Handle single-row result (Steady State or Final Value)
                if data.ndim == 1:
                    print("DEBUG: Single-row CSV detected. Reshaping to (1, N).", file=sys.stderr)
                    data = data.reshape(1, -1)

                if data.ndim > 1:
                    # Print first and last rows
                    print(f"DEBUG: First 5 rows:\n{data[:5]}", file=sys.stderr)
                    
                    cols = data.shape[1]
                    
                    # --- NEW PARSING FOR FLYBACK_EFFI.PLECS ---
                # Expected columns:
                # Col 0: Time
                # Col 1: Ps_cond
                # Col 2: Ps_sw
                # Col 3: Pd_cond
                # Col 4: Pcu_pri
                # Col 5: Pcu_sec
                # Col 6: Pfe
                # Col 7: Total Loss (Sum1)
                # Col 8: Pout (Periodic Average5)
                
                # Note: np.genfromtxt might result in standard floats.
                # PLECS CSV usually has header so data[0] might be NaN if using genfromtxt without skip_header.
                # But previous code didn't skip header, implying PLECS ToFile might not write header?
                # Usually ToFile writes header "Time", "Signal 1", etc.
                # If first row is NaN, skip it.
                if np.isnan(data[0,0]):
                    data = data[1:]
                
                cutoff = int(len(data) * 0.5) # Steady state
                
                # Robust Column Extraction
                if cols >= 9:
                    total_loss_col = data[:, 7]
                    pout_col = data[:, 8]
                    
                    # Calculate Averages in Steady State
                    p_loss_avg = np.mean(total_loss_col[cutoff:])
                    p_out_avg = np.mean(pout_col[cutoff:])
                    
                    # Calculate Efficiency
                    # Eff = Pout / (Pout + Ploss)
                    # Use absolute values just in case signs are flipped (though usually Power is positive)
                    p_loss_avg = abs(p_loss_avg)
                    p_out_avg = abs(p_out_avg)
                    
                    if (p_out_avg + p_loss_avg) > 0:
                        eff = p_out_avg / (p_out_avg + p_loss_avg)
                    else:
                        eff = 0.0
                        
                    print(f"DEBUG: Pout={p_out_avg:.2f}W, Ploss={p_loss_avg:.2f}W, Calc Eff={eff:.2%}", file=sys.stderr)
                    
                    # Detailed Breakdown
                    breakdown = {
                        "Ps_cond": float(np.mean(data[cutoff:, 1])),
                        "Ps_sw": float(np.mean(data[cutoff:, 2])),
                        "Pd_cond": float(np.mean(data[cutoff:, 3])),
                        "Pcu_pri": float(np.mean(data[cutoff:, 4])),
                        "Pcu_sec": float(np.mean(data[cutoff:, 5])),
                        "Pfe": float(np.mean(data[cutoff:, 6]))
                    }
                    print(f"DEBUG: Loss Breakdown: {breakdown}", file=sys.stderr)
                    
                    # Vout Ripple Estimation?
                    # This model doesn't output Vout directly in CSV anymore.
                    # Use Pout and Ro to estimate Vout? P = V^2/R => V = sqrt(P*R)
                    ro = float(p.get('Ro', 10.0))
                    v_out_est = np.sqrt(p_out_avg * ro)
                    
                    # Prefer ripple from the captured Vout waveform when available.
                    v_ripple = 0.05
                    ripple_method = "placeholder_no_vout_waveform"
                    waveform_values = []

                    if os.path.exists(waveforms_file) and os.path.getsize(waveforms_file) > 100:
                        try:
                            wdata = np.genfromtxt(waveforms_file, delimiter=',')
                            print(f"DEBUG: Reading Waveforms CSV: {wdata.shape}", file=sys.stderr)
                            # Expected Cols: Time, IL, Id, IDiode, Vout
                            # If Vout is Col 4 (0-index), let's check
                            if wdata.ndim > 1 and wdata.shape[1] >= 5:
                                if np.isnan(wdata[0, 0]):
                                    wdata = wdata[1:]
                                # Resample or take slice
                                # PLECS steps are small (1e-7), file might be huge.
                                # Take last period or just decimate
                                w_cutoff = max(0, len(wdata) - 2000) # Last 2000 points
                                vout_trace = wdata[w_cutoff:, 4] # Assuming 5th column is Vout
                                finite_vout = vout_trace[np.isfinite(vout_trace)]
                                waveform_values = finite_vout.tolist()
                                if finite_vout.size >= 2:
                                    v_ripple = float(np.max(finite_vout) - np.min(finite_vout))
                                    ripple_method = "waveform_peak_to_peak"
                                elif finite_vout.size == 1:
                                    v_ripple = 0.0
                                    ripple_method = "waveform_single_point"
                                print(
                                    f"DEBUG: Extracted {len(waveform_values)} points from Waveforms CSV. "
                                    f"Ripple={v_ripple:.6f}V ({ripple_method})",
                                    file=sys.stderr,
                                )
                        except Exception as w_err:
                            print(f"DEBUG: Waveforms CSV Read Error: {w_err}", file=sys.stderr)
                    
                    # Vds Estimate from Input + Reflected
                    n_ratio = float(p.get('n', 2.0))
                    vor = (v_out_est + 0.7) * n_ratio
                    vds_est = float(p['Vin']) + vor + 50.0 # Spike margin
                    
                    # Create artificial waveforms for visualization if real ones missing
                    if not waveform_values:
                        waveform_values = pout_col.tolist() if len(pout_col) < 1000 else pout_col[::10].tolist()
                    if len(waveform_values) == 1:
                        # Create a dummy flat line for visualization
                        waveform_values = [float(waveform_values[0])] * 100
                    
                    result_dict = {
                        "is_converged": True,
                        "raw_data": {
                            "Efficiency": float(eff),
                            "Pout": float(p_out_avg),
                            "Ploss": float(p_loss_avg),
                            "LossBreakdown": breakdown,
                            "Vout_Ripple": float(v_ripple), 
                            "Ripple_Method": ripple_method,
                            "Vds_Max": float(vds_est),
                            "Vds_Method": "Estimated from Metrics",
                            # Return Pout waveform for plotting
                            "Values": waveform_values,
                            "waveforms_absolute_path": waveforms_file,
                            "waveform_extras_absolute_path": waveform_extras_file,
                        },
                        "control_mode": "closed_loop" if closed_loop else "open_loop",
                        "control_switch_state": int(switch_val),
                    }
                    
                    # [NEW] Try to fetch real waveforms from Scope via XML-RPC if available
                    # Since we have a valid result, we can check if 'res' has scope data
                    if res and 'Values' in res and len(res['Values']) > 0:
                        print("DEBUG: Enhancing CSV Metrics with Scope Waveforms from XML-RPC", file=sys.stderr)
                        # Assume Scope 1 is usually Vout or relevant
                        # res['Values'] is a list of arrays corresponding to scope signals.
                        # We don't know the order, but we can return raw traces.
                        result_dict['raw_data']['Scope_Traces'] = res['Values']
                        # Try to find one that looks like Vout (avg ~ 5V or similar)
                        # or just pass it all.
                        
                    return result_dict


            except Exception as parse_err:
                 print(f"DEBUG: CSV Parsing Failed: {parse_err}", file=sys.stderr)

        print("DEBUG: CSV not found or empty.", file=sys.stderr)
        return {}

    except Exception as e:
        print(f"DEBUG: Loop Exception: {e}", file=sys.stderr)
        return {}

def inject_params_into_plecs_file(model_path, params):
    pass
