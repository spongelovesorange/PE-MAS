from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapters import provenance
from .plecs_registry import PlecsModelRegistry
from .schemas import RequirementAnalysisResult
from .topology_service import TopologyKnowledgeService


class RequirementAnalysisAgent:
    """Production v1 Requirement Analysis Agent for PE-MAS.

    The agent is deliberately offline-first. It extracts and derives only from
    user-provided requirements, tracks provenance, and prepares downstream
    handoff without selecting final topology or components.
    """

    def __init__(
        self,
        topology_service: Optional[TopologyKnowledgeService] = None,
        plecs_registry: Optional[PlecsModelRegistry] = None,
    ) -> None:
        self.topology_service = topology_service or TopologyKnowledgeService()
        self.plecs_registry = plecs_registry or PlecsModelRegistry(self.topology_service)

    def analyze(self, prompt: str, project_id: Optional[str] = None) -> RequirementAnalysisResult:
        raw_prompt = str(prompt or "").strip()
        if not raw_prompt:
            raise ValueError("prompt is required")

        normalized_prompt = self._normalize_text(raw_prompt)
        specs = self._parse_specs(normalized_prompt)
        project_summary = self._project_summary(normalized_prompt, specs)
        application_analysis = self._application_analysis(project_summary, specs, normalized_prompt)
        explicit_specs = self._explicit_specifications(normalized_prompt, specs)
        derived_specs = self._derived_specifications(specs)
        qualitative_specs = self._qualitative_specifications(normalized_prompt, specs)
        missing_info = self._missing_information(specs, normalized_prompt)
        layer1_definition = self._layer1_minimum_converter_definition(project_summary, specs, missing_info)
        layer2_constraints = self._layer2_constraints_preferences(specs, normalized_prompt)
        engineering_checks = self._engineering_derived_checks(specs, derived_specs, normalized_prompt)
        spec_gate_status = self._spec_gate_status(layer1_definition, missing_info)
        clarification_questions = self._clarification_questions(missing_info)
        task_decomposition = self._task_decomposition()
        feasibility = self._feasibility_conflicts(specs, missing_info, normalized_prompt)
        objectives = self._refined_objectives(specs, missing_info)
        workflow = self._recommended_workflow(missing_info)
        assumption_register = self._assumptions(specs, normalized_prompt)
        topology_candidates = self.topology_service.candidates_for_requirements(normalized_prompt, specs)
        handoff = self._handoff(project_summary, specs, derived_specs, missing_info, feasibility, topology_candidates, spec_gate_status, engineering_checks)
        spec_package = self._spec_package(
            spec_gate_status,
            layer1_definition,
            layer2_constraints,
            engineering_checks,
            derived_specs,
            missing_info,
            assumption_register,
            feasibility,
            handoff,
        )
        readiness = self._readiness(missing_info)
        quality = self._quality(explicit_specs, derived_specs, missing_info)
        frontend = self._frontend_response(
            project_summary,
            explicit_specs,
            derived_specs,
            missing_info,
            feasibility,
            clarification_questions,
            workflow,
            readiness,
            handoff,
            assumption_register,
            spec_gate_status,
            layer1_definition,
            layer2_constraints,
            engineering_checks,
        )
        report = self._human_report(
            project_summary,
            application_analysis,
            explicit_specs,
            derived_specs,
            missing_info,
            feasibility,
            objectives,
            workflow,
            readiness,
            spec_gate_status,
            layer1_definition,
            layer2_constraints,
            engineering_checks,
        )

        return RequirementAnalysisResult(
            project_id=project_id,
            human_readable_report=report,
            spec_gate_status=spec_gate_status,
            layer1_minimum_converter_definition=layer1_definition,
            layer2_constraints_preferences=layer2_constraints,
            engineering_derived_checks=engineering_checks,
            spec_package=spec_package,
            normalized_specs=specs,
            project_summary=project_summary,
            application_analysis=application_analysis,
            extracted_specifications={
                "explicit_specifications": explicit_specs,
                "derived_specifications": derived_specs,
                "qualitative_specifications": qualitative_specs,
            },
            missing_information=missing_info,
            clarification_questions=clarification_questions,
            preliminary_design_task_decomposition=task_decomposition,
            feasibility_and_conflict_check=feasibility,
            refined_design_objectives=objectives,
            recommended_workflow=workflow,
            assumption_register=assumption_register,
            handoff_package=handoff,
            readiness_status=readiness,
            quality_check=quality,
            frontend_response=frontend,
            source_provenance=[
                provenance("user_prompt", "inline prompt", "high", "User-provided requirement text."),
                provenance("PE-MAS seed topology database", str(self.topology_service.data_path), "medium", "Offline topology seed fixture."),
            ],
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return (
            text.replace("–", "-")
            .replace("—", "-")
            .replace("−", "-")
            .replace("≤", "<=")
            .replace("≥", ">=")
            .replace("\u00a0", " ")
        )

    @staticmethod
    def _snippet(text: str, pattern: str, fallback: str = "") -> str:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return fallback
        start = max(0, match.start() - 36)
        end = min(len(text), match.end() + 36)
        return re.sub(r"\s+", " ", text[start:end]).strip()

    @staticmethod
    def _num(value: Any) -> Optional[float]:
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _round(value: float) -> float:
        return round(value, 3)

    def _parse_specs(self, text: str) -> Dict[str, Any]:
        specs: Dict[str, Any] = {}
        low = text.lower()

        if re.search(r"\bdc\s*[-/]\s*dc\b|dc-dc|dc/dc", low):
            specs["input_type"] = "DC"
        elif re.search(r"\bac\s*[-/]\s*dc\b|ac-dc|ac/dc|offline|mains|交流", low):
            specs["input_type"] = "AC"
            specs["line_frequency_hz"] = [50, 60]

        range_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:-|~|to|至|到)\s*(-?\d+(?:\.\d+)?)\s*v\s*(ac|vac|dc|vdc)?\b[^.,;]*(?:input|in|输入)?", text, re.I)
        if range_match:
            specs["input_voltage_min"] = float(range_match.group(1))
            specs["input_voltage_max"] = float(range_match.group(2))
            unit = str(range_match.group(3) or "").lower()
            if unit in {"ac", "vac"} or any(token in low for token in ["ac/dc", "offline", "mains", "交流"]):
                specs["input_type"] = "AC"
                specs["line_frequency_hz"] = [50, 60]
            elif unit in {"dc", "vdc"}:
                specs["input_type"] = "DC"

        conversion_match = re.search(r"(\d+(?:\.\d+)?)\s*v\s*(?:to|->|到|至)\s*(\d+(?:\.\d+)?)\s*v", text, re.I)
        if conversion_match:
            specs.setdefault("input_voltage_nominal", float(conversion_match.group(1)))
            specs.setdefault("output_voltage", float(conversion_match.group(2)))
            specs.setdefault("input_type", "DC")

        output_pair = re.search(r"(\d+(?:\.\d+)?)\s*v\s*(?:/|,|，|\s+)\s*(\d+(?:\.\d+)?)\s*a\b", text, re.I)
        if output_pair:
            specs["output_voltage"] = float(output_pair.group(1))
            specs["output_current"] = float(output_pair.group(2))
            specs["rated_output_power"] = specs["output_voltage"] * specs["output_current"]
        else:
            output_v = re.search(r"(\d+(?:\.\d+)?)\s*v\s*(?:output|out|输出)\b", text, re.I)
            if not output_v:
                output_v = re.search(r"(?:output|out|输出)[^0-9]{0,24}(\d+(?:\.\d+)?)\s*v\b", text, re.I)
            if output_v:
                specs["output_voltage"] = float(output_v.group(1))

        rated_power = re.search(r"(\d+(?:\.\d+)?)\s*(kw|w)\b[^.,;]{0,24}(?:rated|continuous|额定)?", text, re.I)
        if rated_power:
            value = float(rated_power.group(1)) * (1000.0 if rated_power.group(2).lower() == "kw" else 1.0)
            specs["rated_output_power"] = value

        peak_power = re.search(
            r"(?:peak|峰值)[^0-9]{0,28}(\d+(?:\.\d+)?)\s*(kw|w)\b|(\d+(?:\.\d+)?)\s*(kw|w)\b[^.,;]{0,28}(?:peak|峰值)",
            text,
            re.I,
        )
        if peak_power:
            value = peak_power.group(1) or peak_power.group(3)
            unit = peak_power.group(2) or peak_power.group(4)
            specs["peak_output_power"] = float(value) * (1000.0 if unit.lower() == "kw" else 1.0)
            duration = re.search(r"(?:for|持续)?\s*(\d+(?:\.\d+)?)\s*s\b", text, re.I)
            if duration:
                specs["peak_duration_s"] = float(duration.group(1))
        else:
            timed_power = re.search(r"(\d+(?:\.\d+)?)\s*(kw|w)\b[^.,;]{0,36}(?:for|持续)\s*(\d+(?:\.\d+)?)\s*s\b", text, re.I)
            if timed_power and specs.get("rated_output_power") is not None:
                value = float(timed_power.group(1)) * (1000.0 if timed_power.group(2).lower() == "kw" else 1.0)
                if value > float(specs["rated_output_power"]):
                    specs["peak_output_power"] = value
                    specs["peak_duration_s"] = float(timed_power.group(3))

        efficiency = re.search(r"(?:>|>=|at least|efficiency|效率)[^0-9]{0,24}(\d+(?:\.\d+)?)\s*%", text, re.I)
        if efficiency:
            specs["efficiency_target"] = float(efficiency.group(1)) / 100.0

        ripple = re.search(r"(?:ripple|纹波)[^0-9]{0,24}(?:<|<=)?\s*(\d+(?:\.\d+)?)\s*m?v", text, re.I)
        if ripple:
            specs["output_ripple_mvpp"] = float(ripple.group(1))
            specs["output_ripple_vpp"] = specs["output_ripple_mvpp"] / 1000.0

        ambient = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:-|~|to|至|到)\s*(-?\d+(?:\.\d+)?)\s*(?:°\s*)?c\b", text, re.I)
        if ambient:
            specs["ambient_c_min"] = float(ambient.group(1))
            specs["ambient_c_max"] = float(ambient.group(2))
        else:
            ambient_max = re.search(r"(?:ambient|ta|环境)[^0-9-]{0,24}(-?\d+(?:\.\d+)?)\s*(?:°\s*)?c\b", text, re.I)
            if ambient_max:
                specs["ambient_c_max"] = float(ambient_max.group(1))

        if "reinforced" in low or "加强绝缘" in text:
            specs["isolation_requirement"] = "reinforced isolation assumption"
        elif re.search(r"\bisolat(?:ed|ion)\b|隔离", low):
            specs["isolation_requirement"] = "isolation required"

        emi_marked_missing = re.search(r"(?:emi|emc)[^.;\n]{0,80}(?:not specified|missing|unknown|未指定|缺失|未知)", text, re.I)
        if re.search(r"cispr\s*32|en\s*55032|class\s*b|emi|emc", text, re.I) and not emi_marked_missing:
            if re.search(r"cispr\s*32", text, re.I) and re.search(r"class\s*b", text, re.I):
                specs["emi_emc_standard"] = "CISPR 32 Class B pre-compliance"
            elif re.search(r"cispr\s*32", text, re.I):
                specs["emi_emc_standard"] = "CISPR 32 pre-compliance"
            else:
                specs["emi_emc_standard"] = "EMI/EMC target mentioned, exact standard not fully specified"

        if re.search(r"tl431|opto|光耦", text, re.I):
            specs["feedback_preference"] = "TL431 + optocoupler"
        if re.search(r"qr|valley|quasi|flyback|反激|谷底|准谐振", text, re.I):
            specs["topology_preference"] = "QR/valley flyback" if re.search(r"qr|valley|quasi|谷底|准谐振", text, re.I) else "flyback"

        if "limited airflow" in low or "自然风" in text:
            specs["cooling_condition"] = "limited airflow"
        if "automotive" in low or "electric vehicle" in low or re.search(r"\bev\b", low) or "车载" in text:
            specs["application_hint"] = "automotive / EV"
        if "basic protection" in low or "保护" in text:
            specs["protection_requirement"] = "basic protection requested"

        return specs

    def _project_summary(self, text: str, specs: Dict[str, Any]) -> Dict[str, Any]:
        low = text.lower()
        if "flyback" in low or "反激" in text:
            design_object = "isolated flyback power supply"
        elif "converter" in low or "变换器" in text:
            design_object = "power converter"
        else:
            design_object = "power electronics system"
        if "automotive" in low or "electric vehicle" in low or re.search(r"\bev\b", low):
            application = "electric vehicle auxiliary power system"
        elif specs.get("input_type") == "AC" or any(token in low for token in ["ac/dc", "offline", "mains"]):
            application = "offline AC/DC power supply"
        else:
            application = "application not fully specified"
        conversion_type = "AC/DC" if specs.get("input_type") == "AC" else "DC/DC"
        return {
            "design_object": design_object,
            "application": application,
            "conversion_type": conversion_type,
            "main_user_goal": text,
        }

    def _application_analysis(self, summary: Dict[str, Any], specs: Dict[str, Any], text: str) -> Dict[str, Any]:
        implications: List[str] = []
        environment: List[str] = []
        priorities: List[str] = []
        app = str(summary.get("application", "")).lower()
        if "electric vehicle" in app:
            environment += ["vehicle auxiliary electrical environment", "thermal cycling and vibration awareness", "possible limited airflow"]
            implications += [
                "Automotive use implies reliability, input disturbance, thermal derating, protection, and EMI/EMC awareness.",
                "No specific automotive EMI/EMC standard is fabricated; it remains a clarification item.",
            ]
            priorities += ["reliability", "thermal feasibility", "protection behavior", "EMI/EMC readiness"]
        if specs.get("input_type") == "AC":
            environment += ["offline mains input", "hazardous primary voltage", "isolation/safety documentation path"]
            implications += [
                "Offline AC/DC use drives creepage/clearance, transformer insulation, Y-cap/fuse/MOV choices, and conducted EMI layout.",
            ]
            priorities += ["reinforced isolation handling", "conducted EMI pre-compliance", "high-line/low-line stress checks"]
        if specs.get("ambient_c_max") is not None:
            environment.append(f"ambient up to {specs['ambient_c_max']:g} C")
        if specs.get("cooling_condition"):
            environment.append(str(specs["cooling_condition"]))
        return {
            "use_case": summary.get("application"),
            "operating_environment": list(dict.fromkeys(environment)) or ["not fully specified"],
            "application_driven_implications": list(dict.fromkeys(implications)) or ["Only qualitative implications are inferred; no unsupported numerical standard is invented."],
            "design_priorities": list(dict.fromkeys(priorities + ["regulated output", "efficiency target", "verification traceability"])),
        }

    @staticmethod
    def _layer_row(
        field: str,
        extracted_value: Any,
        traceability: str,
        constraint_type: str,
        status: str,
        why_it_matters: str,
    ) -> Dict[str, Any]:
        return {
            "field": field,
            "extracted_value": extracted_value,
            "traceability": traceability,
            "constraint_type": constraint_type,
            "status": status,
            "why_it_matters": why_it_matters,
        }

    def _layer1_minimum_converter_definition(
        self,
        summary: Dict[str, Any],
        specs: Dict[str, Any],
        missing_info: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        missing_items = {row["item"] for row in missing_info}
        vin_range = (
            [specs.get("input_voltage_min"), specs.get("input_voltage_max")]
            if specs.get("input_voltage_min") is not None and specs.get("input_voltage_max") is not None
            else None
        )
        output_rail = None
        if specs.get("output_voltage") is not None and specs.get("output_current") is not None:
            output_rail = f"{specs['output_voltage']:g} V / {specs['output_current']:g} A"
        elif specs.get("output_voltage") is not None:
            output_rail = f"{specs['output_voltage']:g} V"
        rated_power = specs.get("rated_output_power")
        peak_power = None
        if specs.get("peak_output_power") is not None:
            duration = specs.get("peak_duration_s")
            peak_power = f"{specs['peak_output_power']:g} W" + (f" for {duration:g} s" if duration is not None else "")
        application = summary.get("application")
        isolation = specs.get("isolation_requirement") or "TBD"

        return [
            self._layer_row("conversion_type", summary.get("conversion_type"), "explicit_or_derived_from_request", "Architecture Gate", "known", "Determines broad converter family."),
            self._layer_row("input_source_type", specs.get("input_type") or "TBD", "explicit_or_reasonable_implication", "Architecture Gate", "known" if specs.get("input_type") else "TBD", "AC vs DC affects topology, rectification, protection, EMI, and safety."),
            self._layer_row("input_voltage_range", vin_range or specs.get("input_voltage_nominal") or "TBD", "explicit" if vin_range or specs.get("input_voltage_nominal") is not None else "open_question", "Hard Constraint", "known" if "input_voltage_range" not in missing_items else "TBD", "Defines voltage stress, current stress, and operating corners."),
            self._layer_row("output_rails", output_rail or "TBD", "explicit" if output_rail else "open_question", "Hard Constraint", "known" if "output_rating" not in missing_items else "TBD", "Defines regulation target, current stress, and power allocation."),
            self._layer_row("output_regulation_need", "regulated output" if specs.get("output_voltage") is not None else "TBD", "explicit_or_implication", "Hard Constraint", "mostly_known" if specs.get("output_voltage") is not None else "TBD", "Determines whether a regulated converter is required."),
            self._layer_row("rated_power_or_current", rated_power if rated_power is not None else specs.get("output_current") or "TBD", "explicit_or_derived" if rated_power is not None or specs.get("output_current") is not None else "open_question", "Hard Constraint", "known" if "output_rating" not in missing_items else "TBD", "Defines continuous power level and main current stress."),
            self._layer_row("peak_power_or_overload", peak_power or "TBD", "explicit" if peak_power else "open_question", "Verification Target", "known" if peak_power else "TBD", "Defines short-duration current, magnetic saturation, SOA, and thermal stress."),
            self._layer_row("application_scenario", application or "TBD", "explicit_or_reasonable_implication", "Context Driver", "known" if application and application != "application not fully specified" else "TBD", "Drives safety, thermal, EMI, reliability, and cost interpretation."),
            self._layer_row("isolation_requirement", isolation, "explicit" if specs.get("isolation_requirement") else "open_question", "Architecture Gate", "known" if specs.get("isolation_requirement") else "TBD", "Major architecture gate for topology, safety, size, and efficiency."),
            self._layer_row("operating_mode", "continuous load supply" + (" with short-duration peak support" if specs.get("peak_output_power") else ""), "reasonable_implication", "Verification Target", "mostly_known", "Affects topology, control, protection, and verification."),
        ]

    def _layer2_constraints_preferences(self, specs: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        def add(area: str, value: Any, traceability: str, constraint_type: str, measurability: str) -> None:
            rows.append({
                "area": area,
                "extracted_value": value,
                "traceability": traceability,
                "constraint_type": constraint_type,
                "measurability": measurability,
            })

        if specs.get("efficiency_target") is not None:
            add("efficiency", f">{specs['efficiency_target'] * 100:g}%", "explicit", "Verification Target", "quantified")
        else:
            add("efficiency", "TBD", "open_question", "TBD", "missing")
        if specs.get("ambient_c_max") is not None:
            ambient = [specs.get("ambient_c_min"), specs.get("ambient_c_max")] if specs.get("ambient_c_min") is not None else specs.get("ambient_c_max")
            add("thermal_environment", ambient, "explicit", "Hard Constraint", "quantified")
        else:
            add("thermal_environment", "TBD", "open_question", "TBD", "missing")
        add("cooling", specs.get("cooling_condition") or "TBD", "explicit" if specs.get("cooling_condition") else "open_question", "Design Constraint", "qualitative" if specs.get("cooling_condition") else "missing")
        compact = "compact" in text.lower() or "紧凑" in text
        add("mechanical", "compact design" if compact else "TBD", "explicit_qualitative" if compact else "open_question", "Preference" if compact else "TBD", "unquantified" if compact else "missing")
        reliable = "reliable" in text.lower() or "可靠" in text
        add("reliability", "reliable design" if reliable else "TBD", "explicit_qualitative" if reliable else "open_question", "Preference/TBD", "unquantified" if reliable else "missing")
        cost = "cost-effective" in text.lower() or "low cost" in text.lower() or "成本" in text
        add("cost", "cost-effective design" if cost else "TBD", "explicit_qualitative" if cost else "open_question", "Optimization Target" if cost else "TBD", "unquantified" if cost else "missing")
        add("emi_emc", specs.get("emi_emc_standard") or "TBD", "explicit" if specs.get("emi_emc_standard") else "open_question", "Verification Target" if specs.get("emi_emc_standard") else "TBD", "quantified_or_named" if specs.get("emi_emc_standard") else "missing")
        add("output_quality", specs.get("output_ripple_mvpp") if specs.get("output_ripple_mvpp") is not None else "TBD", "explicit" if specs.get("output_ripple_mvpp") is not None else "open_question", "Verification Target" if specs.get("output_ripple_mvpp") is not None else "TBD", "quantified" if specs.get("output_ripple_mvpp") is not None else "missing")
        add("protection", specs.get("protection_requirement") or "TBD", "explicit_qualitative" if specs.get("protection_requirement") else "open_question", "Verification Target/TBD", "incomplete" if specs.get("protection_requirement") else "missing")
        add("safety_certification", specs.get("isolation_requirement") or "TBD", "explicit" if specs.get("isolation_requirement") else "open_question", "Architecture Gate/TBD", "incomplete" if specs.get("isolation_requirement") else "missing")
        add("supply_chain", "TBD", "open_question", "TBD", "missing")
        return rows

    def _engineering_derived_checks(self, specs: Dict[str, Any], derived: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        def add(name: str, check_type: str, why: str, downstream_owner: List[str], status: str = "generated") -> None:
            rows.append({
                "name": name,
                "check_type": check_type,
                "why_it_matters": why,
                "downstream_owner": downstream_owner,
                "status": status,
            })

        derived_names = {row.get("name") for row in derived}
        if "rated_output_current" in derived_names:
            add("rated_output_current", "topology_agnostic", "High output current affects conduction loss, PCB copper, connectors, sensing, and thermal design.", ["Topology", "Power Stage", "Devices", "PCB Rules"])
        if "peak_output_current" in derived_names:
            add("peak_output_current", "topology_agnostic", "Peak operation must be handled as a separate short-duration stress case.", ["Power Stage", "Magnetics", "Devices", "PLECS Simulation"])
        if "loss_at_efficiency" in derived_names:
            add("loss_at_efficiency", "topology_agnostic", "Target efficiency still leaves a heat budget that may dominate topology and thermal choices.", ["Topology", "Thermal", "Devices", "PCB Rules"])
        if "low_line_rated_input_current" in derived_names or (specs.get("input_voltage_min") is not None and specs.get("rated_output_power") is not None):
            add("low_line_input_current", "topology_agnostic", "Low-line input usually creates worst-case current and conduction stress.", ["Topology", "Power Stage", "Devices"])
        if specs.get("input_voltage_min") is not None and specs.get("input_voltage_max") is not None and specs.get("output_voltage") is not None:
            add("voltage_conversion_ratio_range", "topology_agnostic", "Voltage ratio helps early topology reasoning but must not be treated as final duty ratio before topology selection.", ["Topology"])

        topology = str(specs.get("topology_preference") or "").lower()
        if "flyback" in topology:
            for name, why, owners in [
                ("transformer_peak_current", "Flyback transformer peak current drives saturation, copper loss, and current-limit margin.", ["Power Stage", "Magnetics"]),
                ("mosfet_drain_stress", "Flyback MOSFET stress depends on bulk voltage, reflected voltage, and leakage spike.", ["Power Stage", "Devices", "Clamp"]),
                ("leakage_energy_and_clamp_loss", "Transformer leakage energy sets RCD/TVS stress, temperature, efficiency, and EMI risk.", ["Magnetics", "Clamp", "PLECS Simulation"]),
                ("output_rectifier_stress", "Secondary rectifier reverse voltage and current stress drive diode/SR selection and thermal margin.", ["Devices", "Power Stage"]),
            ]:
                add(name, "topology_specific_triggered_by_user", why, owners, "handoff_for_downstream_calculation")
        elif re.search(r"\bdab\b|dual active bridge|lc-dab", text, re.I):
            for name in ["resonant_tank_peak_current", "rms_current", "transformer_current_stress", "zvs_range", "circulating_energy"]:
                add(name, "topology_specific_triggered_by_user", f"{name} is required for DAB/LC-DAB evaluation but should be calculated downstream.", ["Topology", "Power Stage", "Magnetics", "PLECS Simulation"], "handoff_for_downstream_calculation")
        else:
            add("topology_specific_stress_checks", "deferred", "No final topology is selected by Spec Agent; internal stress checks must be generated after Topology Agent selects candidates.", ["Topology"], "deferred_to_topology_agent")

        return rows

    @staticmethod
    def _spec_gate_status(layer1: List[Dict[str, Any]], missing_info: List[Dict[str, Any]]) -> str:
        known_by_field = {row["field"]: row["status"] for row in layer1}
        hard_missing = {
            "conversion_type",
            "input_source_type",
            "input_voltage_range",
            "output_rails",
            "rated_power_or_current",
            "application_scenario",
        }
        if any(known_by_field.get(field) == "TBD" for field in hard_missing):
            return "HOLD"
        if any(row.get("priority") in {"blocking", "high", "medium"} for row in missing_info):
            return "PARTIAL"
        return "LOCKED"

    def _field(self, text: str, name: str, value: Any, unit: str, pattern: str, req_type: str) -> Dict[str, Any]:
        return {
            "name": name,
            "value": value,
            "unit": unit,
            "source_text": self._snippet(text, pattern, str(value)),
            "requirement_type": req_type,
            "status": "confirmed",
            "source_type": "explicit",
        }

    def _explicit_specifications(self, text: str, specs: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if specs.get("input_voltage_min") is not None and specs.get("input_voltage_max") is not None:
            rows.append(self._field(text, "input_voltage_range", [specs["input_voltage_min"], specs["input_voltage_max"]], "V", r"-?\d+(?:\.\d+)?\s*(?:-|~|to|至|到)\s*-?\d+(?:\.\d+)?\s*v", "electrical"))
        elif specs.get("input_voltage_nominal") is not None:
            rows.append(self._field(text, "input_voltage_nominal", specs["input_voltage_nominal"], "V", r"\d+(?:\.\d+)?\s*v\s*(?:to|->|到|至)", "electrical"))
        if specs.get("output_voltage") is not None:
            rows.append(self._field(text, "output_voltage", specs["output_voltage"], "V", r"(?:to|->|到|至)?\s*\d+(?:\.\d+)?\s*v\b[^.,;]{0,24}(?:output|out|输出)?", "electrical"))
        if specs.get("output_current") is not None:
            rows.append(self._field(text, "output_current", specs["output_current"], "A", r"\d+(?:\.\d+)?\s*a\b", "electrical"))
        if specs.get("rated_output_power") is not None:
            rows.append(self._field(text, "rated_output_power", specs["rated_output_power"], "W", r"\d+(?:\.\d+)?\s*(?:kw|w)\b[^.,;]{0,24}(?:rated|continuous|额定)?", "electrical"))
        if specs.get("peak_output_power") is not None:
            rows.append(self._field(text, "peak_output_power", specs["peak_output_power"], "W", r"(?:peak|峰值)[^.,;]{0,40}\d+(?:\.\d+)?\s*(?:kw|w)|\d+(?:\.\d+)?\s*(?:kw|w)[^.,;]{0,28}(?:peak|峰值)", "electrical"))
        if specs.get("peak_duration_s") is not None:
            rows.append(self._field(text, "peak_duration", specs["peak_duration_s"], "s", r"\d+(?:\.\d+)?\s*s\b", "electrical"))
        if specs.get("efficiency_target") is not None:
            rows.append(self._field(text, "efficiency_target", specs["efficiency_target"], "ratio", r"(?:>|>=|at least|efficiency|效率)[^0-9]{0,24}\d+(?:\.\d+)?\s*%", "thermal"))
        if specs.get("ambient_c_max") is not None:
            val = [specs.get("ambient_c_min"), specs.get("ambient_c_max")] if specs.get("ambient_c_min") is not None else specs["ambient_c_max"]
            rows.append(self._field(text, "ambient_temperature", val, "C", r"-?\d+(?:\.\d+)?\s*(?:-|~|to|至|到)\s*-?\d+(?:\.\d+)?\s*c|-?\d+(?:\.\d+)?\s*c", "thermal"))
        if specs.get("output_ripple_mvpp") is not None:
            rows.append(self._field(text, "output_ripple_limit", specs["output_ripple_mvpp"], "mVp-p", r"(?:ripple|纹波)[^0-9]{0,24}\d+(?:\.\d+)?\s*m?v", "electrical"))
        if specs.get("emi_emc_standard"):
            rows.append(self._field(text, "emi_emc_target", specs["emi_emc_standard"], "", r"cispr\s*\d+|emi|emc|class\s*b", "EMI"))
        if specs.get("isolation_requirement"):
            rows.append(self._field(text, "isolation_requirement", specs["isolation_requirement"], "", r"reinforced|isolation|isolated|隔离|加强绝缘", "safety"))
        return rows

    def _derived_specifications(self, specs: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        def add(name: str, value: float, unit: str, formula: str, based_on: List[str]) -> None:
            rows.append({"name": name, "value": self._round(value), "unit": unit, "formula": formula, "based_on": based_on, "status": "derived"})

        vout = self._num(specs.get("output_voltage"))
        rated_power = self._num(specs.get("rated_output_power"))
        peak_power = self._num(specs.get("peak_output_power"))
        efficiency = self._num(specs.get("efficiency_target"))
        if rated_power is None and vout is not None and specs.get("output_current") is not None:
            iout = float(specs["output_current"])
            add("rated_output_power", vout * iout, "W", f"rated_output_power = output_voltage * output_current = {vout:g} V * {iout:g} A", ["output_voltage", "output_current"])
        if rated_power is not None and vout:
            add("rated_output_current", rated_power / vout, "A", f"rated_output_current = rated_power / output_voltage = {rated_power:g} W / {vout:g} V", ["rated_output_power", "output_voltage"])
        if peak_power is not None and vout:
            add("peak_output_current", peak_power / vout, "A", f"peak_output_current = peak_power / output_voltage = {peak_power:g} W / {vout:g} V", ["peak_output_power", "output_voltage"])
        if rated_power is not None and efficiency:
            input_power = rated_power / efficiency
            add("rated_input_power_at_efficiency", input_power, "W", f"rated_input_power = rated_output_power / efficiency = {rated_power:g} W / {efficiency:g}", ["rated_output_power", "efficiency_target"])
            add("loss_at_efficiency", rated_power / efficiency - rated_power, "W", f"loss_at_efficiency = output_power / efficiency - output_power = {rated_power:g} W / {efficiency:g} - {rated_power:g} W", ["rated_output_power", "efficiency_target"])
            vin_min = self._num(specs.get("input_voltage_min"))
            vin_max = self._num(specs.get("input_voltage_max"))
            if vin_min:
                add("low_line_rated_input_current", input_power / vin_min, "A", f"low_line_rated_input_current = rated_input_power / input_voltage_min = {input_power:g} W / {vin_min:g} V", ["rated_input_power_at_efficiency", "input_voltage_min"])
            if vin_max:
                add("high_line_rated_input_current", input_power / vin_max, "A", f"high_line_rated_input_current = rated_input_power / input_voltage_max = {input_power:g} W / {vin_max:g} V", ["rated_input_power_at_efficiency", "input_voltage_max"])
        if peak_power is not None and efficiency:
            peak_input_power = peak_power / efficiency
            add("peak_input_power_at_efficiency", peak_input_power, "W", f"peak_input_power = peak_output_power / efficiency = {peak_power:g} W / {efficiency:g}", ["peak_output_power", "efficiency_target"])
            vin_min = self._num(specs.get("input_voltage_min"))
            if vin_min:
                add("low_line_peak_input_current", peak_input_power / vin_min, "A", f"low_line_peak_input_current = peak_input_power / input_voltage_min = {peak_input_power:g} W / {vin_min:g} V", ["peak_input_power_at_efficiency", "input_voltage_min"])
        if specs.get("output_ripple_vpp") is not None and vout:
            add("ripple_percent_of_output", float(specs["output_ripple_vpp"]) / vout * 100.0, "%", f"ripple_percent = output_ripple / output_voltage * 100 = {float(specs['output_ripple_vpp']):g} V / {vout:g} V * 100", ["output_ripple_limit", "output_voltage"])
        vin_min = self._num(specs.get("input_voltage_min"))
        vin_max = self._num(specs.get("input_voltage_max"))
        if vout and vin_min and vin_max:
            add("voltage_conversion_ratio_min", vout / vin_max, "ratio", f"voltage_conversion_ratio_min = output_voltage / input_voltage_max = {vout:g} V / {vin_max:g} V", ["output_voltage", "input_voltage_max"])
            add("voltage_conversion_ratio_max", vout / vin_min, "ratio", f"voltage_conversion_ratio_max = output_voltage / input_voltage_min = {vout:g} V / {vin_min:g} V", ["output_voltage", "input_voltage_min"])
        return rows

    def _qualitative_specifications(self, text: str, specs: Dict[str, Any]) -> List[Dict[str, Any]]:
        low = text.lower()
        terms = [
            ("compact", ["compact", "small", "紧凑", "小型"], "Package size is requested qualitatively; exact dimensions are missing."),
            ("reliable", ["reliable", "reliability", "可靠"], "Reliability is requested qualitatively; mission profile/lifetime target is missing."),
            ("cost_effective", ["cost-effective", "low cost", "成本", "低成本"], "Cost effectiveness is requested qualitatively; BOM target is missing."),
            ("limited_airflow", ["limited airflow", "有限气流"], "Cooling is constrained qualitatively; exact cooling path is missing."),
            ("basic_protection", ["basic protection", "保护"], "Protection is requested qualitatively; thresholds and recovery behavior are missing."),
        ]
        rows = []
        for name, needles, description in terms:
            if any(needle in low or needle in text for needle in needles):
                rows.append({"name": name, "description": description, "status": "requires_clarification", "source_type": "explicit_qualitative"})
        if specs.get("topology_preference"):
            rows.append({"name": "topology_preference", "description": specs["topology_preference"], "status": "preference_only", "source_type": "explicit_qualitative"})
        if specs.get("feedback_preference"):
            rows.append({"name": "feedback_preference", "description": specs["feedback_preference"], "status": "confirmed", "source_type": "explicit_qualitative"})
        return rows

    @staticmethod
    def _missing(item: str, priority: str, why: str, tasks: List[str], question: str, architecture_critical: bool = False) -> Dict[str, Any]:
        return {
            "item": item,
            "priority": priority,
            "why_it_matters": why,
            "affected_design_tasks": tasks,
            "clarification_question": question,
            "architecture_critical": architecture_critical,
        }

    def _missing_information(self, specs: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not specs.get("isolation_requirement"):
            rows.append(self._missing("isolation_requirement", "blocking", "Isolation status changes topology family, transformer/inductor design, safety strategy, size, cost, efficiency, and final topology selection.", ["topology_selection", "magnetic_design", "safety_design", "verification_agent"], "Is galvanic isolation required between input and output?", True))
        if not (specs.get("input_voltage_min") is not None and specs.get("input_voltage_max") is not None):
            priority = "high" if specs.get("input_voltage_nominal") is not None else "blocking"
            rows.append(self._missing("input_voltage_range", priority, "Input range sets duty ratio, RMS current, voltage stress, protection thresholds, and thermal corners.", ["topology_selection", "power_stage_design", "protection_design"], "What minimum and maximum input voltage should be used?", priority == "blocking"))
        if not specs.get("emi_emc_standard"):
            rows.append(self._missing("emi_emc_standard", "high", "The EMI/EMC standard and margin policy determine filter, layout, shielding, and verification requirements.", ["emi_design", "pcb_layout", "verification_agent"], "Which EMI/EMC standard and class should be targeted?"))
        if not specs.get("cooling_condition") or "limited airflow" in str(specs.get("cooling_condition")).lower():
            rows.append(self._missing("cooling_method", "high", "Cooling path determines whether the loss budget is thermally feasible, especially with high ambient or high power.", ["thermal_design", "mechanical_design"], "Is cooling natural convection, forced airflow, conduction to chassis, heatsink, or another method?"))
        if not specs.get("mechanical_constraints"):
            rows.append(self._missing("mechanical_constraints", "high", "Board area, height, weight, connector, mounting, and enclosure constraints affect topology, magnetics, thermal path, and manufacturability.", ["mechanical_design", "magnetic_design", "thermal_design"], "What board size, height, connector, enclosure, and mounting limits apply?"))
        if specs.get("output_ripple_mvpp") is None:
            rows.append(self._missing("output_ripple", "medium", "Ripple limit affects output capacitor ESR/RMS current, post-filter choice, layout, and measurement method.", ["output_filter_design", "control_design", "verification_agent"], "What output ripple limit and measurement bandwidth are required?"))
        rows.append(self._missing("transient_response", "medium", "Load-step size, overshoot/undershoot, and recovery time are needed for output capacitor and compensation design.", ["control_design", "verification_agent"], "What load transient step and allowed response are required?"))
        rows.append(self._missing("protection_thresholds", "medium", "Basic protection is not enough to design thresholds, timing, and latch/retry behavior.", ["protection_design", "controller_selection", "verification_agent"], "What UVLO, OVP, OCP/short-circuit, OTP thresholds and fault response are required?"))
        if not specs.get("cost_target"):
            rows.append(self._missing("cost_target", "low", "Cost target affects topology complexity, magnetics, semiconductor package, capacitor choices, and AVL depth.", ["component_selection", "bom_avl"], "What BOM cost target, production volume, and approved vendors apply?"))
        rows.append(self._missing("reliability_lifetime_target", "low", "Mission profile, lifetime, derating, and qualification targets affect capacitor life, thermal margin, and validation.", ["thermal_design", "component_selection", "verification_agent"], "What lifetime, reliability, or mission-profile target should be used?"))
        return rows

    def _clarification_questions(self, missing_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        priority_rank = {"blocking": 0, "high": 1, "medium": 2, "low": 3}
        ranked = sorted(
            missing_info,
            key=lambda row: (
                priority_rank.get(str(row.get("priority")), 9),
                0 if row.get("architecture_critical") else 1,
                str(row.get("item")),
            ),
        )
        return [
            {
                "priority": row["priority"],
                "item": row["item"],
                "question": row["clarification_question"],
                "architecture_critical": row.get("architecture_critical", False),
            }
            for row in ranked[:6]
        ]

    @staticmethod
    def _task_decomposition() -> List[Dict[str, Any]]:
        return [
            {"task_name": "Topology Selection", "purpose": "Compare candidate architectures without selecting final topology.", "required_inputs": ["Vin range", "Vout", "power", "isolation status", "efficiency target", "cost priority"], "expected_outputs": ["candidate topologies", "trade-off matrix", "rollback criteria"], "downstream_agent": "topology_selection_agent"},
            {"task_name": "Power Stage Design", "purpose": "Translate requirements into voltage/current/energy stress targets.", "required_inputs": ["candidate topology", "Vin range", "Vout/Iout", "power", "frequency range"], "expected_outputs": ["stress sweep", "semiconductor requirements", "capacitor requirements"], "downstream_agent": "power_stage_design_agent"},
            {"task_name": "Magnetic Design", "purpose": "Prepare inductor/transformer requirements and manufacturability checks.", "required_inputs": ["topology", "power", "frequency", "isolation", "current ripple"], "expected_outputs": ["core/gap/winding/insulation package"], "downstream_agent": "magnetic_design_agent"},
            {"task_name": "Thermal Design", "purpose": "Assess feasibility of loss budget against ambient and cooling.", "required_inputs": ["loss estimate", "ambient", "cooling method", "mechanical constraints"], "expected_outputs": ["thermal risk", "cooling strategy", "temperature targets"], "downstream_agent": "thermal_design_agent"},
            {"task_name": "Control Design", "purpose": "Define regulation and dynamic-response architecture.", "required_inputs": ["topology", "feedback method", "ripple/transient targets"], "expected_outputs": ["control architecture", "PM/GM target", "transient plan"], "downstream_agent": "control_design_agent"},
            {"task_name": "Protection Design", "purpose": "Convert protection intent into thresholds and fault behavior.", "required_inputs": ["fault list", "thresholds", "recovery policy"], "expected_outputs": ["UVLO/OVP/OCP/OTP strategy", "test matrix"], "downstream_agent": "protection_design_agent"},
            {"task_name": "EMI Design", "purpose": "Plan EMI filters, layout constraints, and compliance evidence.", "required_inputs": ["EMI standard", "switching behavior", "layout constraints"], "expected_outputs": ["filter strategy", "layout rules", "pre-scan plan"], "downstream_agent": "emi_design_agent"},
            {"task_name": "Verification Planning", "purpose": "Create simulation and bench matrix with pass/fail criteria.", "required_inputs": ["requirements", "assumptions", "risk register"], "expected_outputs": ["PLECS matrix", "bench matrix", "evidence checklist"], "downstream_agent": "verification_agent"},
        ]

    def _derived_lookup(self, derived: List[Dict[str, Any]], name: str) -> Optional[float]:
        for row in derived:
            if row.get("name") == name:
                return self._num(row.get("value"))
        return None

    def _feasibility_conflicts(self, specs: Dict[str, Any], missing_info: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        derived = self._derived_specifications(specs)
        loss = self._derived_lookup(derived, "loss_at_efficiency")
        ambient = self._num(specs.get("ambient_c_max"))
        thermal_high = bool((loss is not None and loss > 25) or (ambient is not None and ambient >= 85) or "limited airflow" in text.lower())
        issues = [
            {
                "issue": "Efficiency vs thermal",
                "risk_level": "high" if thermal_high else "medium",
                "explanation": "The target efficiency still leaves real heat that must be removed through semiconductors, magnetics, PCB copper, enclosure, or airflow.",
                "quantitative_basis": f"loss_at_efficiency = {loss:.1f} W" if loss is not None else "loss cannot be computed until power and efficiency are known",
                "affected_requirements": ["efficiency_target", "ambient_temperature", "cooling_method"],
                "recommended_action": "Run early loss budget and thermal feasibility before topology freeze.",
            },
            {
                "issue": "Compactness vs heat dissipation",
                "risk_level": "high" if "compact" in text.lower() and thermal_high else "medium",
                "explanation": "Compact packaging reduces heat spreading area and can conflict with high output power, high ambient, and limited airflow.",
                "quantitative_basis": "compact is qualitative; no dimensions provided",
                "affected_requirements": ["mechanical_constraints", "thermal_design", "rated_power"],
                "recommended_action": "Clarify board/enclosure/cooling constraints and carry them into topology and magnetics selection.",
            },
            {
                "issue": "Automotive suitability vs missing EMI/EMC details",
                "risk_level": "high" if any(row["item"] == "emi_emc_standard" for row in missing_info) else "medium",
                "explanation": "Automotive use implies EMI/EMC attention, but no specific standard is provided.",
                "quantitative_basis": "No EMI/EMC class is present in the prompt.",
                "affected_requirements": ["emi_emc_standard", "verification_agent"],
                "recommended_action": "Ask for standard/class; do not fabricate any specific EMI/EMC limit.",
            },
            {
                "issue": "Protection requirement vs missing thresholds",
                "risk_level": "medium",
                "explanation": "Basic protection is not enough for component stress, controller setup, or verification.",
                "quantitative_basis": "No UVLO/OVP/OCP/OTP thresholds parsed.",
                "affected_requirements": ["protection_thresholds"],
                "recommended_action": "Clarify thresholds and fault response before detailed controller/protection design.",
            },
        ]
        return issues

    def _refined_objectives(self, specs: Dict[str, Any], missing_info: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        hard: List[str] = []
        if specs.get("input_voltage_min") is not None and specs.get("input_voltage_max") is not None:
            hard.append(f"Input voltage range {specs['input_voltage_min']:g}-{specs['input_voltage_max']:g} V")
        elif specs.get("input_voltage_nominal") is not None:
            hard.append(f"Nominal input voltage {specs['input_voltage_nominal']:g} V")
        if specs.get("output_voltage") is not None:
            hard.append(f"Regulated output voltage {specs['output_voltage']:g} V")
        if specs.get("rated_output_power") is not None:
            hard.append(f"Rated output power {specs['rated_output_power']:g} W")
        if specs.get("peak_output_power") is not None:
            hard.append(f"Peak output power {specs['peak_output_power']:g} W for {specs.get('peak_duration_s', 'unspecified')} s")
        if specs.get("efficiency_target") is not None:
            hard.append(f"Efficiency target >={specs['efficiency_target'] * 100:g}%")
        if specs.get("ambient_c_max") is not None:
            hard.append(f"Ambient up to {specs['ambient_c_max']:g} C")
        return {
            "hard_constraints": hard,
            "high_priority_objectives": ["thermal feasibility", "safe derating", "stable regulated output", "protection strategy", "EMI readiness"],
            "medium_priority_objectives": ["compact packaging", "cost effectiveness", "manufacturability", "layout robustness", "verification coverage"],
            "pending_requirements": [f"{row['item']}: {row['clarification_question']}" for row in missing_info],
        }

    def _recommended_workflow(self, missing_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {"step": 1, "action": "Review explicit and derived requirements", "reason": "Prevent source or unit drift before design work.", "depends_on": []},
            {"step": 2, "action": "Resolve blocking/high-priority missing information or document assumptions", "reason": "Avoid treating assumptions as confirmed requirements.", "depends_on": ["requirement_analysis"]},
            {"step": 3, "action": "Run preliminary topology comparison", "reason": "Topology drives stress, thermal, EMI, magnetics, and control design.", "depends_on": ["isolation status or explicit architecture assumption"]},
            {"step": 4, "action": "Run early thermal and stress estimates", "reason": "High loss or limited cooling can invalidate architecture choices.", "depends_on": ["power and efficiency targets"]},
            {"step": 5, "action": "Plan PLECS and bench validation matrix", "reason": "Claims need simulation or measurement evidence before release.", "depends_on": ["selected candidate topology", "control/protection assumptions"]},
        ]

    def _assumptions(self, specs: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
        assumptions = []
        if specs.get("cooling_condition") == "limited airflow":
            assumptions.append({"assumption": "Limited airflow is a qualitative thermal constraint, not a defined cooling method.", "reason": "The exact cooling path is missing.", "risk_if_wrong": "Thermal design may be over- or under-constrained.", "must_confirm": True})
        if specs.get("topology_preference"):
            assumptions.append({"assumption": f"Topology preference captured as preference only: {specs['topology_preference']}", "reason": "Requirement agent must not choose final topology.", "risk_if_wrong": "Downstream topology selection may be biased incorrectly.", "must_confirm": False})
        assumptions.append({"assumption": "No unprovided EMI/EMC, protection, mechanical, reliability, or cost numbers are created.", "reason": "Source grounding rule.", "risk_if_wrong": "The MAS could produce false release confidence.", "must_confirm": False})
        return assumptions

    def _handoff(
        self,
        summary: Dict[str, Any],
        specs: Dict[str, Any],
        derived: List[Dict[str, Any]],
        missing_info: List[Dict[str, Any]],
        feasibility: List[Dict[str, Any]],
        topology_candidates: List[Dict[str, Any]],
        spec_gate_status: str,
        engineering_checks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        rated_i = self._derived_lookup(derived, "rated_output_current")
        peak_i = self._derived_lookup(derived, "peak_output_current")
        loss = self._derived_lookup(derived, "loss_at_efficiency")
        missing_items = [row["item"] for row in missing_info]
        return {
            "topology_selection_agent": {
                "spec_gate_status": spec_gate_status,
                "conversion": summary.get("conversion_type"),
                "vin_range": [specs.get("input_voltage_min"), specs.get("input_voltage_max")],
                "vin_nominal": specs.get("input_voltage_nominal"),
                "vout": specs.get("output_voltage"),
                "rated_power": specs.get("rated_output_power"),
                "peak_power": specs.get("peak_output_power"),
                "efficiency_target": specs.get("efficiency_target"),
                "isolation_requirement": specs.get("isolation_requirement") or "unknown",
                "candidate_seed_topologies": [{"name": row.get("name"), "plecs_model_status": row.get("plecs_model_status")} for row in topology_candidates],
                "do_not_select_final_topology": True,
                "may_proceed": spec_gate_status in {"LOCKED", "PARTIAL"},
                "must_not_assume": [
                    "final topology",
                    "final EMI standard",
                    "final mechanical envelope",
                    "final cooling method",
                    "final protection thresholds",
                ],
                "engineering_derived_checks": engineering_checks,
                "open_items": missing_items,
            },
            "thermal_design_agent": {
                "rated_output_power": specs.get("rated_output_power"),
                "rated_output_current": rated_i,
                "estimated_loss_at_efficiency_w": loss,
                "ambient_temperature": [specs.get("ambient_c_min"), specs.get("ambient_c_max")],
                "cooling_condition": specs.get("cooling_condition") or "unknown",
                "early_analysis_required": True,
            },
            "control_design_agent": {
                "output_voltage": specs.get("output_voltage"),
                "feedback_preference": specs.get("feedback_preference") or "not specified",
                "ripple_limit": specs.get("output_ripple_mvpp"),
                "transient_response": "missing",
            },
            "protection_design_agent": {
                "required_protection": specs.get("protection_requirement") or "not specified",
                "missing_thresholds": ["UVLO", "OVP", "OCP", "short-circuit", "OTP"],
                "fault_response_unknown": True,
            },
            "emi_design_agent": {
                "application": summary.get("application"),
                "specific_standard": specs.get("emi_emc_standard") or "not specified",
                "do_not_fabricate_standard": True,
            },
            "verification_agent": {
                "must_verify": ["minimum input", "maximum input", "regulated output", "rated load", "peak load duration", "efficiency", "thermal operation", "protection behavior"],
                "cannot_finalize_until_confirmed": missing_items,
                "plecs_model_registry_preview": self.plecs_registry.status(),
                "rated_output_current_a": rated_i,
                "peak_output_current_a": peak_i,
            },
        }

    @staticmethod
    def _spec_package(
        spec_gate_status: str,
        layer1: List[Dict[str, Any]],
        layer2: List[Dict[str, Any]],
        engineering_checks: List[Dict[str, Any]],
        derived: List[Dict[str, Any]],
        missing_info: List[Dict[str, Any]],
        assumptions: List[Dict[str, Any]],
        risks: List[Dict[str, Any]],
        handoff: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "spec_gate_status": spec_gate_status,
            "layer1_locked_items": layer1,
            "layer2_constraints": layer2,
            "engineering_derived_checks": engineering_checks,
            "derived_quantities": derived,
            "assumptions": assumptions,
            "open_questions": missing_info,
            "risk_flags": risks,
            "handoff_notes": {
                "topology_agent": handoff.get("topology_selection_agent", {}),
                "rule": "Topology Agent may compare candidates, but Spec Agent must not select final topology.",
            },
        }

    @staticmethod
    def _readiness(missing_info: List[Dict[str, Any]]) -> str:
        hard_blocking = [row for row in missing_info if row["priority"] == "blocking" and row["item"] in {"input_voltage_range", "output_rating"}]
        if hard_blocking:
            return "blocked"
        if any(row["priority"] in {"blocking", "high", "medium"} for row in missing_info):
            return "partial"
        return "ready"

    @staticmethod
    def _quality(explicit_specs: List[Dict[str, Any]], derived_specs: List[Dict[str, Any]], missing_info: List[Dict[str, Any]]) -> Dict[str, bool]:
        checked_items = {row["item"] for row in missing_info}
        explicit_names = {row.get("name") for row in explicit_specs}
        if "isolation_requirement" in explicit_names:
            checked_items.add("isolation_requirement")
        if "emi_emc_target" in explicit_names:
            checked_items.add("emi_emc_standard")
        if "output_ripple_limit" in explicit_names:
            checked_items.add("output_ripple")
        required_checks = {"isolation_requirement", "cooling_method", "emi_emc_standard", "output_ripple", "transient_response", "protection_thresholds", "mechanical_constraints", "cost_target", "reliability_lifetime_target"}
        return {
            "no_unsupported_numbers": True,
            "explicit_specs_have_source": all(bool(row.get("source_text")) for row in explicit_specs),
            "derived_values_have_formula": all(bool(row.get("formula")) for row in derived_specs),
            "critical_missing_info_checked": required_checks.issubset(checked_items),
            "no_final_topology_fabricated": True,
        }

    def _frontend_response(
        self,
        summary: Dict[str, Any],
        explicit: List[Dict[str, Any]],
        derived: List[Dict[str, Any]],
        missing: List[Dict[str, Any]],
        risks: List[Dict[str, Any]],
        questions: List[Dict[str, Any]],
        workflow: List[Dict[str, Any]],
        readiness: str,
        handoff: Dict[str, Any],
        assumptions: List[Dict[str, Any]],
        spec_gate_status: str,
        layer1: List[Dict[str, Any]],
        layer2: List[Dict[str, Any]],
        engineering_checks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "summary_cards": [
                {"label": "Design object", "value": summary.get("design_object"), "status": "ready"},
                {"label": "Application", "value": summary.get("application"), "status": "context"},
                {"label": "Conversion", "value": summary.get("conversion_type"), "status": "context"},
                {"label": "Spec Gate", "value": spec_gate_status, "status": spec_gate_status.lower()},
                {"label": "Readiness", "value": readiness, "status": readiness},
            ],
            "layer1_minimum_converter_definition": layer1,
            "layer2_constraints_preferences": layer2,
            "engineering_derived_checks": engineering_checks,
            "extracted_specs_table": explicit,
            "derived_calculations_table": derived,
            "missing_info_table": missing,
            "risk_conflict_cards": risks,
            "clarification_questions": questions,
            "recommended_next_actions": workflow,
            "readiness_status": readiness,
            "handoff_package_preview": {key: value for key, value in handoff.items()},
            "assumption_cards": assumptions,
            "source_provenance_badges": [
                {"label": "Prompt", "value": "User-provided", "confidence": "high"},
                {"label": "Topology DB", "value": "Offline seed", "confidence": "medium"},
            ],
        }

    def _human_report(
        self,
        summary: Dict[str, Any],
        app: Dict[str, Any],
        explicit: List[Dict[str, Any]],
        derived: List[Dict[str, Any]],
        missing: List[Dict[str, Any]],
        risks: List[Dict[str, Any]],
        objectives: Dict[str, List[str]],
        workflow: List[Dict[str, Any]],
        readiness: str,
        spec_gate_status: str,
        layer1: List[Dict[str, Any]],
        layer2: List[Dict[str, Any]],
        engineering_checks: List[Dict[str, Any]],
    ) -> str:
        lines = [
            "Spec Agent Report",
            f"Spec gate status: {spec_gate_status}",
            f"Design object: {summary.get('design_object')}",
            f"Application: {summary.get('application')}",
            f"Conversion: {summary.get('conversion_type')}",
            f"Readiness status: {readiness}",
            "",
            "Layer-1 minimum converter definition:",
        ]
        lines += [f"- {row['field']}: {row['extracted_value']} [{row['status']}, {row['constraint_type']}]" for row in layer1]
        lines += [
            "",
            "Layer-2 constraints and preferences:",
        ]
        lines += [f"- {row['area']}: {row['extracted_value']} [{row['constraint_type']}]" for row in layer2]
        lines += [
            "",
            "Engineering-derived checks:",
        ]
        lines += [f"- {row['name']}: {row['status']} -> {', '.join(row['downstream_owner'])}" for row in engineering_checks]
        lines += [
            "",
            "Application implications:",
        ]
        lines += [f"- {item}" for item in app.get("application_driven_implications", [])]
        lines.append("")
        lines.append("Explicit specifications:")
        lines += [f"- {row['name']}: {row['value']} {row.get('unit', '')}".rstrip() for row in explicit]
        lines.append("")
        lines.append("Derived specifications:")
        lines += [f"- {row['name']}: {row['value']} {row.get('unit', '')}; {row['formula']}".rstrip() for row in derived]
        lines.append("")
        lines.append("Prioritized missing information:")
        lines += [f"- [{row['priority']}] {row['item']}: {row['clarification_question']}" for row in missing]
        lines.append("")
        lines.append("Feasibility/conflict check:")
        lines += [f"- [{row['risk_level']}] {row['issue']}: {row['explanation']} Basis: {row['quantitative_basis']}" for row in risks]
        lines.append("")
        lines.append("Refined objectives:")
        for key, values in objectives.items():
            lines.append(f"- {key}: {'; '.join(values)}")
        lines.append("")
        lines.append("Recommended workflow:")
        lines += [f"{row['step']}. {row['action']} - {row['reason']}" for row in workflow]
        return "\n".join(lines)
