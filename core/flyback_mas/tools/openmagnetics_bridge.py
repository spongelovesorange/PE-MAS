from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "auto"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _python_has_openmagnetics(py_executable: str) -> bool:
    try:
        proc = subprocess.run(
            [
                py_executable,
                "-c",
                (
                    "import importlib.util; "
                    "mods=('PyOpenMagnetics','pyopenmagnetics'); "
                    "raise SystemExit(0 if any(importlib.util.find_spec(m) for m in mods) else 1)"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=4,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _helper_python() -> str:
    explicit = str(os.getenv("PE_MAS_OPENMAGNETICS_PYTHON") or "").strip()
    if explicit:
        return explicit
    if _python_has_openmagnetics(sys.executable):
        return sys.executable
    helper_root = _repo_root() / ".pe_mas_runtime" / "openmagnetics-helper"
    candidates = [
        helper_root / "bin" / "python",
        helper_root / "conda-env" / "bin" / "python",
        helper_root / "venv" / "bin" / "python",
    ]
    pointer_file = helper_root / ".helper_python_path"
    if pointer_file.exists():
        try:
            pointed = pointer_file.read_text(encoding="utf-8").strip()
            if pointed:
                candidates.insert(0, Path(pointed))
        except Exception:
            pass
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _python_label(path_value: str) -> str:
    return Path(str(path_value or "python")).name or "python"


def _probe_timeout_sec() -> float:
    try:
        return max(2.0, float(os.getenv("PE_MAS_OPENMAGNETICS_PROBE_TIMEOUT_SEC") or 8.0))
    except Exception:
        return 8.0


def _advise_timeout_sec() -> float:
    try:
        return max(6.0, float(os.getenv("PE_MAS_OPENMAGNETICS_ADVISE_TIMEOUT_SEC") or 15.0))
    except Exception:
        return 15.0


def _enable_mode() -> str:
    return str(os.getenv("PE_MAS_OPENMAGNETICS_MODE") or "auto").strip().lower()


def _require_package() -> bool:
    return _truthy(os.getenv("PE_MAS_OPENMAGNETICS_REQUIRE_PACKAGE") or "0")


def _sidecar_command() -> list[str]:
    custom = str(os.getenv("PE_MAS_OPENMAGNETICS_SIDECAR_CMD") or "").strip()
    if custom:
        return custom.split()
    sidecar_script = _repo_root() / "core" / "flyback_mas" / "tools" / "openmagnetics_sidecar.py"
    return [_helper_python(), str(sidecar_script)]


def _run_sidecar(payload: Dict[str, Any], timeout_sec: float | None = None) -> Dict[str, Any]:
    cmd = _sidecar_command()
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_sec if timeout_sec is not None else _probe_timeout_sec(),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"sidecar_timeout: exceeded {exc.timeout} seconds",
            "command": cmd,
        }
    if proc.returncode != 0 and not proc.stdout.strip():
        return {
            "ok": False,
            "error": proc.stderr.strip() or f"sidecar exited with code {proc.returncode}",
            "command": cmd,
        }
    try:
        response = json.loads(proc.stdout or "{}")
    except Exception as exc:
        return {
            "ok": False,
            "error": f"invalid_sidecar_json: {exc}",
            "stdout": proc.stdout[:500],
            "stderr": proc.stderr[:500],
            "command": cmd,
        }
    response["command"] = cmd
    if proc.stderr.strip():
        response["stderr"] = proc.stderr.strip()
    return response


def probe_openmagnetics_availability() -> Dict[str, Any]:
    mode = _enable_mode()
    if mode == "off":
        return {
            "enabled": False,
            "available": False,
            "package_detected": False,
            "mode": mode,
            "helper_python": _helper_python(),
            "reason": "OpenMagnetics integration disabled by PE_MAS_OPENMAGNETICS_MODE=off",
        }
    response = _run_sidecar({"action": "probe"}, timeout_sec=_probe_timeout_sec())
    probe = response.get("probe") if isinstance(response.get("probe"), dict) else {}
    if not response.get("ok"):
        return {
            "enabled": True,
            "available": False,
            "package_detected": False,
            "mode": mode,
            "helper_python": _python_label(_helper_python()),
            "reason": response.get("error") or "openmagnetics sidecar probe failed",
            "command": response.get("command") or _sidecar_command(),
        }
    return {
        "enabled": True,
        "available": bool(probe.get("available")),
        "package_detected": bool(probe.get("package_detected")),
        "package_version": probe.get("package_version") or "",
        "mode": mode,
        "helper_python": _python_label(probe.get("python_executable") or _helper_python()),
        "python_version": probe.get("python_version") or "",
        "reason": probe.get("reason") or "",
        "command": response.get("command") or _sidecar_command(),
        "platform": probe.get("platform") or {},
        "source_build_likely": bool(probe.get("source_build_likely")),
        "recommended_install_mode": probe.get("recommended_install_mode") or "",
        "import_strategy": probe.get("import_strategy") or "",
        "database_loaded": bool(probe.get("database_loaded")),
        "functions_detected": probe.get("functions_detected") or [],
        "diagnostics": probe.get("diagnostics") or [],
    }


def run_openmagnetics_advisor(specifications: Dict[str, Any], theoretical_design: Dict[str, Any]) -> Dict[str, Any]:
    probe = probe_openmagnetics_availability()
    mode = str(probe.get("mode") or "auto")
    if mode == "off":
        return {
            "status": "disabled",
            "engine": "disabled",
            "probe": probe,
            "notes": [probe.get("reason") or "OpenMagnetics integration is disabled."],
            "next_actions": ["Set PE_MAS_OPENMAGNETICS_MODE=auto to enable helper probing."],
        }

    response = _run_sidecar(
        {
            "action": "advise",
            "specifications": specifications or {},
            "theoretical_design": theoretical_design or {},
        },
        timeout_sec=_advise_timeout_sec(),
    )
    if not response.get("ok"):
        return {
            "status": "unavailable",
            "engine": "sidecar_error",
            "probe": probe,
            "notes": [response.get("error") or "OpenMagnetics sidecar failed."],
            "next_actions": [
                "Verify helper Python path and PyOpenMagnetics installation.",
                "Keep heuristic magnetics selection in the selector path until helper health is restored.",
            ],
        }

    advisor = response.get("advisor") if isinstance(response.get("advisor"), dict) else {}
    advisor["probe"] = probe
    if _require_package() and not probe.get("available"):
        advisor["status"] = "unavailable"
        advisor["engine"] = "package_required"
        advisor["notes"] = list(advisor.get("notes") or []) + [
            "PyOpenMagnetics package is required by configuration, but it was not detected in the helper environment."
        ]
    return advisor
