from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .config import rpc_url
from .rpc import call, call_first_available, server


def _looks_like_pe_mas_root(path: str) -> bool:
    probe = os.path.join(path, "core", "flyback_mas", "tools", "plecs_interface.py")
    return os.path.isfile(probe)


def _find_pe_mas_root() -> Optional[str]:
    # 1) explicit override for deployment users
    env_root = str(os.getenv("PE_MAS_ROOT", "")).strip()
    if env_root and _looks_like_pe_mas_root(env_root):
        return env_root

    # 2) search upward from CWD
    cwd = os.getcwd()
    cur = cwd
    for _ in range(8):
        if _looks_like_pe_mas_root(cur):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # 3) search upward from this file location
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(10):
        if _looks_like_pe_mas_root(cur):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    return None


def register_tools(mcp: FastMCP) -> None:
    def _safe_name(name: str) -> str:
        return str(name or "").strip().replace(" ", "_")

    def _join_path(model_name: str, component_path: str) -> str:
        model = str(model_name or "").strip().strip("/")
        comp = str(component_path or "").strip().strip("/")
        return f"{model}/{comp}" if comp else model

    def _script_for_action(model_name: str, action: Dict[str, Any]) -> str:
        # Fallback script snippets when direct RPC methods are unavailable.
        op = str(action.get("op") or "").strip().lower()
        if op == "set_param":
            path = _join_path(model_name, str(action.get("path") or ""))
            param = str(action.get("param") or "")
            value = action.get("value")
            return f'plecs.set("{path}", "{param}", "{value}");'
        if op == "connect":
            src = _join_path(model_name, str(action.get("src") or ""))
            dst = _join_path(model_name, str(action.get("dst") or ""))
            return f'plecs.connect("{src}", "{dst}");'
        if op == "disconnect":
            src = _join_path(model_name, str(action.get("src") or ""))
            dst = _join_path(model_name, str(action.get("dst") or ""))
            return f'plecs.disconnect("{src}", "{dst}");'
        if op == "move_component":
            path = _join_path(model_name, str(action.get("path") or ""))
            x = action.get("x", 0)
            y = action.get("y", 0)
            return f'plecs.set("{path}", "Location", "{x},{y}");'
        if op == "rename_component":
            path = _join_path(model_name, str(action.get("path") or ""))
            new_name = _safe_name(str(action.get("new_name") or ""))
            return f'plecs.set("{path}", "Name", "{new_name}");'
        if op == "delete_component":
            path = _join_path(model_name, str(action.get("path") or ""))
            return f'plecs.delete("{path}");'
        if op == "add_component":
            parent = _join_path(model_name, str(action.get("parent") or ""))
            block_type = str(action.get("block_type") or "")
            name = _safe_name(str(action.get("name") or ""))
            x = action.get("x", 0)
            y = action.get("y", 0)
            return f'plecs.add("{parent}", "{block_type}", "{name}", "{x},{y}");'
        return ""

    def _direct_action_candidates(model_name: str, action: Dict[str, Any]) -> Dict[str, Any]:
        op = str(action.get("op") or "").strip().lower()

        if op == "set_param":
            path = _join_path(model_name, str(action.get("path") or ""))
            param = str(action.get("param") or "")
            value = action.get("value")
            return {
                "methods": ["plecs.set", "plecs.setParameter"],
                "args": [path, param, value],
            }

        if op == "connect":
            src = _join_path(model_name, str(action.get("src") or ""))
            dst = _join_path(model_name, str(action.get("dst") or ""))
            return {
                "methods": ["plecs.connect", "plecs.addConnection"],
                "args": [src, dst],
            }

        if op == "disconnect":
            src = _join_path(model_name, str(action.get("src") or ""))
            dst = _join_path(model_name, str(action.get("dst") or ""))
            return {
                "methods": ["plecs.disconnect", "plecs.removeConnection"],
                "args": [src, dst],
            }

        if op == "move_component":
            path = _join_path(model_name, str(action.get("path") or ""))
            x = action.get("x", 0)
            y = action.get("y", 0)
            return {
                "methods": ["plecs.set", "plecs.setParameter"],
                "args": [path, "Location", f"{x},{y}"],
            }

        if op == "rename_component":
            path = _join_path(model_name, str(action.get("path") or ""))
            new_name = _safe_name(str(action.get("new_name") or ""))
            return {
                "methods": ["plecs.set", "plecs.setParameter", "plecs.rename"],
                "args": [path, "Name", new_name],
            }

        if op == "delete_component":
            path = _join_path(model_name, str(action.get("path") or ""))
            return {
                "methods": ["plecs.delete", "plecs.remove"],
                "args": [path],
            }

        if op == "add_component":
            parent = _join_path(model_name, str(action.get("parent") or ""))
            block_type = str(action.get("block_type") or "")
            name = _safe_name(str(action.get("name") or ""))
            x = action.get("x", 0)
            y = action.get("y", 0)
            return {
                "methods": ["plecs.add", "plecs.addComponent", "plecs.create"],
                "args": [parent, block_type, name, f"{x},{y}"],
            }

        return {"methods": [], "args": []}

    def _execute_circuit_action(model_name: str, action: Dict[str, Any], allow_script_fallback: bool = True) -> Dict[str, Any]:
        op = str(action.get("op") or "").strip().lower()
        if not op:
            return {"ok": False, "error": "action.op is required", "action": action}

        direct = _direct_action_candidates(model_name, action)
        methods = direct.get("methods") or []
        args = direct.get("args") or []
        if methods:
            ok, method, result = call_first_available(methods, args)
            if ok:
                return {
                    "ok": True,
                    "mode": "rpc",
                    "op": op,
                    "method": method,
                    "result": result,
                    "action": action,
                }

        if allow_script_fallback:
            script = _script_for_action(model_name, action)
            if script:
                script_pack = run_script(script=script)
                if script_pack.get("ok"):
                    return {
                        "ok": True,
                        "mode": "script_fallback",
                        "op": op,
                        "action": action,
                        "script": script,
                        "result": script_pack.get("result"),
                    }
                return {
                    "ok": False,
                    "mode": "script_fallback",
                    "op": op,
                    "action": action,
                    "script": script,
                    "error": script_pack.get("error"),
                    "notes": script_pack.get("notes", []),
                }

        return {
            "ok": False,
            "mode": "rpc",
            "op": op,
            "action": action,
            "error": "no compatible RPC method found",
            "tried_methods": methods,
        }

    @mcp.tool()
    def ping() -> Dict[str, Any]:
        try:
            s = server()
            methods = s.system.listMethods()
            return {
                "ok": True,
                "rpc_url": rpc_url(),
                "method_count": len(methods) if isinstance(methods, list) else 0,
            }
        except Exception as e:
            return {"ok": False, "rpc_url": rpc_url(), "error": str(e)}

    @mcp.tool()
    def list_methods(prefix: str = "") -> Dict[str, Any]:
        try:
            s = server()
            methods = s.system.listMethods()
            out = [m for m in methods if str(m).startswith(prefix)] if prefix else list(methods)
            return {"ok": True, "methods": out}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mcp.tool()
    def inspect_method(method: str) -> Dict[str, Any]:
        try:
            s = server()
            out: Dict[str, Any] = {"ok": True, "method": method}
            try:
                out["help"] = s.system.methodHelp(method)
            except Exception as e:
                out["help_error"] = str(e)
            try:
                out["signature"] = s.system.methodSignature(method)
            except Exception as e:
                out["signature_error"] = str(e)
            return out
        except Exception as e:
            return {"ok": False, "error": str(e), "method": method}

    @mcp.tool()
    def rpc_call(method: str, args: Optional[List[Any]] = None) -> Dict[str, Any]:
        try:
            if not method:
                return {"ok": False, "error": "method is required"}
            result = call(method, args or [])
            return {"ok": True, "result": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mcp.tool()
    def rpc_catalog(
        prefix: str = "",
        include_help: bool = False,
        include_signature: bool = False,
        max_methods: int = 300,
    ) -> Dict[str, Any]:
        """Enumerate RPC methods with optional help/signature expansion."""
        try:
            methods_pack = list_methods(prefix=prefix)
            if not methods_pack.get("ok"):
                return methods_pack

            methods = list(methods_pack.get("methods") or [])[: max(1, int(max_methods))]
            if not (include_help or include_signature):
                return {"ok": True, "count": len(methods), "methods": methods}

            rows: List[Dict[str, Any]] = []
            for name in methods:
                row: Dict[str, Any] = {"method": name}
                details = inspect_method(name)
                if include_help:
                    row["help"] = details.get("help")
                    if details.get("help_error"):
                        row["help_error"] = details.get("help_error")
                if include_signature:
                    row["signature"] = details.get("signature")
                    if details.get("signature_error"):
                        row["signature_error"] = details.get("signature_error")
                rows.append(row)
            return {"ok": True, "count": len(rows), "methods": rows}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mcp.tool()
    def rpc_try_methods(methods: List[str], args: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Try method name candidates in order and return the first success."""
        try:
            method_list = [str(m).strip() for m in (methods or []) if str(m).strip()]
            if not method_list:
                return {"ok": False, "error": "methods list is empty"}

            ok, method, result = call_first_available(method_list, args or [])
            if ok:
                return {"ok": True, "method": method, "result": result, "tried": method_list}
            return {
                "ok": False,
                "error": "all candidate methods failed",
                "tried": method_list,
                "failures": result,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mcp.tool()
    def rpc_profile(method: str, args: Optional[List[Any]] = None, repeat: int = 1) -> Dict[str, Any]:
        """Profile RPC latency for repeated calls."""
        runs = max(1, int(repeat or 1))
        timings_ms: List[float] = []
        errors: List[str] = []
        last_result: Any = None

        for _ in range(runs):
            t0 = time.perf_counter()
            try:
                last_result = call(method, args or [])
            except Exception as e:
                errors.append(str(e))
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)

        success_count = runs - len(errors)
        return {
            "ok": success_count > 0,
            "method": method,
            "repeat": runs,
            "success_count": success_count,
            "failure_count": len(errors),
            "avg_ms": (sum(timings_ms) / len(timings_ms)) if timings_ms else 0.0,
            "min_ms": min(timings_ms) if timings_ms else None,
            "max_ms": max(timings_ms) if timings_ms else None,
            "timings_ms": timings_ms,
            "errors": errors,
            "last_result": last_result,
        }

    @mcp.tool()
    def rpc_batch(calls: List[Dict[str, Any]], stop_on_error: bool = False) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for idx, item in enumerate(calls or []):
            method = str(item.get("method") or "").strip()
            args = item.get("args") or []
            if not method:
                row = {"index": idx, "ok": False, "error": "missing method"}
                results.append(row)
                if stop_on_error:
                    break
                continue

            try:
                result = call(method, args)
                results.append({"index": idx, "method": method, "ok": True, "result": result})
            except Exception as e:
                row = {"index": idx, "method": method, "ok": False, "error": str(e)}
                results.append(row)
                if stop_on_error:
                    break

        all_ok = all(bool(r.get("ok")) for r in results) if results else True
        return {"ok": all_ok, "count": len(results), "results": results}

    @mcp.tool()
    def open_model(model_path: str) -> Dict[str, Any]:
        try:
            ok, method, result = call_first_available(["plecs.load"], [model_path])
            if ok:
                return {"ok": True, "method": method, "result": result, "model_path": model_path}
            return {"ok": False, "error": result, "model_path": model_path}
        except Exception as e:
            return {"ok": False, "error": str(e), "model_path": model_path}

    @mcp.tool()
    def close_model(model_name: str) -> Dict[str, Any]:
        try:
            ok, method, result = call_first_available(["plecs.close"], [model_name])
            if ok:
                return {"ok": True, "method": method, "result": result, "model_name": model_name}
            return {"ok": False, "error": result, "model_name": model_name}
        except Exception as e:
            return {"ok": False, "error": str(e), "model_name": model_name}

    @mcp.tool()
    def list_open_models() -> Dict[str, Any]:
        try:
            ok, method, result = call_first_available(
                ["plecs.getLoadedModels", "plecs.listModels", "plecs.getModels"],
                [],
            )
            if ok:
                return {"ok": True, "method": method, "models": result}
            return {"ok": False, "error": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mcp.tool()
    def save_model(model_name: str) -> Dict[str, Any]:
        try:
            ok, method, result = call_first_available(["plecs.save"], [model_name])
            if ok:
                return {"ok": True, "method": method, "result": result, "model_name": model_name}
            return {"ok": False, "error": result, "model_name": model_name}
        except Exception as e:
            return {"ok": False, "error": str(e), "model_name": model_name}

    @mcp.tool()
    def save_model_as(model_name: str, output_path: str) -> Dict[str, Any]:
        try:
            ok, method, result = call_first_available(["plecs.saveAs", "plecs.save_as"], [model_name, output_path])
            if ok:
                return {
                    "ok": True,
                    "method": method,
                    "result": result,
                    "model_name": model_name,
                    "output_path": output_path,
                }
            return {"ok": False, "error": result, "model_name": model_name, "output_path": output_path}
        except Exception as e:
            return {"ok": False, "error": str(e), "model_name": model_name, "output_path": output_path}

    @mcp.tool()
    def simulate(
        model_name: str,
        start_time: float = 0.0,
        stop_time: float = 0.1,
        timeout: float = 20.0,
        model_vars: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            opts: Dict[str, Any] = {
                "StartTime": float(start_time),
                "StopTime": float(stop_time),
                "TimeOut": float(timeout),
            }
            if isinstance(model_vars, dict) and model_vars:
                opts["ModelVars"] = model_vars
            res = call("plecs.simulate", [model_name, opts])
            return {"ok": True, "result": res, "options": opts}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mcp.tool()
    def simulate_advanced(model_name: str, options: Dict[str, Any]) -> Dict[str, Any]:
        try:
            res = call("plecs.simulate", [model_name, options or {}])
            return {"ok": True, "result": res, "options": options or {}}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mcp.tool()
    def get_component_param(path: str, param: str) -> Dict[str, Any]:
        try:
            ok, method, result = call_first_available(["plecs.get", "plecs.getParameter"], [path, param])
            if ok:
                return {"ok": True, "method": method, "path": path, "param": param, "value": result}
            return {"ok": False, "error": result, "path": path, "param": param}
        except Exception as e:
            return {"ok": False, "error": str(e), "path": path, "param": param}

    @mcp.tool()
    def set_component_param(path: str, param: str, value: Any) -> Dict[str, Any]:
        try:
            ok, method, result = call_first_available(["plecs.set", "plecs.setParameter"], [path, param, value])
            if ok:
                return {
                    "ok": True,
                    "method": method,
                    "result": result,
                    "path": path,
                    "param": param,
                    "value": value,
                }
            return {"ok": False, "error": result, "path": path, "param": param, "value": value}
        except Exception as e:
            return {"ok": False, "error": str(e), "path": path, "param": param, "value": value}

    @mcp.tool()
    def set_component_params_batch(path: str, params: Dict[str, Any], stop_on_error: bool = False) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for k, v in (params or {}).items():
            row = set_component_param(path=path, param=str(k), value=v)
            results.append(row)
            if stop_on_error and not row.get("ok"):
                break
        all_ok = all(bool(r.get("ok")) for r in results) if results else True
        return {"ok": all_ok, "path": path, "results": results}

    @mcp.tool()
    def clear_console() -> Dict[str, Any]:
        try:
            ok, method, result = call_first_available(["plecs.clearConsole"], [])
            if ok:
                return {"ok": True, "method": method, "result": result}
            return {"ok": False, "error": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mcp.tool()
    def get_console_output() -> Dict[str, Any]:
        try:
            ok, method, result = call_first_available(["plecs.getConsoleOutput", "plecs.consoleOutput"], [])
            if ok:
                return {"ok": True, "method": method, "console": result}
            return {"ok": False, "error": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mcp.tool()
    def run_script(script: str, language: str = "plecs") -> Dict[str, Any]:
        try:
            candidates = [
                "plecs.eval",
                "plecs.execute",
                "plecs.runScript",
                "plecs.script",
            ]
            ok, method, result = call_first_available(candidates, [script])
            if ok:
                return {"ok": True, "method": method, "result": result, "language": language}
            return {
                "ok": False,
                "error": result,
                "language": language,
                "notes": [
                    "Script API may not be exposed in your PLECS XML-RPC build. Use rpc_call/list_methods to discover supported methods."
                ],
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "language": language}

    @mcp.tool()
    def circuit_action(model_name: str, action: Dict[str, Any], allow_script_fallback: bool = True) -> Dict[str, Any]:
        """
        High-level circuit editing action.

        Supported action.op:
        - add_component: {parent, block_type, name, x, y}
        - delete_component: {path}
        - set_param: {path, param, value}
        - move_component: {path, x, y}
        - rename_component: {path, new_name}
        - connect: {src, dst}
        - disconnect: {src, dst}
        """
        return _execute_circuit_action(model_name, action, allow_script_fallback=allow_script_fallback)

    @mcp.tool()
    def circuit_patch(
        model_name: str,
        actions: List[Dict[str, Any]],
        transactional: bool = True,
        stop_on_error: bool = True,
        backup_dir: str = "/tmp",
    ) -> Dict[str, Any]:
        """
        Apply a batch of circuit actions with optional transactional backup.
        If transactional is enabled, the model is first saved to a backup file.
        """
        backup_path = ""
        backup_pack: Dict[str, Any] = {"ok": True}
        if transactional:
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, f"{_safe_name(model_name)}_mcp_backup.plecs")
            backup_pack = save_model_as(model_name=model_name, output_path=backup_path)
            if not backup_pack.get("ok"):
                return {
                    "ok": False,
                    "error": "failed to create transactional backup",
                    "backup": backup_pack,
                }

        results: List[Dict[str, Any]] = []
        for idx, act in enumerate(actions or []):
            row = _execute_circuit_action(model_name, act, allow_script_fallback=True)
            row["index"] = idx
            results.append(row)
            if stop_on_error and not row.get("ok"):
                break

        success = all(bool(r.get("ok")) for r in results) if results else True
        out: Dict[str, Any] = {
            "ok": success,
            "model_name": model_name,
            "count": len(results),
            "results": results,
            "transactional": transactional,
        }

        if transactional:
            out["backup_path"] = backup_path

        if not success and transactional and backup_path:
            restore = open_model(backup_path)
            out["rollback"] = {
                "attempted": True,
                "result": restore,
                "note": "Backup model opened for manual restore/save workflow.",
            }

        return out

    @mcp.tool()
    def script_transaction(
        model_name: str,
        script: str,
        transactional: bool = True,
        backup_dir: str = "/tmp",
    ) -> Dict[str, Any]:
        """
        Execute a raw script with optional backup + rollback-on-failure behavior.
        """
        backup_path = ""
        if transactional:
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, f"{_safe_name(model_name)}_script_backup.plecs")
            backup_pack = save_model_as(model_name=model_name, output_path=backup_path)
            if not backup_pack.get("ok"):
                return {
                    "ok": False,
                    "error": "failed to create transactional backup",
                    "backup": backup_pack,
                }

        run = run_script(script=script)
        out: Dict[str, Any] = {
            "ok": bool(run.get("ok")),
            "run": run,
            "transactional": transactional,
        }
        if backup_path:
            out["backup_path"] = backup_path

        if transactional and not run.get("ok") and backup_path:
            out["rollback"] = {
                "attempted": True,
                "result": open_model(backup_path),
                "note": "Backup model opened for manual restore/save workflow.",
            }
        return out

    @mcp.tool()
    def ui_action_catalog() -> Dict[str, Any]:
        """
        UI-like operation catalog so agents can operate by semantic action names
        instead of low-level RPC details.
        """
        return {
            "ok": True,
            "operations": [
                {"id": "system.ping", "required": []},
                {"id": "methods.list", "required": []},
                {"id": "capabilities.discover", "required": []},
                {"id": "model.open", "required": ["model_path"]},
                {"id": "model.close", "required": ["model_name"]},
                {"id": "model.save", "required": ["model_name"]},
                {"id": "model.save_as", "required": ["model_name", "output_path"]},
                {"id": "simulate.run", "required": ["model_name"]},
                {"id": "component.get", "required": ["path", "param"]},
                {"id": "component.set", "required": ["path", "param", "value"]},
                {"id": "component.set_many", "required": ["path", "params"]},
                {"id": "circuit.action", "required": ["model_name", "action"]},
                {"id": "circuit.patch", "required": ["model_name", "actions"]},
                {"id": "script.run", "required": ["script"]},
                {"id": "script.run_tx", "required": ["model_name", "script"]},
                {"id": "console.get", "required": []},
                {"id": "console.clear", "required": []},
                {"id": "rpc.call", "required": ["method"]},
                {"id": "rpc.batch", "required": ["calls"]},
                {"id": "rpc.catalog", "required": []},
                {"id": "rpc.try", "required": ["methods"]},
                {"id": "rpc.profile", "required": ["method"]},
            ],
            "notes": [
                "Use ui_invoke(operation, payload) for single UI-like operations.",
                "Use ui_macro(steps, transactional_model_name) for multi-step UI workflows with rollback.",
            ],
        }

    @mcp.tool()
    def ui_invoke(operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified UI-style operation gateway.
        This hides low-level RPC method details and exposes semantic operation IDs.
        """
        op = str(operation or "").strip().lower()
        data = payload or {}

        if op == "system.ping":
            return ping()
        if op == "methods.list":
            return list_methods(prefix=str(data.get("prefix") or ""))
        if op == "capabilities.discover":
            return discover_capabilities()

        if op == "model.open":
            return open_model(model_path=str(data.get("model_path") or ""))
        if op == "model.close":
            return close_model(model_name=str(data.get("model_name") or ""))
        if op == "model.save":
            return save_model(model_name=str(data.get("model_name") or ""))
        if op == "model.save_as":
            return save_model_as(
                model_name=str(data.get("model_name") or ""),
                output_path=str(data.get("output_path") or ""),
            )

        if op == "simulate.run":
            if isinstance(data.get("options"), dict):
                return simulate_advanced(
                    model_name=str(data.get("model_name") or ""),
                    options=data.get("options") or {},
                )
            return simulate(
                model_name=str(data.get("model_name") or ""),
                start_time=float(data.get("start_time", 0.0) or 0.0),
                stop_time=float(data.get("stop_time", 0.1) or 0.1),
                timeout=float(data.get("timeout", 20.0) or 20.0),
                model_vars=data.get("model_vars") if isinstance(data.get("model_vars"), dict) else None,
            )

        if op == "component.get":
            return get_component_param(path=str(data.get("path") or ""), param=str(data.get("param") or ""))
        if op == "component.set":
            return set_component_param(
                path=str(data.get("path") or ""),
                param=str(data.get("param") or ""),
                value=data.get("value"),
            )
        if op == "component.set_many":
            return set_component_params_batch(
                path=str(data.get("path") or ""),
                params=data.get("params") if isinstance(data.get("params"), dict) else {},
                stop_on_error=bool(data.get("stop_on_error", False)),
            )

        if op == "circuit.action":
            return circuit_action(
                model_name=str(data.get("model_name") or ""),
                action=data.get("action") if isinstance(data.get("action"), dict) else {},
                allow_script_fallback=bool(data.get("allow_script_fallback", True)),
            )
        if op == "circuit.patch":
            return circuit_patch(
                model_name=str(data.get("model_name") or ""),
                actions=data.get("actions") if isinstance(data.get("actions"), list) else [],
                transactional=bool(data.get("transactional", True)),
                stop_on_error=bool(data.get("stop_on_error", True)),
                backup_dir=str(data.get("backup_dir") or "/tmp"),
            )

        if op == "script.run":
            return run_script(
                script=str(data.get("script") or ""),
                language=str(data.get("language") or "plecs"),
            )
        if op == "script.run_tx":
            return script_transaction(
                model_name=str(data.get("model_name") or ""),
                script=str(data.get("script") or ""),
                transactional=bool(data.get("transactional", True)),
                backup_dir=str(data.get("backup_dir") or "/tmp"),
            )

        if op == "console.get":
            return get_console_output()
        if op == "console.clear":
            return clear_console()

        if op == "rpc.call":
            return rpc_call(
                method=str(data.get("method") or ""),
                args=data.get("args") if isinstance(data.get("args"), list) else [],
            )
        if op == "rpc.batch":
            return rpc_batch(
                calls=data.get("calls") if isinstance(data.get("calls"), list) else [],
                stop_on_error=bool(data.get("stop_on_error", False)),
            )
        if op == "rpc.catalog":
            return rpc_catalog(
                prefix=str(data.get("prefix") or ""),
                include_help=bool(data.get("include_help", False)),
                include_signature=bool(data.get("include_signature", False)),
                max_methods=int(data.get("max_methods", 300) or 300),
            )
        if op == "rpc.try":
            return rpc_try_methods(
                methods=data.get("methods") if isinstance(data.get("methods"), list) else [],
                args=data.get("args") if isinstance(data.get("args"), list) else [],
            )
        if op == "rpc.profile":
            return rpc_profile(
                method=str(data.get("method") or ""),
                args=data.get("args") if isinstance(data.get("args"), list) else [],
                repeat=int(data.get("repeat", 1) or 1),
            )

        return {
            "ok": False,
            "error": f"unknown operation: {operation}",
            "hint": "Call ui_action_catalog() to list supported operations.",
        }

    @mcp.tool()
    def ui_macro(
        steps: List[Dict[str, Any]],
        transactional_model_name: str = "",
        backup_dir: str = "/tmp",
        stop_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a sequence of UI operations with optional model-level backup/rollback.
        steps item format: {"operation": "...", "payload": {...}}
        """
        backup_path = ""
        backup_info: Dict[str, Any] = {"ok": True}
        model_name = str(transactional_model_name or "").strip()
        if model_name:
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, f"{_safe_name(model_name)}_ui_macro_backup.plecs")
            backup_info = save_model_as(model_name=model_name, output_path=backup_path)
            if not backup_info.get("ok"):
                return {
                    "ok": False,
                    "error": "failed to create ui_macro backup",
                    "backup": backup_info,
                }

        results: List[Dict[str, Any]] = []
        for idx, step in enumerate(steps or []):
            operation = str(step.get("operation") or "")
            payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}
            row = ui_invoke(operation=operation, payload=payload)
            row["index"] = idx
            row["operation"] = operation
            results.append(row)
            if stop_on_error and not row.get("ok"):
                break

        success = all(bool(r.get("ok")) for r in results) if results else True
        out: Dict[str, Any] = {
            "ok": success,
            "count": len(results),
            "results": results,
        }
        if backup_path:
            out["backup_path"] = backup_path
            out["backup"] = backup_info

        if not success and backup_path:
            out["rollback"] = {
                "attempted": True,
                "result": open_model(backup_path),
                "note": "Backup model opened for manual restore/save workflow.",
            }
        return out

    @mcp.tool()
    def discover_capabilities() -> Dict[str, Any]:
        capabilities: Dict[str, Any] = {
            "open_model": ["plecs.load"],
            "close_model": ["plecs.close"],
            "simulate": ["plecs.simulate"],
            "get_component_param": ["plecs.get", "plecs.getParameter"],
            "set_component_param": ["plecs.set", "plecs.setParameter"],
            "list_open_models": ["plecs.getLoadedModels", "plecs.listModels", "plecs.getModels"],
            "console_read": ["plecs.getConsoleOutput", "plecs.consoleOutput"],
            "console_clear": ["plecs.clearConsole"],
            "save_model": ["plecs.save"],
            "save_model_as": ["plecs.saveAs", "plecs.save_as"],
            "script_eval": ["plecs.eval", "plecs.execute", "plecs.runScript", "plecs.script"],
            "circuit_editing": [
                "plecs.add",
                "plecs.addComponent",
                "plecs.create",
                "plecs.delete",
                "plecs.remove",
                "plecs.connect",
                "plecs.addConnection",
                "plecs.disconnect",
                "plecs.removeConnection",
                "plecs.set",
                "plecs.setParameter",
            ],
            "ui_layer": [
                "ui_action_catalog",
                "ui_invoke",
                "ui_macro",
            ],
            "rpc_diagnostics": [
                "rpc_catalog",
                "rpc_try_methods",
                "rpc_profile",
            ],
        }

        supported: Dict[str, Any] = {}
        unsupported: Dict[str, Any] = {}
        methods_pack = list_methods()
        all_methods = set(methods_pack.get("methods", [])) if methods_pack.get("ok") else set()

        for cap, candidate_methods in capabilities.items():
            hit = [m for m in candidate_methods if m in all_methods]
            if hit:
                supported[cap] = hit
            else:
                unsupported[cap] = candidate_methods

        return {
            "ok": bool(methods_pack.get("ok")),
            "rpc_url": rpc_url(),
            "supported": supported,
            "unsupported": unsupported,
            "notes": [
                "Unsupported items can still be attempted via rpc_call/rpc_batch if your server exposes alternative names.",
                "For complete freedom, use rpc_call with method names from list_methods().",
            ],
        }

    @mcp.tool()
    def simulate_flyback(params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            root = _find_pe_mas_root()
            if not root:
                return {
                    "ok": False,
                    "error": "PE-MAS root not found. Set PE_MAS_ROOT to your project root.",
                    "notes": [
                        "simulate_flyback needs PE-MAS source tree (core/flyback_mas/tools/plecs_interface.py)."
                    ],
                }

            if root not in sys.path:
                sys.path.insert(0, root)

            from core.flyback_mas.tools.plecs_interface import run_plecs_simulation

            prev_cwd = os.getcwd()
            try:
                os.chdir(root)
                result = run_plecs_simulation(params or {})
            finally:
                os.chdir(prev_cwd)
            if not isinstance(result, dict) or not result:
                return {
                    "ok": False,
                    "error": "PLECS simulation returned empty result",
                    "result": result,
                    "notes": [
                        "Check PLECS is running and XML-RPC is enabled.",
                        "If model parameters are undefined, verify model initialization commands.",
                    ],
                }
            return {
                "ok": True,
                "result": result,
                "notes": ["Executed via PE-MAS flyback compatibility wrapper."],
            }
        except Exception as e:
            return {
                "ok": False,
                "error": str(e),
                "notes": [
                    "simulate_flyback is PE-MAS specific. Use generic tools rpc_call/open_model/simulate for standalone usage."
                ],
            }
