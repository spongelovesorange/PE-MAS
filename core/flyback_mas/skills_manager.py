from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _safe_read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _tokenize(text: str) -> List[str]:
    raw = str(text or "").replace("_", " ").replace("-", " ").lower()
    return [token for token in raw.split() if token]


class Skill:
    def __init__(self, path: str):
        self.path = path
        self.id = os.path.basename(path)
        self.manifest_path = os.path.join(self.path, "manifest.json")
        self.prompt_path = os.path.join(self.path, "prompt.txt")
        self.tools_path = os.path.join(self.path, "tools.py")
        self._prompt = ""
        self._manifest: Dict[str, Any] = {}
        self._tools_module = None
        self._mtime_cache = 0.0
        self.reload()

    def reload(self) -> None:
        self._manifest = _safe_read_json(self.manifest_path)
        self._prompt = _safe_read_text(self.prompt_path)
        self._tools_module = self._load_tools()
        self._mtime_cache = self._latest_mtime()

    def _latest_mtime(self) -> float:
        mtimes = []
        for path in [self.manifest_path, self.prompt_path, self.tools_path]:
            if os.path.exists(path):
                mtimes.append(os.path.getmtime(path))
        return max(mtimes) if mtimes else 0.0

    def refresh_if_stale(self) -> None:
        current = self._latest_mtime()
        if current > self._mtime_cache:
            self.reload()

    @property
    def manifest(self) -> Dict[str, Any]:
        self.refresh_if_stale()
        return self._manifest

    @property
    def prompt(self) -> str:
        self.refresh_if_stale()
        return self._prompt

    @property
    def tools_module(self):
        self.refresh_if_stale()
        return self._tools_module

    def _load_tools(self):
        if not os.path.exists(self.tools_path):
            return None

        module_name = f"skills.{self.id}.tools"
        spec = importlib.util.spec_from_file_location(module_name, self.tools_path)
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"Error loading skill tools {self.id}: {e}")
            return None

    def tool_names(self) -> List[str]:
        module = self.tools_module
        if not module:
            return []
        names: List[str] = []
        for name, func in vars(module).items():
            if callable(func) and not name.startswith("_"):
                names.append(name)
        return sorted(names)

    def entry_point(self) -> str:
        entry = str(self.manifest.get("entry_point") or "").strip()
        if entry:
            return entry
        names = self.tool_names()
        return names[0] if names else ""

    def capabilities(self) -> List[str]:
        caps = self.manifest.get("capabilities")
        if isinstance(caps, list):
            return [str(x).strip() for x in caps if str(x).strip()]

        derived = []
        for value in [
            self.id,
            self.manifest.get("name"),
            self.manifest.get("description"),
            " ".join(self.tool_names()),
        ]:
            derived.extend(_tokenize(str(value or "")))
        return sorted(set(derived))

    def health(self) -> str:
        if self.tools_module is None and os.path.exists(self.tools_path):
            return "degraded"
        return "ok"

    def score_against(self, query: str) -> float:
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return 0.0
        cand_tokens = set(
            self.capabilities()
            + _tokenize(self.manifest.get("name", ""))
            + _tokenize(self.manifest.get("description", ""))
            + _tokenize(" ".join(self.tool_names()))
        )
        overlap = len(q_tokens & cand_tokens)
        desc_bonus = 1.0 if str(self.manifest.get("description") or "").strip() else 0.0
        entry_bonus = 1.0 if self.entry_point() else 0.0
        return float(overlap) + 0.25 * desc_bonus + 0.15 * entry_bonus

    def get_info(self):
        return {
            "id": self.id,
            "name": self.manifest.get("name", self.id),
            "description": self.manifest.get("description", ""),
            "version": self.manifest.get("version", "1.0.0"),
            "author": self.manifest.get("author", ""),
            "entry_point": self.entry_point(),
            "tool_names": self.tool_names(),
            "capabilities": self.capabilities(),
            "health": self.health(),
        }


class SkillManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SkillManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, skills_dir: str = None):
        if self.initialized and self.skills_dir == skills_dir:
            return

        self.skills_dir = skills_dir
        self.skills: Dict[str, Skill] = {}
        if skills_dir:
            self.refresh_skills()
            self.initialized = True

    def refresh_skills(self):
        existing = dict(self.skills)
        self.skills = {}
        if not self.skills_dir:
            return
        if not os.path.exists(self.skills_dir):
            os.makedirs(self.skills_dir, exist_ok=True)
            return

        for item in sorted(os.listdir(self.skills_dir)):
            item_path = os.path.join(self.skills_dir, item)
            if not os.path.isdir(item_path):
                continue
            manifest_path = os.path.join(item_path, "manifest.json")
            tools_path = os.path.join(item_path, "tools.py")
            prompt_path = os.path.join(item_path, "prompt.txt")
            if not (os.path.exists(manifest_path) or os.path.exists(tools_path) or os.path.exists(prompt_path)):
                continue
            skill = existing.get(item)
            if skill and skill.path == item_path:
                skill.refresh_if_stale()
                self.skills[item] = skill
            else:
                self.skills[item] = Skill(item_path)

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        self.refresh_skills()
        return self.skills.get(skill_id)

    def list_skills(self) -> List[Dict[str, Any]]:
        self.refresh_skills()
        return [skill.get_info() for skill in self.skills.values()]

    def catalog_snapshot(self) -> List[Dict[str, Any]]:
        out = []
        for skill in self.skills.values():
            info = skill.get_info()
            out.append(
                {
                    "id": info["id"],
                    "name": info["name"],
                    "entry_point": info["entry_point"],
                    "tool_count": len(info["tool_names"]),
                    "health": info["health"],
                    "capabilities": info["capabilities"][:10],
                }
            )
        return out

    def recommend_skills(self, query: str, *, active_skill_id: str = "", limit: int = 5) -> List[Dict[str, Any]]:
        self.refresh_skills()
        ranked: List[Dict[str, Any]] = []
        for skill in self.skills.values():
            score = skill.score_against(query)
            if active_skill_id and skill.id == active_skill_id:
                score += 100.0
            reason = "Primary request skill" if skill.id == active_skill_id else "Capability overlap"
            ranked.append(
                {
                    **skill.get_info(),
                    "score": round(score, 3),
                    "reason": reason,
                }
            )
        ranked.sort(key=lambda item: (float(item.get("score") or 0.0), item.get("id")), reverse=True)
        return ranked[: max(1, limit)]

    def get_tool_functions(self, skill_id: str) -> List[Any]:
        skill = self.get_skill(skill_id)
        if skill and skill.tools_module:
            return [
                func for name, func in vars(skill.tools_module).items()
                if callable(func) and not name.startswith("__")
            ]
        return []


def get_skill_manager(skills_dir: str = None) -> SkillManager:
    return SkillManager(skills_dir)
