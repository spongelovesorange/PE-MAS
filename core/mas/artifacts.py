from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ArtifactStore:
    """Filesystem artifact writer for long-running MAS runs."""

    def __init__(self, root: Path | str = "runs") -> None:
        self.root = Path(root)

    def run_dir(self, project_id: str, run_id: str) -> Path:
        path = self.root / project_id / run_id
        path.mkdir(parents=True, exist_ok=True)
        (path / "artifacts").mkdir(exist_ok=True)
        return path

    def write_json(self, project_id: str, run_id: str, name: str, payload: Dict[str, Any]) -> str:
        path = self.run_dir(project_id, run_id) / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def append_trace(self, project_id: str, run_id: str, record: Dict[str, Any]) -> str:
        path = self.run_dir(project_id, run_id) / "trace.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return str(path)

