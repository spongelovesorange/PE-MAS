from __future__ import annotations

from pydantic import BaseModel


class PlecsModelEntry(BaseModel):
    topology: str
    status: str
    local_model_path: str = ""
    validation_status: str = "unknown"
    notes: str = ""

