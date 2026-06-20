from __future__ import annotations

from pydantic import BaseModel


class AgentDescriptor(BaseModel):
    name: str
    status: str = "placeholder"
