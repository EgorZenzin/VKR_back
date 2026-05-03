"""Схемы истории запусков."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class SolveHistoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    task_name: str
    algorithm: str
    input_data: dict[str, Any]
    params: dict[str, Any] | None
    result: dict[str, Any]
    objective_value: float | None
    elapsed_ms: float | None
    created_at: datetime


class ComparisonHistoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    task_name: str
    algorithms: list[str]
    input_data: dict[str, Any]
    params: dict[str, Any] | None
    result: dict[str, Any]
    algorithms_count: int
    created_at: datetime
