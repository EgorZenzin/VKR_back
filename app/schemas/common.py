"""Общие Pydantic-схемы для запросов и ответов API."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any


# ── Запросы ─────────────────────────────────────────────────────────

class SolveRequest(BaseModel):
    """Запрос на решение одной задачи одним алгоритмом."""

    task_name: str = Field(..., description="Идентификатор задачи (tsp, knapsack, assignment)")
    algorithm: str = Field(..., description="Идентификатор алгоритма (greedy, genetic и т.д.)")
    input_data: dict = Field(..., description="Входные данные задачи")
    params: dict | None = Field(default=None, description="Параметры алгоритма (опционально)")


class CompareRequest(BaseModel):
    """Запрос на сравнение нескольких алгоритмов на одной задаче."""

    task_name: str = Field(..., description="Идентификатор задачи")
    algorithms: list[str] = Field(
        ..., min_length=2,
        description="Список алгоритмов для сравнения (минимум 2)",
    )
    input_data: dict = Field(..., description="Входные данные задачи")
    params: dict[str, dict] | None = Field(
        default=None,
        description="Параметры для каждого алгоритма: {algorithm_name: {param: value}}",
    )


# ── Ответы ──────────────────────────────────────────────────────────

class AlgorithmResultResponse(BaseModel):
    """Результат работы одного алгоритма (используется и в solve, и в compare)."""

    algorithm_name: str
    display_name: str
    solution: Any
    objective_value: float
    execution_time: float
    iterations: int | None = None
    convergence_history: list[float] = []
    visualization_data: dict = {}
    ml_metrics: dict | None = None


class SolveResponse(BaseModel):
    """Полный ответ на POST /solve."""

    task_name: str
    task_display_name: str
    algorithm_name: str
    display_name: str
    input_data: dict
    solution: Any
    objective_value: float
    execution_time: float
    iterations: int | None = None
    convergence_history: list[float] = []
    visualization_data: dict = {}
    ml_metrics: dict | None = None


class ConvergenceSeries(BaseModel):
    """Данные одной серии для графика сходимости."""

    algorithm_name: str
    display_name: str
    data: list[dict[str, float]]  # [{iteration: 0, value: 123.4}, ...]


class BarChartData(BaseModel):
    """Данные для столбчатой диаграммы."""

    labels: list[str]
    values: list[float]


class ComparisonCharts(BaseModel):
    """Подготовленные данные для графиков сравнения."""

    convergence: list[ConvergenceSeries]
    time_comparison: BarChartData
    quality_comparison: BarChartData


class CompareResponse(BaseModel):
    """Полный ответ на POST /compare."""

    task_name: str
    task_display_name: str
    input_data: dict
    algorithms: list[str]
    results: list[AlgorithmResultResponse]
    comparison_charts: ComparisonCharts


# ── Вспомогательные схемы ───────────────────────────────────────────

class TaskInfo(BaseModel):
    """Информация о задаче для GET /tasks."""

    name: str
    display_name: str
    description: str
    optimization: str  # minimize / maximize
    algorithms: list[AlgorithmInfo]


class AlgorithmInfo(BaseModel):
    """Информация об алгоритме."""

    name: str
    display_name: str


class ErrorResponse(BaseModel):
    """Стандартный ответ об ошибке."""

    detail: str
