"""Пакет Pydantic-схем."""

from app.schemas.common import (
    SolveRequest,
    SolveResponse,
    CompareRequest,
    CompareResponse,
    AlgorithmResultResponse,
    TaskInfo,
    AlgorithmInfo,
    ErrorResponse,
    ComparisonCharts,
    ConvergenceSeries,
    BarChartData,
)

__all__ = [
    "SolveRequest",
    "SolveResponse",
    "CompareRequest",
    "CompareResponse",
    "AlgorithmResultResponse",
    "TaskInfo",
    "AlgorithmInfo",
    "ErrorResponse",
    "ComparisonCharts",
    "ConvergenceSeries",
    "BarChartData",
]
