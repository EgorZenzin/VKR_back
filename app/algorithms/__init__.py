"""Пакет алгоритмов комбинаторной оптимизации."""

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.algorithms.registry import (
    TASK_REGISTRY,
    get_task_names,
    get_algorithms_for_task,
    get_algorithm,
)

__all__ = [
    "BaseAlgorithm",
    "AlgorithmResult",
    "TASK_REGISTRY",
    "get_task_names",
    "get_algorithms_for_task",
    "get_algorithm",
]
