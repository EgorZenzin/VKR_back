"""Сервис решения задач.

Оркестрирует запуск алгоритма: получает алгоритм из реестра,
вызывает solve(), формирует унифицированный ответ с данными визуализации.
"""

from __future__ import annotations

from app.algorithms.registry import (
    get_algorithm,
    TASK_DISPLAY_NAMES,
)
from app.algorithms.base import AlgorithmResult
from app.schemas.common import SolveResponse
from app.utils.helpers import convert_numpy
from app.visualization_data.chart_data import build_visualization_data


def solve_problem(
    task_name: str,
    algorithm_name: str,
    input_data: dict,
    params: dict | None = None,
) -> SolveResponse:
    """Решить задачу выбранным алгоритмом и вернуть SolveResponse."""

    algo = get_algorithm(task_name, algorithm_name)
    result: AlgorithmResult = algo.solve(input_data, params)

    # Подготовка данных визуализации
    viz_data = build_visualization_data(
        task_name, input_data, result.solution, result.extra,
    )

    # Извлечение ML-метрик из extra (если алгоритм с ML).
    # Берём все поля extra, кроме известных полей предметных данных.
    ml_metrics = None
    if result.extra.get("ml_used"):
        _NON_ML_KEYS = {"total_weight"}
        ml_metrics = {
            k: v for k, v in result.extra.items() if k not in _NON_ML_KEYS
        }

    return SolveResponse(
        task_name=task_name,
        task_display_name=TASK_DISPLAY_NAMES.get(task_name, task_name),
        algorithm_name=algorithm_name,
        display_name=algo.display_name,
        input_data=convert_numpy(input_data),
        solution=convert_numpy(result.solution),
        objective_value=float(result.cost),
        execution_time=round(result.execution_time, 6),
        iterations=result.iterations,
        convergence_history=[float(v) for v in result.convergence_history],
        visualization_data=convert_numpy(viz_data),
        ml_metrics=ml_metrics,
    )
