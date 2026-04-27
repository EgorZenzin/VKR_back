"""Сервис сравнения алгоритмов.

Запускает несколько алгоритмов на одних и тех же входных данных
и формирует сводный ответ с данными для графиков сравнения.
"""

from __future__ import annotations

from app.algorithms.registry import (
    get_algorithm,
    get_algorithms_for_task,
    TASK_DISPLAY_NAMES,
)
from app.schemas.common import (
    AlgorithmResultResponse,
    CompareResponse,
)
from app.comparison.comparator import build_comparison_charts
from app.utils.helpers import convert_numpy
from app.visualization_data.chart_data import build_visualization_data


def compare_algorithms(
    task_name: str,
    algorithm_names: list[str],
    input_data: dict,
    params: dict[str, dict] | None = None,
) -> CompareResponse:
    """Сравнить несколько алгоритмов на одних входных данных."""

    # Проверяем, что все запрошенные алгоритмы существуют
    available = get_algorithms_for_task(task_name)
    for name in algorithm_names:
        if name not in available:
            raise KeyError(
                f"Алгоритм '{name}' не найден для задачи '{task_name}'. "
                f"Доступны: {list(available.keys())}"
            )

    params = params or {}
    results: list[AlgorithmResultResponse] = []

    for algo_name in algorithm_names:
        algo = get_algorithm(task_name, algo_name)
        algo_params = params.get(algo_name)

        result = algo.solve(input_data, algo_params)

        viz_data = build_visualization_data(
            task_name, input_data, result.solution, result.extra,
        )

        # Извлечение ML-метрик: все поля extra, кроме предметных данных.
        ml_metrics = None
        if result.extra.get("ml_used"):
            _NON_ML_KEYS = {"total_weight"}
            ml_metrics = {
                k: v for k, v in result.extra.items() if k not in _NON_ML_KEYS
            }

        results.append(AlgorithmResultResponse(
            algorithm_name=algo_name,
            display_name=algo.display_name,
            solution=convert_numpy(result.solution),
            objective_value=float(result.cost),
            execution_time=round(result.execution_time, 6),
            iterations=result.iterations,
            convergence_history=[float(v) for v in result.convergence_history],
            visualization_data=convert_numpy(viz_data),
            ml_metrics=ml_metrics,
        ))

    # Формируем данные для графиков сравнения
    comparison_charts = build_comparison_charts(results)

    return CompareResponse(
        task_name=task_name,
        task_display_name=TASK_DISPLAY_NAMES.get(task_name, task_name),
        input_data=convert_numpy(input_data),
        algorithms=algorithm_names,
        results=results,
        comparison_charts=comparison_charts,
    )
