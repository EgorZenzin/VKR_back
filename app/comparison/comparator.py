"""Модуль сравнения алгоритмов.

Формирует данные для графиков сравнения: сходимость, время, качество.
"""

from __future__ import annotations

from app.schemas.common import (
    AlgorithmResultResponse,
    ComparisonCharts,
    ConvergenceSeries,
    BarChartData,
)


def build_comparison_charts(results: list[AlgorithmResultResponse]) -> ComparisonCharts:
    """Подготовить данные графиков сравнения на основе списка результатов."""

    # 1. График сходимости
    convergence_series = []
    for r in results:
        data_points = []
        if r.convergence_history:
            for i, value in enumerate(r.convergence_history):
                data_points.append({"iteration": i, "value": value})
        else:
            # Для неитерационных алгоритмов — одна финальная точка
            data_points.append({"iteration": 0, "value": r.objective_value})

        convergence_series.append(ConvergenceSeries(
            algorithm_name=r.algorithm_name,
            display_name=r.display_name,
            data=data_points,
        ))

    # 2. Столбчатая диаграмма: сравнение времени работы
    time_chart = BarChartData(
        labels=[r.display_name for r in results],
        values=[r.execution_time for r in results],
    )

    # 3. Столбчатая диаграмма: сравнение качества решения
    quality_chart = BarChartData(
        labels=[r.display_name for r in results],
        values=[r.objective_value for r in results],
    )

    return ComparisonCharts(
        convergence=convergence_series,
        time_comparison=time_chart,
        quality_comparison=quality_chart,
    )
