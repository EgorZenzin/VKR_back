"""Подготовка данных для визуализации на фронтенде.

Модуль формирует структурированные словари, которые фронтенд
может напрямую использовать для отрисовки графиков (chart.js, plotly и т.д.).
"""

from __future__ import annotations

from typing import Any
import numpy as np


def prepare_tsp_visualization(input_data: dict, solution: list[int]) -> dict:
    """Данные для визуализации маршрута TSP.

    Возвращает:
        - cities: координаты городов
        - route_order: порядок обхода
        - route_coordinates: координаты точек маршрута (замкнутый)
    """
    cities = input_data.get("cities", [])

    viz: dict[str, Any] = {
        "cities": cities,
        "route_order": solution,
    }

    if cities and solution:
        # Замкнутый маршрут: добавляем первый город в конец
        route_coords = []
        for idx in solution:
            if 0 <= idx < len(cities):
                route_coords.append({"x": cities[idx]["x"], "y": cities[idx]["y"]})
        # Замыкаем маршрут
        if route_coords:
            route_coords.append(route_coords[0])
        viz["route_coordinates"] = route_coords

    return viz


def prepare_knapsack_visualization(input_data: dict, solution: list[int], extra: dict) -> dict:
    """Данные для визуализации решения задачи о рюкзаке.

    Возвращает:
        - items: все предметы
        - selected_indices: индексы выбранных предметов
        - selected_items: выбранные предметы с деталями
        - total_weight: суммарный вес
        - total_value: суммарная ценность
        - capacity: ёмкость рюкзака
    """
    items = input_data.get("items", [])
    capacity = input_data.get("capacity", 0)

    selected_items = []
    for idx in solution:
        if 0 <= idx < len(items):
            item = items[idx].copy() if isinstance(items[idx], dict) else items[idx]
            selected_items.append({"index": idx, **item})

    return {
        "items": items,
        "selected_indices": solution,
        "selected_items": selected_items,
        "total_weight": extra.get("total_weight", 0),
        "total_value": sum(items[i]["value"] for i in solution if 0 <= i < len(items)),
        "capacity": capacity,
    }


def prepare_assignment_visualization(input_data: dict, solution: list) -> dict:
    """Данные для визуализации решения задачи назначения.

    Возвращает:
        - cost_matrix: матрица стоимости
        - assignments: итоговое распределение [[worker, task], ...]
        - assignment_costs: стоимость каждого назначения
    """
    cost_matrix = input_data.get("cost_matrix", [])

    assignment_costs = []
    for pair in solution:
        i, j = pair[0], pair[1]
        if 0 <= i < len(cost_matrix) and 0 <= j < len(cost_matrix[i]):
            assignment_costs.append({
                "worker": i,
                "task": j,
                "cost": cost_matrix[i][j],
            })

    return {
        "cost_matrix": cost_matrix,
        "assignments": solution,
        "assignment_costs": assignment_costs,
    }


# ── Диспетчер по типу задачи ───────────────────────────────────────

_VIZ_BUILDERS: dict[str, Any] = {
    "tsp": lambda inp, sol, extra: prepare_tsp_visualization(inp, sol),
    "knapsack": lambda inp, sol, extra: prepare_knapsack_visualization(inp, sol, extra),
    "assignment": lambda inp, sol, extra: prepare_assignment_visualization(inp, sol),
}


def build_visualization_data(task_name: str, input_data: dict, solution: Any, extra: dict) -> dict:
    """Построить данные для визуализации по типу задачи."""
    builder = _VIZ_BUILDERS.get(task_name)
    if builder:
        return builder(input_data, solution, extra)
    return {}
