"""Жадный алгоритм для задачи о рюкзаке."""

import time

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class GreedyKnapsack(BaseAlgorithm):
    """Жадный алгоритм по удельной ценности (value / weight).

    Предметы сортируются по убыванию удельной ценности и добавляются
    в рюкзак, пока позволяет ёмкость. Сложность: O(n log n).
    """

    name = "greedy"
    display_name = "Жадный алгоритм"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        items = input_data["items"]
        capacity = input_data["capacity"]

        start = time.perf_counter()

        indexed = []
        for i, item in enumerate(items):
            ratio = item["value"] / item["weight"] if item["weight"] > 0 else 0
            indexed.append((ratio, i, item))

        indexed.sort(reverse=True)

        selected = []
        total_value = 0.0
        total_weight = 0.0

        for ratio, i, item in indexed:
            if total_weight + item["weight"] <= capacity:
                selected.append(i)
                total_value += item["value"]
                total_weight += item["weight"]

        elapsed = time.perf_counter() - start
        selected.sort()

        return AlgorithmResult(
            solution=selected,
            cost=total_value,
            execution_time=elapsed,
            iterations=None,
            convergence_history=[total_value],
            extra={"total_weight": total_weight},
        )
import time
