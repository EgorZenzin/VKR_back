"""Полный перебор для задачи о рюкзаке (малые n)."""

import time
import itertools

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class BruteForceKnapsack(BaseAlgorithm):
    """Полный перебор всех подмножеств предметов.

    Гарантирует оптимальное решение. Применим только для n ≤ 20,
    так как сложность O(2^n).
    """

    name = "brute_force"
    display_name = "Полный перебор"

    MAX_ITEMS = 20

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        items = input_data["items"]
        capacity = input_data["capacity"]
        n = len(items)

        if n > self.MAX_ITEMS:
            raise ValueError(
                f"Полный перебор допустим для n ≤ {self.MAX_ITEMS}, получено n = {n}"
            )

        weights = [item["weight"] for item in items]
        values = [item["value"] for item in items]

        start = time.perf_counter()

        best_value = 0.0
        best_weight = 0.0
        best_subset: list[int] = []
        checked = 0

        # Перебор всех подмножеств через битовые маски
        for mask in range(1 << n):
            total_w = 0.0
            total_v = 0.0
            subset = []
            for i in range(n):
                if mask & (1 << i):
                    total_w += weights[i]
                    total_v += values[i]
                    subset.append(i)
            checked += 1

            if total_w <= capacity and total_v > best_value:
                best_value = total_v
                best_weight = total_w
                best_subset = subset

        elapsed = time.perf_counter() - start

        return AlgorithmResult(
            solution=best_subset,
            cost=best_value,
            execution_time=elapsed,
            iterations=checked,
            convergence_history=[best_value],
            extra={"total_weight": best_weight},
        )
