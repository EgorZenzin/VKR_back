"""Динамическое программирование для задачи о рюкзаке."""

import time

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class DynamicKnapsack(BaseAlgorithm):
    """Динамическое программирование (0-1 Knapsack).

    Гарантирует оптимальное решение для целочисленных весов.
    Для вещественных весов производится масштабирование до целых.
    Сложность: O(n · W).
    """

    name = "dynamic_programming"
    display_name = "Динамическое программирование"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        items = input_data["items"]
        capacity = input_data["capacity"]
        n = len(items)

        start = time.perf_counter()

        weights = [item["weight"] for item in items]
        values = [item["value"] for item in items]

        # Масштабирование вещественных весов
        scale = 1
        if any(isinstance(w, float) and w != int(w) for w in weights):
            scale = 100
            int_weights = [int(round(w * scale)) for w in weights]
            int_capacity = int(round(capacity * scale))
        else:
            int_weights = [int(w) for w in weights]
            int_capacity = int(capacity)

        dp = [0.0] * (int_capacity + 1)
        keep = [[False] * (int_capacity + 1) for _ in range(n)]

        for i in range(n):
            for w in range(int_capacity, int_weights[i] - 1, -1):
                if dp[w - int_weights[i]] + values[i] > dp[w]:
                    dp[w] = dp[w - int_weights[i]] + values[i]
                    keep[i][w] = True

        # Восстановление решения (backtracking)
        selected = []
        w = int_capacity
        for i in range(n - 1, -1, -1):
            if keep[i][w]:
                selected.append(i)
                w -= int_weights[i]
        selected.reverse()

        total_value = sum(values[i] for i in selected)
        total_weight = sum(weights[i] for i in selected)

        elapsed = time.perf_counter() - start

        return AlgorithmResult(
            solution=selected,
            cost=total_value,
            execution_time=elapsed,
            iterations=None,
            convergence_history=[total_value],
            extra={"total_weight": total_weight},
        )
import time
