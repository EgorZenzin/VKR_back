"""Имитация отжига для задачи о рюкзаке."""

import random
import math
import time

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class SimulatedAnnealingKnapsack(BaseAlgorithm):
    """Имитация отжига для задачи о рюкзаке."""

    name = "simulated_annealing"
    display_name = "Имитация отжига"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        initial_temp = params.get("initial_temp", 1000.0)
        cooling_rate = params.get("cooling_rate", 0.999)
        min_temp = params.get("min_temp", 1e-6)

        items = input_data["items"]
        capacity = input_data["capacity"]
        n = len(items)

        weights = [item["weight"] for item in items]
        values = [item["value"] for item in items]

        start = time.perf_counter()

        # Начальное решение — жадное по ценности/весу
        indexed = sorted(range(n), key=lambda i: values[i] / max(weights[i], 1e-9), reverse=True)
        current = [0] * n
        total_w = 0.0
        for i in indexed:
            if total_w + weights[i] <= capacity:
                current[i] = 1
                total_w += weights[i]

        current_value = self._evaluate(current, weights, values, capacity)

        best = current[:]
        best_value = current_value
        temp = initial_temp
        convergence = []
        iteration = 0

        while temp > min_temp:
            # Сосед: переключить один предмет
            neighbor = current[:]
            j = random.randint(0, n - 1)
            neighbor[j] = 1 - neighbor[j]
            neighbor_value = self._evaluate(neighbor, weights, values, capacity)

            # Максимизация → delta = current - neighbor (принимаем если neighbor лучше)
            delta = current_value - neighbor_value
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current = neighbor
                current_value = neighbor_value

            if current_value > best_value:
                best = current[:]
                best_value = current_value

            temp *= cooling_rate
            iteration += 1
            if iteration % 100 == 0:
                convergence.append(best_value)

        elapsed = time.perf_counter() - start

        selected = [i for i in range(n) if best[i] == 1]
        total_weight = sum(weights[i] for i in selected)

        return AlgorithmResult(
            solution=selected,
            cost=best_value,
            execution_time=elapsed,
            convergence_history=convergence,
            extra={"total_weight": total_weight},
        )

    def _evaluate(self, solution: list[int], weights: list, values: list, capacity: float) -> float:
        total_w = sum(weights[i] * solution[i] for i in range(len(solution)))
        total_v = sum(values[i] * solution[i] for i in range(len(solution)))
        if total_w > capacity:
            return 0.0
        return total_v
