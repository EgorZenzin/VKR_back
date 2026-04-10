"""Имитация отжига для задачи о назначениях."""

import random
import math
import time
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class SimulatedAnnealingAssignment(BaseAlgorithm):
    """Имитация отжига для задачи о назначениях."""

    name = "simulated_annealing"
    display_name = "Имитация отжига"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        initial_temp = params.get("initial_temp", 1000.0)
        cooling_rate = params.get("cooling_rate", 0.999)
        min_temp = params.get("min_temp", 1e-6)

        cost_matrix = np.array(input_data["cost_matrix"])
        n = len(cost_matrix)

        start = time.perf_counter()

        # Начальное решение — случайная перестановка столбцов
        current = list(np.random.permutation(n))
        current_cost = self._calc_cost(current, cost_matrix)

        best = current[:]
        best_cost = current_cost
        temp = initial_temp
        convergence = []
        iteration = 0

        while temp > min_temp:
            # Сосед: обмен двух назначений
            neighbor = current[:]
            i, j = random.sample(range(n), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbor_cost = self._calc_cost(neighbor, cost_matrix)

            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current = neighbor
                current_cost = neighbor_cost

            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

            temp *= cooling_rate
            iteration += 1
            if iteration % 100 == 0:
                convergence.append(best_cost)

        elapsed = time.perf_counter() - start

        assignments = [(i, best[i]) for i in range(n)]
        assignments.sort()

        return AlgorithmResult(
            solution=assignments,
            cost=float(best_cost),
            execution_time=elapsed,
            convergence_history=convergence,
        )

    def _calc_cost(self, assignment: list[int], cost_matrix: np.ndarray) -> float:
        return float(sum(cost_matrix[i][assignment[i]] for i in range(len(assignment))))
