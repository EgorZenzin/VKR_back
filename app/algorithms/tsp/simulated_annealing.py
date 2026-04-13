"""Имитация отжига для задачи коммивояжёра."""

import time
import math
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class SimulatedAnnealingTSP(BaseAlgorithm):
    """Имитация отжига (Simulated Annealing) с 2-opt соседством.

    Начальное решение — случайная перестановка. Соседнее решение получается
    реверсом подмассива (2-opt). История сходимости записывается каждые 100 шагов.
    """

    name = "simulated_annealing"
    display_name = "Имитация отжига"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        initial_temp = params.get("initial_temp", 10000.0)
        cooling_rate = params.get("cooling_rate", 0.9995)
        min_temp = params.get("min_temp", 1e-8)

        dist_matrix = _get_distance_matrix(input_data)
        n = len(dist_matrix)

        start = time.perf_counter()

        current = list(np.random.permutation(n))
        current_cost = _route_cost(current, dist_matrix)

        best = current[:]
        best_cost = current_cost
        temp = initial_temp
        convergence = []
        iteration = 0

        while temp > min_temp:
            # 2-opt: реверс случайного подмассива
            i, j = sorted(random.sample(range(n), 2))
            neighbor = current[:]
            neighbor[i:j + 1] = reversed(neighbor[i:j + 1])
            neighbor_cost = _route_cost(neighbor, dist_matrix)

            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current = neighbor
                current_cost = neighbor_cost

            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

            temp *= cooling_rate
            iteration += 1

            if iteration % 100 == 0:
                convergence.append(float(best_cost))

        elapsed = time.perf_counter() - start

        # Финальная точка, если не записана
        if not convergence or convergence[-1] != best_cost:
            convergence.append(float(best_cost))

        return AlgorithmResult(
            solution=best,
            cost=float(best_cost),
            execution_time=elapsed,
            iterations=iteration,
            convergence_history=convergence,
        )


def _route_cost(route: list[int], dist_matrix: np.ndarray) -> float:
    cost = sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
    cost += dist_matrix[route[-1]][route[0]]
    return float(cost)


def _get_distance_matrix(input_data: dict) -> np.ndarray:
    if "distance_matrix" in input_data and input_data["distance_matrix"]:
        return np.array(input_data["distance_matrix"])
    cities = input_data["cities"]
    coords = np.array([[c["x"], c["y"]] for c in cities])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))
import numpy as np
import random
import math
import time
