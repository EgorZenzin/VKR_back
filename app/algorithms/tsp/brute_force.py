"""Полный перебор для задачи коммивояжёра (малые n)."""

import time
import itertools
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class BruteForceTSP(BaseAlgorithm):
    """Полный перебор всех перестановок.

    Гарантирует оптимальное решение. Применим только для малых n (≤ 12),
    так как сложность O(n!).
    """

    name = "brute_force"
    display_name = "Полный перебор"

    MAX_CITIES = 12  # Ограничение по числу городов

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        dist_matrix = _get_distance_matrix(input_data)
        n = len(dist_matrix)

        if n > self.MAX_CITIES:
            raise ValueError(
                f"Полный перебор допустим для n ≤ {self.MAX_CITIES}, получено n = {n}"
            )

        start = time.perf_counter()

        # Фиксируем город 0 как начальный (симметрия маршрута)
        best_route = None
        best_cost = float("inf")
        checked = 0

        for perm in itertools.permutations(range(1, n)):
            route = [0] + list(perm)
            cost = sum(dist_matrix[route[i]][route[i + 1]] for i in range(n - 1))
            cost += dist_matrix[route[-1]][route[0]]
            checked += 1

            if cost < best_cost:
                best_cost = cost
                best_route = route

        elapsed = time.perf_counter() - start

        return AlgorithmResult(
            solution=best_route,
            cost=float(best_cost),
            execution_time=elapsed,
            iterations=checked,
            convergence_history=[float(best_cost)],
        )


def _get_distance_matrix(input_data: dict) -> np.ndarray:
    """Построить матрицу расстояний из координат городов или готовой матрицы."""
    if "distance_matrix" in input_data and input_data["distance_matrix"]:
        return np.array(input_data["distance_matrix"])
    cities = input_data["cities"]
    coords = np.array([[c["x"], c["y"]] for c in cities])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))
