"""Жадный алгоритм (ближайший сосед) для задачи коммивояжёра."""

import time
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class GreedyTSP(BaseAlgorithm):
    """Жадный алгоритм ближайшего соседа.

    Стратегия: из каждого города выбирается ближайший непосещённый сосед.
    Перебираются все стартовые города, возвращается лучший маршрут.
    Сложность: O(n²).
    """

    name = "greedy"
    display_name = "Жадный алгоритм (ближайший сосед)"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        dist_matrix = _get_distance_matrix(input_data)
        n = len(dist_matrix)

        start = time.perf_counter()

        best_route = None
        best_cost = float("inf")

        for start_city in range(n):
            route = [start_city]
            visited = {start_city}
            current = start_city
            cost = 0.0

            for _ in range(n - 1):
                nearest = None
                nearest_dist = float("inf")
                for j in range(n):
                    if j not in visited and dist_matrix[current][j] < nearest_dist:
                        nearest = j
                        nearest_dist = dist_matrix[current][j]
                route.append(nearest)
                visited.add(nearest)
                cost += nearest_dist
                current = nearest

            cost += dist_matrix[current][start_city]

            if cost < best_cost:
                best_cost = cost
                best_route = route

        elapsed = time.perf_counter() - start

        return AlgorithmResult(
            solution=best_route,
            cost=float(best_cost),
            execution_time=elapsed,
            iterations=None,
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
