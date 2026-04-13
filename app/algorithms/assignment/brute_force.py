"""Полный перебор для задачи назначения (малые n)."""

import time
import itertools
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class BruteForceAssignment(BaseAlgorithm):
    """Полный перебор всех перестановок назначений.

    Гарантирует оптимальное решение. Применим для n ≤ 10,
    так как сложность O(n!).
    """

    name = "brute_force"
    display_name = "Полный перебор"

    MAX_SIZE = 10

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        cost_matrix = np.array(input_data["cost_matrix"])
        n = len(cost_matrix)

        if n > self.MAX_SIZE:
            raise ValueError(
                f"Полный перебор допустим для n ≤ {self.MAX_SIZE}, получено n = {n}"
            )

        start = time.perf_counter()

        best_perm = None
        best_cost = float("inf")
        checked = 0

        for perm in itertools.permutations(range(n)):
            cost = sum(cost_matrix[i][perm[i]] for i in range(n))
            checked += 1
            if cost < best_cost:
                best_cost = cost
                best_perm = perm

        elapsed = time.perf_counter() - start

        assignments = [[i, best_perm[i]] for i in range(n)]

        return AlgorithmResult(
            solution=assignments,
            cost=float(best_cost),
            execution_time=elapsed,
            iterations=checked,
            convergence_history=[float(best_cost)],
        )
