"""Венгерский алгоритм для задачи назначения."""

import time
import numpy as np
from scipy.optimize import linear_sum_assignment

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class HungarianAssignment(BaseAlgorithm):
    """Венгерский алгоритм (метод Куна-Манкреса).

    Гарантирует оптимальное решение за полиномиальное время O(n³).
    Использует реализацию scipy.optimize.linear_sum_assignment.
    """

    name = "hungarian"
    display_name = "Венгерский алгоритм"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        cost_matrix = np.array(input_data["cost_matrix"])

        start = time.perf_counter()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        elapsed = time.perf_counter() - start

        assignments = [[int(r), int(c)] for r, c in zip(row_ind, col_ind)]
        total_cost = float(cost_matrix[row_ind, col_ind].sum())

        return AlgorithmResult(
            solution=assignments,
            cost=total_cost,
            execution_time=elapsed,
            iterations=None,
            convergence_history=[total_cost],
        )
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
