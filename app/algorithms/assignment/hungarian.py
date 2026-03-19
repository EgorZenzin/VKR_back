import numpy as np
import time
from scipy.optimize import linear_sum_assignment

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class HungarianAssignment(BaseAlgorithm):
    """Венгерский алгоритм для задачи о назначениях."""

    name = "hungarian"
    display_name = "Венгерский алгоритм"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        cost_matrix = np.array(input_data["cost_matrix"])

        start = time.perf_counter()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        elapsed = time.perf_counter() - start

        assignments = list(zip(row_ind.tolist(), col_ind.tolist()))
        total_cost = cost_matrix[row_ind, col_ind].sum()

        return AlgorithmResult(
            solution=assignments,
            cost=float(total_cost),
            execution_time=elapsed,
        )
