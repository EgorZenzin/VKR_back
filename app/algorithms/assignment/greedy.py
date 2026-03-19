import numpy as np
import time

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class GreedyAssignment(BaseAlgorithm):
    """Жадный алгоритм для задачи о назначениях."""

    name = "greedy"
    display_name = "Жадный алгоритм"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        cost_matrix = np.array(input_data["cost_matrix"])
        n = len(cost_matrix)

        start = time.perf_counter()

        assigned_cols = set()
        assignments = []
        total_cost = 0.0

        # Сортируем все ячейки по стоимости
        cells = []
        for i in range(n):
            for j in range(n):
                cells.append((cost_matrix[i][j], i, j))
        cells.sort()

        assigned_rows = set()
        for cost, i, j in cells:
            if i not in assigned_rows and j not in assigned_cols:
                assignments.append((i, j))
                assigned_rows.add(i)
                assigned_cols.add(j)
                total_cost += cost
            if len(assignments) == n:
                break

        elapsed = time.perf_counter() - start

        assignments.sort()
        return AlgorithmResult(
            solution=assignments,
            cost=float(total_cost),
            execution_time=elapsed,
        )
