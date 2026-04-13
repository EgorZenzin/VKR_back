"""Жадный алгоритм для задачи назначения."""

import time
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class GreedyAssignment(BaseAlgorithm):
    """Жадный алгоритм назначения.

    Сортирует все ячейки матрицы стоимости по возрастанию и последовательно
    назначает пары (работник, задание), пропуская уже занятые строки/столбцы.
    Сложность: O(n² log n).
    """

    name = "greedy"
    display_name = "Жадный алгоритм"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        cost_matrix = np.array(input_data["cost_matrix"])
        n = len(cost_matrix)

        start = time.perf_counter()

        # Сортируем все ячейки по стоимости
        cells = []
        for i in range(n):
            for j in range(n):
                cells.append((float(cost_matrix[i][j]), i, j))
        cells.sort()

        assigned_rows: set[int] = set()
        assigned_cols: set[int] = set()
        assignments = []
        total_cost = 0.0

        for cost, i, j in cells:
            if i not in assigned_rows and j not in assigned_cols:
                assignments.append([i, j])
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
            iterations=None,
            convergence_history=[float(total_cost)],
        )
import numpy as np
import time
