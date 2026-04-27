"""Жадный алгоритм с ML-суррогатом для задачи назначения.

Подход: ML-суррогат отбирает лучшие жадные назначения из множества
рандомизированных вариантов.

1. Обучение суррогата на случайных перестановках с известной стоимостью.
2. Генерация пула жадных решений со случайным выбором из top-k самых
   дешёвых доступных ячеек.
3. Быстрая оценка всех вариантов суррогатом.
4. Точный пересчёт top-K назначений и возврат лучшего.
"""

import time
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.ml.surrogate import MLPSurrogateModel


class GreedyMLAssignment(BaseAlgorithm):
    """Жадный алгоритм + ML-суррогат для задачи назначения."""

    name = "greedy_ml"
    display_name = "Жадный алгоритм + ML"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        warmup_samples = params.get("warmup_samples", 150)
        candidate_count = params.get("candidate_count", 200)
        top_k_exact = params.get("top_k_exact", 20)
        top_k_cells = params.get("top_k_cells", 3)

        cost_matrix = np.array(input_data["cost_matrix"])
        n = len(cost_matrix)

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (32, 16)),
        )

        start = time.perf_counter()

        # ── Фаза 1: обучение суррогата на случайных перестановках ──
        sample_pool = [list(np.random.permutation(n)) for _ in range(warmup_samples)]
        train_X = np.array([_encode_perm(p, n) for p in sample_pool])
        train_y = np.array([_perm_cost(p, cost_matrix) for p in sample_pool])
        surrogate.fit(train_X, train_y)

        exact_evals = warmup_samples

        # ── Фаза 2: рандомизированные жадные кандидаты ─────────────
        candidates = []
        for _ in range(candidate_count):
            perm = _randomized_greedy_assignment(cost_matrix, top_k_cells)
            candidates.append(perm)

        X_cand = np.array([_encode_perm(c, n) for c in candidates])
        pred_costs = surrogate.predict(X_cand)
        surrogate_evals = candidate_count

        # ── Фаза 3: точный пересчёт top-K ──────────────────────────
        top_indices = np.argsort(pred_costs)[:top_k_exact]
        best_perm = None
        best_cost = float("inf")

        actual_costs = []
        for idx in top_indices:
            c = _perm_cost(candidates[idx], cost_matrix)
            actual_costs.append(c)
            exact_evals += 1
            if c < best_cost:
                best_cost = c
                best_perm = candidates[idx][:]

        surrogate_r2 = surrogate.score(
            X_cand[top_indices], np.array(actual_costs)
        )

        elapsed = time.perf_counter() - start

        assignments = [[i, best_perm[i]] for i in range(n)] if best_perm else []

        return AlgorithmResult(
            solution=assignments,
            cost=float(best_cost),
            execution_time=elapsed,
            iterations=None,
            convergence_history=[float(best_cost)],
            extra={
                "ml_used": True,
                "surrogate_model": "MLPRegressor",
                "exact_evaluations": exact_evals,
                "surrogate_evaluations": surrogate_evals,
                "surrogate_accuracy_r2": round(float(surrogate_r2), 4),
                "warmup_samples": warmup_samples,
                "candidate_count": candidate_count,
                "top_k_exact": top_k_exact,
                "top_k_cells": top_k_cells,
                "training_samples": warmup_samples,
            },
        )


# ── Вспомогательные функции ─────────────────────────────────────────

def _randomized_greedy_assignment(
    cost_matrix: np.ndarray, top_k: int,
) -> list[int]:
    """Жадный алгоритм со случайным выбором из top-k самых дешёвых ячеек."""
    n = len(cost_matrix)
    cells = []
    for i in range(n):
        for j in range(n):
            cells.append((float(cost_matrix[i][j]), i, j))
    cells.sort()

    assigned_rows: set[int] = set()
    assigned_cols: set[int] = set()
    perm = [-1] * n

    while len(assigned_rows) < n:
        # Доступные ячейки (строка и столбец ещё свободны)
        available = [
            cell for cell in cells
            if cell[1] not in assigned_rows and cell[2] not in assigned_cols
        ]
        if not available:
            break
        k = min(top_k, len(available))
        _, i, j = random.choice(available[:k])
        perm[i] = j
        assigned_rows.add(i)
        assigned_cols.add(j)

    # Если что-то не назначилось (edge case), заполняем остаточным
    if -1 in perm:
        free_cols = [c for c in range(n) if c not in assigned_cols]
        random.shuffle(free_cols)
        for i in range(n):
            if perm[i] == -1 and free_cols:
                perm[i] = free_cols.pop()

    return perm


def _perm_cost(perm: list[int], cost_matrix: np.ndarray) -> float:
    return float(sum(cost_matrix[i][perm[i]] for i in range(len(perm))))


def _encode_perm(perm: list[int], n: int) -> list[float]:
    encoding = [0.0] * n
    for pos, task in enumerate(perm):
        encoding[task] = pos / max(n - 1, 1)
    return encoding
