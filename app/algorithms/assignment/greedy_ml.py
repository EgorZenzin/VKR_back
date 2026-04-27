"""Жадный алгоритм с ML-суррогатом для задачи назначения.

Стратегия:
1. Если n маленькое — fallback к обычному жадному.
2. Иначе: лёгкий MLP, рандомизированные жадные назначения,
   ML-скрининг, точная переоценка top-K.
"""

import time
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.algorithms.assignment.greedy import GreedyAssignment
from app.ml.surrogate import MLPSurrogateModel


ML_MIN_N = 12


class GreedyMLAssignment(BaseAlgorithm):
    """Жадный алгоритм + ML-суррогат для задачи назначения."""

    name = "greedy_ml"
    display_name = "Жадный алгоритм + ML"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        cost_matrix = np.array(input_data["cost_matrix"])
        n = len(cost_matrix)

        # ── Fallback для маленьких задач ───────────────────────────
        if n < ML_MIN_N:
            base = GreedyAssignment().solve(input_data, params)
            extra = dict(base.extra or {})
            extra.update({
                "ml_used": False,
                "fallback_reason": "input_too_small_for_ml",
                "fallback_threshold": ML_MIN_N,
                "fallback_to": "greedy",
            })
            return AlgorithmResult(
                solution=base.solution,
                cost=base.cost,
                execution_time=base.execution_time,
                iterations=base.iterations,
                convergence_history=base.convergence_history,
                extra=extra,
            )

        warmup_samples = params.get("warmup_samples", 200)
        candidate_count = params.get("candidate_count", 80)
        top_k_exact = params.get("top_k_exact", 10)
        top_k_cells = params.get("top_k_cells", 3)

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (32, 16)),
            max_iter=params.get("max_iter", 80),
            learning_rate_init=params.get("learning_rate_init", 0.02),
        )

        start = time.perf_counter()

        sample_pool = [list(np.random.permutation(n)) for _ in range(warmup_samples)]
        train_X = np.array([_encode_perm(p, n) for p in sample_pool])
        train_y = np.array([_perm_cost(p, cost_matrix) for p in sample_pool])
        surrogate.fit(train_X, train_y)
        exact_evals = warmup_samples

        # Гарантированный детерминированный жадный — всегда точно оценивается.
        guaranteed = [_deterministic_greedy_assignment(cost_matrix)]
        candidates = list(guaranteed)
        for _ in range(candidate_count - len(guaranteed)):
            perm = _randomized_greedy_assignment(cost_matrix, top_k_cells)
            candidates.append(perm)
        guaranteed_count = len(guaranteed)

        X_cand = np.array([_encode_perm(c, n) for c in candidates])
        pred_costs = surrogate.predict(X_cand)
        surrogate_evals = candidate_count

        ml_top = np.argsort(pred_costs)[:top_k_exact].tolist()
        eval_indices = sorted(set(range(guaranteed_count)) | set(ml_top))
        best_perm = None
        best_cost = float("inf")

        for idx in eval_indices:
            c = _perm_cost(candidates[idx], cost_matrix)
            exact_evals += 1
            if c < best_cost:
                best_cost = c
                best_perm = candidates[idx][:]

        # R² на отдельной валидационной выборке.
        val_size = min(30, max(10, warmup_samples // 2))
        val_perms = [list(np.random.permutation(n)) for _ in range(val_size)]
        val_X = np.array([_encode_perm(p, n) for p in val_perms])
        val_y = np.array([_perm_cost(p, cost_matrix) for p in val_perms])
        exact_evals += val_size
        surrogate_r2 = surrogate.score(val_X, val_y)

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

def _deterministic_greedy_assignment(cost_matrix: np.ndarray) -> list[int]:
    """Обычное жадное назначение: всегда берём самую дешёвую ячейку."""
    n = len(cost_matrix)
    cells = sorted(
        ((float(cost_matrix[i][j]), i, j) for i in range(n) for j in range(n))
    )
    perm = [-1] * n
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for _, i, j in cells:
        if i in used_rows or j in used_cols:
            continue
        perm[i] = j
        used_rows.add(i)
        used_cols.add(j)
        if len(used_rows) == n:
            break
    return perm


def _randomized_greedy_assignment(
    cost_matrix: np.ndarray, top_k: int,
) -> list[int]:
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
