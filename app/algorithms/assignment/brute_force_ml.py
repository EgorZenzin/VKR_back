"""ML-усечённый «полный перебор» для задачи назначения.

Стратегия: перечисление всех n! перестановок назначений с оценкой
суррогатом, точная оценка только для top-K по предсказанию.

ВНИМАНИЕ: оптимальность НЕ гарантируется.
"""

import time
import itertools
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.ml.surrogate import MLPSurrogateModel


class BruteForceMLAssignment(BaseAlgorithm):
    """ML-усечённый перебор для задачи назначения."""

    name = "brute_force_ml"
    display_name = "Полный перебор + ML"

    MAX_SIZE = 9

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        warmup_samples = params.get("warmup_samples", 200)
        top_k_exact = params.get("top_k_exact", 50)

        cost_matrix = np.array(input_data["cost_matrix"])
        n = len(cost_matrix)

        if n > self.MAX_SIZE:
            raise ValueError(
                f"Полный перебор + ML допустим для n ≤ {self.MAX_SIZE}, получено n = {n}"
            )

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (32, 16)),
        )

        start = time.perf_counter()

        # ── Фаза 1: прогрев ────────────────────────────────────────
        sample_pool = [list(np.random.permutation(n)) for _ in range(warmup_samples)]
        train_X = np.array([_encode_perm(p, n) for p in sample_pool])
        train_y = np.array([_perm_cost(p, cost_matrix) for p in sample_pool])
        surrogate.fit(train_X, train_y)

        exact_evals = warmup_samples

        # ── Фаза 2: ML-скрининг всех перестановок ──────────────────
        all_perms = [list(p) for p in itertools.permutations(range(n))]
        X_all = np.array([_encode_perm(p, n) for p in all_perms])
        pred_costs = surrogate.predict(X_all)
        surrogate_evals = len(all_perms)

        # ── Фаза 3: точная переоценка top-K ────────────────────────
        top_k = min(top_k_exact, len(all_perms))
        top_indices = np.argsort(pred_costs)[:top_k]

        best_perm = None
        best_cost = float("inf")
        actual_costs = []
        for idx in top_indices:
            c = _perm_cost(all_perms[idx], cost_matrix)
            actual_costs.append(c)
            exact_evals += 1
            if c < best_cost:
                best_cost = c
                best_perm = all_perms[idx][:]

        surrogate_r2 = surrogate.score(X_all[top_indices], np.array(actual_costs))

        elapsed = time.perf_counter() - start

        assignments = [[i, best_perm[i]] for i in range(n)] if best_perm else []

        return AlgorithmResult(
            solution=assignments,
            cost=float(best_cost),
            execution_time=elapsed,
            iterations=len(all_perms),
            convergence_history=[float(best_cost)],
            extra={
                "ml_used": True,
                "surrogate_model": "MLPRegressor",
                "exact_evaluations": exact_evals,
                "surrogate_evaluations": surrogate_evals,
                "surrogate_accuracy_r2": round(float(surrogate_r2), 4),
                "warmup_samples": warmup_samples,
                "top_k_exact": top_k,
                "total_candidates": len(all_perms),
                "training_samples": warmup_samples,
                "optimality_guaranteed": False,
            },
        )


def _perm_cost(perm: list[int], cost_matrix: np.ndarray) -> float:
    return float(sum(cost_matrix[i][perm[i]] for i in range(len(perm))))


def _encode_perm(perm: list[int], n: int) -> list[float]:
    encoding = [0.0] * n
    for pos, task in enumerate(perm):
        encoding[task] = pos / max(n - 1, 1)
    return encoding
