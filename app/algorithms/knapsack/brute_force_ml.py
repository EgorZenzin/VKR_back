"""ML-усечённый «полный перебор» для задачи о рюкзаке.

Стратегия: перечисление всех 2^n подмножеств с оценкой суррогатом,
точная оценка только для top-K по предсказанию.

ВНИМАНИЕ: оптимальность НЕ гарантируется.
"""

import time
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.ml.surrogate import MLPSurrogateModel


class BruteForceMLKnapsack(BaseAlgorithm):
    """ML-усечённый перебор для задачи о рюкзаке."""

    name = "brute_force_ml"
    display_name = "Полный перебор + ML"

    MAX_ITEMS = 20

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        warmup_samples = params.get("warmup_samples", 200)
        top_k_exact = params.get("top_k_exact", 50)

        items = input_data["items"]
        capacity = input_data["capacity"]
        n = len(items)

        if n > self.MAX_ITEMS:
            raise ValueError(
                f"Полный перебор + ML допустим для n ≤ {self.MAX_ITEMS}, получено n = {n}"
            )

        weights = [item["weight"] for item in items]
        values = [item["value"] for item in items]

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (32, 16)),
        )

        start = time.perf_counter()

        # ── Фаза 1: прогрев ────────────────────────────────────────
        sample_pool = [
            [random.randint(0, 1) for _ in range(n)] for _ in range(warmup_samples)
        ]
        train_X = np.array(sample_pool, dtype=float)
        train_y = np.array(
            [_exact_fitness(ind, weights, values, capacity) for ind in sample_pool]
        )
        surrogate.fit(train_X, train_y)

        exact_evals = warmup_samples

        # ── Фаза 2: ML-скрининг всех подмножеств ───────────────────
        total = 1 << n
        all_solutions = np.zeros((total, n), dtype=float)
        for mask in range(total):
            for i in range(n):
                if mask & (1 << i):
                    all_solutions[mask, i] = 1.0

        pred_fitness = surrogate.predict(all_solutions)
        surrogate_evals = total

        # ── Фаза 3: точная переоценка top-K ────────────────────────
        top_k = min(top_k_exact, total)
        top_indices = np.argsort(-pred_fitness)[:top_k]

        best_solution = None
        best_value = -1.0
        actual_fitness = []
        for idx in top_indices:
            ind = all_solutions[idx].astype(int).tolist()
            f = _exact_fitness(ind, weights, values, capacity)
            actual_fitness.append(f)
            exact_evals += 1
            if f > best_value:
                best_value = f
                best_solution = ind

        surrogate_r2 = surrogate.score(
            all_solutions[top_indices], np.array(actual_fitness)
        )

        elapsed = time.perf_counter() - start

        selected = [i for i in range(n) if best_solution and best_solution[i] == 1]
        total_weight = sum(weights[i] for i in selected)

        return AlgorithmResult(
            solution=selected,
            cost=float(best_value),
            execution_time=elapsed,
            iterations=total,
            convergence_history=[float(best_value)],
            extra={
                "total_weight": total_weight,
                "ml_used": True,
                "surrogate_model": "MLPRegressor",
                "exact_evaluations": exact_evals,
                "surrogate_evaluations": surrogate_evals,
                "surrogate_accuracy_r2": round(float(surrogate_r2), 4),
                "warmup_samples": warmup_samples,
                "top_k_exact": top_k,
                "total_candidates": total,
                "training_samples": warmup_samples,
                "optimality_guaranteed": False,
            },
        )


def _exact_fitness(
    ind: list[int],
    weights: list[float],
    values: list[float],
    capacity: float,
) -> float:
    n = len(ind)
    w = sum(weights[i] * ind[i] for i in range(n))
    v = sum(values[i] * ind[i] for i in range(n))
    return v if w <= capacity else 0.0
