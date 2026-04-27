"""Жадный алгоритм с ML-суррогатом для задачи о рюкзаке.

Подход: ML-суррогат отбирает лучшие жадные решения из множества
вариантов с разными критериями сортировки и рандомизацией.

1. Обучение суррогата на случайных бинарных решениях.
2. Генерация пула жадных решений с варьируемым критерием сортировки
   (value/weight, value, value/weight^α при разных α) и случайным выбором
   среди эквивалентных кандидатов.
3. Быстрая оценка всех вариантов суррогатом.
4. Точный пересчёт top-K решений и возврат лучшего.
"""

import time
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.ml.surrogate import MLPSurrogateModel


class GreedyMLKnapsack(BaseAlgorithm):
    """Жадный алгоритм + ML-суррогат для задачи о рюкзаке."""

    name = "greedy_ml"
    display_name = "Жадный алгоритм + ML"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        warmup_samples = params.get("warmup_samples", 150)
        candidate_count = params.get("candidate_count", 200)
        top_k_exact = params.get("top_k_exact", 20)

        items = input_data["items"]
        capacity = input_data["capacity"]
        n = len(items)

        weights = [item["weight"] for item in items]
        values = [item["value"] for item in items]

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (32, 16)),
        )

        start = time.perf_counter()

        # ── Фаза 1: обучение суррогата на случайных решениях ────────
        sample_pool = [
            [random.randint(0, 1) for _ in range(n)] for _ in range(warmup_samples)
        ]
        train_X = np.array(sample_pool, dtype=float)
        train_y = np.array(
            [_exact_fitness(ind, weights, values, capacity) for ind in sample_pool]
        )
        surrogate.fit(train_X, train_y)

        exact_evals = warmup_samples

        # ── Фаза 2: генерация жадных кандидатов с вариациями ───────
        candidates = []
        for _ in range(candidate_count):
            alpha = random.choice([0.5, 1.0, 1.0, 1.5, 2.0])
            mode = random.choice(["ratio", "value", "ratio_alpha"])
            sol = _randomized_greedy_knapsack(
                weights, values, capacity, mode=mode, alpha=alpha,
            )
            candidates.append(sol)

        X_cand = np.array(candidates, dtype=float)
        pred_fitness = surrogate.predict(X_cand)
        surrogate_evals = candidate_count

        # ── Фаза 3: точный пересчёт top-K ──────────────────────────
        top_indices = np.argsort(-pred_fitness)[:top_k_exact]
        best_solution = None
        best_value = -1.0

        actual_fitness = []
        for idx in top_indices:
            f = _exact_fitness(candidates[idx], weights, values, capacity)
            actual_fitness.append(f)
            exact_evals += 1
            if f > best_value:
                best_value = f
                best_solution = candidates[idx][:]

        surrogate_r2 = surrogate.score(
            X_cand[top_indices], np.array(actual_fitness)
        )

        elapsed = time.perf_counter() - start

        selected = (
            [i for i in range(n) if best_solution and best_solution[i] == 1]
        )
        total_weight = sum(weights[i] for i in selected)

        return AlgorithmResult(
            solution=selected,
            cost=float(best_value),
            execution_time=elapsed,
            iterations=None,
            convergence_history=[float(best_value)],
            extra={
                "total_weight": total_weight,
                "ml_used": True,
                "surrogate_model": "MLPRegressor",
                "exact_evaluations": exact_evals,
                "surrogate_evaluations": surrogate_evals,
                "surrogate_accuracy_r2": round(float(surrogate_r2), 4),
                "warmup_samples": warmup_samples,
                "candidate_count": candidate_count,
                "top_k_exact": top_k_exact,
                "training_samples": warmup_samples,
            },
        )


# ── Вспомогательные функции ─────────────────────────────────────────

def _randomized_greedy_knapsack(
    weights: list[float],
    values: list[float],
    capacity: float,
    mode: str = "ratio",
    alpha: float = 1.0,
) -> list[int]:
    """Жадный отбор предметов с рандомизацией и варьируемым критерием."""
    n = len(weights)
    scores = []
    for i in range(n):
        w = weights[i]
        v = values[i]
        if mode == "value":
            s = v
        elif mode == "ratio_alpha":
            s = v / (w ** alpha) if w > 0 else 0.0
        else:  # ratio
            s = v / w if w > 0 else 0.0
        # Лёгкое случайное возмущение для разнообразия
        s *= 1.0 + random.uniform(-0.1, 0.1)
        scores.append((s, i))

    scores.sort(reverse=True)

    solution = [0] * n
    total_weight = 0.0
    for _, i in scores:
        if total_weight + weights[i] <= capacity:
            solution[i] = 1
            total_weight += weights[i]

    return solution


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
