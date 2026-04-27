"""Жадный алгоритм с ML-суррогатом для задачи о рюкзаке.

Стратегия:
1. Если n маленькое — fallback к обычному жадному.
2. Иначе: лёгкий MLP, рандомизированные жадные кандидаты с разными
   критериями сортировки, ML-скрининг, точная переоценка top-K.
"""

import time
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.algorithms.knapsack.greedy import GreedyKnapsack
from app.ml.surrogate import MLPSurrogateModel


ML_MIN_N = 20


class GreedyMLKnapsack(BaseAlgorithm):
    """Жадный алгоритм + ML-суррогат для задачи о рюкзаке."""

    name = "greedy_ml"
    display_name = "Жадный алгоритм + ML"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        items = input_data["items"]
        capacity = input_data["capacity"]
        n = len(items)

        # ── Fallback для маленьких задач ───────────────────────────
        if n < ML_MIN_N:
            base = GreedyKnapsack().solve(input_data, params)
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

        weights = [item["weight"] for item in items]
        values = [item["value"] for item in items]

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (32, 16)),
            max_iter=params.get("max_iter", 80),
            learning_rate_init=params.get("learning_rate_init", 0.02),
        )

        start = time.perf_counter()

        sample_pool = [
            [random.randint(0, 1) for _ in range(n)] for _ in range(warmup_samples)
        ]
        train_X = np.array(sample_pool, dtype=float)
        train_y = np.array(
            [_exact_fitness(ind, weights, values, capacity) for ind in sample_pool]
        )
        surrogate.fit(train_X, train_y)
        exact_evals = warmup_samples

        # Гарантированные детерминированные кандидаты (жадные по всем критериям).
        guaranteed = [
            _deterministic_greedy(weights, values, capacity, mode="ratio"),
            _deterministic_greedy(weights, values, capacity, mode="value"),
        ]
        candidates = list(guaranteed)
        for _ in range(candidate_count - len(guaranteed)):
            alpha = random.choice([0.5, 1.0, 1.0, 1.5, 2.0])
            mode = random.choice(["ratio", "value", "ratio_alpha"])
            sol = _randomized_greedy_knapsack(
                weights, values, capacity, mode=mode, alpha=alpha,
            )
            candidates.append(sol)
        guaranteed_count = len(guaranteed)

        X_cand = np.array(candidates, dtype=float)
        pred_fitness = surrogate.predict(X_cand)
        surrogate_evals = candidate_count

        ml_top = np.argsort(-pred_fitness)[:top_k_exact].tolist()
        eval_indices = sorted(set(range(guaranteed_count)) | set(ml_top))
        best_solution = None
        best_value = -1.0

        for idx in eval_indices:
            f = _exact_fitness(candidates[idx], weights, values, capacity)
            exact_evals += 1
            if f > best_value:
                best_value = f
                best_solution = candidates[idx][:]

        # R² на отдельной валидационной выборке.
        val_size = min(30, max(10, warmup_samples // 2))
        val_pool = [
            [random.randint(0, 1) for _ in range(n)] for _ in range(val_size)
        ]
        val_X = np.array(val_pool, dtype=float)
        val_y = np.array(
            [_exact_fitness(ind, weights, values, capacity) for ind in val_pool]
        )
        exact_evals += val_size
        surrogate_r2 = surrogate.score(val_X, val_y)

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

def _deterministic_greedy(
    weights: list[float],
    values: list[float],
    capacity: float,
    mode: str = "ratio",
) -> list[int]:
    """Обычный жадный без рандома."""
    n = len(weights)
    if mode == "value":
        order = sorted(range(n), key=lambda i: -values[i])
    else:
        order = sorted(
            range(n),
            key=lambda i: -(values[i] / weights[i] if weights[i] > 0 else 0.0),
        )
    sol = [0] * n
    total = 0.0
    for i in order:
        if total + weights[i] <= capacity:
            sol[i] = 1
            total += weights[i]
    return sol


def _randomized_greedy_knapsack(
    weights: list[float],
    values: list[float],
    capacity: float,
    mode: str = "ratio",
    alpha: float = 1.0,
) -> list[int]:
    n = len(weights)
    scores = []
    for i in range(n):
        w = weights[i]
        v = values[i]
        if mode == "value":
            s = v
        elif mode == "ratio_alpha":
            s = v / (w ** alpha) if w > 0 else 0.0
        else:
            s = v / w if w > 0 else 0.0
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
