"""Генетический алгоритм с ML-суррогатом для задачи о рюкзаке.

Подход: ML-суррогат используется как предварительный этап оценки.
1. Обучение суррогата на случайной выборке точных решений.
2. Генерация большого пула кандидатов, быстрая оценка суррогатом.
3. Отбор лучших кандидатов как начальная популяция ГА.
4. Стандартный ГА с точными оценками, но меньшим числом поколений.
"""

import time
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.ml.surrogate import MLPSurrogateModel


class GeneticMLKnapsack(BaseAlgorithm):
    """Генетический алгоритм + ML-суррогат для задачи о рюкзаке."""

    name = "genetic_ml"
    display_name = "Генетический алгоритм + ML"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        pop_size = params.get("population_size", 100)
        generations = params.get("generations", 300)
        mutation_rate = params.get("mutation_rate", 0.05)
        # ML параметры
        warmup_samples = params.get("warmup_samples", 150)
        screening_factor = params.get("screening_factor", 5)

        items = input_data["items"]
        capacity = input_data["capacity"]
        n = len(items)

        weights = [item["weight"] for item in items]
        values = [item["value"] for item in items]

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (32, 16)),
        )

        start = time.perf_counter()

        # ── Фаза 1: Обучение суррогата на случайной выборке ─────────
        sample_pool = [[random.randint(0, 1) for _ in range(n)] for _ in range(warmup_samples)]
        train_X = np.array(sample_pool, dtype=float)
        train_y = np.array([_exact_fitness(ind, weights, values, capacity) for ind in sample_pool])
        surrogate.fit(train_X, train_y)

        exact_evals = warmup_samples

        # ── Фаза 2: ML-скрининг начальной популяции ────────────────
        candidate_count = pop_size * screening_factor
        candidates = [[random.randint(0, 1) for _ in range(n)] for _ in range(candidate_count)]
        X_cand = np.array(candidates, dtype=float)
        pred_fitness = surrogate.predict(X_cand)

        top_indices = np.argsort(-pred_fitness)[:pop_size]
        population = [candidates[i] for i in top_indices]

        surrogate_evals = candidate_count

        # R² суррогата на независимой валидационной выборке.
        val_size = min(40, max(15, warmup_samples // 3))
        val_solutions = [
            [random.randint(0, 1) for _ in range(n)] for _ in range(val_size)
        ]
        val_X = np.array(val_solutions, dtype=float)
        val_y = np.array(
            [_exact_fitness(s, weights, values, capacity) for s in val_solutions]
        )
        exact_evals += val_size
        surrogate_r2 = surrogate.score(val_X, val_y)

        # ── Фаза 3: Стандартный ГА (сокращённые поколения) ─────────
        reduced_gens = max(1, int(generations * 0.4))

        best_solution = None
        best_value = -1.0
        convergence = []

        for gen in range(reduced_gens):
            fitness = [_exact_fitness(ind, weights, values, capacity) for ind in population]
            exact_evals += pop_size

            for i, f in enumerate(fitness):
                if f > best_value:
                    best_value = f
                    best_solution = population[i][:]

            convergence.append(float(best_value))

            # ── Генетические операторы ──────────────────────────────
            new_pop = []
            for _ in range(pop_size):
                t1, t2 = random.sample(range(pop_size), 2)
                winner = population[t1] if fitness[t1] >= fitness[t2] else population[t2]
                new_pop.append(winner[:])

            for i in range(0, pop_size - 1, 2):
                if random.random() < 0.8:
                    pt = random.randint(1, n - 1)
                    c1 = new_pop[i][:pt] + new_pop[i + 1][pt:]
                    c2 = new_pop[i + 1][:pt] + new_pop[i][pt:]
                    new_pop[i] = c1
                    new_pop[i + 1] = c2

            for ind in new_pop:
                for j in range(n):
                    if random.random() < mutation_rate:
                        ind[j] = 1 - ind[j]

            population = new_pop

        elapsed = time.perf_counter() - start

        selected = [i for i in range(n) if best_solution and best_solution[i] == 1]
        total_weight = sum(weights[i] for i in selected)

        return AlgorithmResult(
            solution=selected,
            cost=float(best_value),
            execution_time=elapsed,
            iterations=reduced_gens,
            convergence_history=convergence,
            extra={
                "total_weight": total_weight,
                "ml_used": True,
                "surrogate_model": "MLPRegressor",
                "exact_evaluations": exact_evals,
                "surrogate_evaluations": surrogate_evals,
                "surrogate_accuracy_r2": round(surrogate_r2, 4),
                "warmup_samples": warmup_samples,
                "screening_factor": screening_factor,
                "reduced_generations": reduced_gens,
                "training_samples": warmup_samples,
            },
        )


def _exact_fitness(
    ind: list[int],
    weights: list,
    values: list,
    capacity: float,
) -> float:
    """Точная оценка fitness: суммарная ценность (0 при превышении ёмкости)."""
    n = len(ind)
    w = sum(weights[i] * ind[i] for i in range(n))
    v = sum(values[i] * ind[i] for i in range(n))
    return v if w <= capacity else 0.0
