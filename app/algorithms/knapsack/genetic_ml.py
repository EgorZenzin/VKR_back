"""Генетический алгоритм с ML-суррогатом для задачи о рюкзаке.

Суррогатная модель (MLP) обучается предсказывать суммарную ценность
по бинарному вектору выбора предметов. Часть популяции оценивается
суррогатом вместо точного вычисления.
"""

import time
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.ml.surrogate import MLPSurrogateModel


class GeneticMLKnapsack(BaseAlgorithm):
    """Генетический алгоритм + ML-суррогат для задачи о рюкзаке.

    Кодировка для суррогата: бинарный вектор (0/1) — то же представление,
    что используется в ГА.
    """

    name = "genetic_ml"
    display_name = "Генетический алгоритм + ML"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        pop_size = params.get("population_size", 100)
        generations = params.get("generations", 300)
        mutation_rate = params.get("mutation_rate", 0.05)
        warmup_gens = params.get("warmup_generations", 15)
        surrogate_ratio = params.get("surrogate_ratio", 0.5)
        retrain_every = params.get("retrain_every", 10)

        items = input_data["items"]
        capacity = input_data["capacity"]
        n = len(items)

        weights = [item["weight"] for item in items]
        values = [item["value"] for item in items]

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (64, 32)),
        )

        start = time.perf_counter()

        population = [[random.randint(0, 1) for _ in range(n)] for _ in range(pop_size)]
        convergence = []
        best_solution = None
        best_value = -1.0

        train_X: list[list[float]] = []
        train_y: list[float] = []
        exact_evals = 0
        surrogate_evals = 0
        surrogate_scores: list[float] = []

        for gen in range(generations):
            fitness = [0.0] * pop_size

            if gen < warmup_gens or not surrogate.is_ready():
                for i, ind in enumerate(population):
                    fitness[i] = _exact_fitness(ind, weights, values, capacity)
                    train_X.append(ind[:])
                    train_y.append(fitness[i])
                exact_evals += pop_size
            else:
                n_exact = max(2, int(pop_size * (1 - surrogate_ratio)))
                exact_indices = set(random.sample(range(pop_size), n_exact))

                exact_X_batch = []
                exact_y_batch = []

                for i, ind in enumerate(population):
                    if i in exact_indices:
                        fitness[i] = _exact_fitness(ind, weights, values, capacity)
                        exact_X_batch.append(ind[:])
                        exact_y_batch.append(fitness[i])
                        train_X.append(ind[:])
                        train_y.append(fitness[i])
                        exact_evals += 1
                    else:
                        surrogate_evals += 1

                if [i for i in range(pop_size) if i not in exact_indices]:
                    surrogate_indices = [i for i in range(pop_size) if i not in exact_indices]
                    X_pred = np.array([population[i] for i in surrogate_indices], dtype=float)
                    predictions = surrogate.predict(X_pred)
                    for idx, s_idx in enumerate(surrogate_indices):
                        # Суррогат может предсказать отрицательное — обнуляем
                        fitness[s_idx] = max(0.0, float(predictions[idx]))

                if exact_X_batch:
                    score = surrogate.score(
                        np.array(exact_X_batch), np.array(exact_y_batch),
                    )
                    surrogate_scores.append(score)

            # Обновление лучшего
            for i, f in enumerate(fitness):
                if f > best_value:
                    # Проверяем точным вычислением
                    true_val = _exact_fitness(population[i], weights, values, capacity)
                    if true_val > best_value:
                        best_value = true_val
                        best_solution = population[i][:]

            convergence.append(float(best_value))

            # Дообучение суррогата
            if (gen == warmup_gens - 1) or (
                gen >= warmup_gens and gen % retrain_every == 0
            ):
                if len(train_X) >= 10:
                    surrogate.fit(
                        np.array(train_X, dtype=float), np.array(train_y),
                    )

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

        avg_score = float(np.mean(surrogate_scores)) if surrogate_scores else 0.0

        return AlgorithmResult(
            solution=selected,
            cost=float(best_value),
            execution_time=elapsed,
            iterations=generations,
            convergence_history=convergence,
            extra={
                "total_weight": total_weight,
                "ml_used": True,
                "surrogate_model": "MLPRegressor",
                "exact_evaluations": exact_evals,
                "surrogate_evaluations": surrogate_evals,
                "surrogate_accuracy_r2": round(avg_score, 4),
                "warmup_generations": warmup_gens,
                "surrogate_ratio": surrogate_ratio,
                "training_samples": len(train_X),
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
