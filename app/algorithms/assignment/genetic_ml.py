"""Генетический алгоритм с ML-суррогатом для задачи назначения.

Суррогатная модель (MLP) обучается предсказывать стоимость назначения
по кодировке перестановки. Часть популяции оценивается суррогатом
вместо точного вычисления.
"""

import time
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.ml.surrogate import MLPSurrogateModel


class GeneticMLAssignment(BaseAlgorithm):
    """Генетический алгоритм + ML-суррогат для задачи назначения.

    Кодировка для суррогата: нормализованная перестановка (аналогично TSP).
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

        cost_matrix = np.array(input_data["cost_matrix"])
        n = len(cost_matrix)

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (64, 32)),
        )

        start = time.perf_counter()

        population = [list(np.random.permutation(n)) for _ in range(pop_size)]
        convergence = []
        best_perm = None
        best_cost = float("inf")

        train_X: list[list[float]] = []
        train_y: list[float] = []
        exact_evals = 0
        surrogate_evals = 0
        surrogate_scores: list[float] = []

        for gen in range(generations):
            costs = [0.0] * pop_size

            if gen < warmup_gens or not surrogate.is_ready():
                for i, ind in enumerate(population):
                    costs[i] = _perm_cost(ind, cost_matrix)
                    train_X.append(_encode_perm(ind, n))
                    train_y.append(costs[i])
                exact_evals += pop_size
            else:
                n_exact = max(2, int(pop_size * (1 - surrogate_ratio)))
                exact_indices = set(random.sample(range(pop_size), n_exact))

                exact_X_batch = []
                exact_y_batch = []

                for i, ind in enumerate(population):
                    if i in exact_indices:
                        costs[i] = _perm_cost(ind, cost_matrix)
                        enc = _encode_perm(ind, n)
                        exact_X_batch.append(enc)
                        exact_y_batch.append(costs[i])
                        train_X.append(enc)
                        train_y.append(costs[i])
                        exact_evals += 1
                    else:
                        surrogate_evals += 1

                surrogate_indices = [i for i in range(pop_size) if i not in exact_indices]
                if surrogate_indices:
                    X_pred = np.array([_encode_perm(population[i], n) for i in surrogate_indices])
                    predictions = surrogate.predict(X_pred)
                    for idx, s_idx in enumerate(surrogate_indices):
                        costs[s_idx] = float(predictions[idx])

                if exact_X_batch:
                    score = surrogate.score(
                        np.array(exact_X_batch), np.array(exact_y_batch),
                    )
                    surrogate_scores.append(score)

            # Обновление лучшего (всегда проверяем точной оценкой)
            for i, ind in enumerate(population):
                true_cost = _perm_cost(ind, cost_matrix)
                if true_cost < best_cost:
                    best_cost = true_cost
                    best_perm = ind[:]

            convergence.append(float(best_cost))

            # Дообучение суррогата
            if (gen == warmup_gens - 1) or (
                gen >= warmup_gens and gen % retrain_every == 0
            ):
                if len(train_X) >= 10:
                    surrogate.fit(np.array(train_X), np.array(train_y))

            # ── Генетические операторы ──────────────────────────────
            fitness = [1.0 / max(c, 1e-10) for c in costs]

            new_pop = []
            for _ in range(pop_size):
                t1, t2 = random.sample(range(pop_size), 2)
                winner = population[t1] if fitness[t1] > fitness[t2] else population[t2]
                new_pop.append(winner[:])

            for i in range(0, pop_size - 1, 2):
                if random.random() < 0.8:
                    c1, c2 = _pmx(new_pop[i], new_pop[i + 1], n)
                    new_pop[i] = c1
                    new_pop[i + 1] = c2

            for ind in new_pop:
                if random.random() < mutation_rate:
                    a, b = random.sample(range(n), 2)
                    ind[a], ind[b] = ind[b], ind[a]

            population = new_pop

        elapsed = time.perf_counter() - start

        assignments = [[i, best_perm[i]] for i in range(n)] if best_perm else []

        avg_score = float(np.mean(surrogate_scores)) if surrogate_scores else 0.0

        return AlgorithmResult(
            solution=assignments,
            cost=float(best_cost),
            execution_time=elapsed,
            iterations=generations,
            convergence_history=convergence,
            extra={
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


# ── Вспомогательные функции ─────────────────────────────────────────

def _perm_cost(perm: list[int], cost_matrix: np.ndarray) -> float:
    return float(sum(cost_matrix[i][perm[i]] for i in range(len(perm))))


def _encode_perm(perm: list[int], n: int) -> list[float]:
    """Кодирование перестановки в вектор признаков для суррогата."""
    encoding = [0.0] * n
    for pos, task in enumerate(perm):
        encoding[task] = pos / max(n - 1, 1)
    return encoding


def _pmx(p1: list, p2: list, n: int) -> tuple[list, list]:
    a, b = sorted(random.sample(range(n), 2))
    c1, c2 = [-1] * n, [-1] * n
    c1[a:b + 1] = p1[a:b + 1]
    c2[a:b + 1] = p2[a:b + 1]

    for child, parent in [(c1, p2), (c2, p1)]:
        for i in range(a, b + 1):
            if parent[i] not in child:
                val = parent[i]
                pos = i
                while a <= pos <= b:
                    mapped = child[pos]
                    idx = list(parent).index(mapped) if mapped in parent else -1
                    if idx == -1 or not (a <= idx <= b):
                        pos = idx if idx != -1 else pos
                        break
                    pos = idx
                if 0 <= pos < n and child[pos] == -1:
                    child[pos] = val

        for i in range(n):
            if child[i] == -1:
                child[i] = parent[i]

    return c1, c2
