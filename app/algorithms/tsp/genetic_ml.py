"""Генетический алгоритм с ML-суррогатом для задачи коммивояжёра.

Отличие от обычного ГА: часть популяции оценивается суррогатной моделью
(MLP) вместо точного вычисления целевой функции. Суррогат периодически
дообучается на накопленных точных оценках.

Метрики ML записываются в extra-поле результата для сравнения с базовым ГА.
"""

import time
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.ml.surrogate import MLPSurrogateModel


class GeneticMLTSP(BaseAlgorithm):
    """Генетический алгоритм + ML-суррогат для TSP.

    Кодировка решения для суррогата: нормализованная перестановка городов
    (вектор длины n, значения от 0 до 1).
    """

    name = "genetic_ml"
    display_name = "Генетический алгоритм + ML"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        pop_size = params.get("population_size", 100)
        generations = params.get("generations", 500)
        mutation_rate = params.get("mutation_rate", 0.02)
        crossover_rate = params.get("crossover_rate", 0.8)
        # ML параметры
        warmup_gens = params.get("warmup_generations", 20)
        surrogate_ratio = params.get("surrogate_ratio", 0.5)
        retrain_every = params.get("retrain_every", 10)

        dist_matrix = _get_distance_matrix(input_data)
        n = len(dist_matrix)

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (64, 32)),
        )

        start = time.perf_counter()

        population = [list(np.random.permutation(n)) for _ in range(pop_size)]
        convergence = []
        best_route = None
        best_cost = float("inf")

        # Накопитель обучающих данных для суррогата
        train_X: list[list[float]] = []
        train_y: list[float] = []

        exact_evals = 0
        surrogate_evals = 0
        surrogate_scores: list[float] = []

        for gen in range(generations):
            # ── Оценка fitness ──────────────────────────────────────
            costs = [0.0] * pop_size

            if gen < warmup_gens or not surrogate.is_ready():
                # Фаза прогрева: точная оценка всех
                for i, ind in enumerate(population):
                    costs[i] = _route_cost(ind, dist_matrix)
                    train_X.append(_encode_route(ind, n))
                    train_y.append(costs[i])
                exact_evals += pop_size
            else:
                # ML-фаза: часть точно, часть суррогатом
                n_exact = max(2, int(pop_size * (1 - surrogate_ratio)))
                exact_indices = set(random.sample(range(pop_size), n_exact))

                exact_X_batch = []
                exact_y_batch = []

                for i, ind in enumerate(population):
                    if i in exact_indices:
                        costs[i] = _route_cost(ind, dist_matrix)
                        enc = _encode_route(ind, n)
                        exact_X_batch.append(enc)
                        exact_y_batch.append(costs[i])
                        train_X.append(enc)
                        train_y.append(costs[i])
                        exact_evals += 1
                    else:
                        surrogate_evals += 1

                # Суррогатные предсказания для оставшихся
                surrogate_indices = [i for i in range(pop_size) if i not in exact_indices]
                if surrogate_indices:
                    X_pred = np.array([_encode_route(population[i], n) for i in surrogate_indices])
                    predictions = surrogate.predict(X_pred)
                    for idx, s_idx in enumerate(surrogate_indices):
                        costs[s_idx] = float(predictions[idx])

                # Оценка качества суррогата на точных данных этого поколения
                if exact_X_batch:
                    score = surrogate.score(
                        np.array(exact_X_batch), np.array(exact_y_batch),
                    )
                    surrogate_scores.append(score)

            # Обновление лучшего решения
            for i, ind in enumerate(population):
                true_cost = _route_cost(ind, dist_matrix)
                if true_cost < best_cost:
                    best_cost = true_cost
                    best_route = ind[:]

            convergence.append(float(best_cost))

            # ── Дообучение суррогата ────────────────────────────────
            if (gen == warmup_gens - 1) or (
                gen >= warmup_gens and gen % retrain_every == 0
            ):
                if len(train_X) >= 10:
                    surrogate.fit(np.array(train_X), np.array(train_y))

            # ── Генетические операторы (те же, что в базовом ГА) ────
            fitness = [1.0 / max(c, 1e-10) for c in costs]

            new_pop = []
            for _ in range(pop_size):
                t1, t2 = random.sample(range(pop_size), 2)
                winner = population[t1] if fitness[t1] > fitness[t2] else population[t2]
                new_pop.append(winner[:])

            for i in range(0, pop_size - 1, 2):
                if random.random() < crossover_rate:
                    child1, child2 = _ox_crossover(new_pop[i], new_pop[i + 1], n)
                    new_pop[i] = child1
                    new_pop[i + 1] = child2

            for ind in new_pop:
                if random.random() < mutation_rate:
                    a, b = random.sample(range(n), 2)
                    ind[a], ind[b] = ind[b], ind[a]

            population = new_pop

        elapsed = time.perf_counter() - start

        avg_score = float(np.mean(surrogate_scores)) if surrogate_scores else 0.0

        return AlgorithmResult(
            solution=best_route,
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

def _route_cost(route: list[int], dist_matrix: np.ndarray) -> float:
    cost = sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
    cost += dist_matrix[route[-1]][route[0]]
    return float(cost)


def _encode_route(route: list[int], n: int) -> list[float]:
    """Кодирование маршрута в вектор признаков для суррогата.

    Нормализуем индексы городов в [0, 1] — порядок в перестановке.
    """
    encoding = [0.0] * n
    for pos, city in enumerate(route):
        encoding[city] = pos / max(n - 1, 1)
    return encoding


def _ox_crossover(p1: list, p2: list, n: int) -> tuple[list, list]:
    a, b = sorted(random.sample(range(n), 2))
    child1, child2 = [-1] * n, [-1] * n
    child1[a:b + 1] = p1[a:b + 1]
    child2[a:b + 1] = p2[a:b + 1]
    _fill_ox(child1, p2, b, n)
    _fill_ox(child2, p1, b, n)
    return child1, child2


def _fill_ox(child: list, parent: list, b: int, n: int):
    pos = (b + 1) % n
    for gene in parent[b + 1:] + parent[:b + 1]:
        if gene not in child:
            child[pos] = gene
            pos = (pos + 1) % n


def _get_distance_matrix(input_data: dict) -> np.ndarray:
    if "distance_matrix" in input_data and input_data["distance_matrix"]:
        return np.array(input_data["distance_matrix"])
    cities = input_data["cities"]
    coords = np.array([[c["x"], c["y"]] for c in cities])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))
