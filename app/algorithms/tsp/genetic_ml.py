"""Генетический алгоритм с ML-суррогатом для задачи коммивояжёра.

Подход: ML-суррогат используется как предварительный этап оценки.
1. Обучение суррогата на случайной выборке точных решений.
2. Генерация большого пула кандидатов, быстрая оценка суррогатом.
3. Отбор лучших кандидатов как начальная популяция ГА.
4. Стандартный ГА с точными оценками, но меньшим числом поколений
   (лучшая стартовая популяция → быстрая сходимость).

Метрики ML записываются в extra-поле результата для сравнения с базовым ГА.
"""

import time
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.ml.surrogate import MLPSurrogateModel


class GeneticMLTSP(BaseAlgorithm):
    """Генетический алгоритм + ML-суррогат для TSP."""

    name = "genetic_ml"
    display_name = "Генетический алгоритм + ML"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        pop_size = params.get("population_size", 100)
        generations = params.get("generations", 500)
        mutation_rate = params.get("mutation_rate", 0.02)
        crossover_rate = params.get("crossover_rate", 0.8)
        # ML параметры
        warmup_samples = params.get("warmup_samples", 150)
        screening_factor = params.get("screening_factor", 5)

        dist_matrix = _get_distance_matrix(input_data)
        n = len(dist_matrix)

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (32, 16)),
        )

        start = time.perf_counter()

        # ── Фаза 1: Обучение суррогата на случайной выборке ─────────
        sample_pool = [list(np.random.permutation(n)) for _ in range(warmup_samples)]
        train_X = np.array([_encode_route(r, n) for r in sample_pool])
        train_y = np.array([_route_cost(r, dist_matrix) for r in sample_pool])
        surrogate.fit(train_X, train_y)

        exact_evals = warmup_samples

        # ── Фаза 2: ML-скрининг начальной популяции ────────────────
        candidate_count = pop_size * screening_factor
        candidates = [list(np.random.permutation(n)) for _ in range(candidate_count)]
        X_cand = np.array([_encode_route(c, n) for c in candidates])
        pred_costs = surrogate.predict(X_cand)

        top_indices = np.argsort(pred_costs)[:pop_size]
        population = [candidates[i] for i in top_indices]

        # Сидируем популяцию детерминированными жадными маршрутами с разных
        # стартовых городов — гарантирует, что начальная популяция не хуже
        # обычного жадного решения.
        seed_count = min(n, max(3, pop_size // 10))
        for s in range(seed_count):
            population[s] = _greedy_route(dist_matrix, start_city=s)

        surrogate_evals = candidate_count

        # R² суррогата на независимой валидационной выборке (не на top-K!).
        val_size = min(40, max(15, warmup_samples // 3))
        val_routes = [list(np.random.permutation(n)) for _ in range(val_size)]
        val_X = np.array([_encode_route(r, n) for r in val_routes])
        val_y = np.array([_route_cost(r, dist_matrix) for r in val_routes])
        exact_evals += val_size
        surrogate_r2 = surrogate.score(val_X, val_y)

        # ── Фаза 3: Стандартный ГА (сокращённые поколения) ─────────
        reduced_gens = max(1, int(generations * 0.4))

        best_route = None
        best_cost = float("inf")
        convergence = []

        for gen in range(reduced_gens):
            costs = [_route_cost(ind, dist_matrix) for ind in population]
            exact_evals += pop_size

            for i, ind in enumerate(population):
                if costs[i] < best_cost:
                    best_cost = costs[i]
                    best_route = ind[:]

            convergence.append(float(best_cost))

            # ── Генетические операторы ──────────────────────────────
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

        return AlgorithmResult(
            solution=best_route,
            cost=float(best_cost),
            execution_time=elapsed,
            iterations=reduced_gens,
            convergence_history=convergence,
            extra={
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


# ── Вспомогательные функции ─────────────────────────────────────────

def _route_cost(route: list[int], dist_matrix: np.ndarray) -> float:
    cost = sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
    cost += dist_matrix[route[-1]][route[0]]
    return float(cost)


def _greedy_route(dist_matrix: np.ndarray, start_city: int = 0) -> list[int]:
    """Детерминированный жадный маршрут от заданного стартового города."""
    n = len(dist_matrix)
    visited = [False] * n
    route = [start_city]
    visited[start_city] = True
    for _ in range(n - 1):
        cur = route[-1]
        best_j = -1
        best_d = float("inf")
        for j in range(n):
            if not visited[j] and dist_matrix[cur][j] < best_d:
                best_d = dist_matrix[cur][j]
                best_j = j
        route.append(best_j)
        visited[best_j] = True
    return route


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
