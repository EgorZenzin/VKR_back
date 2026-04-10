"""
NN-Guided Генетический алгоритм для TSP.

Генетический алгоритм, усиленный суррогатной нейросетью.
Нейросеть обучается на уже оценённых маршрутах и предсказывает
стоимость новых кандидатов, отсеивая неперспективные варианты
до полного вычисления целевой функции.
"""

import numpy as np
import random
import time

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.algorithms.ml.surrogate_model import SurrogateModel


class NNGuidedGeneticTSP(BaseAlgorithm):
    """NN-Guided Генетический алгоритм для TSP."""

    name = "nn_guided_genetic"
    display_name = "NN-Guided Генетический алгоритм"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        pop_size = params.get("population_size", 100)
        generations = params.get("generations", 500)
        mutation_rate = params.get("mutation_rate", 0.02)
        crossover_rate = params.get("crossover_rate", 0.8)
        # Коэффициент перегенерации: создаём больше кандидатов, НС отсеивает лишних
        oversample_ratio = params.get("oversample_ratio", 2.0)

        dist_matrix = self._get_distance_matrix(input_data)
        n = len(dist_matrix)

        # Суррогатная модель
        surrogate = SurrogateModel(
            hidden_layers=(64, 32),
            warmup_size=min(pop_size, 50),
            retrain_every=pop_size,
        )

        start = time.perf_counter()

        population = [list(np.random.permutation(n)) for _ in range(pop_size)]
        convergence = []
        nn_active_generations = 0

        best_route = None
        best_cost = float("inf")

        for gen in range(generations):
            # Полное вычисление стоимости для текущей популяции
            costs = [self._route_cost(ind, dist_matrix) for ind in population]
            fitness = [1.0 / c for c in costs]

            # Собираем обучающие данные для НС
            features_batch = [self._encode_route(ind, dist_matrix) for ind in population]
            surrogate.add_batch(features_batch, costs)

            # Обучаем/переобучаем НС при необходимости
            if surrogate.should_train():
                surrogate.train()

            for i, cost in enumerate(costs):
                if cost < best_cost:
                    best_cost = cost
                    best_route = population[i][:]

            convergence.append(best_cost)

            # --- Селекция (турнирная) ---
            new_pop = []
            for _ in range(pop_size):
                t1, t2 = random.sample(range(pop_size), 2)
                winner = population[t1] if fitness[t1] > fitness[t2] else population[t2]
                new_pop.append(winner[:])

            # --- Кроссовер (OX) ---
            for i in range(0, pop_size - 1, 2):
                if random.random() < crossover_rate:
                    child1, child2 = self._ox_crossover(new_pop[i], new_pop[i + 1], n)
                    new_pop[i] = child1
                    new_pop[i + 1] = child2

            # --- Мутация (swap) ---
            for ind in new_pop:
                if random.random() < mutation_rate:
                    i, j = random.sample(range(n), 2)
                    ind[i], ind[j] = ind[j], ind[i]

            # --- NN-фильтрация: генерируем доп. кандидатов, НС отсеивает слабых ---
            if surrogate.is_trained:
                nn_active_generations += 1
                n_extra = int(pop_size * (oversample_ratio - 1.0))
                extra_candidates = []
                for _ in range(n_extra):
                    # Создаём кандидатов мутацией лучших особей
                    base = random.choice(new_pop)[:]
                    i, j = random.sample(range(n), 2)
                    base[i], base[j] = base[j], base[i]
                    extra_candidates.append(base)

                all_candidates = new_pop + extra_candidates
                all_features = [self._encode_route(c, dist_matrix) for c in all_candidates]

                # НС предсказывает стоимость, отбираем лучших
                predictions = surrogate.predict_batch(all_features)
                top_indices = np.argsort(predictions)[:pop_size]
                population = [all_candidates[i] for i in top_indices]
            else:
                population = new_pop

        elapsed = time.perf_counter() - start

        return AlgorithmResult(
            solution=best_route,
            cost=best_cost,
            execution_time=elapsed,
            convergence_history=convergence,
            extra={
                "nn_active_generations": nn_active_generations,
                "total_generations": generations,
                "surrogate_stats": surrogate.get_stats(),
            },
        )

    def _encode_route(self, route: list[int], dist_matrix: np.ndarray) -> np.ndarray:
        """Кодировка маршрута в вектор признаков для НС.

        Признаки: вектор расстояний между последовательными городами
        + статистики (мин, макс, среднее, стд).
        """
        n = len(route)
        edges = [dist_matrix[route[i]][route[(i + 1) % n]] for i in range(n)]
        edges = np.array(edges)
        stats = [edges.min(), edges.max(), edges.mean(), edges.std()]
        return np.concatenate([edges, stats])

    def _route_cost(self, route: list[int], dist_matrix: np.ndarray) -> float:
        cost = sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
        cost += dist_matrix[route[-1]][route[0]]
        return cost

    def _ox_crossover(self, p1: list, p2: list, n: int) -> tuple[list, list]:
        a, b = sorted(random.sample(range(n), 2))
        child1 = [-1] * n
        child2 = [-1] * n
        child1[a:b + 1] = p1[a:b + 1]
        child2[a:b + 1] = p2[a:b + 1]
        self._fill_ox(child1, p2, b, n)
        self._fill_ox(child2, p1, b, n)
        return child1, child2

    def _fill_ox(self, child: list, parent: list, b: int, n: int):
        pos = (b + 1) % n
        for gene in parent[b + 1:] + parent[:b + 1]:
            if gene not in child:
                child[pos] = gene
                pos = (pos + 1) % n

    def _get_distance_matrix(self, input_data: dict) -> np.ndarray:
        if "distance_matrix" in input_data and input_data["distance_matrix"]:
            return np.array(input_data["distance_matrix"])
        cities = input_data["cities"]
        coords = np.array([[c["x"], c["y"]] for c in cities])
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))
