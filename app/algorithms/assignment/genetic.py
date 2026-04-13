"""Генетический алгоритм для задачи назначения."""

import time
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class GeneticAssignment(BaseAlgorithm):
    """Генетический алгоритм с PMX-кроссовером.

    Особь — перестановка (perm[i] = j означает «работник i → задание j»).
    Селекция турнирная, кроссовер PMX, мутация — обмен двух позиций.
    """

    name = "genetic"
    display_name = "Генетический алгоритм"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        pop_size = params.get("population_size", 100)
        generations = params.get("generations", 300)
        mutation_rate = params.get("mutation_rate", 0.05)

        cost_matrix = np.array(input_data["cost_matrix"])
        n = len(cost_matrix)

        start = time.perf_counter()

        population = [list(np.random.permutation(n)) for _ in range(pop_size)]
        convergence = []
        best_perm = None
        best_cost = float("inf")

        for gen in range(generations):
            costs = [self._perm_cost(ind, cost_matrix) for ind in population]

            for i, c in enumerate(costs):
                if c < best_cost:
                    best_cost = c
                    best_perm = population[i][:]

            convergence.append(float(best_cost))

            fitness = [1.0 / c if c > 0 else 1.0 for c in costs]

            # Турнирная селекция
            new_pop = []
            for _ in range(pop_size):
                t1, t2 = random.sample(range(pop_size), 2)
                winner = population[t1] if fitness[t1] > fitness[t2] else population[t2]
                new_pop.append(winner[:])

            # PMX-кроссовер
            for i in range(0, pop_size - 1, 2):
                if random.random() < 0.8:
                    c1, c2 = self._pmx(new_pop[i], new_pop[i + 1], n)
                    new_pop[i] = c1
                    new_pop[i + 1] = c2

            # Мутация (swap)
            for ind in new_pop:
                if random.random() < mutation_rate:
                    a, b = random.sample(range(n), 2)
                    ind[a], ind[b] = ind[b], ind[a]

            population = new_pop

        elapsed = time.perf_counter() - start

        assignments = [[i, best_perm[i]] for i in range(n)] if best_perm else []

        return AlgorithmResult(
            solution=assignments,
            cost=float(best_cost),
            execution_time=elapsed,
            iterations=generations,
            convergence_history=convergence,
        )

    # ── вспомогательные методы ──────────────────────────────────────

    @staticmethod
    def _perm_cost(perm: list[int], cost_matrix: np.ndarray) -> float:
        return float(sum(cost_matrix[i][perm[i]] for i in range(len(perm))))

    @staticmethod
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
