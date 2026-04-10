"""
NN-Guided Генетический алгоритм для раскраски графа.

Генетический алгоритм, усиленный суррогатной нейросетью.
Нейросеть обучается предсказывать количество конфликтов и цветов,
отсеивая неперспективные раскраски до полного вычисления.
"""

import numpy as np
import random
import time

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.algorithms.ml.surrogate_model import SurrogateModel


class NNGuidedGeneticGraphColoring(BaseAlgorithm):
    """NN-Guided Генетический алгоритм для раскраски графа."""

    name = "nn_guided_genetic"
    display_name = "NN-Guided Генетический алгоритм"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        pop_size = params.get("population_size", 100)
        generations = params.get("generations", 300)
        mutation_rate = params.get("mutation_rate", 0.05)
        oversample_ratio = params.get("oversample_ratio", 2.0)

        n = input_data["n_vertices"]
        edges = input_data["edges"]
        edge_list = [(e["u"], e["v"]) for e in edges]

        surrogate = SurrogateModel(
            hidden_layers=(64, 32),
            warmup_size=min(pop_size, 50),
            retrain_every=pop_size,
        )

        start = time.perf_counter()

        max_colors = n
        population = [
            [random.randint(0, max_colors - 1) for _ in range(n)]
            for _ in range(pop_size)
        ]

        convergence = []
        nn_active_generations = 0
        best_solution = None
        best_fitness = float("inf")

        for gen in range(generations):
            # Полное вычисление fitness
            fitness_list = []
            features_batch = []
            for ind in population:
                conflicts = sum(1 for u, v in edge_list if ind[u] == ind[v])
                n_colors = len(set(ind))
                f = conflicts * 1000 + n_colors
                fitness_list.append(f)
                features_batch.append(self._encode(ind, n, edge_list))

                if f < best_fitness:
                    best_fitness = f
                    best_solution = ind[:]

            surrogate.add_batch(features_batch, fitness_list)

            if surrogate.should_train():
                surrogate.train()

            convergence.append(best_fitness)

            # --- Селекция ---
            new_pop = []
            for _ in range(pop_size):
                t1, t2 = random.sample(range(pop_size), 2)
                winner = population[t1] if fitness_list[t1] < fitness_list[t2] else population[t2]
                new_pop.append(winner[:])

            # --- Кроссовер (uniform) ---
            for i in range(0, pop_size - 1, 2):
                if random.random() < 0.8:
                    for j in range(n):
                        if random.random() < 0.5:
                            new_pop[i][j], new_pop[i + 1][j] = new_pop[i + 1][j], new_pop[i][j]

            # --- Мутация ---
            for ind in new_pop:
                for j in range(n):
                    if random.random() < mutation_rate:
                        ind[j] = random.randint(0, max_colors - 1)

            # --- NN-фильтрация ---
            if surrogate.is_trained:
                nn_active_generations += 1
                n_extra = int(pop_size * (oversample_ratio - 1.0))
                extra_candidates = []
                for _ in range(n_extra):
                    base = random.choice(new_pop)[:]
                    j = random.randint(0, n - 1)
                    base[j] = random.randint(0, max_colors - 1)
                    extra_candidates.append(base)

                all_candidates = new_pop + extra_candidates
                all_features = [self._encode(c, n, edge_list) for c in all_candidates]

                # argsort по возрастанию — меньше fitness = лучше
                predictions = surrogate.predict_batch(all_features)
                top_indices = np.argsort(predictions)[:pop_size]
                population = [all_candidates[i] for i in top_indices]
            else:
                population = new_pop

        # Перенумеровка цветов
        if best_solution:
            unique_colors = sorted(set(best_solution))
            remap = {c: i for i, c in enumerate(unique_colors)}
            best_solution = [remap[c] for c in best_solution]

        n_colors = len(set(best_solution)) if best_solution else 0
        elapsed = time.perf_counter() - start

        return AlgorithmResult(
            solution={str(i): best_solution[i] for i in range(n)},
            cost=float(n_colors),
            execution_time=elapsed,
            convergence_history=convergence,
            extra={
                "nn_active_generations": nn_active_generations,
                "total_generations": generations,
                "surrogate_stats": surrogate.get_stats(),
            },
        )

    def _encode(self, ind: list[int], n: int, edge_list: list) -> np.ndarray:
        """Кодировка раскраски в вектор признаков для НС.

        Признаки: one-hot по цветам для каждой вершины сжатый в
        кол-во вершин каждого цвета + конфликты на каждое ребро + статистики.
        """
        max_c = max(ind) + 1 if ind else 1
        color_counts = np.zeros(n)
        for c in ind:
            if c < n:
                color_counts[c] += 1

        edge_conflicts = np.array([1.0 if ind[u] == ind[v] else 0.0 for u, v in edge_list])

        total_conflicts = edge_conflicts.sum()
        n_colors = len(set(ind))
        stats = np.array([total_conflicts, n_colors, total_conflicts / max(len(edge_list), 1)])

        return np.concatenate([color_counts, edge_conflicts, stats])
