"""
NN-Guided Генетический алгоритм для задачи о рюкзаке.

Генетический алгоритм, усиленный суррогатной нейросетью.
Нейросеть обучается предсказывать ценность набора предметов
и отсеивает неперспективных кандидатов до полного вычисления.
"""

import numpy as np
import random
import time

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.algorithms.ml.surrogate_model import SurrogateModel


class NNGuidedGeneticKnapsack(BaseAlgorithm):
    """NN-Guided Генетический алгоритм для задачи о рюкзаке."""

    name = "nn_guided_genetic"
    display_name = "NN-Guided Генетический алгоритм"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        pop_size = params.get("population_size", 100)
        generations = params.get("generations", 300)
        mutation_rate = params.get("mutation_rate", 0.05)
        oversample_ratio = params.get("oversample_ratio", 2.0)

        items = input_data["items"]
        capacity = input_data["capacity"]
        n = len(items)

        weights = np.array([item["weight"] for item in items])
        values = np.array([item["value"] for item in items])

        surrogate = SurrogateModel(
            hidden_layers=(64, 32),
            warmup_size=min(pop_size, 50),
            retrain_every=pop_size,
        )

        start = time.perf_counter()

        population = [[random.randint(0, 1) for _ in range(n)] for _ in range(pop_size)]
        convergence = []
        nn_active_generations = 0
        best_solution = None
        best_value = -1.0

        for gen in range(generations):
            # Полное вычисление fitness
            fitness_list = []
            features_batch = []
            for ind in population:
                ind_arr = np.array(ind)
                w = float(np.dot(weights, ind_arr))
                v = float(np.dot(values, ind_arr))
                f = v if w <= capacity else 0.0
                fitness_list.append(f)
                features_batch.append(self._encode(ind_arr, weights, values, capacity))

            # Для НС: отрицательный fitness (минимизация стоимости → максимизация ценности)
            # Передаём -fitness чтобы select_top_k (argsort по возрастанию) выбирал лучших
            surrogate.add_batch(features_batch, [-f for f in fitness_list])

            if surrogate.should_train():
                surrogate.train()

            for i, f in enumerate(fitness_list):
                if f > best_value:
                    best_value = f
                    best_solution = population[i][:]

            convergence.append(best_value)

            # --- Турнирная селекция ---
            new_pop = []
            for _ in range(pop_size):
                t1, t2 = random.sample(range(pop_size), 2)
                winner = population[t1] if fitness_list[t1] >= fitness_list[t2] else population[t2]
                new_pop.append(winner[:])

            # --- Одноточечный кроссовер ---
            for i in range(0, pop_size - 1, 2):
                if random.random() < 0.8:
                    pt = random.randint(1, n - 1)
                    c1 = new_pop[i][:pt] + new_pop[i + 1][pt:]
                    c2 = new_pop[i + 1][:pt] + new_pop[i][pt:]
                    new_pop[i] = c1
                    new_pop[i + 1] = c2

            # --- Мутация ---
            for ind in new_pop:
                for j in range(n):
                    if random.random() < mutation_rate:
                        ind[j] = 1 - ind[j]

            # --- NN-фильтрация ---
            if surrogate.is_trained:
                nn_active_generations += 1
                n_extra = int(pop_size * (oversample_ratio - 1.0))
                extra_candidates = []
                for _ in range(n_extra):
                    base = random.choice(new_pop)[:]
                    j = random.randint(0, n - 1)
                    base[j] = 1 - base[j]
                    extra_candidates.append(base)

                all_candidates = new_pop + extra_candidates
                all_features = [
                    self._encode(np.array(c), weights, values, capacity)
                    for c in all_candidates
                ]

                predictions = surrogate.predict_batch(all_features)
                # argsort по возрастанию — меньше = лучше (мы передали -fitness)
                top_indices = np.argsort(predictions)[:pop_size]
                population = [all_candidates[i] for i in top_indices]
            else:
                population = new_pop

        elapsed = time.perf_counter() - start

        selected = [i for i in range(n) if best_solution[i] == 1]
        total_weight = sum(weights[i] for i in selected)

        return AlgorithmResult(
            solution=selected,
            cost=best_value,
            execution_time=elapsed,
            convergence_history=convergence,
            extra={
                "total_weight": float(total_weight),
                "nn_active_generations": nn_active_generations,
                "total_generations": generations,
                "surrogate_stats": surrogate.get_stats(),
            },
        )

    def _encode(
        self, ind: np.ndarray, weights: np.ndarray, values: np.ndarray, capacity: float
    ) -> np.ndarray:
        """Кодировка решения в вектор признаков для НС.

        Признаки: бинарный вектор выбора + суммарный вес/ёмкость
        + суммарная ценность + плотность ценности выбранных.
        """
        total_w = float(np.dot(weights, ind))
        total_v = float(np.dot(values, ind))
        ratio = total_w / capacity if capacity > 0 else 0.0
        density = total_v / max(total_w, 1e-9)
        stats = np.array([total_w, total_v, ratio, density])
        return np.concatenate([ind, stats])
