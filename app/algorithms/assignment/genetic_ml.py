"""Генетический алгоритм с ML-суррогатом для задачи назначения.

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


class GeneticMLAssignment(BaseAlgorithm):
    """Генетический алгоритм + ML-суррогат для задачи назначения."""

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

        cost_matrix = np.array(input_data["cost_matrix"])
        n = len(cost_matrix)

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (32, 16)),
        )

        start = time.perf_counter()

        # ── Фаза 1: Обучение суррогата на случайной выборке ─────────
        sample_pool = [list(np.random.permutation(n)) for _ in range(warmup_samples)]
        train_X = np.array([_encode_perm(r, n) for r in sample_pool])
        train_y = np.array([_perm_cost(r, cost_matrix) for r in sample_pool])
        surrogate.fit(train_X, train_y)

        exact_evals = warmup_samples

        # ── Фаза 2: ML-скрининг начальной популяции ────────────────
        candidate_count = pop_size * screening_factor
        candidates = [list(np.random.permutation(n)) for _ in range(candidate_count)]
        X_cand = np.array([_encode_perm(c, n) for c in candidates])
        pred_costs = surrogate.predict(X_cand)

        top_indices = np.argsort(pred_costs)[:pop_size]
        population = [candidates[i] for i in top_indices]

        # Сидируем популяцию детерминированным жадным назначением.
        population[0] = _greedy_assignment(cost_matrix)

        surrogate_evals = candidate_count

        # R² суррогата на независимой валидационной выборке.
        val_size = min(40, max(15, warmup_samples // 3))
        val_perms = [list(np.random.permutation(n)) for _ in range(val_size)]
        val_X = np.array([_encode_perm(p, n) for p in val_perms])
        val_y = np.array([_perm_cost(p, cost_matrix) for p in val_perms])
        exact_evals += val_size
        surrogate_r2 = surrogate.score(val_X, val_y)

        # ── Фаза 3: Стандартный ГА (сокращённые поколения) ─────────
        reduced_gens = max(1, int(generations * 0.6))

        best_perm = None
        best_cost = float("inf")
        convergence = []

        for gen in range(reduced_gens):
            costs = [_perm_cost(ind, cost_matrix) for ind in population]
            exact_evals += pop_size

            for i, ind in enumerate(population):
                if costs[i] < best_cost:
                    best_cost = costs[i]
                    best_perm = ind[:]

            convergence.append(float(best_cost))

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

            # Элитизм: всегда сохраняем лучшего найденного.
            if best_perm is not None:
                new_pop[0] = best_perm[:]

            population = new_pop

        elapsed = time.perf_counter() - start

        assignments = [[i, best_perm[i]] for i in range(n)] if best_perm else []

        return AlgorithmResult(
            solution=assignments,
            cost=float(best_cost),
            execution_time=elapsed,
            iterations=reduced_gens,
            convergence_history=convergence,
            extra={
                "ml_used": True,
                "surrogate_model": "Ridge",
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

def _greedy_assignment(cost_matrix: np.ndarray) -> list[int]:
    """Детерминированное жадное назначение: для каждой строки выбираем
    минимальный неиспользованный столбец."""
    n = len(cost_matrix)
    used = [False] * n
    perm: list[int] = []
    for i in range(n):
        best_j = -1
        best_c = float("inf")
        for j in range(n):
            if not used[j] and cost_matrix[i][j] < best_c:
                best_c = cost_matrix[i][j]
                best_j = j
        perm.append(best_j)
        used[best_j] = True
    return perm


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
