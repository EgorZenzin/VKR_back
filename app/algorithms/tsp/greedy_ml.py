"""Жадный алгоритм с ML-суррогатом для задачи коммивояжёра.

Подход: ML-суррогат используется для отбора лучших жадных маршрутов
из множества вариантов с рандомизированным выбором соседа.

1. Обучение суррогата на случайной выборке маршрутов с известной длиной.
2. Генерация большого пула рандомизированных жадных маршрутов
   (разные стартовые города + случайный выбор из top-k ближайших).
3. Быстрая оценка всех вариантов суррогатом.
4. Точный пересчёт длины для top-K маршрутов и возврат лучшего.
"""

import time
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.ml.surrogate import MLPSurrogateModel


class GreedyMLTSP(BaseAlgorithm):
    """Жадный алгоритм + ML-суррогат для TSP."""

    name = "greedy_ml"
    display_name = "Жадный алгоритм + ML"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        warmup_samples = params.get("warmup_samples", 150)
        candidate_count = params.get("candidate_count", 200)
        top_k_exact = params.get("top_k_exact", 20)
        top_k_neighbors = params.get("top_k_neighbors", 3)

        dist_matrix = _get_distance_matrix(input_data)
        n = len(dist_matrix)

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (32, 16)),
        )

        start = time.perf_counter()

        # ── Фаза 1: обучение суррогата на случайных маршрутах ───────
        sample_pool = [list(np.random.permutation(n)) for _ in range(warmup_samples)]
        train_X = np.array([_encode_route(r, n) for r in sample_pool])
        train_y = np.array([_route_cost(r, dist_matrix) for r in sample_pool])
        surrogate.fit(train_X, train_y)

        exact_evals = warmup_samples

        # ── Фаза 2: генерация рандомизированных жадных кандидатов ──
        candidates = []
        for _ in range(candidate_count):
            start_city = random.randint(0, n - 1)
            route = _randomized_greedy(start_city, dist_matrix, top_k_neighbors)
            candidates.append(route)

        X_cand = np.array([_encode_route(c, n) for c in candidates])
        pred_costs = surrogate.predict(X_cand)
        surrogate_evals = candidate_count

        # ── Фаза 3: точный пересчёт top-K кандидатов ───────────────
        top_indices = np.argsort(pred_costs)[:top_k_exact]
        best_route = None
        best_cost = float("inf")

        actual_costs = []
        for idx in top_indices:
            cost = _route_cost(candidates[idx], dist_matrix)
            actual_costs.append(cost)
            exact_evals += 1
            if cost < best_cost:
                best_cost = cost
                best_route = candidates[idx][:]

        # R² суррогата на проверенных кандидатах
        surrogate_r2 = surrogate.score(
            X_cand[top_indices], np.array(actual_costs)
        )

        elapsed = time.perf_counter() - start

        return AlgorithmResult(
            solution=best_route,
            cost=float(best_cost),
            execution_time=elapsed,
            iterations=None,
            convergence_history=[float(best_cost)],
            extra={
                "ml_used": True,
                "surrogate_model": "MLPRegressor",
                "exact_evaluations": exact_evals,
                "surrogate_evaluations": surrogate_evals,
                "surrogate_accuracy_r2": round(float(surrogate_r2), 4),
                "warmup_samples": warmup_samples,
                "candidate_count": candidate_count,
                "top_k_exact": top_k_exact,
                "top_k_neighbors": top_k_neighbors,
                "training_samples": warmup_samples,
            },
        )


# ── Вспомогательные функции ─────────────────────────────────────────

def _randomized_greedy(
    start_city: int, dist_matrix: np.ndarray, top_k: int,
) -> list[int]:
    """Жадный маршрут со случайным выбором из top-k ближайших соседей."""
    n = len(dist_matrix)
    route = [start_city]
    visited = {start_city}
    current = start_city

    for _ in range(n - 1):
        candidates = [
            (dist_matrix[current][j], j) for j in range(n) if j not in visited
        ]
        candidates.sort()
        k = min(top_k, len(candidates))
        _, nxt = random.choice(candidates[:k])
        route.append(nxt)
        visited.add(nxt)
        current = nxt

    return route


def _route_cost(route: list[int], dist_matrix: np.ndarray) -> float:
    cost = sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
    cost += dist_matrix[route[-1]][route[0]]
    return float(cost)


def _encode_route(route: list[int], n: int) -> list[float]:
    """Кодирование маршрута: позиция каждого города, нормированная в [0, 1]."""
    encoding = [0.0] * n
    for pos, city in enumerate(route):
        encoding[city] = pos / max(n - 1, 1)
    return encoding


def _get_distance_matrix(input_data: dict) -> np.ndarray:
    if "distance_matrix" in input_data and input_data["distance_matrix"]:
        return np.array(input_data["distance_matrix"])
    cities = input_data["cities"]
    coords = np.array([[c["x"], c["y"]] for c in cities])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))
