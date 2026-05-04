"""Жадный алгоритм с ML-суррогатом для задачи коммивояжёра.

Подход: ML-суррогат используется для отбора лучших жадных маршрутов
из множества вариантов с рандомизированным выбором соседа.

Стратегия:
1. Если n маленькое — fallback к обычному жадному алгоритму
   (overhead ML здесь больше, чем сам алгоритм).
2. Иначе: лёгкий MLP, обучение на небольшой выборке,
   генерация рандомизированных кандидатов, ML-скрининг,
   точный пересчёт top-K.
"""

import time
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.algorithms.tsp.greedy import GreedyTSP
from app.ml.surrogate import MLPSurrogateModel


# Минимальный размер задачи, при котором ML имеет смысл.
# Для n<20 жадный работает за доли миллисекунды и близок к оптимуму.
ML_MIN_N = 20


class GreedyMLTSP(BaseAlgorithm):
    """Жадный алгоритм + ML-суррогат для TSP."""

    name = "greedy_ml"
    display_name = "Жадный алгоритм + ML"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        dist_matrix = _get_distance_matrix(input_data)
        n = len(dist_matrix)

        # ── Fallback для маленьких задач ───────────────────────────
        if n < ML_MIN_N:
            base_result = GreedyTSP().solve(input_data, params)
            extra = dict(base_result.extra or {})
            extra.update({
                "ml_used": False,
                "fallback_reason": "input_too_small_for_ml",
                "fallback_threshold": ML_MIN_N,
                "fallback_to": "greedy",
            })
            return AlgorithmResult(
                solution=base_result.solution,
                cost=base_result.cost,
                execution_time=base_result.execution_time,
                iterations=base_result.iterations,
                convergence_history=base_result.convergence_history,
                extra=extra,
            )

        # ── Облегчённый ML-режим ───────────────────────────────────
        warmup_samples = params.get("warmup_samples", 200)
        candidate_count = params.get("candidate_count", 80)
        top_k_exact = params.get("top_k_exact", 10)
        top_k_neighbors = params.get("top_k_neighbors", 3)

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (32, 16)),
            max_iter=params.get("max_iter", 80),
            learning_rate_init=params.get("learning_rate_init", 0.02),
        )

        start = time.perf_counter()

        sample_pool = [list(np.random.permutation(n)) for _ in range(warmup_samples)]
        train_X = np.array([_encode_route(r, n) for r in sample_pool])
        train_y = np.array([_route_cost(r, dist_matrix) for r in sample_pool])
        surrogate.fit(train_X, train_y)
        exact_evals = warmup_samples

        # Гарантированные детерминированные кандидаты (жадные с разных стартов).
        # Они всегда точно оцениваются — гарантия качества не хуже базы.
        guaranteed = [
            _deterministic_greedy(s, dist_matrix)
            for s in range(min(n, 4))
        ]
        candidates = list(guaranteed)
        for _ in range(candidate_count - len(guaranteed)):
            start_city = random.randint(0, n - 1)
            route = _randomized_greedy(start_city, dist_matrix, top_k_neighbors)
            candidates.append(route)
        guaranteed_count = len(guaranteed)

        X_cand = np.array([_encode_route(c, n) for c in candidates])
        pred_costs = surrogate.predict(X_cand)
        surrogate_evals = candidate_count

        # Объединяем top-K по ML + гарантированных кандидатов.
        ml_top = np.argsort(pred_costs)[:top_k_exact].tolist()
        eval_indices = sorted(set(range(guaranteed_count)) | set(ml_top))
        best_route = None
        best_cost = float("inf")

        actual_costs_map: dict[int, float] = {}
        for idx in eval_indices:
            cost = _route_cost(candidates[idx], dist_matrix)
            actual_costs_map[idx] = cost
            exact_evals += 1
            if cost < best_cost:
                best_cost = cost
                best_route = candidates[idx][:]

        # R² на отдельной валидационной выборке (случайные перестановки с разным
        # разбросом стоимости) — top-K не годится, там почти нет вариации.
        val_size = min(30, max(10, warmup_samples // 2))
        val_routes = [list(np.random.permutation(n)) for _ in range(val_size)]
        val_X = np.array([_encode_route(r, n) for r in val_routes])
        val_y = np.array([_route_cost(r, dist_matrix) for r in val_routes])
        exact_evals += val_size
        surrogate_r2 = surrogate.score(val_X, val_y)

        elapsed = time.perf_counter() - start

        return AlgorithmResult(
            solution=best_route,
            cost=float(best_cost),
            execution_time=elapsed,
            iterations=None,
            convergence_history=[float(best_cost)],
            extra={
                "ml_used": True,
                "surrogate_model": "Ridge",
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

def _deterministic_greedy(start_city: int, dist_matrix: np.ndarray) -> list[int]:
    """Обычный жадный маршрут (без рандома)."""
    n = len(dist_matrix)
    route = [start_city]
    visited = {start_city}
    current = start_city
    for _ in range(n - 1):
        nxt = min(
            (j for j in range(n) if j not in visited),
            key=lambda j: dist_matrix[current][j],
        )
        route.append(nxt)
        visited.add(nxt)
        current = nxt
    return route


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
