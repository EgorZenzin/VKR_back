"""Имитация отжига с ML-суррогатом для задачи коммивояжёра.

Стратегия (упрощённая и надёжная):
1. Всегда запускаем baseline SimulatedAnnealingTSP — это гарантирует,
   что итоговый результат не хуже чистого SA.
2. Дополнительно: ML обучается на случайной выборке, отбирает лучший
   стартовый маршрут из пула кандидатов, и SA с этого старта пытается
   улучшить решение.
3. Возвращаем лучший из двух (ML-вариант и baseline) маршрутов.
"""

import time
import math
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.algorithms.tsp.simulated_annealing import SimulatedAnnealingTSP
from app.ml.surrogate import LinearSurrogateModel
from app.ml.encoders import encode_route_adjacency


# Ниже этого порога ML-фаза просто пропускается — overhead не окупается.
ML_MIN_N = 15


class SimulatedAnnealingMLTSP(BaseAlgorithm):
    """Имитация отжига + ML-суррогат для TSP.

    Гарантирует качество ≥ baseline SimulatedAnnealingTSP за счёт
    финального сравнения и выбора лучшего из двух запусков.
    """

    name = "simulated_annealing_ml"
    display_name = "Имитация отжига + ML"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        dist_matrix = _get_distance_matrix(input_data)
        n = len(dist_matrix)

        start = time.perf_counter()

        # ── Этап 1: baseline SA (всегда). Гарантирует качество ≥ SA.
        baseline = SimulatedAnnealingTSP().solve(input_data, params)
        best_route = list(baseline.solution)
        best_cost = float(baseline.cost)
        convergence: list[float] = list(baseline.convergence_history or [])
        total_iterations = int(baseline.iterations or 0)

        ml_used = False
        ml_info: dict = {
            "ml_used": False,
            "surrogate_model": "Ridge",
            "fallback_reason": None,
        }

        # ── Этап 2: ML-фаза (только для достаточно больших n) ──────
        if n < ML_MIN_N:
            ml_info["fallback_reason"] = f"n<{ML_MIN_N}, ML overhead не окупается"
        else:
            ml_used = True
            warmup_samples = params.get("warmup_samples", max(80, n * 6))
            screening_factor = params.get("screening_factor", 30)
            ridge_alpha = params.get("ridge_alpha", 1.0)

            sample_pool = [list(np.random.permutation(n)) for _ in range(warmup_samples)]
            train_X = np.array([encode_route_adjacency(r, n) for r in sample_pool])
            train_y = np.array([_route_cost(r, dist_matrix) for r in sample_pool])

            surrogate = LinearSurrogateModel(alpha=ridge_alpha)
            surrogate.fit(train_X, train_y)
            r2 = float(surrogate.score(train_X, train_y))

            candidates = [list(np.random.permutation(n)) for _ in range(screening_factor)]
            X_cand = np.array([encode_route_adjacency(r, n) for r in candidates])
            preds = surrogate.predict(X_cand)
            initial = candidates[int(np.argmin(preds))]

            ml_result = _run_sa_from(initial, dist_matrix, params)
            total_iterations += ml_result["iterations"]

            if ml_result["cost"] < best_cost:
                best_route = ml_result["route"]
                best_cost = ml_result["cost"]
                convergence.extend(float(c) for c in ml_result["convergence"])

            ml_info.update({
                "ml_used": True,
                "surrogate_model": "Ridge",
                "warmup_samples": warmup_samples,
                "screening_factor": screening_factor,
                "surrogate_accuracy_r2": round(r2, 4),
                "ml_cost": round(float(ml_result["cost"]), 6),
                "baseline_cost": round(float(baseline.cost), 6),
                "ml_won": ml_result["cost"] < float(baseline.cost),
            })

        elapsed = time.perf_counter() - start

        if not convergence or convergence[-1] != best_cost:
            convergence.append(float(best_cost))

        extra = ml_info
        if not ml_used and ml_info.get("fallback_reason"):
            extra["fallback_to"] = "simulated_annealing"

        return AlgorithmResult(
            solution=best_route,
            cost=float(best_cost),
            execution_time=elapsed,
            iterations=total_iterations,
            convergence_history=convergence,
            extra=extra,
        )


# ── Вспомогательные функции ─────────────────────────────────────────

def _run_sa_from(initial: list[int], dist_matrix: np.ndarray, params: dict) -> dict:
    """Запустить чистый SA с заданного стартового маршрута."""
    initial_temp = params.get("initial_temp", 10000.0)
    cooling_rate = params.get("cooling_rate", 0.9995)
    min_temp = params.get("min_temp", 1e-8)

    n = len(dist_matrix)
    current = list(initial)
    current_cost = _route_cost(current, dist_matrix)

    best = current[:]
    best_cost = current_cost
    temp = initial_temp
    convergence: list[float] = []
    iteration = 0

    while temp > min_temp:
        i, j = sorted(random.sample(range(n), 2))
        neighbor = current[:]
        neighbor[i:j + 1] = reversed(neighbor[i:j + 1])
        neighbor_cost = _route_cost(neighbor, dist_matrix)

        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-12)):
            current = neighbor
            current_cost = neighbor_cost

        if current_cost < best_cost:
            best = current[:]
            best_cost = current_cost

        temp *= cooling_rate
        iteration += 1

        if iteration % 100 == 0:
            convergence.append(float(best_cost))

    return {
        "route": best,
        "cost": float(best_cost),
        "iterations": iteration,
        "convergence": convergence,
    }


def _route_cost(route: list[int], dist_matrix: np.ndarray) -> float:
    cost = sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
    cost += dist_matrix[route[-1]][route[0]]
    return float(cost)


def _get_distance_matrix(input_data: dict) -> np.ndarray:
    if "distance_matrix" in input_data and input_data["distance_matrix"]:
        return np.array(input_data["distance_matrix"])
    cities = input_data["cities"]
    coords = np.array([[c["x"], c["y"]] for c in cities])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))
