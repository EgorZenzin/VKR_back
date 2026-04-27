"""ML-усечённый «полный перебор» для задачи коммивояжёра.

Стратегия: исчерпывающее перечисление всех перестановок,
но точная оценка выполняется только для top-K кандидатов,
отобранных суррогатом.

ВНИМАНИЕ: оптимальность НЕ гарантируется. Этот алгоритм нужен
для сравнения времени и качества с честным brute-force.
"""

import time
import itertools
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.ml.surrogate import MLPSurrogateModel


class BruteForceMLTSP(BaseAlgorithm):
    """ML-усечённый перебор для TSP."""

    name = "brute_force_ml"
    display_name = "Полный перебор + ML"

    MAX_CITIES = 11

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        warmup_samples = params.get("warmup_samples", 200)
        top_k_exact = params.get("top_k_exact", 50)

        dist_matrix = _get_distance_matrix(input_data)
        n = len(dist_matrix)

        if n > self.MAX_CITIES:
            raise ValueError(
                f"Полный перебор + ML допустим для n ≤ {self.MAX_CITIES}, получено n = {n}"
            )

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (32, 16)),
        )

        start = time.perf_counter()

        # ── Фаза 1: прогрев суррогата ──────────────────────────────
        sample_pool = [list(np.random.permutation(n)) for _ in range(warmup_samples)]
        train_X = np.array([_encode_route(r, n) for r in sample_pool])
        train_y = np.array([_route_cost(r, dist_matrix) for r in sample_pool])
        surrogate.fit(train_X, train_y)

        exact_evals = warmup_samples

        # ── Фаза 2: ML-скрининг всех перестановок ──────────────────
        all_routes = [[0] + list(p) for p in itertools.permutations(range(1, n))]
        X_all = np.array([_encode_route(r, n) for r in all_routes])
        pred_costs = surrogate.predict(X_all)
        surrogate_evals = len(all_routes)

        # ── Фаза 3: точный пересчёт top-K ──────────────────────────
        top_k = min(top_k_exact, len(all_routes))
        top_indices = np.argsort(pred_costs)[:top_k]

        best_route = None
        best_cost = float("inf")
        actual_costs = []
        for idx in top_indices:
            cost = _route_cost(all_routes[idx], dist_matrix)
            actual_costs.append(cost)
            exact_evals += 1
            if cost < best_cost:
                best_cost = cost
                best_route = all_routes[idx][:]

        surrogate_r2 = surrogate.score(X_all[top_indices], np.array(actual_costs))

        elapsed = time.perf_counter() - start

        return AlgorithmResult(
            solution=best_route,
            cost=float(best_cost),
            execution_time=elapsed,
            iterations=len(all_routes),
            convergence_history=[float(best_cost)],
            extra={
                "ml_used": True,
                "surrogate_model": "MLPRegressor",
                "exact_evaluations": exact_evals,
                "surrogate_evaluations": surrogate_evals,
                "surrogate_accuracy_r2": round(float(surrogate_r2), 4),
                "warmup_samples": warmup_samples,
                "top_k_exact": top_k,
                "total_candidates": len(all_routes),
                "training_samples": warmup_samples,
                "optimality_guaranteed": False,
            },
        )


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
