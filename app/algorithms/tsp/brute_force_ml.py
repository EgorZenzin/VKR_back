"""ML-усечённый «полный перебор» для задачи коммивояжёра.

Стратегия:
1. Если n маленькое — fallback к честному brute_force (точный оптимум).
2. Иначе: вместо полного перебора (n-1)! используется случайная
   выборка кандидатов; ML-суррогат отбирает top-K по предсказанию,
   точная оценка только для top-K.

Это даёт время O(sample_count) вместо O((n-1)!), и при n ≥ 8
становится быстрее честного brute_force. Оптимальность не гарантируется.
"""

import time
import random
import math
import itertools
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.algorithms.tsp.brute_force import BruteForceTSP
from app.ml.surrogate import MLPSurrogateModel


ML_MIN_N = 8
MAX_CITIES = 14


class BruteForceMLTSP(BaseAlgorithm):
    """ML-усечённый перебор для TSP (sampling + суррогат)."""

    name = "brute_force_ml"
    display_name = "Полный перебор + ML"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        dist_matrix = _get_distance_matrix(input_data)
        n = len(dist_matrix)

        if n > MAX_CITIES:
            raise ValueError(
                f"Полный перебор + ML допустим для n ≤ {MAX_CITIES}, получено n = {n}"
            )

        # ── Fallback: для маленьких n честный brute_force работает быстро
        if n < ML_MIN_N:
            base = BruteForceTSP().solve(input_data, params)
            extra = dict(base.extra or {})
            extra.update({
                "ml_used": False,
                "fallback_reason": "input_too_small_for_ml",
                "fallback_threshold": ML_MIN_N,
                "fallback_to": "brute_force",
                "optimality_guaranteed": True,
            })
            return AlgorithmResult(
                solution=base.solution,
                cost=base.cost,
                execution_time=base.execution_time,
                iterations=base.iterations,
                convergence_history=base.convergence_history,
                extra=extra,
            )

        warmup_samples = params.get("warmup_samples", 120)
        sample_count = params.get("sample_count", 800)
        top_k_exact = params.get("top_k_exact", 30)

        total_perms = math.factorial(n - 1)
        sample_count = min(sample_count, total_perms)

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (16,)),
            max_iter=params.get("max_iter", 25),
            learning_rate_init=params.get("learning_rate_init", 0.03),
        )

        start = time.perf_counter()

        # ── Фаза 1: прогрев суррогата ──────────────────────────────
        warmup_routes = [
            [0] + list(np.random.permutation(range(1, n)))
            for _ in range(warmup_samples)
        ]
        train_X = np.array([_encode_route(r, n) for r in warmup_routes])
        train_y = np.array([_route_cost(r, dist_matrix) for r in warmup_routes])
        surrogate.fit(train_X, train_y)
        exact_evals = warmup_samples

        # Гарантированные кандидаты: жадные маршруты с разных стартов.
        # Берём ровно эти маршруты в канонической форме с нулевым первым городом.
        guaranteed_routes = []
        seen_canon: set[tuple[int, ...]] = set()
        for s in range(min(n, 5)):
            r = _deterministic_greedy_route(s, dist_matrix)
            r_canon = _canonical_route(r)
            key = tuple(r_canon)
            if key not in seen_canon:
                seen_canon.add(key)
                guaranteed_routes.append(r_canon)

        # ── Фаза 2: sampling кандидатов вместо полного перебора ────
        if total_perms <= sample_count:
            sampled = [
                [0] + list(p) for p in itertools.permutations(range(1, n))
            ]
        else:
            seen: set[tuple[int, ...]] = set()
            sampled = []
            for r in guaranteed_routes:
                key = tuple(r[1:])
                if key not in seen:
                    seen.add(key)
                    sampled.append(r)
            while len(sampled) < sample_count:
                tail = list(np.random.permutation(range(1, n)))
                key = tuple(tail)
                if key in seen:
                    continue
                seen.add(key)
                sampled.append([0] + tail)
        guaranteed_count = len(guaranteed_routes)

        X_all = np.array([_encode_route(r, n) for r in sampled])
        pred_costs = surrogate.predict(X_all)
        surrogate_evals = len(sampled)

        # ── Фаза 3: top-K + гарантированные кандидаты ───
        top_k = min(top_k_exact, len(sampled))
        ml_top = np.argsort(pred_costs)[:top_k].tolist()
        # Гарантированные кандидаты лежат в sampled[:guaranteed_count] по конструкции
        # (либо все перестановки, тогда гарантированные обязательно в переборе).
        if total_perms > sample_count:
            eval_indices = sorted(set(range(guaranteed_count)) | set(ml_top))
        else:
            eval_indices = ml_top

        best_route = None
        best_cost = float("inf")
        evaluated: list[tuple[float, list[int]]] = []
        for idx in eval_indices:
            cost = _route_cost(sampled[idx], dist_matrix)
            exact_evals += 1
            evaluated.append((cost, sampled[idx][:]))
            if cost < best_cost:
                best_cost = cost
                best_route = sampled[idx][:]

        # Дополнительно: точно оцениваем все детерминированные жадные маршруты
        # (если они не попали в sampled из-за ограничения выборки).
        for r in guaranteed_routes:
            cost = _route_cost(r, dist_matrix)
            exact_evals += 1
            evaluated.append((cost, r[:]))
            if cost < best_cost:
                best_cost = cost
                best_route = r[:]

        # ── Фаза 4: 2-opt локальный поиск на нескольких лучших стартах ─
        evaluated.sort(key=lambda t: t[0])
        n_starts = min(5, len(evaluated))
        for start_cost, start_route in evaluated[:n_starts]:
            ls_route, ls_cost, two_opt_iters = _two_opt(
                start_route, dist_matrix, start_cost
            )
            exact_evals += two_opt_iters
            if ls_cost < best_cost:
                best_cost = ls_cost
                best_route = ls_route

        # R² на отдельной валидационной выборке.
        val_size = min(30, max(10, warmup_samples // 2))
        val_routes = [
            [0] + list(np.random.permutation(range(1, n)))
            for _ in range(val_size)
        ]
        val_X = np.array([_encode_route(r, n) for r in val_routes])
        val_y = np.array([_route_cost(r, dist_matrix) for r in val_routes])
        exact_evals += val_size
        surrogate_r2 = surrogate.score(val_X, val_y)

        elapsed = time.perf_counter() - start

        return AlgorithmResult(
            solution=best_route,
            cost=float(best_cost),
            execution_time=elapsed,
            iterations=len(sampled),
            convergence_history=[float(best_cost)],
            extra={
                "ml_used": True,
                "surrogate_model": "Ridge",
                "exact_evaluations": exact_evals,
                "surrogate_evaluations": surrogate_evals,
                "surrogate_accuracy_r2": round(float(surrogate_r2), 4),
                "warmup_samples": warmup_samples,
                "sample_count": len(sampled),
                "top_k_exact": top_k,
                "total_candidates": total_perms,
                "training_samples": warmup_samples,
                "optimality_guaranteed": False,
            },
        )


def _randomized_greedy_route(
    start_city: int, dist_matrix: np.ndarray, top_k: int = 3,
) -> list[int]:
    """Жадный маршрут со случайным выбором из top-k ближайших соседей."""
    n = len(dist_matrix)
    route = [start_city]
    visited = {start_city}
    current = start_city
    for _ in range(n - 1):
        cands = [
            (dist_matrix[current][j], j) for j in range(n) if j not in visited
        ]
        cands.sort()
        k = min(top_k, len(cands))
        _, nxt = random.choice(cands[:k])
        route.append(nxt)
        visited.add(nxt)
        current = nxt
    return route


def _two_opt(
    route: list[int], dist_matrix: np.ndarray, cost: float,
) -> tuple[list[int], float, int]:
    """Локальный поиск 2-opt до первого локального оптимума."""
    n = len(route)
    improved = True
    evals = 0
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = route[i], route[(i + 1) % n]
                c, d = route[j], route[(j + 1) % n]
                delta = (
                    dist_matrix[a][c] + dist_matrix[b][d]
                    - dist_matrix[a][b] - dist_matrix[c][d]
                )
                evals += 1
                if delta < -1e-12:
                    route[i + 1:j + 1] = route[i + 1:j + 1][::-1]
                    cost += float(delta)
                    improved = True
    return route, cost, evals


def _deterministic_greedy_route(start_city: int, dist_matrix: np.ndarray) -> list[int]:
    """Жадный маршрут от заданного стартового города."""
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


def _canonical_route(route: list[int]) -> list[int]:
    """Привести маршрут к виду с первым городом в позиции 0 (без изменения порядка)."""
    if route[0] == 0:
        return route[:]
    idx = route.index(0)
    return route[idx:] + route[:idx]


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
