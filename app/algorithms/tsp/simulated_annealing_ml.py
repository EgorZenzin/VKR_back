"""Имитация отжига с ML-суррогатом для задачи коммивояжёра.

Подход: ML-суррогат используется для отбора перспективных соседей
на каждом шаге отжига.

1. Прогрев суррогата на случайных маршрутах с известной длиной.
2. На каждом шаге генерируется несколько 2-opt соседей.
3. Суррогат предсказывает их стоимость, выбирается лучший предсказанный.
4. Точная оценка только для выбранного кандидата → Метрополис.
5. Периодическое дообучение суррогата на накопленных точных данных.
"""

import time
import math
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.ml.surrogate import MLPSurrogateModel


class SimulatedAnnealingMLTSP(BaseAlgorithm):
    """Имитация отжига + ML-суррогат для TSP."""

    name = "simulated_annealing_ml"
    display_name = "Имитация отжига + ML"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        initial_temp = params.get("initial_temp", 10000.0)
        cooling_rate = params.get("cooling_rate", 0.9995)
        min_temp = params.get("min_temp", 1e-8)
        warmup_samples = params.get("warmup_samples", 150)
        neighbor_candidates = params.get("neighbor_candidates", 5)
        retrain_every = params.get("retrain_every", 500)

        dist_matrix = _get_distance_matrix(input_data)
        n = len(dist_matrix)

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (32, 16)),
        )

        start = time.perf_counter()

        # ── Фаза 1: прогрев суррогата ──────────────────────────────
        sample_pool = [list(np.random.permutation(n)) for _ in range(warmup_samples)]
        train_X_list = [_encode_route(r, n) for r in sample_pool]
        train_y_list = [_route_cost(r, dist_matrix) for r in sample_pool]

        surrogate.fit(np.array(train_X_list), np.array(train_y_list))

        exact_evals = warmup_samples
        surrogate_evals = 0

        # ── Фаза 2: SA с ML-отбором соседей ────────────────────────
        current = list(np.random.permutation(n))
        current_cost = _route_cost(current, dist_matrix)
        exact_evals += 1

        best = current[:]
        best_cost = current_cost
        temp = initial_temp
        convergence: list[float] = []
        iteration = 0

        # Накопленные точные данные для дообучения
        train_X_list.append(_encode_route(current, n))
        train_y_list.append(current_cost)

        while temp > min_temp:
            # Генерация нескольких 2-opt соседей
            neighbors = []
            for _ in range(neighbor_candidates):
                i, j = sorted(random.sample(range(n), 2))
                nb = current[:]
                nb[i:j + 1] = reversed(nb[i:j + 1])
                neighbors.append(nb)

            # ML-скрининг: выбираем соседа с лучшей предсказанной стоимостью
            X_nb = np.array([_encode_route(nb, n) for nb in neighbors])
            pred = surrogate.predict(X_nb)
            surrogate_evals += len(neighbors)

            best_pred_idx = int(np.argmin(pred))
            chosen = neighbors[best_pred_idx]

            # Точная оценка только для выбранного
            chosen_cost = _route_cost(chosen, dist_matrix)
            exact_evals += 1

            # Обновляем обучающую выборку
            train_X_list.append(_encode_route(chosen, n))
            train_y_list.append(chosen_cost)

            # Критерий Метрополиса
            delta = chosen_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-12)):
                current = chosen
                current_cost = chosen_cost

            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

            temp *= cooling_rate
            iteration += 1

            if iteration % 100 == 0:
                convergence.append(float(best_cost))

            # Периодическое дообучение
            if iteration % retrain_every == 0 and len(train_X_list) > 10:
                surrogate.fit(np.array(train_X_list), np.array(train_y_list))

        # Финальная оценка R²
        final_X = np.array(train_X_list[-min(50, len(train_X_list)):])
        final_y = np.array(train_y_list[-min(50, len(train_y_list)):])
        surrogate_r2 = surrogate.score(final_X, final_y)

        elapsed = time.perf_counter() - start

        if not convergence or convergence[-1] != best_cost:
            convergence.append(float(best_cost))

        return AlgorithmResult(
            solution=best,
            cost=float(best_cost),
            execution_time=elapsed,
            iterations=iteration,
            convergence_history=convergence,
            extra={
                "ml_used": True,
                "surrogate_model": "MLPRegressor",
                "exact_evaluations": exact_evals,
                "surrogate_evaluations": surrogate_evals,
                "surrogate_accuracy_r2": round(float(surrogate_r2), 4),
                "warmup_samples": warmup_samples,
                "neighbor_candidates": neighbor_candidates,
                "retrain_every": retrain_every,
                "training_samples": len(train_X_list),
            },
        )


# ── Вспомогательные функции ─────────────────────────────────────────

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
