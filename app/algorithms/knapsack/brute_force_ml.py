"""ML-усечённый «полный перебор» для задачи о рюкзаке.

Стратегия:
1. Если n маленькое — fallback к точному brute_force.
2. Иначе: sampling 2^n подмножеств вместо полного перебора;
   суррогат отбирает top-K, точный пересчёт только top-K.
"""

import time
import random
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.algorithms.knapsack.brute_force import BruteForceKnapsack
from app.ml.surrogate import MLPSurrogateModel


ML_MIN_N = 12
MAX_ITEMS = 25


class BruteForceMLKnapsack(BaseAlgorithm):
    """ML-усечённый перебор для задачи о рюкзаке (sampling)."""

    name = "brute_force_ml"
    display_name = "Полный перебор + ML"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        items = input_data["items"]
        capacity = input_data["capacity"]
        n = len(items)

        if n > MAX_ITEMS:
            raise ValueError(
                f"Полный перебор + ML допустим для n ≤ {MAX_ITEMS}, получено n = {n}"
            )

        if n < ML_MIN_N:
            base = BruteForceKnapsack().solve(input_data, params)
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

        warmup_samples = params.get("warmup_samples", 250)
        sample_count = params.get("sample_count", 1500)
        top_k_exact = params.get("top_k_exact", 30)

        total = 1 << n
        sample_count = min(sample_count, total)

        weights = [item["weight"] for item in items]
        values = [item["value"] for item in items]

        surrogate = MLPSurrogateModel(
            hidden_layers=params.get("hidden_layers", (32, 16)),
            max_iter=params.get("max_iter", 80),
            learning_rate_init=params.get("learning_rate_init", 0.02),
        )

        start = time.perf_counter()

        # ── Фаза 1: прогрев (случайные + возмущённые жадные) ────────
        rand_count = warmup_samples // 2
        greedy_count = warmup_samples - rand_count
        warmup_pool = [
            [random.randint(0, 1) for _ in range(n)] for _ in range(rand_count)
        ]
        for _ in range(greedy_count):
            sol = _deterministic_greedy_kn(
                weights, values, capacity, mode=random.choice(["ratio", "value"])
            )
            for _ in range(random.randint(0, 3)):
                idx = random.randint(0, n - 1)
                sol[idx] ^= 1
            warmup_pool.append(sol)
        train_X = np.array(warmup_pool, dtype=float)
        train_y = np.array(
            [_exact_fitness(ind, weights, values, capacity) for ind in warmup_pool]
        )
        surrogate.fit(train_X, train_y)
        exact_evals = warmup_samples

        # Гарантированные кандидаты: детерминированный жадный по ratio и по value.
        guaranteed_solutions = [
            _deterministic_greedy_kn(weights, values, capacity, mode="ratio"),
            _deterministic_greedy_kn(weights, values, capacity, mode="value"),
        ]

        # ── Фаза 2: sampling подмножеств ───────────────────────────
        if total <= sample_count:
            sampled = np.zeros((total, n), dtype=float)
            for mask in range(total):
                for i in range(n):
                    if mask & (1 << i):
                        sampled[mask, i] = 1.0
            guaranteed_in_sample = True
        else:
            seen: set[int] = set()
            sampled_list = []
            for sol in guaranteed_solutions:
                mask = sum((1 << i) for i in range(n) if sol[i])
                if mask not in seen:
                    seen.add(mask)
                    sampled_list.append([float(b) for b in sol])
            while len(sampled_list) < sample_count:
                mask = random.randint(0, total - 1)
                if mask in seen:
                    continue
                seen.add(mask)
                row = [1.0 if mask & (1 << i) else 0.0 for i in range(n)]
                sampled_list.append(row)
            sampled = np.array(sampled_list, dtype=float)
            guaranteed_in_sample = False
        guaranteed_count = len(guaranteed_solutions)

        pred_fitness = surrogate.predict(sampled)
        surrogate_evals = len(sampled)

        top_k = min(top_k_exact, len(sampled))
        ml_top = np.argsort(-pred_fitness)[:top_k].tolist()
        if not guaranteed_in_sample:
            eval_indices = sorted(set(range(guaranteed_count)) | set(ml_top))
        else:
            eval_indices = ml_top

        best_solution = None
        best_value = -1.0
        evaluated: list[tuple[float, list[int]]] = []
        for idx in eval_indices:
            ind = sampled[idx].astype(int).tolist()
            f = _exact_fitness(ind, weights, values, capacity)
            exact_evals += 1
            evaluated.append((f, ind))
            if f > best_value:
                best_value = f
                best_solution = ind

        # Дополнительно точно оцениваем гарантированные жадные решения.
        for sol in guaranteed_solutions:
            f = _exact_fitness(sol, weights, values, capacity)
            exact_evals += 1
            evaluated.append((f, sol[:]))
            if f > best_value:
                best_value = f
                best_solution = sol[:]

        # ── Локальный поиск (bit-flip + swap) на нескольких лучших ──
        # стартах, чтобы избежать застревания в локальных максимумах.
        evaluated.sort(key=lambda t: -t[0])
        n_starts = min(5, len(evaluated))
        for start_val, start_sol in evaluated[:n_starts]:
            ls_sol, ls_val, ls_iters = _local_search_kn(
                start_sol, weights, values, capacity, start_val
            )
            exact_evals += ls_iters
            if ls_val > best_value:
                best_value = ls_val
                best_solution = ls_sol

        # R² на отдельной валидационной выборке.
        val_size = min(30, max(10, warmup_samples // 2))
        val_pool = [
            [random.randint(0, 1) for _ in range(n)] for _ in range(val_size)
        ]
        val_X = np.array(val_pool, dtype=float)
        val_y = np.array(
            [_exact_fitness(ind, weights, values, capacity) for ind in val_pool]
        )
        exact_evals += val_size
        surrogate_r2 = surrogate.score(val_X, val_y)

        elapsed = time.perf_counter() - start

        selected = [i for i in range(n) if best_solution and best_solution[i] == 1]
        total_weight = sum(weights[i] for i in selected)

        return AlgorithmResult(
            solution=selected,
            cost=float(best_value),
            execution_time=elapsed,
            iterations=len(sampled),
            convergence_history=[float(best_value)],
            extra={
                "total_weight": total_weight,
                "ml_used": True,
                "surrogate_model": "MLPRegressor",
                "exact_evaluations": exact_evals,
                "surrogate_evaluations": surrogate_evals,
                "surrogate_accuracy_r2": round(float(surrogate_r2), 4),
                "warmup_samples": warmup_samples,
                "sample_count": len(sampled),
                "top_k_exact": top_k,
                "total_candidates": total,
                "training_samples": warmup_samples,
                "optimality_guaranteed": False,
            },
        )


def _local_search_kn(
    sol: list[int],
    weights: list[float],
    values: list[float],
    capacity: float,
    cur_value: float,
) -> tuple[list[int], float, int]:
    """Локальный поиск: flip одного бита и swap двух битов."""
    n = len(sol)
    sol = sol[:]
    evals = 0
    improved = True
    while improved:
        improved = False
        # 1-flip
        for i in range(n):
            cand = sol[:]
            cand[i] ^= 1
            v = _exact_fitness(cand, weights, values, capacity)
            evals += 1
            if v > cur_value:
                sol = cand
                cur_value = v
                improved = True
        # swap (вывести один, ввести другой)
        for i in range(n):
            if sol[i] != 1:
                continue
            for j in range(n):
                if sol[j] != 0:
                    continue
                cand = sol[:]
                cand[i] = 0
                cand[j] = 1
                v = _exact_fitness(cand, weights, values, capacity)
                evals += 1
                if v > cur_value:
                    sol = cand
                    cur_value = v
                    improved = True
    return sol, cur_value, evals


def _deterministic_greedy_kn(
    weights: list[float],
    values: list[float],
    capacity: float,
    mode: str = "ratio",
) -> list[int]:
    """Детерминированный жадный по ratio или value."""
    n = len(weights)
    if mode == "value":
        order = sorted(range(n), key=lambda i: -values[i])
    else:
        order = sorted(
            range(n),
            key=lambda i: -(values[i] / weights[i] if weights[i] > 0 else 0.0),
        )
    sol = [0] * n
    total = 0.0
    for i in order:
        if total + weights[i] <= capacity:
            sol[i] = 1
            total += weights[i]
    return sol


def _exact_fitness(
    ind: list[int],
    weights: list[float],
    values: list[float],
    capacity: float,
) -> float:
    n = len(ind)
    w = sum(weights[i] * ind[i] for i in range(n))
    v = sum(values[i] * ind[i] for i in range(n))
    return v if w <= capacity else 0.0
