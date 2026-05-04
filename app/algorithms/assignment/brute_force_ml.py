"""ML-усечённый «полный перебор» для задачи назначения.

Стратегия:
1. Если n маленькое — fallback к точному brute_force.
2. Иначе: sampling n! перестановок вместо полного перебора;
   суррогат отбирает top-K, точная переоценка только top-K.
"""

import time
import math
import random
import itertools
import numpy as np

from app.algorithms.base import BaseAlgorithm, AlgorithmResult
from app.algorithms.assignment.brute_force import BruteForceAssignment
from app.ml.surrogate import MLPSurrogateModel


ML_MIN_N = 7
MAX_SIZE = 11


class BruteForceMLAssignment(BaseAlgorithm):
    """ML-усечённый перебор для задачи назначения (sampling)."""

    name = "brute_force_ml"
    display_name = "Полный перебор + ML"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        cost_matrix = np.array(input_data["cost_matrix"])
        n = len(cost_matrix)

        if n > MAX_SIZE:
            raise ValueError(
                f"Полный перебор + ML допустим для n ≤ {MAX_SIZE}, получено n = {n}"
            )

        if n < ML_MIN_N:
            base = BruteForceAssignment().solve(input_data, params)
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

        warmup_samples = params.get("warmup_samples", 200)
        sample_count = params.get("sample_count", 1000)
        top_k_exact = params.get("top_k_exact", 30)

        total_perms = math.factorial(n)
        sample_count = min(sample_count, total_perms)

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
            list(np.random.permutation(n)) for _ in range(rand_count)
        ]
        base_greedy = _deterministic_greedy_assignment(cost_matrix)
        for _ in range(greedy_count):
            p = base_greedy[:]
            # 1-3 случайных swap-возмущений
            for _ in range(random.randint(1, 3)):
                i, j = random.sample(range(n), 2)
                p[i], p[j] = p[j], p[i]
            warmup_pool.append(p)
        train_X = np.array([_encode_perm(p, n) for p in warmup_pool])
        train_y = np.array([_perm_cost(p, cost_matrix) for p in warmup_pool])
        surrogate.fit(train_X, train_y)
        exact_evals = warmup_samples

        # Гарантированный кандидат: детерминированное жадное назначение.
        guaranteed_perms = [_deterministic_greedy_assignment(cost_matrix)]

        if total_perms <= sample_count:
            sampled = [list(p) for p in itertools.permutations(range(n))]
            guaranteed_in_sample = True
        else:
            seen: set[tuple[int, ...]] = set()
            sampled = []
            for p in guaranteed_perms:
                key = tuple(p)
                if key not in seen:
                    seen.add(key)
                    sampled.append(p)
            while len(sampled) < sample_count:
                p = list(np.random.permutation(n))
                key = tuple(p)
                if key in seen:
                    continue
                seen.add(key)
                sampled.append(p)
            guaranteed_in_sample = False
        guaranteed_count = len(guaranteed_perms)

        X_all = np.array([_encode_perm(p, n) for p in sampled])
        pred_costs = surrogate.predict(X_all)
        surrogate_evals = len(sampled)

        top_k = min(top_k_exact, len(sampled))
        ml_top = np.argsort(pred_costs)[:top_k].tolist()
        if not guaranteed_in_sample:
            eval_indices = sorted(set(range(guaranteed_count)) | set(ml_top))
        else:
            eval_indices = ml_top

        best_perm = None
        best_cost = float("inf")
        evaluated: list[tuple[float, list[int]]] = []
        for idx in eval_indices:
            c = _perm_cost(sampled[idx], cost_matrix)
            exact_evals += 1
            evaluated.append((c, sampled[idx][:]))
            if c < best_cost:
                best_cost = c
                best_perm = sampled[idx][:]

        # Дополнительно оцениваем гарантированные перестановки.
        for p in guaranteed_perms:
            c = _perm_cost(p, cost_matrix)
            exact_evals += 1
            evaluated.append((c, p[:]))
            if c < best_cost:
                best_cost = c
                best_perm = p[:]

        # ── Локальный поиск swap: запускаем на нескольких лучших ──
        # стартовых точках, чтобы избежать застревания в локальном минимуме.
        evaluated.sort(key=lambda t: t[0])
        n_starts = min(5, len(evaluated))
        for start_cost, start_perm in evaluated[:n_starts]:
            ls_perm, ls_cost, ls_iters = _local_search_assignment(
                start_perm, cost_matrix, start_cost
            )
            exact_evals += ls_iters
            if ls_cost < best_cost:
                best_cost = ls_cost
                best_perm = ls_perm

        # R² на отдельной валидационной выборке.
        val_size = min(30, max(10, warmup_samples // 2))
        val_perms = [list(np.random.permutation(n)) for _ in range(val_size)]
        val_X = np.array([_encode_perm(p, n) for p in val_perms])
        val_y = np.array([_perm_cost(p, cost_matrix) for p in val_perms])
        exact_evals += val_size
        surrogate_r2 = surrogate.score(val_X, val_y)

        elapsed = time.perf_counter() - start

        assignments = [[i, best_perm[i]] for i in range(n)] if best_perm else []

        return AlgorithmResult(
            solution=assignments,
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


def _local_search_assignment(
    perm: list[int], cost_matrix: np.ndarray, cur_cost: float,
) -> tuple[list[int], float, int]:
    """Локальный поиск: попарный swap столбцов."""
    n = len(perm)
    perm = perm[:]
    evals = 0
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                delta = (
                    cost_matrix[i][perm[j]] + cost_matrix[j][perm[i]]
                    - cost_matrix[i][perm[i]] - cost_matrix[j][perm[j]]
                )
                evals += 1
                if delta < -1e-12:
                    perm[i], perm[j] = perm[j], perm[i]
                    cur_cost += float(delta)
                    improved = True
    return perm, cur_cost, evals


def _deterministic_greedy_assignment(cost_matrix: np.ndarray) -> list[int]:
    """Жадное назначение: всегда берём самую дешёвую доступную ячейку."""
    n = len(cost_matrix)
    cells = sorted(
        ((float(cost_matrix[i][j]), i, j) for i in range(n) for j in range(n))
    )
    perm = [-1] * n
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for _, i, j in cells:
        if i in used_rows or j in used_cols:
            continue
        perm[i] = j
        used_rows.add(i)
        used_cols.add(j)
        if len(used_rows) == n:
            break
    return perm


def _perm_cost(perm: list[int], cost_matrix: np.ndarray) -> float:
    return float(sum(cost_matrix[i][perm[i]] for i in range(len(perm))))


def _encode_perm(perm: list[int], n: int) -> list[float]:
    encoding = [0.0] * n
    for pos, task in enumerate(perm):
        encoding[task] = pos / max(n - 1, 1)
    return encoding
