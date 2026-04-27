"""Бенчмарк ML-версий vs базовых алгоритмов после оптимизации."""
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np

random.seed(42)
np.random.seed(42)

from app.algorithms.registry import get_algorithm


def bench(task, algo, inp, params=None, runs=3):
    times, costs, r2s = [], [], []
    fallback = None
    for _ in range(runs):
        r = get_algorithm(task, algo).solve(inp, params)
        times.append(r.execution_time)
        costs.append(r.cost)
        if r.extra:
            if r.extra.get("ml_used") is False:
                fallback = r.extra.get("fallback_to")
            elif "surrogate_accuracy_r2" in r.extra:
                r2s.append(r.extra["surrogate_accuracy_r2"])
    avg_r2 = sum(r2s) / len(r2s) if r2s else None
    return sum(times) / runs, sum(costs) / runs, fallback, avg_r2


def fmt_row(name, t, c, fb, r2, value_label):
    tag = f" [fallback->{fb}]" if fb else ""
    r2_str = f" R2={r2:+.3f}" if r2 is not None else ""
    return f"  {name:30s} time={t * 1000:8.2f}ms {value_label}={c:.2f}{tag}{r2_str}"


tsp_algos_full = ["greedy", "greedy_ml", "brute_force", "brute_force_ml",
                  "simulated_annealing", "simulated_annealing_ml",
                  "genetic", "genetic_ml"]

for n in [7, 10, 14]:
    cities = [{"x": random.uniform(0, 10), "y": random.uniform(0, 10)} for _ in range(n)]
    inp = {"cities": cities}
    print(f"=== tsp n={n} ===")
    algos = tsp_algos_full if n <= 11 else [a for a in tsp_algos_full if "brute_force" not in a]
    for a in algos:
        try:
            t, c, fb, r2 = bench("tsp", a, inp, runs=2)
            print(fmt_row(a, t, c, fb, r2, "cost"))
        except Exception as e:
            print(f"  {a:30s} ERR: {e}")

print()

for n in [10, 15, 20]:
    items = [{"weight": random.uniform(1, 10), "value": random.uniform(1, 20)} for _ in range(n)]
    inp = {"items": items, "capacity": n * 3.0}
    print(f"=== knapsack n={n} ===")
    algos = ["greedy", "greedy_ml", "brute_force", "brute_force_ml",
             "dynamic_programming", "genetic", "genetic_ml"]
    for a in algos:
        try:
            t, c, fb, r2 = bench("knapsack", a, inp, runs=2)
            print(fmt_row(a, t, c, fb, r2, "value"))
        except Exception as e:
            print(f"  {a:30s} ERR: {e}")

print()

for n in [5, 8, 10]:
    cm = [[random.uniform(1, 10) for _ in range(n)] for _ in range(n)]
    inp = {"cost_matrix": cm}
    print(f"=== assignment n={n} ===")
    algos = ["greedy", "greedy_ml", "brute_force", "brute_force_ml",
             "hungarian", "genetic", "genetic_ml"]
    if n > 9:
        algos = [a for a in algos if a != "brute_force"]
    for a in algos:
        try:
            t, c, fb, r2 = bench("assignment", a, inp, runs=2)
            print(fmt_row(a, t, c, fb, r2, "cost"))
        except Exception as e:
            print(f"  {a:30s} ERR: {e}")
