import random
import math
import time

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class SimulatedAnnealingGraphColoring(BaseAlgorithm):
    """Имитация отжига для раскраски графа."""

    name = "simulated_annealing"
    display_name = "Имитация отжига"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        initial_temp = params.get("initial_temp", 1000.0)
        cooling_rate = params.get("cooling_rate", 0.999)
        min_temp = params.get("min_temp", 1e-6)

        n = input_data["n_vertices"]
        edges = input_data["edges"]
        edge_list = [(e["u"], e["v"]) for e in edges]

        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)

        start = time.perf_counter()

        # Начальная раскраска — жадная
        current = [-1] * n
        for v in range(n):
            neighbor_colors = {current[u] for u in adj[v] if current[u] != -1}
            c = 0
            while c in neighbor_colors:
                c += 1
            current[v] = c

        max_color = max(current)
        current_cost = self._evaluate(current, edge_list)
        best = current[:]
        best_cost = current_cost
        temp = initial_temp
        convergence = []
        iteration = 0

        while temp > min_temp:
            neighbor = current[:]
            v = random.randint(0, n - 1)
            neighbor[v] = random.randint(0, max_color)
            neighbor_cost = self._evaluate(neighbor, edge_list)

            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current = neighbor
                current_cost = neighbor_cost

            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

            temp *= cooling_rate
            iteration += 1
            if iteration % 100 == 0:
                convergence.append(best_cost)

        # Перенумеровка
        unique = sorted(set(best))
        remap = {c: i for i, c in enumerate(unique)}
        best = [remap[c] for c in best]
        n_colors = len(unique)

        elapsed = time.perf_counter() - start

        return AlgorithmResult(
            solution={str(i): best[i] for i in range(n)},
            cost=float(n_colors),
            execution_time=elapsed,
            convergence_history=convergence,
        )

    def _evaluate(self, coloring: list[int], edges: list[tuple]) -> float:
        conflicts = sum(1 for u, v in edges if coloring[u] == coloring[v])
        n_colors = len(set(coloring))
        return conflicts * 1000 + n_colors
