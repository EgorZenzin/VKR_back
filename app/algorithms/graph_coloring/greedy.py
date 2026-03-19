import time

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class GreedyGraphColoring(BaseAlgorithm):
    """Жадный алгоритм для раскраски графа."""

    name = "greedy"
    display_name = "Жадный алгоритм"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        n = input_data["n_vertices"]
        edges = input_data["edges"]

        start = time.perf_counter()

        adj = [set() for _ in range(n)]
        for e in edges:
            u, v = e["u"], e["v"]
            adj[u].add(v)
            adj[v].add(u)

        # Сортировка по убыванию степени
        order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)

        coloring = [-1] * n
        for v in order:
            neighbor_colors = {coloring[u] for u in adj[v] if coloring[u] != -1}
            color = 0
            while color in neighbor_colors:
                color += 1
            coloring[v] = color

        n_colors = max(coloring) + 1 if coloring else 0
        elapsed = time.perf_counter() - start

        return AlgorithmResult(
            solution={str(i): coloring[i] for i in range(n)},
            cost=float(n_colors),
            execution_time=elapsed,
        )
