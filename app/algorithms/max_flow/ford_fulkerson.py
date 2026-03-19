import time
from collections import defaultdict

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class FordFulkersonMaxFlow(BaseAlgorithm):
    """Алгоритм Форда-Фалкерсона (DFS) для максимального потока."""

    name = "ford_fulkerson"
    display_name = "Форда-Фалкерсона"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        n = input_data["n_vertices"]
        edges = input_data["edges"]
        source = input_data["source"]
        sink = input_data["sink"]

        start = time.perf_counter()

        graph = defaultdict(lambda: defaultdict(float))
        for e in edges:
            graph[e["u"]][e["v"]] += e["capacity"]

        max_flow = 0.0
        while True:
            visited = set()
            path_flow = self._dfs(graph, source, sink, float("inf"), visited)
            if path_flow == 0:
                break
            max_flow += path_flow

        # Вычисление потоков по рёбрам
        original_cap = defaultdict(lambda: defaultdict(float))
        for e in edges:
            original_cap[e["u"]][e["v"]] += e["capacity"]

        flow_edges = []
        for e in edges:
            u, v, cap = e["u"], e["v"], e["capacity"]
            flow = original_cap[u][v] - graph[u][v]
            if flow < 0:
                flow = 0
            flow_edges.append({"u": u, "v": v, "capacity": cap, "flow": flow})

        elapsed = time.perf_counter() - start

        return AlgorithmResult(
            solution=flow_edges,
            cost=float(max_flow),
            execution_time=elapsed,
        )

    def _dfs(self, graph, u, sink, flow, visited):
        if u == sink:
            return flow
        visited.add(u)
        for v in list(graph[u].keys()):
            if v not in visited and graph[u][v] > 0:
                min_flow = min(flow, graph[u][v])
                result = self._dfs(graph, v, sink, min_flow, visited)
                if result > 0:
                    graph[u][v] -= result
                    graph[v][u] += result
                    return result
        return 0
