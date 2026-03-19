import time
from collections import defaultdict, deque

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class EdmondsKarpMaxFlow(BaseAlgorithm):
    """Алгоритм Эдмондса-Карпа (BFS) для максимального потока."""

    name = "edmonds_karp"
    display_name = "Эдмондса-Карпа"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        n = input_data["n_vertices"]
        edges = input_data["edges"]
        source = input_data["source"]
        sink = input_data["sink"]

        start = time.perf_counter()

        graph = defaultdict(lambda: defaultdict(float))
        original_cap = defaultdict(lambda: defaultdict(float))
        for e in edges:
            graph[e["u"]][e["v"]] += e["capacity"]
            original_cap[e["u"]][e["v"]] += e["capacity"]

        adj = defaultdict(set)
        for e in edges:
            adj[e["u"]].add(e["v"])
            adj[e["v"]].add(e["u"])

        max_flow = 0.0

        while True:
            parent = self._bfs(graph, adj, source, sink, n)
            if parent is None:
                break

            # Найти мин. пропускную способность на пути
            path_flow = float("inf")
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, graph[u][v])
                v = u

            # Обновить остаточный граф
            v = sink
            while v != source:
                u = parent[v]
                graph[u][v] -= path_flow
                graph[v][u] += path_flow
                adj[u].add(v)
                adj[v].add(u)
                v = u

            max_flow += path_flow

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

    def _bfs(self, graph, adj, source, sink, n):
        visited = {source}
        queue = deque([source])
        parent = {}

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in visited and graph[u][v] > 0:
                    visited.add(v)
                    parent[v] = u
                    if v == sink:
                        return parent
                    queue.append(v)

        return None
