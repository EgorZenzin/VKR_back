import time
from collections import defaultdict, deque

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class DinicMaxFlow(BaseAlgorithm):
    """Алгоритм Диница для максимального потока."""

    name = "dinic"
    display_name = "Алгоритм Диница"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        n = input_data["n_vertices"]
        edges_input = input_data["edges"]
        source = input_data["source"]
        sink = input_data["sink"]

        start = time.perf_counter()

        # Построение графа в формате списка рёбер
        graph = [[] for _ in range(n)]
        edge_list = []

        def add_edge(u, v, cap):
            graph[u].append(len(edge_list))
            edge_list.append([v, cap, 0])
            graph[v].append(len(edge_list))
            edge_list.append([u, 0, 0])

        original_edge_indices = []
        for e in edges_input:
            original_edge_indices.append(len(edge_list))
            add_edge(e["u"], e["v"], e["capacity"])

        max_flow = 0.0

        while True:
            level = self._bfs_level(graph, edge_list, source, sink, n)
            if level[sink] < 0:
                break
            iter_ptr = [0] * n
            while True:
                f = self._dfs_blocking(graph, edge_list, source, sink, float("inf"), level, iter_ptr)
                if f == 0:
                    break
                max_flow += f

        flow_edges = []
        for i, e in enumerate(edges_input):
            idx = original_edge_indices[i]
            flow = edge_list[idx][2]
            flow_edges.append({
                "u": e["u"], "v": e["v"],
                "capacity": e["capacity"], "flow": flow,
            })

        elapsed = time.perf_counter() - start

        return AlgorithmResult(
            solution=flow_edges,
            cost=float(max_flow),
            execution_time=elapsed,
        )

    def _bfs_level(self, graph, edge_list, source, sink, n):
        level = [-1] * n
        level[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for eid in graph[u]:
                v, cap, flow = edge_list[eid]
                if level[v] < 0 and cap - flow > 0:
                    level[v] = level[u] + 1
                    queue.append(v)
        return level

    def _dfs_blocking(self, graph, edge_list, u, sink, pushed, level, iter_ptr):
        if u == sink:
            return pushed
        while iter_ptr[u] < len(graph[u]):
            eid = graph[u][iter_ptr[u]]
            v, cap, flow = edge_list[eid]
            if level[v] == level[u] + 1 and cap - flow > 0:
                d = self._dfs_blocking(graph, edge_list, v, sink, min(pushed, cap - flow), level, iter_ptr)
                if d > 0:
                    edge_list[eid][2] += d
                    edge_list[eid ^ 1][2] -= d
                    return d
            iter_ptr[u] += 1
        return 0
