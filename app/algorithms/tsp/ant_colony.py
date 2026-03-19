import numpy as np
import random
import time

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class AntColonyTSP(BaseAlgorithm):
    """Муравьиный алгоритм (ACO) для TSP."""

    name = "ant_colony"
    display_name = "Муравьиный алгоритм"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        n_ants = params.get("n_ants", 30)
        n_iterations = params.get("n_iterations", 200)
        alpha = params.get("alpha", 1.0)
        beta = params.get("beta", 2.0)
        evaporation = params.get("evaporation_rate", 0.5)

        dist_matrix = self._get_distance_matrix(input_data)
        n = len(dist_matrix)

        start = time.perf_counter()

        pheromone = np.ones((n, n))
        heuristic = np.where(dist_matrix > 0, 1.0 / dist_matrix, 0)

        best_route = None
        best_cost = float("inf")
        convergence = []

        for iteration in range(n_iterations):
            all_routes = []
            all_costs = []

            for _ in range(n_ants):
                route = self._construct_route(n, pheromone, heuristic, alpha, beta)
                cost = self._route_cost(route, dist_matrix)
                all_routes.append(route)
                all_costs.append(cost)

                if cost < best_cost:
                    best_cost = cost
                    best_route = route[:]

            convergence.append(best_cost)

            # Обновление феромонов
            pheromone *= (1 - evaporation)
            for route, cost in zip(all_routes, all_costs):
                deposit = 1.0 / cost
                for i in range(n - 1):
                    pheromone[route[i]][route[i + 1]] += deposit
                    pheromone[route[i + 1]][route[i]] += deposit
                pheromone[route[-1]][route[0]] += deposit
                pheromone[route[0]][route[-1]] += deposit

        elapsed = time.perf_counter() - start

        return AlgorithmResult(
            solution=best_route,
            cost=best_cost,
            execution_time=elapsed,
            convergence_history=convergence,
        )

    def _construct_route(
        self, n: int, pheromone: np.ndarray, heuristic: np.ndarray,
        alpha: float, beta: float
    ) -> list[int]:
        start_city = random.randint(0, n - 1)
        route = [start_city]
        visited = {start_city}

        for _ in range(n - 1):
            current = route[-1]
            probabilities = []
            candidates = []

            for j in range(n):
                if j not in visited:
                    prob = (pheromone[current][j] ** alpha) * (heuristic[current][j] ** beta)
                    probabilities.append(prob)
                    candidates.append(j)

            total = sum(probabilities)
            if total == 0:
                next_city = random.choice(candidates)
            else:
                probabilities = [p / total for p in probabilities]
                next_city = random.choices(candidates, weights=probabilities, k=1)[0]

            route.append(next_city)
            visited.add(next_city)

        return route

    def _route_cost(self, route: list[int], dist_matrix: np.ndarray) -> float:
        cost = sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
        cost += dist_matrix[route[-1]][route[0]]
        return cost

    def _get_distance_matrix(self, input_data: dict) -> np.ndarray:
        if "distance_matrix" in input_data and input_data["distance_matrix"]:
            return np.array(input_data["distance_matrix"])
        cities = input_data["cities"]
        coords = np.array([[c["x"], c["y"]] for c in cities])
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))
