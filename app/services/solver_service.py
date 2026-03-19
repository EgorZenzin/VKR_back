from app.algorithms.base import BaseAlgorithm, AlgorithmResult

from app.algorithms.tsp.nearest_neighbor import NearestNeighborTSP
from app.algorithms.tsp.genetic import GeneticTSP
from app.algorithms.tsp.ant_colony import AntColonyTSP
from app.algorithms.tsp.simulated_annealing import SimulatedAnnealingTSP

from app.algorithms.assignment.hungarian import HungarianAssignment
from app.algorithms.assignment.greedy import GreedyAssignment
from app.algorithms.assignment.genetic import GeneticAssignment

from app.algorithms.knapsack.dynamic import DynamicKnapsack
from app.algorithms.knapsack.greedy import GreedyKnapsack
from app.algorithms.knapsack.genetic import GeneticKnapsack

from app.algorithms.graph_coloring.greedy import GreedyGraphColoring
from app.algorithms.graph_coloring.genetic import GeneticGraphColoring
from app.algorithms.graph_coloring.simulated_annealing import SimulatedAnnealingGraphColoring

from app.algorithms.max_flow.ford_fulkerson import FordFulkersonMaxFlow
from app.algorithms.max_flow.edmonds_karp import EdmondsKarpMaxFlow
from app.algorithms.max_flow.dinic import DinicMaxFlow


PROBLEM_ALGORITHMS: dict[str, dict[str, BaseAlgorithm]] = {
    "tsp": {
        "nearest_neighbor": NearestNeighborTSP(),
        "genetic": GeneticTSP(),
        "ant_colony": AntColonyTSP(),
        "simulated_annealing": SimulatedAnnealingTSP(),
    },
    "assignment": {
        "hungarian": HungarianAssignment(),
        "greedy": GreedyAssignment(),
        "genetic": GeneticAssignment(),
    },
    "knapsack": {
        "dynamic_programming": DynamicKnapsack(),
        "greedy": GreedyKnapsack(),
        "genetic": GeneticKnapsack(),
    },
    "graph_coloring": {
        "greedy": GreedyGraphColoring(),
        "genetic": GeneticGraphColoring(),
        "simulated_annealing": SimulatedAnnealingGraphColoring(),
    },
    "max_flow": {
        "ford_fulkerson": FordFulkersonMaxFlow(),
        "edmonds_karp": EdmondsKarpMaxFlow(),
        "dinic": DinicMaxFlow(),
    },
}

PROBLEM_DISPLAY_NAMES = {
    "tsp": "Задача коммивояжёра (TSP)",
    "assignment": "Задача о назначениях",
    "knapsack": "Задача о рюкзаке",
    "graph_coloring": "Раскраска графа",
    "max_flow": "Максимальный поток",
}


def get_problems_info() -> list[dict]:
    result = []
    for problem_key, algorithms in PROBLEM_ALGORITHMS.items():
        algo_list = []
        for algo_key, algo_instance in algorithms.items():
            algo_list.append({
                "key": algo_key,
                "name": algo_instance.display_name,
            })
        result.append({
            "key": problem_key,
            "name": PROBLEM_DISPLAY_NAMES.get(problem_key, problem_key),
            "algorithms": algo_list,
        })
    return result


def solve_problem(
    problem_type: str, algorithm: str, input_data: dict, params: dict | None = None
) -> AlgorithmResult:
    if problem_type not in PROBLEM_ALGORITHMS:
        raise ValueError(f"Неизвестная задача: {problem_type}")
    algorithms = PROBLEM_ALGORITHMS[problem_type]
    if algorithm not in algorithms:
        raise ValueError(
            f"Неизвестный алгоритм '{algorithm}' для задачи '{problem_type}'. "
            f"Доступные: {list(algorithms.keys())}"
        )
    return algorithms[algorithm].solve(input_data, params)
