"""Реестр задач и алгоритмов.

Центральный реестр связывает названия задач с экземплярами алгоритмов.
Для добавления новой задачи или алгоритма достаточно зарегистрировать
его в словаре TASK_REGISTRY.
"""

from app.algorithms.base import BaseAlgorithm

from app.algorithms.tsp.greedy import GreedyTSP
from app.algorithms.tsp.brute_force import BruteForceTSP
from app.algorithms.tsp.genetic import GeneticTSP
from app.algorithms.tsp.simulated_annealing import SimulatedAnnealingTSP

from app.algorithms.knapsack.greedy import GreedyKnapsack
from app.algorithms.knapsack.brute_force import BruteForceKnapsack
from app.algorithms.knapsack.dynamic import DynamicKnapsack
from app.algorithms.knapsack.genetic import GeneticKnapsack

from app.algorithms.assignment.greedy import GreedyAssignment
from app.algorithms.assignment.brute_force import BruteForceAssignment
from app.algorithms.assignment.hungarian import HungarianAssignment
from app.algorithms.assignment.genetic import GeneticAssignment

from app.algorithms.tsp.genetic_ml import GeneticMLTSP
from app.algorithms.knapsack.genetic_ml import GeneticMLKnapsack
from app.algorithms.assignment.genetic_ml import GeneticMLAssignment

from app.algorithms.tsp.greedy_ml import GreedyMLTSP
from app.algorithms.knapsack.greedy_ml import GreedyMLKnapsack
from app.algorithms.assignment.greedy_ml import GreedyMLAssignment
from app.algorithms.tsp.simulated_annealing_ml import SimulatedAnnealingMLTSP
from app.algorithms.tsp.brute_force_ml import BruteForceMLTSP
from app.algorithms.knapsack.brute_force_ml import BruteForceMLKnapsack
from app.algorithms.assignment.brute_force_ml import BruteForceMLAssignment


# ── Описания задач ──────────────────────────────────────────────────

TASK_DISPLAY_NAMES: dict[str, str] = {
    "tsp": "Задача коммивояжёра",
    "knapsack": "Задача о рюкзаке",
    "assignment": "Задача назначения",
}

TASK_DESCRIPTIONS: dict[str, str] = {
    "tsp": "Найти кратчайший замкнутый маршрут, проходящий через все города ровно по одному разу.",
    "knapsack": "Выбрать подмножество предметов максимальной суммарной ценности, не превышая ёмкость рюкзака.",
    "assignment": "Распределить n работников по n заданиям с минимальной суммарной стоимостью назначений.",
}

# ── Тип оптимизации (для корректного сравнения) ─────────────────────

TASK_OPTIMIZATION: dict[str, str] = {
    "tsp": "minimize",
    "knapsack": "maximize",
    "assignment": "minimize",
}

# ── Центральный реестр: задача → {имя_алгоритма → экземпляр} ────────

TASK_REGISTRY: dict[str, dict[str, BaseAlgorithm]] = {
    "tsp": {
        "greedy": GreedyTSP(),
        "greedy_ml": GreedyMLTSP(),
        "brute_force": BruteForceTSP(),
        "brute_force_ml": BruteForceMLTSP(),
        "genetic": GeneticTSP(),
        "genetic_ml": GeneticMLTSP(),
        "simulated_annealing": SimulatedAnnealingTSP(),
        "simulated_annealing_ml": SimulatedAnnealingMLTSP(),
    },
    "knapsack": {
        "greedy": GreedyKnapsack(),
        "greedy_ml": GreedyMLKnapsack(),
        "brute_force": BruteForceKnapsack(),
        "brute_force_ml": BruteForceMLKnapsack(),
        "dynamic_programming": DynamicKnapsack(),
        "genetic": GeneticKnapsack(),
        "genetic_ml": GeneticMLKnapsack(),
    },
    "assignment": {
        "greedy": GreedyAssignment(),
        "greedy_ml": GreedyMLAssignment(),
        "brute_force": BruteForceAssignment(),
        "brute_force_ml": BruteForceMLAssignment(),
        "hungarian": HungarianAssignment(),
        "genetic": GeneticAssignment(),
        "genetic_ml": GeneticMLAssignment(),
    },
}


# ── Вспомогательные функции ─────────────────────────────────────────

def get_task_names() -> list[str]:
    """Список доступных задач."""
    return list(TASK_REGISTRY.keys())


def get_algorithms_for_task(task_name: str) -> dict[str, BaseAlgorithm]:
    """Словарь алгоритмов для задачи (KeyError если задача не найдена)."""
    if task_name not in TASK_REGISTRY:
        raise KeyError(f"Задача '{task_name}' не найдена. Доступны: {get_task_names()}")
    return TASK_REGISTRY[task_name]


def get_algorithm(task_name: str, algorithm_name: str) -> BaseAlgorithm:
    """Получить конкретный алгоритм для задачи."""
    algorithms = get_algorithms_for_task(task_name)
    if algorithm_name not in algorithms:
        available = list(algorithms.keys())
        raise KeyError(
            f"Алгоритм '{algorithm_name}' не найден для задачи '{task_name}'. "
            f"Доступны: {available}"
        )
    return algorithms[algorithm_name]
