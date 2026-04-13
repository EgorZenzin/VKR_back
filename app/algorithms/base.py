"""Базовый класс для всех алгоритмов комбинаторной оптимизации."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class AlgorithmResult:
    """Унифицированный результат работы алгоритма.

    Все алгоритмы возвращают экземпляр этого класса, что обеспечивает
    единообразие при сравнении и визуализации.
    """

    solution: Any                                       # Найденное решение
    cost: float                                         # Значение целевой функции
    execution_time: float                               # Время выполнения (секунды)
    iterations: int | None = None                       # Количество итераций (None для неитерационных)
    convergence_history: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)           # Дополнительные данные задачи


class BaseAlgorithm(ABC):
    """Абстрактный базовый класс алгоритма.

    Каждый алгоритм обязан реализовать метод solve() и задать атрибуты
    name (латиница) и display_name (русский).
    """

    name: str = ""
    display_name: str = ""

    @abstractmethod
    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        """Решить задачу и вернуть AlgorithmResult."""
        pass
