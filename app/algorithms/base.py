from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import time


@dataclass
class AlgorithmResult:
    solution: Any
    cost: float
    execution_time: float
    convergence_history: list[float] | None = None
    extra: dict | None = None


class BaseAlgorithm(ABC):
    @abstractmethod
    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        pass

    def _timed_solve(self, func, *args, **kwargs) -> tuple:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
