"""Pydantic-схемы для задачи коммивояжёра."""

from pydantic import BaseModel, Field


class City(BaseModel):
    """Город с координатами."""

    x: float
    y: float
    name: str | None = None


class TSPInput(BaseModel):
    """Входные данные для TSP.

    Можно передать либо список городов (координаты), либо готовую матрицу расстояний.
    """

    cities: list[City] | None = None
    distance_matrix: list[list[float]] | None = None

    def model_post_init(self, __context):
        if not self.cities and not self.distance_matrix:
            raise ValueError("Нужно передать cities или distance_matrix")


class TSPParams(BaseModel):
    """Параметры алгоритмов для TSP."""

    # Генетический алгоритм
    population_size: int = Field(default=100, ge=10)
    generations: int = Field(default=500, ge=1)
    mutation_rate: float = Field(default=0.02, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.8, ge=0.0, le=1.0)

    # Имитация отжига
    initial_temp: float = Field(default=10000.0, gt=0)
    cooling_rate: float = Field(default=0.9995, gt=0, lt=1)
    min_temp: float = Field(default=1e-8, gt=0)
from pydantic import BaseModel


class City(BaseModel):
    x: float
    y: float
    name: str | None = None


class TSPInput(BaseModel):
    cities: list[City]
    distance_matrix: list[list[float]] | None = None


class TSPParams(BaseModel):
    # Генетический алгоритм
    population_size: int = 100
    generations: int = 500
    mutation_rate: float = 0.02
    crossover_rate: float = 0.8
    # Муравьиный алгоритм
    n_ants: int = 30
    n_iterations: int = 200
    alpha: float = 1.0
    beta: float = 2.0
    evaporation_rate: float = 0.5
    # Имитация отжига
    initial_temp: float = 10000.0
    cooling_rate: float = 0.9995
    min_temp: float = 1e-8


class TSPResult(BaseModel):
    route: list[int]
    cost: float
    execution_time: float
    convergence_history: list[float] | None = None
