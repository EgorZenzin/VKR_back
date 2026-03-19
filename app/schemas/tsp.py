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
