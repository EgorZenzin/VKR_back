from pydantic import BaseModel


class Edge(BaseModel):
    u: int
    v: int


class GraphColoringInput(BaseModel):
    n_vertices: int
    edges: list[Edge]


class GraphColoringParams(BaseModel):
    # Генетический алгоритм
    population_size: int = 100
    generations: int = 300
    mutation_rate: float = 0.05
    # Имитация отжига
    initial_temp: float = 1000.0
    cooling_rate: float = 0.999
    min_temp: float = 1e-6


class GraphColoringResult(BaseModel):
    coloring: dict[str, int]
    n_colors: int
    execution_time: float
    convergence_history: list[float] | None = None
