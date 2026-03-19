from pydantic import BaseModel


class AssignmentInput(BaseModel):
    cost_matrix: list[list[float]]


class AssignmentParams(BaseModel):
    # Генетический алгоритм
    population_size: int = 100
    generations: int = 300
    mutation_rate: float = 0.05


class AssignmentResult(BaseModel):
    assignments: list[tuple[int, int]]
    total_cost: float
    execution_time: float
    convergence_history: list[float] | None = None
