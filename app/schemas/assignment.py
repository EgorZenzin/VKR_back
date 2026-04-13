"""Pydantic-схемы для задачи назначения."""

from pydantic import BaseModel, Field


class AssignmentInput(BaseModel):
    """Входные данные для задачи назначения.

    cost_matrix[i][j] — стоимость назначения работника i на задание j.
    Матрица должна быть квадратной.
    """

    cost_matrix: list[list[float]] = Field(..., min_length=1)

    def model_post_init(self, __context):
        n = len(self.cost_matrix)
        for i, row in enumerate(self.cost_matrix):
            if len(row) != n:
                raise ValueError(
                    f"Матрица стоимости должна быть квадратной. "
                    f"Строка {i} имеет длину {len(row)}, ожидается {n}."
                )


class AssignmentParams(BaseModel):
    """Параметры алгоритмов для задачи назначения."""

    population_size: int = Field(default=100, ge=10)
    generations: int = Field(default=300, ge=1)
    mutation_rate: float = Field(default=0.05, ge=0.0, le=1.0)
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
