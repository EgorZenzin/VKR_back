"""Pydantic-схемы для задачи о рюкзаке."""

from pydantic import BaseModel, Field


class Item(BaseModel):
    """Предмет для рюкзака."""

    weight: float = Field(..., gt=0)
    value: float = Field(..., ge=0)
    name: str | None = None


class KnapsackInput(BaseModel):
    """Входные данные для задачи о рюкзаке."""

    items: list[Item] = Field(..., min_length=1)
    capacity: float = Field(..., gt=0)


class KnapsackParams(BaseModel):
    """Параметры алгоритмов для задачи о рюкзаке."""

    population_size: int = Field(default=100, ge=10)
    generations: int = Field(default=300, ge=1)
    mutation_rate: float = Field(default=0.05, ge=0.0, le=1.0)


class KnapsackResult(BaseModel):
    """Результат решения задачи о рюкзаке."""

    selected_items: list[int]
    total_value: float
    total_weight: float
    execution_time: float
    convergence_history: list[float] | None = None
