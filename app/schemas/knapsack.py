from pydantic import BaseModel


class Item(BaseModel):
    weight: float
    value: float
    name: str | None = None


class KnapsackInput(BaseModel):
    items: list[Item]
    capacity: float


class KnapsackParams(BaseModel):
    # Генетический алгоритм
    population_size: int = 100
    generations: int = 300
    mutation_rate: float = 0.05


class KnapsackResult(BaseModel):
    selected_items: list[int]
    total_value: float
    total_weight: float
    execution_time: float
    convergence_history: list[float] | None = None
