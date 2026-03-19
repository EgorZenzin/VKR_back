import random
import time

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class GeneticKnapsack(BaseAlgorithm):
    """Генетический алгоритм для задачи о рюкзаке."""

    name = "genetic"
    display_name = "Генетический алгоритм"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        pop_size = params.get("population_size", 100)
        generations = params.get("generations", 300)
        mutation_rate = params.get("mutation_rate", 0.05)

        items = input_data["items"]
        capacity = input_data["capacity"]
        n = len(items)

        weights = [item["weight"] for item in items]
        values = [item["value"] for item in items]

        start = time.perf_counter()

        population = [[random.randint(0, 1) for _ in range(n)] for _ in range(pop_size)]
        convergence = []
        best_solution = None
        best_value = -1

        for gen in range(generations):
            fitness = []
            for ind in population:
                w = sum(weights[i] * ind[i] for i in range(n))
                v = sum(values[i] * ind[i] for i in range(n))
                fitness.append(v if w <= capacity else 0)

            for i, f in enumerate(fitness):
                if f > best_value:
                    best_value = f
                    best_solution = population[i][:]

            convergence.append(best_value)

            # Турнирная селекция
            new_pop = []
            for _ in range(pop_size):
                t1, t2 = random.sample(range(pop_size), 2)
                winner = population[t1] if fitness[t1] >= fitness[t2] else population[t2]
                new_pop.append(winner[:])

            # Одноточечный кроссовер
            for i in range(0, pop_size - 1, 2):
                if random.random() < 0.8:
                    pt = random.randint(1, n - 1)
                    c1 = new_pop[i][:pt] + new_pop[i + 1][pt:]
                    c2 = new_pop[i + 1][:pt] + new_pop[i][pt:]
                    new_pop[i] = c1
                    new_pop[i + 1] = c2

            # Мутация
            for ind in new_pop:
                for j in range(n):
                    if random.random() < mutation_rate:
                        ind[j] = 1 - ind[j]

            population = new_pop

        elapsed = time.perf_counter() - start

        selected = [i for i in range(n) if best_solution[i] == 1]
        total_weight = sum(weights[i] for i in selected)

        return AlgorithmResult(
            solution=selected,
            cost=best_value,
            execution_time=elapsed,
            convergence_history=convergence,
            extra={"total_weight": total_weight},
        )
