import random
import time

from app.algorithms.base import BaseAlgorithm, AlgorithmResult


class GeneticGraphColoring(BaseAlgorithm):
    """Генетический алгоритм для раскраски графа."""

    name = "genetic"
    display_name = "Генетический алгоритм"

    def solve(self, input_data: dict, params: dict | None = None) -> AlgorithmResult:
        params = params or {}
        pop_size = params.get("population_size", 100)
        generations = params.get("generations", 300)
        mutation_rate = params.get("mutation_rate", 0.05)

        n = input_data["n_vertices"]
        edges = input_data["edges"]
        edge_list = [(e["u"], e["v"]) for e in edges]

        start = time.perf_counter()

        max_colors = n
        population = [
            [random.randint(0, max_colors - 1) for _ in range(n)]
            for _ in range(pop_size)
        ]

        convergence = []
        best_solution = None
        best_fitness = float("inf")

        for gen in range(generations):
            fitness = []
            for ind in population:
                conflicts = sum(1 for u, v in edge_list if ind[u] == ind[v])
                n_colors = len(set(ind))
                # Минимизируем: конфликты * 1000 + кол-во цветов
                f = conflicts * 1000 + n_colors
                fitness.append(f)

                if f < best_fitness:
                    best_fitness = f
                    best_solution = ind[:]

            convergence.append(best_fitness)

            # Селекция
            new_pop = []
            for _ in range(pop_size):
                t1, t2 = random.sample(range(pop_size), 2)
                winner = population[t1] if fitness[t1] < fitness[t2] else population[t2]
                new_pop.append(winner[:])

            # Кроссовер (uniform)
            for i in range(0, pop_size - 1, 2):
                if random.random() < 0.8:
                    for j in range(n):
                        if random.random() < 0.5:
                            new_pop[i][j], new_pop[i + 1][j] = new_pop[i + 1][j], new_pop[i][j]

            # Мутация
            for ind in new_pop:
                for j in range(n):
                    if random.random() < mutation_rate:
                        ind[j] = random.randint(0, max_colors - 1)

            population = new_pop

        # Перенумеровка цветов
        if best_solution:
            unique_colors = sorted(set(best_solution))
            remap = {c: i for i, c in enumerate(unique_colors)}
            best_solution = [remap[c] for c in best_solution]

        n_colors = len(set(best_solution)) if best_solution else 0
        elapsed = time.perf_counter() - start

        return AlgorithmResult(
            solution={str(i): best_solution[i] for i in range(n)},
            cost=float(n_colors),
            execution_time=elapsed,
            convergence_history=convergence,
        )
