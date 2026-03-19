import numpy as np

from app.services.solver_service import solve_problem


def _convert_numpy(obj):
    """Конвертирует numpy-типы в стандартные Python-типы для JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(i) for i in obj]
    return obj


def compare_algorithms(
    problem_type: str, algorithms: list[str], input_data: dict, params: dict | None = None
) -> list[dict]:
    results = []
    for algo_name in algorithms:
        try:
            result = solve_problem(problem_type, algo_name, input_data, params)
            results.append(_convert_numpy({
                "algorithm": algo_name,
                "cost": result.cost,
                "execution_time": result.execution_time,
                "solution": result.solution,
                "convergence_history": result.convergence_history,
                "extra": result.extra,
                "error": None,
            }))
        except Exception as e:
            results.append({
                "algorithm": algo_name,
                "cost": None,
                "execution_time": None,
                "solution": None,
                "convergence_history": None,
                "extra": None,
                "error": str(e),
            })
    return results
