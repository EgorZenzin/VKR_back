"""Роутер: решение задач.

POST /solve — решить одну задачу одним алгоритмом.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.common import SolveRequest, SolveResponse
from app.services.solver_service import solve_problem

router = APIRouter(tags=["Решение"])


@router.post("/solve", response_model=SolveResponse)
def solve(request: SolveRequest):
    """Решить задачу комбинаторной оптимизации выбранным алгоритмом.

    Возвращает решение, метрики, историю сходимости и данные для визуализации.
    """
    try:
        return solve_problem(
            task_name=request.task_name,
            algorithm_name=request.algorithm,
            input_data=request.input_data,
            params=request.params,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при решении: {str(e)}")
