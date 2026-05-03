"""Роутер: решение задач.

POST /solve — решить одну задачу одним алгоритмом.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user_optional
from app.db.models import SolveHistory, User
from app.db.session import get_db
from app.schemas.common import SolveRequest, SolveResponse
from app.services.solver_service import solve_problem

router = APIRouter(tags=["Решение"])


@router.post("/solve", response_model=SolveResponse)
async def solve(
    request: SolveRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
):
    """Решить задачу комбинаторной оптимизации выбранным алгоритмом.

    Если запрос авторизован — результат сохраняется в историю пользователя.
    """
    try:
        response = solve_problem(
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

    if current_user is not None:
        try:
            payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
            metrics = payload.get("metrics") or {}
            db.add(
                SolveHistory(
                    user_id=current_user.id,
                    task_name=request.task_name,
                    algorithm=request.algorithm,
                    input_data=request.input_data,
                    params=request.params,
                    result=payload,
                    objective_value=metrics.get("objective_value")
                    or metrics.get("best_value")
                    or metrics.get("value"),
                    elapsed_ms=metrics.get("elapsed_ms") or metrics.get("time_ms"),
                )
            )
            await db.commit()
        except Exception:
            await db.rollback()

    return response
