"""Роутер: сравнение алгоритмов.

POST /compare — сравнить несколько алгоритмов на одной задаче.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user_optional
from app.db.models import ComparisonHistory, User
from app.db.session import get_db
from app.schemas.common import CompareRequest, CompareResponse
from app.services.compare_service import compare_algorithms

router = APIRouter(tags=["Сравнение"])


@router.post("/compare", response_model=CompareResponse)
async def compare(
    request: CompareRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
):
    """Сравнить несколько алгоритмов на одной задаче.

    Если запрос авторизован — сохраняется в историю сравнений пользователя.
    """
    try:
        response = compare_algorithms(
            task_name=request.task_name,
            algorithm_names=request.algorithms,
            input_data=request.input_data,
            params=request.params,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при сравнении: {str(e)}",
        )

    if current_user is not None:
        try:
            payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
            db.add(
                ComparisonHistory(
                    user_id=current_user.id,
                    task_name=request.task_name,
                    algorithms=list(request.algorithms),
                    input_data=request.input_data,
                    params=request.params,
                    result=payload,
                    algorithms_count=len(request.algorithms),
                )
            )
            await db.commit()
        except Exception:
            await db.rollback()

    return response
