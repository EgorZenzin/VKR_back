"""Роутер: сравнение алгоритмов.

POST /compare — сравнить несколько алгоритмов на одной задаче.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.common import CompareRequest, CompareResponse
from app.services.compare_service import compare_algorithms

router = APIRouter(tags=["Сравнение"])


@router.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    """Сравнить несколько алгоритмов на одной задаче.

    Возвращает результаты каждого алгоритма и подготовленные данные
    для графиков сравнения (сходимость, время, качество).
    """
    try:
        return compare_algorithms(
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
