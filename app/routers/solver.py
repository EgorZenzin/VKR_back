from fastapi import APIRouter, HTTPException, Depends
from typing import Any
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np

from app.database import get_db
from app.models.result import Result
from app.services.solver_service import solve_problem, get_problems_info

router = APIRouter(tags=["solver"])


def _convert_numpy(obj):
    """Конвертирует numpy-типы в стандартные Python-типы."""
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_convert_numpy(i) for i in obj]
    return obj


class SolveRequest(BaseModel):
    problem_type: str
    algorithm: str
    input_data: dict
    params: dict | None = None


class SolveResponse(BaseModel):
    id: int | None = None
    problem_type: str
    algorithm: str
    solution: Any
    cost: float
    execution_time: float
    convergence_history: list[float] | None = None
    extra: dict | None = None


@router.get("/problems")
def list_problems():
    return get_problems_info()


@router.post("/solve", response_model=SolveResponse)
def solve(request: SolveRequest, db: Session = Depends(get_db)):
    try:
        result = solve_problem(
            request.problem_type, request.algorithm, request.input_data, request.params
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка выполнения: {str(e)}")

    solution = _convert_numpy(result.solution)
    convergence = _convert_numpy(result.convergence_history)
    cost = float(result.cost)
    exec_time = float(result.execution_time)
    extra = _convert_numpy(result.extra)

    db_result = Result(
        problem_type=request.problem_type,
        algorithm=request.algorithm,
        input_data=request.input_data,
        output_data={"solution": solution},
        cost=cost,
        execution_time=exec_time,
        convergence_history=convergence,
        parameters=request.params,
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)

    return SolveResponse(
        id=db_result.id,
        problem_type=request.problem_type,
        algorithm=request.algorithm,
        solution=solution,
        cost=cost,
        execution_time=exec_time,
        convergence_history=convergence,
        extra=extra,
    )


class PreviewResponse(BaseModel):
    problem_type: str
    algorithm: str
    solution: Any
    cost: float
    execution_time: float
    convergence_history: list[float] | None = None
    extra: dict | None = None
    chart_data: dict | None = None


@router.post("/solve/preview", response_model=PreviewResponse)
def solve_preview(request: SolveRequest):
    """Быстрое решение без сохранения в БД — для динамических ползунков."""
    try:
        result = solve_problem(
            request.problem_type, request.algorithm, request.input_data, request.params
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка выполнения: {str(e)}")

    solution = _convert_numpy(result.solution)
    convergence = _convert_numpy(result.convergence_history)
    cost = float(result.cost)
    exec_time = float(result.execution_time)
    extra = _convert_numpy(result.extra)

    # Данные для построения графиков на фронтенде
    chart_data = {
        "convergence": convergence,
    }
    if request.problem_type == "tsp" and "cities" in request.input_data:
        cities = request.input_data["cities"]
        route = solution if isinstance(solution, list) else []
        chart_data["route"] = {
            "cities": cities,
            "order": route,
        }

    return PreviewResponse(
        problem_type=request.problem_type,
        algorithm=request.algorithm,
        solution=solution,
        cost=cost,
        execution_time=exec_time,
        convergence_history=convergence,
        extra=extra,
        chart_data=chart_data,
    )
