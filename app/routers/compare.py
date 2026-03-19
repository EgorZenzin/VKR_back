from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.result import ComparisonResult
from app.services.compare_service import compare_algorithms

router = APIRouter(tags=["compare"])


class CompareRequest(BaseModel):
    problem_type: str
    algorithms: list[str]
    input_data: dict
    params: dict | None = None


@router.post("/compare")
def compare(request: CompareRequest, db: Session = Depends(get_db)):
    if len(request.algorithms) < 2:
        raise HTTPException(status_code=400, detail="Выберите минимум 2 алгоритма для сравнения")

    try:
        results = compare_algorithms(
            request.problem_type, request.algorithms, request.input_data, request.params
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    db_result = ComparisonResult(
        problem_type=request.problem_type,
        input_data=request.input_data,
        results=results,
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)

    return {
        "id": db_result.id,
        "problem_type": request.problem_type,
        "results": results,
    }
