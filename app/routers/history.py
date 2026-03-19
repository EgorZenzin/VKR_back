from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.result import Result, ComparisonResult

router = APIRouter(prefix="/history", tags=["history"])


@router.get("/")
def list_results(
    problem_type: str | None = None,
    algorithm: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    query = db.query(Result).order_by(Result.created_at.desc())
    if problem_type:
        query = query.filter(Result.problem_type == problem_type)
    if algorithm:
        query = query.filter(Result.algorithm == algorithm)
    total = query.count()
    results = query.offset(offset).limit(limit).all()

    return {
        "total": total,
        "results": [
            {
                "id": r.id,
                "problem_type": r.problem_type,
                "algorithm": r.algorithm,
                "cost": r.cost,
                "execution_time": r.execution_time,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in results
        ],
    }


@router.get("/{result_id}")
def get_result(result_id: int, db: Session = Depends(get_db)):
    result = db.query(Result).filter(Result.id == result_id).first()
    if not result:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Результат не найден")
    return {
        "id": result.id,
        "problem_type": result.problem_type,
        "algorithm": result.algorithm,
        "input_data": result.input_data,
        "output_data": result.output_data,
        "cost": result.cost,
        "execution_time": result.execution_time,
        "convergence_history": result.convergence_history,
        "parameters": result.parameters,
        "created_at": result.created_at.isoformat() if result.created_at else None,
    }


@router.get("/comparisons/")
def list_comparisons(
    problem_type: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    query = db.query(ComparisonResult).order_by(ComparisonResult.created_at.desc())
    if problem_type:
        query = query.filter(ComparisonResult.problem_type == problem_type)
    total = query.count()
    results = query.offset(offset).limit(limit).all()

    return {
        "total": total,
        "results": [
            {
                "id": r.id,
                "problem_type": r.problem_type,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in results
        ],
    }


@router.get("/comparisons/{comparison_id}")
def get_comparison(comparison_id: int, db: Session = Depends(get_db)):
    result = db.query(ComparisonResult).filter(ComparisonResult.id == comparison_id).first()
    if not result:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Сравнение не найдено")
    return {
        "id": result.id,
        "problem_type": result.problem_type,
        "input_data": result.input_data,
        "results": result.results,
        "created_at": result.created_at.isoformat() if result.created_at else None,
    }
