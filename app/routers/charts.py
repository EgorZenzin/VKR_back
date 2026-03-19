from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.result import Result, ComparisonResult
from app.services.chart_service import (
    generate_route_chart,
    generate_convergence_chart,
    generate_comparison_chart,
    generate_graph_visualization,
)

router = APIRouter(prefix="/chart", tags=["charts"])


@router.get("/route/{result_id}")
def chart_route(result_id: int, db: Session = Depends(get_db)):
    result = db.query(Result).filter(Result.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Результат не найден")
    if result.problem_type != "tsp":
        raise HTTPException(status_code=400, detail="График маршрута доступен только для TSP")

    cities = result.input_data.get("cities", [])
    route = result.output_data.get("solution", [])

    if not cities or not route:
        raise HTTPException(status_code=400, detail="Нет данных для построения графика")

    image = generate_route_chart(cities, route)
    return Response(content=image, media_type="image/png")


@router.get("/convergence/{result_id}")
def chart_convergence(result_id: int, db: Session = Depends(get_db)):
    result = db.query(Result).filter(Result.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Результат не найден")
    if not result.convergence_history:
        raise HTTPException(status_code=400, detail="Нет данных сходимости для этого результата")

    image = generate_convergence_chart(result.convergence_history, result.algorithm)
    return Response(content=image, media_type="image/png")


@router.get("/comparison/{comparison_id}")
def chart_comparison(comparison_id: int, db: Session = Depends(get_db)):
    comp = db.query(ComparisonResult).filter(ComparisonResult.id == comparison_id).first()
    if not comp:
        raise HTTPException(status_code=404, detail="Сравнение не найдено")

    image = generate_comparison_chart(comp.results)
    return Response(content=image, media_type="image/png")


@router.get("/graph/{result_id}")
def chart_graph(result_id: int, db: Session = Depends(get_db)):
    result = db.query(Result).filter(Result.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Результат не найден")

    if result.problem_type not in ("graph_coloring", "max_flow"):
        raise HTTPException(
            status_code=400,
            detail="Визуализация графа доступна для graph_coloring и max_flow",
        )

    n_vertices = result.input_data.get("n_vertices", 0)
    edges = result.input_data.get("edges", [])

    coloring = None
    flow_edges = None

    if result.problem_type == "graph_coloring":
        coloring = result.output_data.get("solution", {})
    elif result.problem_type == "max_flow":
        flow_edges = result.output_data.get("solution", [])

    image = generate_graph_visualization(n_vertices, edges, coloring, flow_edges)
    return Response(content=image, media_type="image/png")
