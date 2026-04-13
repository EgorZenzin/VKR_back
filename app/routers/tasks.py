"""Роутер: информация о задачах и алгоритмах.

GET /tasks — список задач.
GET /tasks/{task_name}/algorithms — алгоритмы для конкретной задачи.
"""

from fastapi import APIRouter, HTTPException

from app.algorithms.registry import (
    TASK_REGISTRY,
    TASK_DISPLAY_NAMES,
    TASK_DESCRIPTIONS,
    TASK_OPTIMIZATION,
    get_algorithms_for_task,
)
from app.schemas.common import TaskInfo, AlgorithmInfo

router = APIRouter(tags=["Задачи"])


@router.get("/tasks", response_model=list[TaskInfo])
def list_tasks():
    """Получить список всех доступных задач с их алгоритмами."""
    result = []
    for task_name in TASK_REGISTRY:
        algos = get_algorithms_for_task(task_name)
        algo_list = [
            AlgorithmInfo(name=a.name, display_name=a.display_name)
            for a in algos.values()
        ]
        result.append(TaskInfo(
            name=task_name,
            display_name=TASK_DISPLAY_NAMES.get(task_name, task_name),
            description=TASK_DESCRIPTIONS.get(task_name, ""),
            optimization=TASK_OPTIMIZATION.get(task_name, "minimize"),
            algorithms=algo_list,
        ))
    return result


@router.get("/tasks/{task_name}/algorithms", response_model=list[AlgorithmInfo])
def list_algorithms(task_name: str):
    """Получить список алгоритмов для конкретной задачи."""
    try:
        algos = get_algorithms_for_task(task_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return [
        AlgorithmInfo(name=a.name, display_name=a.display_name)
        for a in algos.values()
    ]
