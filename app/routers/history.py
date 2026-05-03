"""Роутер истории запусков и сравнений текущего пользователя."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.models import ComparisonHistory, SolveHistory, User
from app.db.session import get_db
from app.schemas.history import ComparisonHistoryOut, SolveHistoryOut

router = APIRouter(prefix="/history", tags=["История"])


@router.get("/solve", response_model=list[SolveHistoryOut])
async def list_solve_history(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = (
        await db.execute(
            select(SolveHistory)
            .where(SolveHistory.user_id == current_user.id)
            .order_by(SolveHistory.created_at.desc())
            .offset(offset)
            .limit(min(limit, 200))
        )
    ).scalars().all()
    return rows


@router.get("/compare", response_model=list[ComparisonHistoryOut])
async def list_compare_history(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = (
        await db.execute(
            select(ComparisonHistory)
            .where(ComparisonHistory.user_id == current_user.id)
            .order_by(ComparisonHistory.created_at.desc())
            .offset(offset)
            .limit(min(limit, 200))
        )
    ).scalars().all()
    return rows


@router.delete("/solve/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_solve_item(
    item_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    obj = await db.get(SolveHistory, item_id)
    if obj is None or obj.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Запись не найдена")
    await db.delete(obj)
    await db.commit()


@router.delete("/compare/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_compare_item(
    item_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    obj = await db.get(ComparisonHistory, item_id)
    if obj is None or obj.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Запись не найдена")
    await db.delete(obj)
    await db.commit()
