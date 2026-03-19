import datetime
from sqlalchemy import Integer, String, Float, DateTime, JSON, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Result(Base):
    __tablename__ = "results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    problem_type: Mapped[str] = mapped_column(String(50), index=True)
    algorithm: Mapped[str] = mapped_column(String(50), index=True)
    input_data: Mapped[dict] = mapped_column(JSON)
    output_data: Mapped[dict] = mapped_column(JSON)
    cost: Mapped[float] = mapped_column(Float)
    execution_time: Mapped[float] = mapped_column(Float)
    convergence_history: Mapped[list | None] = mapped_column(JSON, nullable=True)
    parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ComparisonResult(Base):
    __tablename__ = "comparison_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    problem_type: Mapped[str] = mapped_column(String(50), index=True)
    input_data: Mapped[dict] = mapped_column(JSON)
    results: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
