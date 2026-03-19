from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import solver, compare, charts, history

app = FastAPI(title=settings.app_title)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(solver.router)
app.include_router(compare.router)
app.include_router(charts.router)
app.include_router(history.router)


@app.get("/")
def root():
    return {"message": "Combinatorial Optimization API"}
