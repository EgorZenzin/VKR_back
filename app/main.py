"""FastAPI приложение — точка входа.

Регистрирует роутеры, настраивает CORS и health-check.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import tasks, solver, compare, auth, history

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
)

# CORS — разрешает подключение фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Роутеры
app.include_router(auth.router)
app.include_router(tasks.router)
app.include_router(solver.router)
app.include_router(compare.router)
app.include_router(history.router)


@app.get("/health", tags=["Система"])
def health_check():
    """Проверка работоспособности сервиса."""
    return {"status": "ok"}
