"""Конфигурация приложения."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения. Можно переопределять через переменные окружения."""

    APP_TITLE: str = "Combinatorial Optimization API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "REST API для решения задач комбинаторной оптимизации "
        "с возможностью выбора алгоритма и сравнения результатов."
    )
    CORS_ORIGINS: list[str] = ["*"]
    HOST: str = "0.0.0.0"
    PORT: int = 8000


settings = Settings()
