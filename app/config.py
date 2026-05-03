"""Конфигурация приложения."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения. Можно переопределять через переменные окружения / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_TITLE: str = "Combinatorial Optimization API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "REST API для решения задач комбинаторной оптимизации "
        "с возможностью выбора алгоритма и сравнения результатов."
    )
    CORS_ORIGINS: list[str] = ["*"]
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- База данных ---
    # Пример: postgresql+asyncpg://user:password@localhost:5432/vkr
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/vkr"

    # --- JWT ---
    JWT_SECRET: str = "CHANGE_ME_IN_ENV"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 14


settings = Settings()
