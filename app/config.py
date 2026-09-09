from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional, List
from pathlib import Path
import os


class Settings(BaseSettings):
    API_TITLE: str = "Document Q&A Assistant — RAG Pipeline with Open-Source LLM"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    API_KEY_HEADER: str = "X-API-Key"
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    RATE_LIMIT_PER_MINUTE: int = 60

    POSTGRES_USER: str = "raguser"
    POSTGRES_PASSWORD: str = "ragpassword"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "bookrag_db"

    VECTOR_DB_TYPE: str = "pgvector"

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    CACHE_TTL: int = 3600

    LLM_PROVIDER: str = "ollama"
    OPENAI_API_KEY: Optional[str] = None

    OLLAMA_BASE_URL: str = "http://ollama:11434"
    MODEL_NAME: str = "llama3.2:latest"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.2

    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    TOP_K_RETRIEVAL: int = 3
    SIMILARITY_THRESHOLD: float = 0.6

    SENTRY_DSN: Optional[str] = None
    LOG_LEVEL: str = "INFO"
    ENABLE_METRICS: bool = False

    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"

    MAX_FILE_SIZE_MB: int = 25
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".docx", ".txt", ".md"]
    UPLOAD_DIR: str = str(Path("./uploads").resolve())

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}"


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    # langchain-ollama 0.1.0 has no base_url; the ollama client reads OLLAMA_HOST
    if settings.OLLAMA_BASE_URL:
        os.environ["OLLAMA_HOST"] = settings.OLLAMA_BASE_URL
    return settings
