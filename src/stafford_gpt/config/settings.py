from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str
    chat_model: str = "gpt-4o"  # Revert back to gpt-4o
    temperature: float = 0.1

    # Database Configuration
    database_url: str
    postgres_db: str = "staffordgpt"
    postgres_user: str = "stafford"
    postgres_password: str = "password"

    # Application Configuration
    default_namespace: str = "customer"

    # Scraping Configuration
    scrape_timeout: int = 30000
    network_idle_timeout: int = 3000
    content_wait_time: int = 2000

    # Security Configuration
    domain_whitelist: List[str] = ["staffordglobal.org"]

    # CORS Configuration
    cors_origins: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174"
    ]

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
