"""Application settings sourced from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from the environment."""

    model_config = SettingsConfigDict(env_prefix="AETHER_", env_file=".env", extra="ignore")

    app_name: str = "Project Aether"
    debug: bool = False

    # Model settings
    model_path: str = "models/aether_model.pt"
    hidden_dim: int = 64
    trajectory_steps: int = 120

    # Data settings
    data_path: str = "data/"
    cms_data_url: str = "https://data.cms.gov/"
    use_real_data: bool = False

    # Deployment settings
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000",
        description="Comma-separated list of allowed CORS origins.",
    )

    @property
    def parsed_cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
