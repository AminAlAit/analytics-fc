"""Application configuration settings for Analytics FC REST API.

Manages environment variables, database locations, gallery artifacts,
and server parameters using Pydantic Settings.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings with environment variable override support."""

    app_name: str = "Analytics FC Tactical Intelligence API"
    app_version: str = "1.0.0"
    environment: str = "production"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]
    db_path: Path = BASE_DIR / "dbt_project" / "analytics_fc.duckdb"
    gallery_path: Path = BASE_DIR / "data" / "processed" / "superstar_gallery.json"

    model_config = SettingsConfigDict(
        env_prefix="AFC_",
        case_sensitive=False,
    )


settings = Settings()
