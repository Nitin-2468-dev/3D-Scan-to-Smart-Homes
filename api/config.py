"""
Configuration management for the 3D Reconstruction + Multimodal QA API.

Supports environment-based configuration for dev/prod modes.
"""

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application settings
    app_name: str = "3D Reconstruction + Multimodal QA API"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: Literal["development", "production", "test"] = Field(
        default="development", description="Application environment"
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    # CORS settings
    cors_origins: list[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: list[str] = Field(default=["*"])
    cors_allow_headers: list[str] = Field(default=["*"])

    # Redis/Celery settings
    redis_url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0", description="Celery broker URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0", description="Celery result backend URL"
    )

    # File storage settings
    upload_dir: str = Field(
        default="./uploads", description="Directory for uploaded files"
    )
    artifacts_dir: str = Field(
        default="./artifacts", description="Directory for generated artifacts"
    )
    fallback_dir: str = Field(
        default="./fallback", description="Directory for fallback assets"
    )
    max_upload_size_mb: int = Field(
        default=100, description="Maximum upload size in MB"
    )

    # Model settings
    lmm_model_name: str = Field(
        default="Qwen/Qwen2-VL-2B-Instruct",
        description="Multimodal LLM model name",
    )
    lmm_quantization: Literal["none", "4bit", "8bit"] = Field(
        default="4bit", description="LMM quantization level"
    )
    device_map: str = Field(
        default="auto", description="Device map for model loading"
    )

    # Reconstruction settings
    reconstruction_quick_iterations: int = Field(
        default=100, description="Iterations for quick reconstruction"
    )
    reconstruction_full_iterations: int = Field(
        default=500, description="Iterations for full reconstruction"
    )

    # Session settings
    session_ttl_hours: int = Field(
        default=24, description="Session time-to-live in hours"
    )
    max_images_per_session: int = Field(
        default=50, description="Maximum images per session"
    )

    # Demo settings
    demo_mode: bool = Field(
        default=True, description="Enable demo mode with mock responses"
    )
    enable_fallback: bool = Field(
        default=True, description="Enable fallback to precomputed assets"
    )

    model_config = {
        "env_prefix": "APP_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Convenience function to check if running in production
def is_production() -> bool:
    """Check if running in production environment."""
    return get_settings().environment == "production"


# Convenience function to check if demo mode is enabled
def is_demo_mode() -> bool:
    """Check if demo mode is enabled."""
    return get_settings().demo_mode


# Create directories if they don't exist
def ensure_directories():
    """Ensure required directories exist."""
    settings = get_settings()
    for directory in [settings.upload_dir, settings.artifacts_dir]:
        os.makedirs(directory, exist_ok=True)
