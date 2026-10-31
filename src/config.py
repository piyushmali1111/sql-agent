"""Configuration management for the NL2SQL Agent application.

Loads environment variables from .env and exposes strongly typed settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Explicitly load .env from project root
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"

    # LLM Settings
    google_api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))

    # Database
    database_url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'business.db'}")
    )
    schema_descriptions_path: Path = BASE_DIR / "data" / "schema_descriptions.yaml"
    examples_path: Path = BASE_DIR / "data" / "examples.yaml"

    # Vector store & Embeddings
    chroma_dir: Path = BASE_DIR / "data" / "chroma"
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )

    # Agent Limits and Behavior
    requests_per_minute: int = Field(
        default_factory=lambda: int(os.getenv("REQUESTS_PER_MINUTE", "15"))
    )
    max_repair_attempts: int = Field(
        default_factory=lambda: int(os.getenv("MAX_REPAIR_ATTEMPTS", "2"))
    )
    enable_nl_explanation: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_NL_EXPLANATION", "true").lower() in ("true", "1", "yes")
    )
    max_execution_rows: int = Field(
        default_factory=lambda: int(os.getenv("MAX_EXECUTION_ROWS", "100"))
    )
    query_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("QUERY_TIMEOUT_SECONDS", "10.0"))
    )

    # Retrieval tuning
    few_shot_k: int = 3
    table_retrieval_k: int = 4
    column_retrieval_k: int = 8
    conversation_history_turns: int = 5

    # LangSmith Tracing
    langchain_tracing_v2: bool = Field(
        default_factory=lambda: os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes")
    )
    langchain_api_key: str = Field(default_factory=lambda: os.getenv("LANGCHAIN_API_KEY", ""))
    langchain_project: str = Field(
        default_factory=lambda: os.getenv("LANGCHAIN_PROJECT", "nl2sql-agent")
    )


settings = Settings()

