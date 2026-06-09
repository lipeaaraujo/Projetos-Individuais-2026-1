"""Central configuration, loaded from environment / .env.

Everything tunable lives here so the rest of the code never reads os.environ
directly. Uses pydantic-settings for typed, validated config.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the parent of this src/ package.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- LLM (Gemini) -----------------------------------------------------
    # Accepts GEMINI_API_KEY or, as a fallback, GOOGLE_API_KEY (SDK default).
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    # gemini-2.5-flash is multimodal (reads PDFs natively), fast and cheap —
    # the right default for high-volume document extraction.
    gemini_model: str = "gemini-2.5-flash"

    # --- storage ----------------------------------------------------------
    db_path: str = str(DATA_DIR / "catalog.db")
    pdf_dir: str = str(PDF_DIR)

    # --- ingestion --------------------------------------------------------
    # Hour-of-day (0-23) the daily scheduler polls RI pages.
    poll_hour: int = 6
    # Comma-separated list of adapters to enable in the poller/scheduler.
    enabled_adapters: str = "direcional,trisul"
    http_timeout: float = 30.0
    user_agent: str = (
        "Mozilla/5.0 (compatible; ConjunturaBot/1.0; "
        "+pipeline-uda-setor-habitacional)"
    )


def get_settings() -> Settings:
    """Return a Settings instance and ensure data directories exist."""
    s = Settings()
    if not s.gemini_api_key:
        s.gemini_api_key = os.environ.get("GOOGLE_API_KEY", "")
    Path(s.pdf_dir).mkdir(parents=True, exist_ok=True)
    Path(s.db_path).parent.mkdir(parents=True, exist_ok=True)
    return s
