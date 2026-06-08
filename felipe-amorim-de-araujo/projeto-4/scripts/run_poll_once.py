"""Run one discover->ingest cycle across enabled adapters (no scheduler loop)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.ingestion.adapters import build_adapters
from src.pipeline import run_pipeline


def main() -> None:
    settings = get_settings()
    adapters = build_adapters(
        settings.enabled_adapters.split(","),
        user_agent=settings.user_agent,
        timeout=settings.http_timeout,
    )
    print(f"Adapters: {[a.name for a in adapters]}")
    run_pipeline(adapters, settings)


if __name__ == "__main__":
    main()
