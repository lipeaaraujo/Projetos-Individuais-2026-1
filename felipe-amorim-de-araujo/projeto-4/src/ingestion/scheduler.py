"""Daily polling scheduler (APScheduler).

Runs the discover -> ingest cycle once a day. This is the event-driven trigger:
new operational previews are picked up automatically without manual runs.
"""

from __future__ import annotations

from apscheduler.schedulers.blocking import BlockingScheduler

from ..config import get_settings
from .adapters import build_adapters


def run_cycle() -> None:
    from ..pipeline import run_pipeline

    settings = get_settings()
    adapters = build_adapters(
        settings.enabled_adapters.split(","),
        user_agent=settings.user_agent,
        timeout=settings.http_timeout,
    )
    run_pipeline(adapters, settings)


def start() -> None:
    settings = get_settings()
    scheduler = BlockingScheduler(timezone="America/Sao_Paulo")
    scheduler.add_job(run_cycle, "cron", hour=settings.poll_hour, minute=0)
    print(f"Scheduler iniciado: polling diário às {settings.poll_hour:02d}:00 (America/Sao_Paulo).")
    run_cycle()  # run once on startup
    scheduler.start()


if __name__ == "__main__":
    start()
