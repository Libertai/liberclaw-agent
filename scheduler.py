"""Cron scheduler — replaces the simple heartbeat loop with full cron support.

Jobs are defined in ``workspace/cron.json`` and reloaded every tick so the
agent can self-schedule by editing the file with its own tools.  Backward
compatible: if no ``cron.json`` exists but ``heartbeat_interval > 0``, the
scheduler falls back to the legacy HEARTBEAT.md behaviour.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


# ── Cron expression parser ───────────────────────────────────────────

def _parse_field(expr: str, min_val: int, max_val: int) -> set[int]:
    """Parse a single cron field into a set of matching integer values.

    Supports: ``*``, specific values, ranges (``1-5``), lists (``1,3,5``),
    and step intervals (``*/5``, ``1-10/2``).
    """
    values: set[int] = set()
    for part in expr.split(","):
        part = part.strip()
        if "/" in part:
            range_part, step_str = part.split("/", 1)
            step = int(step_str)
            if range_part == "*":
                start, end = min_val, max_val
            elif "-" in range_part:
                start, end = (int(x) for x in range_part.split("-", 1))
            else:
                start, end = int(range_part), max_val
            values.update(range(start, end + 1, step))
        elif part == "*":
            values.update(range(min_val, max_val + 1))
        elif "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    return {v for v in values if min_val <= v <= max_val}


def parse_cron(expression: str) -> tuple[set[int], set[int], set[int], set[int], set[int]]:
    """Parse a 5-field cron expression.

    Returns (minutes, hours, days_of_month, months, days_of_week).
    """
    fields = expression.strip().split()
    if len(fields) != 5:
        raise ValueError(f"Cron expression must have 5 fields, got {len(fields)}: {expression!r}")
    minutes = _parse_field(fields[0], 0, 59)
    hours = _parse_field(fields[1], 0, 23)
    days_of_month = _parse_field(fields[2], 1, 31)
    months = _parse_field(fields[3], 1, 12)
    days_of_week = _parse_field(fields[4], 0, 6)  # 0=Sunday … 6=Saturday
    return minutes, hours, days_of_month, months, days_of_week


def cron_matches(expression: str, dt: datetime) -> bool:
    """Return True if *dt* matches the cron *expression*."""
    minutes, hours, days_of_month, months, days_of_week = parse_cron(expression)
    # datetime.isoweekday(): Mon=1 … Sun=7; cron: Sun=0 … Sat=6
    cron_dow = dt.isoweekday() % 7  # Sun=0, Mon=1, ..., Sat=6
    return (
        dt.minute in minutes
        and dt.hour in hours
        and dt.day in days_of_month
        and dt.month in months
        and cron_dow in days_of_week
    )


# ── Job model ────────────────────────────────────────────────────────

@dataclass
class CronJob:
    id: str
    schedule: str  # 5-field cron expression
    task: str  # message sent to the agent
    enabled: bool = True


# ── Scheduler ────────────────────────────────────────────────────────

class CronScheduler:
    """File-backed cron scheduler.

    Jobs are stored in ``<workspace>/cron.json`` and reloaded every tick
    so the agent can add/remove jobs via its file tools.
    """

    def __init__(self, workspace_path: str, heartbeat_interval: int = 0):
        self.workspace_path = workspace_path
        self.heartbeat_interval = heartbeat_interval
        self.jobs: list[CronJob] = []
        self._last_run: dict[str, float] = {}  # job_id -> unix timestamp
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    # ── public API ───────────────────────────────────────────────────

    async def start(self, run_job_callback: Callable[[str, str], Awaitable[None]]):
        """Start the scheduler loop.

        *run_job_callback(task, job_id)* is called for each due job.
        """
        self._stop_event.clear()
        self._task = asyncio.create_task(self._loop(run_job_callback))

    async def stop(self):
        """Stop the scheduler gracefully."""
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ── job loading ──────────────────────────────────────────────────

    def _cron_file(self) -> Path:
        return Path(self.workspace_path) / "cron.json"

    def load_jobs(self) -> list[CronJob]:
        """Load jobs from ``workspace/cron.json``.  Returns [] on missing/invalid file."""
        path = self._cron_file()
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, list):
                logger.warning("cron.json root is not a list — ignoring")
                return []
            jobs: list[CronJob] = []
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                job_id = entry.get("id")
                schedule = entry.get("schedule")
                task = entry.get("task")
                if not (job_id and schedule and task):
                    continue
                # Validate the cron expression early
                try:
                    parse_cron(schedule)
                except ValueError as e:
                    logger.warning(f"Skipping cron job {job_id!r}: {e}")
                    continue
                jobs.append(CronJob(
                    id=str(job_id),
                    schedule=schedule,
                    task=task,
                    enabled=entry.get("enabled", True),
                ))
            return jobs
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load cron.json: {e}")
            return []

    # ── due check ────────────────────────────────────────────────────

    def is_due(self, job: CronJob, now: datetime | None = None) -> bool:
        """Check if *job* should fire at the current minute.

        A job is due when the cron expression matches AND the job has not
        already been run during this calendar minute (avoids double-exec).
        """
        if not job.enabled:
            return False
        if now is None:
            now = datetime.now(timezone.utc)
        if not cron_matches(job.schedule, now):
            return False
        # Prevent double-execution within the same minute
        minute_start = now.replace(second=0, microsecond=0).timestamp()
        last = self._last_run.get(job.id, 0)
        return last < minute_start

    # ── main loop ────────────────────────────────────────────────────

    async def _loop(self, run_job_callback: Callable[[str, str], Awaitable[None]]):
        """Tick every 60 s, check due jobs, and dispatch them."""
        # Align to the start of the next minute (± 1 s tolerance)
        now = datetime.now(timezone.utc)
        seconds_to_next_minute = 60 - now.second
        if seconds_to_next_minute > 1:
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=seconds_to_next_minute)
                return  # stop requested during initial wait
            except asyncio.TimeoutError:
                pass

        while not self._stop_event.is_set():
            try:
                await self._tick(run_job_callback)
            except Exception as e:
                logger.error(f"Scheduler tick error: {e}")

            # Sleep ~60 s, but break early if stop is requested
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=60)
                return  # stop requested
            except asyncio.TimeoutError:
                pass

    async def _tick(self, run_job_callback: Callable[[str, str], Awaitable[None]]):
        """One scheduler tick: reload jobs, run anything that's due."""
        now = datetime.now(timezone.utc)

        # Reload jobs from disk every tick
        self.jobs = self.load_jobs()

        # If no cron.json exists, fall back to legacy heartbeat behaviour
        if not self.jobs and not self._cron_file().exists():
            if self.heartbeat_interval > 0:
                await self._legacy_heartbeat(run_job_callback, now)
            return

        for job in self.jobs:
            if self.is_due(job, now):
                self._last_run[job.id] = time.time()
                logger.info(f"Cron job {job.id!r} is due — dispatching")
                try:
                    await run_job_callback(job.task, job.id)
                except Exception as e:
                    logger.error(f"Cron job {job.id!r} failed: {e}")

    # ── legacy heartbeat fallback ────────────────────────────────────

    async def _legacy_heartbeat(
        self,
        run_job_callback: Callable[[str, str], Awaitable[None]],
        now: datetime,
    ):
        """Emulate the old heartbeat: run every ``heartbeat_interval`` seconds
        if ``HEARTBEAT.md`` exists and has actionable content.
        """
        last_hb = self._last_run.get("__heartbeat__", 0)
        if (now.timestamp() - last_hb) < self.heartbeat_interval:
            return
        heartbeat_file = Path(self.workspace_path) / "HEARTBEAT.md"
        if not heartbeat_file.exists():
            return
        content = heartbeat_file.read_text()
        if _is_heartbeat_empty(content):
            return
        self._last_run["__heartbeat__"] = now.timestamp()
        task = (
            "Read HEARTBEAT.md and follow any instructions or tasks listed there. "
            "If nothing needs attention, reply with just: HEARTBEAT_OK"
        )
        await run_job_callback(task, "__heartbeat__")


def _is_heartbeat_empty(content: str) -> bool:
    """Check if heartbeat file has no actionable content."""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        if stripped.startswith("- [ ]"):
            return False
        return False
    return True
