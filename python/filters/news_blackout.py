"""High-impact news blackout filter for EUR/USD M5 strategies.

EUR/USD spreads blow from ~1 pip to 8-15 pips on NFP, CPI, FOMC, and ECB
releases. A single unfiltered news bar can erase a month of edge. This
filter is a boolean "is timestamp near a high-impact event" gate that
every entry signal MUST pass before arming an order.

Calendar source: CSV at ``data/news_calendar.csv`` (override via env
``EBB_NEWS_CALENDAR``). Format::

    timestamp_utc,currency,event,impact
    2026-05-01 12:30,USD,NFP,HIGH
    2026-05-07 12:15,EUR,ECB_RATE,HIGH

Only ``impact=HIGH`` and ``currency in {EUR,USD}`` rows are kept. The
CSV is manually maintained so backtests are reproducible. If the CSV is
missing or empty, a hardcoded schedule of recurring Tier-1 events is
generated (2026 FOMC/ECB dates + monthly NFP/CPI templates).

Performance: events are loaded once (module cache) and stored as a
sorted list for ``bisect`` lookup — O(log N) per call. Safe to call on
every M5 bar in multi-year grids. All timestamps are UTC-naive; caller
must convert broker-local to UTC first.
"""

from __future__ import annotations

import bisect
import csv
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

HIGH_IMPACT_CURRENCIES: frozenset[str] = frozenset({"EUR", "USD"})

# Confirmed 2026 Tier-1 dates. Sources: federalreserve.gov,
# ecb.europa.eu. FOMC statement = 14:00 ET (19:00 UTC EST / 18:00 UTC
# EDT). ECB statement = 14:15 CET (13:15 UTC CET / 12:15 UTC CEST).
_FOMC_2026: tuple[datetime, ...] = (
    datetime(2026, 1, 28, 19, 0),
    datetime(2026, 3, 18, 18, 0),
    datetime(2026, 4, 29, 18, 0),
    datetime(2026, 6, 17, 18, 0),
    datetime(2026, 7, 29, 18, 0),
    datetime(2026, 9, 16, 18, 0),
    datetime(2026, 10, 28, 18, 0),
    datetime(2026, 12, 9, 19, 0),
)
_ECB_2026: tuple[datetime, ...] = (
    datetime(2026, 2, 5, 13, 15),
    datetime(2026, 3, 19, 13, 15),
    datetime(2026, 4, 30, 12, 15),
    datetime(2026, 6, 11, 12, 15),
    datetime(2026, 7, 23, 12, 15),
    datetime(2026, 9, 10, 12, 15),
    datetime(2026, 10, 29, 13, 15),
    datetime(2026, 12, 17, 13, 15),
)


@dataclass(frozen=True)
class NewsEvent:
    """Scheduled high-impact event that triggers a blackout window."""

    ts_utc: datetime
    currency: str
    event: str


def _hardcoded_calendar(year: int) -> list[NewsEvent]:
    """Fallback event list for ``year`` when no CSV is present."""
    events: list[NewsEvent] = []
    if year == 2026:
        events.extend(NewsEvent(d, "USD", "FOMC") for d in _FOMC_2026)
        events.extend(NewsEvent(d, "EUR", "ECB_RATE") for d in _ECB_2026)
        events.extend(
            NewsEvent(d + timedelta(minutes=30), "EUR", "ECB_PRESSER")
            for d in _ECB_2026
        )
    # Monthly templates: NFP first Friday, US CPI ~13th, EU CPI ~30th.
    # NFP/CPI release = 08:30 ET = 13:30 UTC (EST) or 12:30 UTC (EDT).
    # Months Jan/Feb/Nov/Dec use EST, Mar-Oct use EDT.
    for month in range(1, 13):
        nfp_hour = 13 if month in (1, 2, 11, 12) else 12
        first = datetime(year, month, 1)
        offset = (4 - first.weekday()) % 7  # Friday = weekday 4
        nfp_day = first + timedelta(days=offset)
        events.append(
            NewsEvent(nfp_day.replace(hour=nfp_hour, minute=30), "USD", "NFP")
        )
        cpi_day = datetime(year, month, 13, nfp_hour, 30)
        events.append(NewsEvent(cpi_day, "USD", "US_CPI"))
        eu_day = datetime(year, month, 28 if month == 2 else 30, 9, 0)
        events.append(NewsEvent(eu_day, "EUR", "EU_CPI"))
    return events


def _default_csv_path() -> Path:
    """``EBB_NEWS_CALENDAR`` env var > ``<repo>/data/news_calendar.csv``."""
    override = os.environ.get("EBB_NEWS_CALENDAR")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    return here.parent.parent.parent / "data" / "news_calendar.csv"


def load_calendar(
    csv_path: Path | None = None,
    *,
    years_for_fallback: Iterable[int] = (2020, 2021, 2022, 2023, 2024, 2025, 2026),
) -> list[NewsEvent]:
    """Load calendar from CSV with hardcoded fallback. Sorted ascending."""
    path = csv_path or _default_csv_path()
    events: list[NewsEvent] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("impact", "").strip().upper() != "HIGH":
                    continue
                ccy = row.get("currency", "").strip().upper()
                if ccy not in HIGH_IMPACT_CURRENCIES:
                    continue
                try:
                    ts = datetime.fromisoformat(row["timestamp_utc"].strip())
                except (KeyError, ValueError):
                    continue
                events.append(NewsEvent(ts, ccy, row.get("event", "").strip()))
    if not events:
        for year in years_for_fallback:
            events.extend(_hardcoded_calendar(year))
    events.sort(key=lambda e: e.ts_utc)
    return events


# Module-level cache. Reload via ``reload_calendar()``.
_CALENDAR: list[NewsEvent] | None = None
_CALENDAR_TIMES: list[datetime] | None = None


def reload_calendar(csv_path: Path | None = None) -> int:
    """Force-reload calendar from disk. Returns number of events."""
    global _CALENDAR, _CALENDAR_TIMES
    _CALENDAR = load_calendar(csv_path)
    _CALENDAR_TIMES = [e.ts_utc for e in _CALENDAR]
    return len(_CALENDAR)


def _ensure_calendar() -> tuple[list[NewsEvent], list[datetime]]:
    if _CALENDAR is None or _CALENDAR_TIMES is None:
        reload_calendar()
    assert _CALENDAR is not None and _CALENDAR_TIMES is not None
    return _CALENDAR, _CALENDAR_TIMES


def is_news_blackout(
    timestamp: datetime,
    blackout_minutes_before: int = 15,
    blackout_minutes_after: int = 30,
) -> bool:
    """Return ``True`` iff ``timestamp`` falls inside a blackout window.

    Parameters
    ----------
    timestamp : datetime
        UTC-naive candle timestamp (typically M5 bar close). Convert
        from broker-local time before calling.
    blackout_minutes_before : int, default 15
        Minutes before release to start the blackout (captures
        pre-release spread widening).
    blackout_minutes_after : int, default 30
        Minutes after release to end the blackout (covers initial spike
        + dealer inventory laydown; post-30min is the drift window
        tradable by the news-drift archetype WG2 #3).
    """
    _, times = _ensure_calendar()
    if not times:
        return False
    lo = timestamp - timedelta(minutes=blackout_minutes_after)
    hi = timestamp + timedelta(minutes=blackout_minutes_before)
    idx = bisect.bisect_left(times, lo)
    if idx >= len(times):
        return False
    return times[idx] <= hi


__all__ = [
    "HIGH_IMPACT_CURRENCIES",
    "NewsEvent",
    "is_news_blackout",
    "load_calendar",
    "reload_calendar",
]
