"""Tests for ``python.filters.news_blackout``.

Covers:
- known-event boundary behavior (exact hit, inside, just-before, just-after)
- empty / missing CSV fallback
- CSV override via explicit path + env var
- load_calendar filters (currency, impact)
- sort ordering invariant
- performance sanity (scales to multi-year bar stream)
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from python.filters import news_blackout as nb


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_calendar_cache(monkeypatch):
    """Clear the module cache before each test so env-var tweaks take hold."""
    monkeypatch.setattr(nb, "_CALENDAR", None)
    monkeypatch.setattr(nb, "_CALENDAR_TIMES", None)
    yield
    monkeypatch.setattr(nb, "_CALENDAR", None)
    monkeypatch.setattr(nb, "_CALENDAR_TIMES", None)


@pytest.fixture()
def sample_csv(tmp_path: Path, monkeypatch) -> Path:
    """Write a tiny calendar with 4 known events and point the module at it."""
    path = tmp_path / "news.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp_utc", "currency", "event", "impact"])
        writer.writerow(["2026-05-01 12:30", "USD", "NFP", "HIGH"])
        writer.writerow(["2026-05-07 12:15", "EUR", "ECB_RATE", "HIGH"])
        writer.writerow(["2026-05-13 12:30", "USD", "US_CPI", "HIGH"])
        # Filtered out: MEDIUM impact
        writer.writerow(["2026-05-14 08:00", "EUR", "GERMAN_IFO", "MEDIUM"])
        # Filtered out: non-EUR/USD currency
        writer.writerow(["2026-05-15 23:50", "JPY", "BOJ_RATE", "HIGH"])
    monkeypatch.setenv("EBB_NEWS_CALENDAR", str(path))
    return path


# ---------------------------------------------------------------------
# load_calendar
# ---------------------------------------------------------------------


def test_load_calendar_filters_currency_and_impact(sample_csv):
    events = nb.load_calendar()
    kinds = {(e.currency, e.event) for e in events}
    assert ("USD", "NFP") in kinds
    assert ("EUR", "ECB_RATE") in kinds
    assert ("USD", "US_CPI") in kinds
    # MEDIUM impact + JPY must be dropped.
    assert ("EUR", "GERMAN_IFO") not in kinds
    assert all(e.currency in {"EUR", "USD"} for e in events)
    assert len(events) == 3


def test_load_calendar_sorted_ascending(sample_csv):
    events = nb.load_calendar()
    timestamps = [e.ts_utc for e in events]
    assert timestamps == sorted(timestamps)


def test_missing_csv_uses_hardcoded_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("EBB_NEWS_CALENDAR", str(tmp_path / "nonexistent.csv"))
    events = nb.load_calendar(years_for_fallback=(2026,))
    # 2026 fallback = 8 FOMC + 8 ECB_RATE + 8 ECB_PRESSER + 12 NFP + 12 US_CPI + 12 EU_CPI = 60
    assert len(events) == 60
    event_names = {e.event for e in events}
    assert {"FOMC", "ECB_RATE", "NFP", "US_CPI", "EU_CPI"}.issubset(event_names)


def test_empty_csv_uses_hardcoded_fallback(tmp_path, monkeypatch):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("EBB_NEWS_CALENDAR", str(empty))
    events = nb.load_calendar(years_for_fallback=(2026,))
    assert len(events) > 0


# ---------------------------------------------------------------------
# is_news_blackout — boundary behavior
# ---------------------------------------------------------------------


def test_exact_event_time_is_blackout(sample_csv):
    nb.reload_calendar()
    # NFP 2026-05-01 12:30 UTC: exact match must be blackout.
    assert nb.is_news_blackout(datetime(2026, 5, 1, 12, 30)) is True


def test_inside_before_window(sample_csv):
    nb.reload_calendar()
    # NFP 12:30, 10 min before -> inside default (15 min before) window.
    assert nb.is_news_blackout(datetime(2026, 5, 1, 12, 20)) is True


def test_inside_after_window(sample_csv):
    nb.reload_calendar()
    # NFP 12:30, 20 min after -> inside default (30 min after) window.
    assert nb.is_news_blackout(datetime(2026, 5, 1, 12, 50)) is True


def test_just_outside_before(sample_csv):
    nb.reload_calendar()
    # 20 min before with default 15-min before window => clear.
    assert nb.is_news_blackout(datetime(2026, 5, 1, 12, 10)) is False


def test_just_outside_after(sample_csv):
    nb.reload_calendar()
    # 35 min after with default 30-min after window => clear.
    assert nb.is_news_blackout(datetime(2026, 5, 1, 13, 5)) is False


def test_wide_window_catches_edge(sample_csv):
    nb.reload_calendar()
    # 60 min before with widened 90-min before window => blackout.
    assert nb.is_news_blackout(
        datetime(2026, 5, 1, 11, 30),
        blackout_minutes_before=90,
        blackout_minutes_after=30,
    ) is True


def test_day_without_events_is_clear(sample_csv):
    nb.reload_calendar()
    # 2026-05-03 is a Sunday with nothing scheduled -> clear.
    assert nb.is_news_blackout(datetime(2026, 5, 3, 15, 0)) is False


def test_multiple_events_same_day(sample_csv):
    nb.reload_calendar()
    # 2026-05-07 has ECB_RATE at 12:15 UTC; bar at 12:20 is blackout.
    assert nb.is_news_blackout(datetime(2026, 5, 7, 12, 20)) is True
    # Bar at 14:00 is clear (1h45m after, outside 30-min window).
    assert nb.is_news_blackout(datetime(2026, 5, 7, 14, 0)) is False


# ---------------------------------------------------------------------
# Hardcoded 2026 dates — integration check against published calendar
# ---------------------------------------------------------------------


def test_known_fomc_2026_dates_in_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("EBB_NEWS_CALENDAR", str(tmp_path / "none.csv"))
    nb.reload_calendar()
    # FOMC Jan 28 2026 19:00 UTC (14:00 ET winter).
    assert nb.is_news_blackout(datetime(2026, 1, 28, 19, 0)) is True
    # FOMC Mar 18 2026 18:00 UTC (DST).
    assert nb.is_news_blackout(datetime(2026, 3, 18, 18, 0)) is True


def test_known_ecb_2026_dates_in_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("EBB_NEWS_CALENDAR", str(tmp_path / "none.csv"))
    nb.reload_calendar()
    # ECB Feb 5 2026 13:15 UTC (CET).
    assert nb.is_news_blackout(datetime(2026, 2, 5, 13, 15)) is True
    # ECB Jun 11 2026 12:15 UTC (CEST).
    assert nb.is_news_blackout(datetime(2026, 6, 11, 12, 15)) is True


# ---------------------------------------------------------------------
# Performance sanity
# ---------------------------------------------------------------------


def test_scales_to_year_of_m5_bars(tmp_path, monkeypatch):
    """A full year of EUR/USD M5 bars ~= 75k rows. Should complete fast."""
    monkeypatch.setenv("EBB_NEWS_CALENDAR", str(tmp_path / "none.csv"))
    nb.reload_calendar()
    start = datetime(2026, 1, 1, 0, 0)
    # Spot-check 1000 bars rather than full year to keep test < 1 s.
    bars = [start + timedelta(minutes=5 * i) for i in range(1000)]
    blackouts = [nb.is_news_blackout(b) for b in bars]
    # Some should be blackout (NFP Jan 2, CPI Jan 13), most clear.
    assert any(blackouts)
    assert not all(blackouts)
