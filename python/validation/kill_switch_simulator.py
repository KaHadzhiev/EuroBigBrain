"""Kill-switch simulator (mirror of mql5/EBB/KillSwitch.mqh).

Replays a trade log through the dual-layer kill-switch and reports
whether / when the switch would have tripped on historical drawdowns.

Layer A (daily): halts NEW entries when today's closed PnL drops below
    -daily_loss_cap_pct of start-of-day equity. Resets each calendar day.
Layer B (acct):  halts ALL trading when equity drops below
    peak * (1 - account_dd_cap_pct). Latched; manual reset only.

Test suite: 2%-daily trip, 8%-account trip, M113-like 96% wipe,
Unchained-like 35% DD, manual override bypass.

CLI: python -m validation.kill_switch_simulator --log trades.csv
CSV schema: close_time (ISO), pnl (float, net of costs).

Per WG4 2026-04-21: daily=2%, account=8%. Research mode only.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import random
import sys
from pathlib import Path
from typing import Iterable


@dataclasses.dataclass
class KillSwitchParams:
    daily_loss_cap_pct: float = 2.0
    account_dd_cap_pct: float = 8.0
    consec_loss_brake: int = 5
    consec_brake_minutes: int = 240
    manual_override: bool = False


@dataclasses.dataclass
class Trade:
    close_time: dt.datetime
    pnl: float          # net USD after costs


@dataclasses.dataclass
class ReplayResult:
    n_trades_seen: int
    n_trades_admitted: int
    layer_a_trips: int
    layer_b_tripped_at: dt.datetime | None
    final_equity: float
    peak_equity: float
    max_observed_dd_pct: float
    consec_brake_trips: int

    @property
    def halted_by_layer_b(self) -> bool:
        return self.layer_b_tripped_at is not None


def _day_key(t: dt.datetime) -> dt.date:
    return t.date()


def simulate(trades: Iterable[Trade], starting_equity: float,
             params: KillSwitchParams | None = None) -> ReplayResult:
    """Replay trades through the dual-layer kill switch.

    Semantics mirror KillSwitch.mqh: Layer B (latching) checked first,
    then Layer A, then consec-brake. A blocked trade contributes nothing
    (no PnL, no consec counter). Peak equity updates only on admitted PnL.
    """
    p = params or KillSwitchParams()
    equity = peak_equity = sod_equity = starting_equity
    today: dt.date | None = None
    today_pnl = 0.0
    consec_losses = 0
    brake_until: dt.datetime | None = None
    layer_b_at: dt.datetime | None = None
    layer_a_trips = consec_brake_trips = n_seen = n_admitted = 0
    max_dd = 0.0

    for trade in trades:
        n_seen += 1
        day = _day_key(trade.close_time)
        if today is None or day != today:
            today, sod_equity, today_pnl = day, equity, 0.0

        allow = True
        if p.manual_override:
            pass
        elif layer_b_at is not None:
            allow = False
        else:
            dd_pct = 100.0 * (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            daily_dd = -100.0 * today_pnl / sod_equity if sod_equity > 0 else 0.0
            if dd_pct >= p.account_dd_cap_pct:
                layer_b_at, allow = trade.close_time, False
            elif daily_dd >= p.daily_loss_cap_pct:
                layer_a_trips += 1; allow = False
            elif brake_until is not None and trade.close_time < brake_until:
                consec_brake_trips += 1; allow = False

        if not allow:
            continue

        n_admitted += 1
        equity += trade.pnl
        today_pnl += trade.pnl
        if equity > peak_equity:
            peak_equity = equity
        cur_dd = 100.0 * (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if cur_dd > max_dd: max_dd = cur_dd

        if trade.pnl < 0:
            consec_losses += 1
            if consec_losses >= p.consec_loss_brake:
                brake_until = trade.close_time + dt.timedelta(minutes=p.consec_brake_minutes)
        elif trade.pnl > 0:
            consec_losses = 0

        # Re-check Layer B so this trade wiping through 8% latches now.
        if not p.manual_override and layer_b_at is None and peak_equity > 0 and \
           100.0 * (peak_equity - equity) / peak_equity >= p.account_dd_cap_pct:
            layer_b_at = trade.close_time

    return ReplayResult(n_trades_seen=n_seen, n_trades_admitted=n_admitted,
                        layer_a_trips=layer_a_trips, layer_b_tripped_at=layer_b_at,
                        final_equity=equity, peak_equity=peak_equity,
                        max_observed_dd_pct=max_dd,
                        consec_brake_trips=consec_brake_trips)


def load_trades_csv(path: Path) -> list[Trade]:
    trades: list[Trade] = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ct = dt.datetime.fromisoformat(row["close_time"])
            pnl = float(row["pnl"])
            trades.append(Trade(close_time=ct, pnl=pnl))
    return trades


# -------- synthetic fixtures for the test suite -------------------------

def _synth_walk(rng: random.Random, n: int, start: dt.datetime, mu: float, sigma: float,
                step_minutes: int = 60) -> list[Trade]:
    out = []
    t = start
    for _ in range(n):
        pnl = rng.gauss(mu, sigma)
        out.append(Trade(close_time=t, pnl=pnl))
        t += dt.timedelta(minutes=step_minutes)
    return out


def _synth_drawdown(rng: random.Random, eq: float, dd_pct: float, n_losers: int,
                    start: dt.datetime) -> list[Trade]:
    """Produce a losing streak whose cumulative PnL = -dd_pct% of eq."""
    total = -eq * dd_pct / 100.0
    per = total / n_losers
    out = []
    t = start
    for _ in range(n_losers):
        out.append(Trade(close_time=t, pnl=per + rng.gauss(0, abs(per) * 0.1)))
        t += dt.timedelta(minutes=5)
    return out


def test_synthetic_2pct_daily():
    """Layer A must trip when a single day crosses -2%."""
    rng = random.Random(42)
    eq0 = 10_000.0
    start = dt.datetime(2026, 4, 21, 9, 0)
    trades = _synth_drawdown(rng, eq0, dd_pct=3.0, n_losers=6, start=start)
    r = simulate(trades, eq0)
    assert r.layer_a_trips >= 1, f"Layer A should trip, got {r}"
    assert r.n_trades_admitted < r.n_trades_seen
    print(f"  [PASS] 2pct_daily: {r.layer_a_trips} trips, "
          f"admitted {r.n_trades_admitted}/{r.n_trades_seen}")


def test_synthetic_8pct_account():
    """Layer B must latch near 8% cumulative DD."""
    rng = random.Random(7)
    eq0 = 10_000.0
    start = dt.datetime(2026, 4, 21, 9, 0)
    trades, t = [], start
    per_day = 4
    per_loss = -eq0 * 0.015 / per_day  # 1.5%/day < 2% cap
    for d in range(20):
        for _ in range(per_day):
            trades.append(Trade(close_time=t, pnl=per_loss + rng.gauss(0, abs(per_loss) * 0.1)))
            t += dt.timedelta(hours=2)
        t = start + dt.timedelta(days=d + 1)
    r = simulate(trades, eq0)
    assert r.halted_by_layer_b, f"Layer B should latch, got {r}"
    assert r.max_observed_dd_pct < 12.0, f"DD should be ~8%, got {r.max_observed_dd_pct:.2f}%"
    print(f"  [PASS] 8pct_account: tripped at {r.layer_b_tripped_at}, "
          f"maxDD {r.max_observed_dd_pct:.2f}%")


def test_m113_like_wipe():
    """XAU-style 96% wipe must trip Layer B far before trough."""
    rng = random.Random(113)
    eq0 = 10_000.0
    start = dt.datetime(2026, 1, 1, 9, 0)
    trades, t = [], start
    per_loss = -eq0 * 0.96 / 200
    for _ in range(200):
        trades.append(Trade(close_time=t, pnl=per_loss + rng.gauss(0, abs(per_loss) * 0.2)))
        t += dt.timedelta(hours=4, minutes=48)  # ~5 per day, avoids Layer A on most days
    r = simulate(trades, eq0)
    assert r.halted_by_layer_b, "Layer B must trip on 96% wipe"
    assert r.final_equity > eq0 * 0.85, f"Wipe not halted: final=${r.final_equity:.0f}"
    saved = eq0 * 0.96 - (eq0 - r.final_equity)
    print(f"  [PASS] m113_wipe: tripped at {r.layer_b_tripped_at}, "
          f"final=${r.final_equity:.0f} saved=${saved:.0f}")


def test_unchained_like_35pct():
    """Unchained 35% DD must trip Layer B well before trough."""
    rng = random.Random(35)
    eq0 = 10_000.0
    start = dt.datetime(2026, 1, 1, 9, 0)
    trades: list[Trade] = []
    t = start
    for _ in range(100):
        trades.append(Trade(close_time=t, pnl=rng.gauss(50, 20)))
        t += dt.timedelta(hours=3)
    for _ in range(60):
        trades.append(Trade(close_time=t, pnl=rng.gauss(-90, 25)))
        t += dt.timedelta(hours=3)
    r = simulate(trades, eq0)
    assert r.halted_by_layer_b, "Layer B must trip on 35% DD"
    assert r.max_observed_dd_pct < 15.0, f"Expected ~8%, got {r.max_observed_dd_pct:.2f}%"
    print(f"  [PASS] unchained_35pct: tripped {r.layer_b_tripped_at}, "
          f"maxDD {r.max_observed_dd_pct:.2f}% final=${r.final_equity:.0f}")


def test_manual_override():
    """Override must let all trades through (test harness only)."""
    rng = random.Random(99)
    eq0 = 10_000.0
    start = dt.datetime(2026, 4, 21, 9, 0)
    trades = _synth_drawdown(rng, eq0, dd_pct=50.0, n_losers=100, start=start)
    r = simulate(trades, eq0, KillSwitchParams(manual_override=True))
    assert r.layer_b_tripped_at is None
    assert r.n_trades_admitted == r.n_trades_seen
    print(f"  [PASS] override: all {r.n_trades_seen} admitted final=${r.final_equity:.0f}")


def run_tests() -> int:
    tests = [test_synthetic_2pct_daily, test_synthetic_8pct_account,
             test_m113_like_wipe, test_unchained_like_35pct, test_manual_override]
    failed = 0
    print("Kill-switch simulator — running test suite")
    for tc in tests:
        try: tc()
        except AssertionError as e:
            failed += 1; print(f"  [FAIL] {tc.__name__}: {e}")
    print("ALL PASS" if failed == 0 else f"{failed} FAILED")
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=None,
                        help="CSV trade log: close_time,pnl")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--daily-cap", type=float, default=2.0)
    parser.add_argument("--acct-cap", type=float, default=8.0)
    parser.add_argument("--test", action="store_true", help="run the test suite")
    args = parser.parse_args(argv)

    if args.test or args.log is None:
        return run_tests()

    trades = load_trades_csv(args.log)
    params = KillSwitchParams(daily_loss_cap_pct=args.daily_cap,
                              account_dd_cap_pct=args.acct_cap)
    r = simulate(trades, args.equity, params)
    print(f"trades={r.n_trades_seen} admitted={r.n_trades_admitted} "
          f"layerA_trips={r.layer_a_trips} layerB_at={r.layer_b_tripped_at} "
          f"peak=${r.peak_equity:.2f} final=${r.final_equity:.2f} "
          f"maxDD={r.max_observed_dd_pct:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
