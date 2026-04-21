"""Quick classifier tests for multi_instrument_null — M113 scenario replay."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multi_instrument_null import InstrumentResult, classify


def _res(sym, pf=None, dd=None, trades=500, err=None):
    return InstrumentResult(symbol=sym, pf=pf, dd_pct=dd, trades=trades,
                            net_profit=0, error=err)


def test_tier_c_m113_replay():
    """Exact M113 2026-04-21 result set: EUR 0 trades, others wiped."""
    base = _res("EURUSD", pf=None, dd=0, trades=0)
    tests = [
        _res("GBPUSD", pf=0.23, dd=99.85, trades=816),
        _res("USDJPY", pf=0.04, dd=99.81, trades=672),
        _res("XAGUSD", pf=0.49, dd=96.44, trades=418),
    ]
    v = classify(base, tests)
    assert v.tier == "C", f"M113 must be Tier C; got {v.tier}"
    assert "M113" in v.reason
    assert len(v.warnings) >= 3  # wipe warnings for 3 test instruments


def test_tier_a_structural():
    base = _res("EURUSD", pf=1.35, dd=8, trades=2000)
    tests = [
        _res("GBPUSD", pf=1.20, dd=10, trades=1800),
        _res("USDJPY", pf=1.15, dd=12, trades=1700),
        _res("XAGUSD", pf=1.40, dd=15, trades=900),
    ]
    v = classify(base, tests)
    assert v.tier == "A", v.reason
    assert v.classification == "ACCEPT"


def test_tier_b_conditional():
    base = _res("EURUSD", pf=1.35, dd=8)
    tests = [
        _res("GBPUSD", pf=1.10, dd=15),
        _res("USDJPY", pf=1.05, dd=18),
        _res("XAGUSD", pf=0.95, dd=22),  # one fails but no wipe
    ]
    v = classify(base, tests)
    assert v.tier == "B", v.reason


def test_zero_trades_is_fail():
    base = _res("EURUSD", pf=1.5, dd=5, trades=1000)
    tests = [
        _res("GBPUSD", pf=None, dd=0, trades=0),    # M113 signature
        _res("USDJPY", pf=None, dd=0, trades=0),
        _res("XAGUSD", pf=1.3, dd=18, trades=500),
    ]
    v = classify(base, tests)
    assert v.tier == "C"
    assert any("zero trades" in w for w in v.warnings)


if __name__ == "__main__":
    test_tier_c_m113_replay()
    test_tier_a_structural()
    test_tier_b_conditional()
    test_zero_trades_is_fail()
    print("all 4 classifier tests PASS")
