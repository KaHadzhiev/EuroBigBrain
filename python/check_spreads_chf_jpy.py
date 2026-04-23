#!/usr/bin/env python3
"""Live Vantage spread check for USDCHF + USDJPY before we sweep them."""
import sys, subprocess
from datetime import datetime
import MetaTrader5 as mt5

if not mt5.initialize():
    print(f"FAIL: {mt5.last_error()}"); sys.exit(1)

ai = mt5.account_info()
ti = mt5.terminal_info()
print(f"Connected: {ai.login}@{ai.server}  ({ti.path})")

for sym in ["EURUSD", "GBPUSD", "USDCHF", "USDJPY"]:
    mt5.symbol_select(sym, True)
    s = mt5.symbol_info(sym)
    t = mt5.symbol_info_tick(sym)
    if s is None or t is None:
        print(f"  {sym}: NOT AVAILABLE")
        continue
    pip_size = 10 ** -(s.digits - 1)  # 5-digit -> 0.0001, 3-digit -> 0.01
    spread_pips = s.spread / 10  # spread in points; 10 points = 1 pip on standard
    live_spread_pips = (t.ask - t.bid) / pip_size
    print(f"  {sym}  digits={s.digits}  pip_size={pip_size}  spread={s.spread}pt = {spread_pips:.1f} pip  live ask-bid = {live_spread_pips:.1f} pip  bid={t.bid}")

mt5.shutdown()
# Kill the spawned terminal per HARD RULE
import time; time.sleep(1)
subprocess.run(['taskkill', '/F', '/IM', 'terminal64.exe', '/FI', f'WINDOWTITLE eq {ai.login} - VantageInternational-Demo*'],
                capture_output=True)
print("done (terminal killed)")
