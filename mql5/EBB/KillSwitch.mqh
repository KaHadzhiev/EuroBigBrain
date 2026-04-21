//+------------------------------------------------------------------+
//| EBB/KillSwitch.mqh — dual-layer kill switch for EuroBigBrain     |
//| Layer A (daily): halt NEW entries when today PnL < -2% of SoD eq |
//| Layer B (acct):  halt ALL + close positions when DD from peak    |
//|                  equity >= 8%. Latched; requires manual reset.    |
//| Persists across EA restarts via GlobalVariables. WG4 2026-04-21. |
//+------------------------------------------------------------------+
#ifndef __EBB_KILLSWITCH_MQH__
#define __EBB_KILLSWITCH_MQH__

// --- tunables (passed via EA inputs; defaults match WG4 spec) ---
struct EBB_KillParams
{
   double daily_loss_cap_pct;     // default 2.0
   double account_dd_cap_pct;     // default 8.0
   int    consec_loss_brake;      // default 5 (pause 4h)
   int    consec_brake_minutes;   // default 240
   bool   manual_override;        // default false — TEST ONLY
   int    magic_number;           // for filtering deals & positions
   string gv_prefix;              // GlobalVariable namespace, e.g. "EBB_KS_"
};

EBB_KillParams g_ks;

// persistent state (mirrored to GlobalVariables on every update)
double   g_ks_startOfDayEquity = 0.0;
double   g_ks_peakEquity       = 0.0;
datetime g_ks_currentDay       = 0;
int      g_ks_consecLosses     = 0;
datetime g_ks_consecBrakeUntil = 0;
bool     g_ks_accountHalted    = false; // latched; reset by operator
double   g_ks_todayPnL         = 0.0;

// --- GV keys ---
string KS_Key(string suffix) { return g_ks.gv_prefix + suffix; }

void KS_PersistState()
{
   GlobalVariableSet(KS_Key("sod_equity"),   g_ks_startOfDayEquity);
   GlobalVariableSet(KS_Key("peak_equity"),  g_ks_peakEquity);
   GlobalVariableSet(KS_Key("current_day"),  (double)g_ks_currentDay);
   GlobalVariableSet(KS_Key("consec_loss"),  (double)g_ks_consecLosses);
   GlobalVariableSet(KS_Key("brake_until"),  (double)g_ks_consecBrakeUntil);
   GlobalVariableSet(KS_Key("acct_halt"),    g_ks_accountHalted ? 1.0 : 0.0);
   GlobalVariableSet(KS_Key("today_pnl"),    g_ks_todayPnL);
}

void KS_RestoreState()
{
   if(GlobalVariableCheck(KS_Key("sod_equity")))
      g_ks_startOfDayEquity = GlobalVariableGet(KS_Key("sod_equity"));
   if(GlobalVariableCheck(KS_Key("peak_equity")))
      g_ks_peakEquity = GlobalVariableGet(KS_Key("peak_equity"));
   if(GlobalVariableCheck(KS_Key("current_day")))
      g_ks_currentDay = (datetime)GlobalVariableGet(KS_Key("current_day"));
   if(GlobalVariableCheck(KS_Key("consec_loss")))
      g_ks_consecLosses = (int)GlobalVariableGet(KS_Key("consec_loss"));
   if(GlobalVariableCheck(KS_Key("brake_until")))
      g_ks_consecBrakeUntil = (datetime)GlobalVariableGet(KS_Key("brake_until"));
   if(GlobalVariableCheck(KS_Key("acct_halt")))
      g_ks_accountHalted = (GlobalVariableGet(KS_Key("acct_halt")) > 0.5);
   if(GlobalVariableCheck(KS_Key("today_pnl")))
      g_ks_todayPnL = GlobalVariableGet(KS_Key("today_pnl"));
}

datetime KS_DayStart(datetime t)
{
   MqlDateTime s; TimeToStruct(t, s);
   s.hour = 0; s.min = 0; s.sec = 0;
   return StructToTime(s);
}

void KS_Init(EBB_KillParams& params)
{
   g_ks = params;
   if(g_ks.gv_prefix == "") g_ks.gv_prefix = "EBB_KS_";
   KS_RestoreState();
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(g_ks_peakEquity <= 0.0)       g_ks_peakEquity = eq;
   if(g_ks_startOfDayEquity <= 0.0) g_ks_startOfDayEquity = eq;
   if(g_ks_currentDay == 0)         g_ks_currentDay = KS_DayStart(TimeCurrent());
   KS_PersistState();
   PrintFormat("[KillSwitch] init eq=%.2f peak=%.2f halted=%s override=%s",
               eq, g_ks_peakEquity, g_ks_accountHalted?"Y":"n",
               g_ks.manual_override?"ON":"off");
}

void KS_RolloverDay(datetime now)
{
   datetime d = KS_DayStart(now);
   if(d != g_ks_currentDay)
   {
      g_ks_currentDay = d;
      g_ks_startOfDayEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      g_ks_todayPnL = 0.0;
      KS_PersistState();
   }
}

// Called on every tick. Updates peak equity and rolls over the day.
void KS_Heartbeat()
{
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq > g_ks_peakEquity)
   {
      g_ks_peakEquity = eq;
      KS_PersistState();
   }
   KS_RolloverDay(TimeCurrent());
}

// Called from OnTradeTransaction for every closing deal (entry == OUT).
void KS_OnTradeClose(double net_pnl)
{
   g_ks_todayPnL += net_pnl;

   if(net_pnl < 0.0)
   {
      g_ks_consecLosses++;
      if(g_ks_consecLosses >= g_ks.consec_loss_brake)
         g_ks_consecBrakeUntil = TimeCurrent() + g_ks.consec_brake_minutes * 60;
   }
   else if(net_pnl > 0.0)
   {
      g_ks_consecLosses = 0;
   }
   KS_PersistState();
}

// --- predicates used by Core before any entry decision ---

bool KS_DailyLossBreached()
{
   if(g_ks_startOfDayEquity <= 0.0) return false;
   double dd_pct = (-g_ks_todayPnL / g_ks_startOfDayEquity) * 100.0;
   return (dd_pct >= g_ks.daily_loss_cap_pct);
}

bool KS_AccountDDBreached()
{
   if(g_ks_peakEquity <= 0.0) return false;
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd_pct = ((g_ks_peakEquity - eq) / g_ks_peakEquity) * 100.0;
   return (dd_pct >= g_ks.account_dd_cap_pct);
}

bool KS_ConsecBrakeActive()
{
   return (TimeCurrent() < g_ks_consecBrakeUntil);
}

// The single checkpoint Core calls. Returns true if trading is allowed.
// If Layer B (account DD) trips, latches halted state until manual reset.
bool KS_AllowNewEntry()
{
   if(g_ks.manual_override) return true;
   if(g_ks_accountHalted)   return false;

   if(KS_AccountDDBreached())
   {
      g_ks_accountHalted = true;
      KS_PersistState();
      PrintFormat("[KillSwitch] LAYER B TRIP eq=%.2f peak=%.2f cap=%.2f%% HALT",
                  AccountInfoDouble(ACCOUNT_EQUITY), g_ks_peakEquity,
                  g_ks.account_dd_cap_pct);
      return false;
   }
   if(KS_DailyLossBreached())
   {
      PrintFormat("[KillSwitch] LAYER A TRIP pnl=%.2f sod=%.2f cap=%.2f%%",
                  g_ks_todayPnL, g_ks_startOfDayEquity, g_ks.daily_loss_cap_pct);
      return false;
   }
   if(KS_ConsecBrakeActive())
   {
      PrintFormat("[KillSwitch] consec-brake until %s",
                  TimeToString(g_ks_consecBrakeUntil, TIME_DATE|TIME_MINUTES));
      return false;
   }
   return true;
}

// Called by Core only when Layer B trips — close all positions for this EA.
void KS_CloseAllForMagic()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != g_ks.magic_number) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      MqlTradeRequest req; MqlTradeResult res;
      ZeroMemory(req); ZeroMemory(res);
      req.action   = TRADE_ACTION_DEAL;
      req.position = ticket;
      req.symbol   = _Symbol;
      req.volume   = PositionGetDouble(POSITION_VOLUME);
      req.type     = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                     ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
      req.price    = (req.type == ORDER_TYPE_BUY)
                     ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                     : SymbolInfoDouble(_Symbol, SYMBOL_BID);
      req.magic    = g_ks.magic_number;
      req.comment  = "KS_halt";
      req.deviation = 20;
      req.type_filling = ORDER_FILLING_IOC;
      OrderSend(req, res);
   }
}

// Manual reset — operator-only. Intended to be called from a script, or
// via input param + OnInit path after the operator confirms the halt.
void KS_ResetAccountHalt()
{
   g_ks_accountHalted = false;
   g_ks_peakEquity    = AccountInfoDouble(ACCOUNT_EQUITY);
   KS_PersistState();
   Print("[KillSwitch] account halt MANUALLY RESET — new peak = current equity");
}

// Debug snapshot for journaling / telemetry.
string KS_StatusLine()
{
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   double daily = (g_ks_startOfDayEquity > 0.0)
                  ? (g_ks_todayPnL / g_ks_startOfDayEquity) * 100.0 : 0.0;
   double acct = (g_ks_peakEquity > 0.0)
                 ? ((g_ks_peakEquity - eq) / g_ks_peakEquity) * 100.0 : 0.0;
   return StringFormat("KS: eq=%.2f sod=%.2f peak=%.2f daily=%.2f%% acctDD=%.2f%% halted=%s",
                       eq, g_ks_startOfDayEquity, g_ks_peakEquity,
                       daily, acct, g_ks_accountHalted ? "YES" : "no");
}

#endif
