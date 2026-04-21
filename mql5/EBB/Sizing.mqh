//+------------------------------------------------------------------+
//| EBB/Sizing.mqh — quarter-Kelly x vol-target, capped 0.5%/trade.   |
//|   base_risk=0.5%, kelly_frac=max(0,(WA-LB)/A)/4,                  |
//|   vol_scalar=target_vol/realized_vol clipped to [0.5,2.0],        |
//|   position_risk=min(base, kelly_frac*base)*vol_scalar (hard cap). |
//|   Fallback 0.3% fixed when n<200 trades or vol missing.           |
//+------------------------------------------------------------------+
#ifndef __EBB_SIZING_MQH__
#define __EBB_SIZING_MQH__

// Packed configuration for the sizer — passed in from Core at OnInit.
struct EBB_SizeParams
{
   double base_risk_pct;       // hard cap, default 0.5
   double fallback_risk_pct;   // used when kelly_frac <= 0 or vol fails, default 0.3
   double kelly_multiplier;    // default 0.25 (quarter-Kelly)
   double vol_scalar_min;      // default 0.5
   double vol_scalar_max;      // default 2.0
   double min_trades_for_kelly;// default 200 — below this, use fallback
   double max_lot_size;        // absolute cap, default 1.0
};

EBB_SizeParams g_sp;

void Sizing_Init(EBB_SizeParams& params)
{
   g_sp = params;
   if(g_sp.base_risk_pct       <= 0) g_sp.base_risk_pct = 0.5;
   if(g_sp.fallback_risk_pct   <= 0) g_sp.fallback_risk_pct = 0.3;
   if(g_sp.kelly_multiplier    <= 0) g_sp.kelly_multiplier = 0.25;
   if(g_sp.vol_scalar_min      <= 0) g_sp.vol_scalar_min = 0.5;
   if(g_sp.vol_scalar_max      <= 0) g_sp.vol_scalar_max = 2.0;
   if(g_sp.min_trades_for_kelly<= 0) g_sp.min_trades_for_kelly = 200;
   if(g_sp.max_lot_size        <= 0) g_sp.max_lot_size = 1.0;
}

// Compute the Kelly fraction f* = (W*A - L*B) / A where
// W = win rate, L = 1 - W, A = avg win magnitude, B = avg loss magnitude.
// Returns 0 if measurement window is too small or edge is non-positive.
double Sizing_KellyFraction(int    win_count,
                            int    loss_count,
                            double avg_win_R,
                            double avg_loss_R)
{
   int n = win_count + loss_count;
   if(n < g_sp.min_trades_for_kelly) return 0.0;
   if(avg_win_R <= 0.0 || avg_loss_R <= 0.0) return 0.0;

   double W = (double)win_count / (double)n;
   double L = 1.0 - W;
   double kelly = (W * avg_win_R - L * avg_loss_R) / avg_win_R;
   if(kelly <= 0.0) return 0.0;
   return kelly;
}

// Clip the vol scalar and handle the vol-measurement-failure fallback.
// realized_vol and target_vol should be in the same unit (e.g. daily σ
// of returns in % terms, or ATR pips — as long as consistent).
double Sizing_VolScalar(double target_vol, double realized_vol)
{
   if(target_vol <= 0.0 || realized_vol <= 0.0) return 1.0; // neutral
   double s = target_vol / realized_vol;
   if(s < g_sp.vol_scalar_min) s = g_sp.vol_scalar_min;
   if(s > g_sp.vol_scalar_max) s = g_sp.vol_scalar_max;
   return s;
}

// Main sizing entry point. Returns the lot size, broker-rounded and
// constrained by min/max lot.
//
// Parameters:
//   account_equity  — current equity (USD-denominated account)
//   sl_pts          — stop distance in price points (already ATR-scaled by caller)
//   kelly_fraction  — from Sizing_KellyFraction()  (0 disables Kelly, falls back to base_risk_pct)
//   vol_scalar      — from Sizing_VolScalar()      (1.0 = neutral)
//
// The formula applied:
//   if kelly == 0:
//      risk_pct = fallback_risk_pct * vol_scalar
//   else:
//      raw  = kelly * kelly_multiplier * 100   # express as pct
//      risk_pct = min(base_risk_pct, raw) * vol_scalar
//   risk_pct is ALWAYS clipped at base_risk_pct (hard cap).
double Sizing_CalculateLotSize(double account_equity,
                               double sl_pts,
                               double kelly_fraction,
                               double vol_scalar)
{
   if(account_equity <= 0.0 || sl_pts <= 0.0) return 0.0;

   double risk_pct;
   if(kelly_fraction <= 0.0)
   {
      risk_pct = g_sp.fallback_risk_pct;
   }
   else
   {
      // quarter-Kelly: scale the optimal fraction down and convert to %.
      // kelly_fraction is a unitless bet-size fraction (e.g. 0.25 = 25%).
      // Quarter-Kelly of that on the base_risk budget caps conservatively.
      double raw_pct = kelly_fraction * g_sp.kelly_multiplier * 100.0;
      risk_pct = MathMin(g_sp.base_risk_pct, raw_pct);
   }

   // vol overlay
   risk_pct *= vol_scalar;

   // hard cap — CRITICAL SAFETY INVARIANT. Even if kelly+vol push higher,
   // we never risk more than base_risk_pct on a single trade.
   if(risk_pct > g_sp.base_risk_pct) risk_pct = g_sp.base_risk_pct;
   if(risk_pct < 0.0)                risk_pct = 0.0;

   double risk_usd  = account_equity * risk_pct / 100.0;
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double lotStep   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLotBrk = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   if(tickSize <= 0.0 || tickValue <= 0.0 || lotStep <= 0.0) return minLot;

   double slTicks = sl_pts / tickSize;
   if(slTicks <= 0.0) return minLot;

   double lot = risk_usd / (slTicks * tickValue);
   lot = MathFloor(lot / lotStep) * lotStep;
   lot = MathMax(lot, minLot);
   lot = MathMin(lot, g_sp.max_lot_size);
   lot = MathMin(lot, maxLotBrk);
   return lot;
}

// Convenience wrapper: everything the Core needs in one call.
// tracker: caller maintains rolling stats from trade log.
struct EBB_SizeTracker
{
   int    win_count;
   int    loss_count;
   double avg_win_R;
   double avg_loss_R;
   double target_vol;     // e.g. 0.01 (1% daily) or ATR-pip target
   double realized_vol;   // measured 20-day realized
};

double Sizing_LotFor(const EBB_SizeTracker& t, double account_equity, double sl_pts)
{
   double k = Sizing_KellyFraction(t.win_count, t.loss_count, t.avg_win_R, t.avg_loss_R);
   double v = Sizing_VolScalar(t.target_vol, t.realized_vol);
   return Sizing_CalculateLotSize(account_equity, sl_pts, k, v);
}

// Debug string for journaling every entry decision.
string Sizing_StatusLine(const EBB_SizeTracker& t)
{
   double k = Sizing_KellyFraction(t.win_count, t.loss_count, t.avg_win_R, t.avg_loss_R);
   double v = Sizing_VolScalar(t.target_vol, t.realized_vol);
   return StringFormat("SZ: kelly=%.4f vol=%.2f W=%d L=%d avgW=%.2fR avgL=%.2fR",
                       k, v, t.win_count, t.loss_count, t.avg_win_R, t.avg_loss_R);
}

#endif
