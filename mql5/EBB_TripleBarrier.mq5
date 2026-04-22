//+------------------------------------------------------------------+
//| EBB_TripleBarrier.mq5 — EuroBigBrain Triple-Barrier ML EA        |
//|                                                                  |
//| Loads LightGBM ONNX classifier (h=10, SL=0.7×ATR, TP=2.0×ATR).   |
//| On each new EURUSD M5 bar:                                       |
//|   1. Compute 23 features matching python/build_eurusd_features.py|
//|   2. Run ONNX inference                                          |
//|   3. If P(up) > Threshold (0.44) -> BUY market with bracket      |
//|   4. Time-stop: close after MaxHoldBars (10) if no SL/TP hit     |
//|                                                                  |
//| Deploy candidate: project_ebb_DEPLOY_CANDIDATE_h10               |
//+------------------------------------------------------------------+
#property copyright "EuroBigBrain"
#property version   "1.00"
#property strict

#resource "eur_tb_h10_22feat.onnx" as uchar OnnxModelData[]

#include <Trade/Trade.mqh>

//--- Inputs ---------------------------------------------------------
input double   ProbThreshold    = 0.44;
input double   SL_ATR_Mult      = 0.7;
input double   TP_ATR_Mult      = 2.0;
input int      MaxHoldBars      = 10;
input double   RiskPercent      = 0.6;
input double   MaxLotSize       = 0.10;
input int      MaxTradesPerDay  = 20;
input double   DailyLossCapPct  = 5.0;
input int      MagicNumber      = 26042201;
input int      DebugEveryNTicks = 100;
input bool     RequireCrossSymbols = true;   // FAIL OnInit if any cross symbol missing

//--- ONNX -----------------------------------------------------------
#define NUM_FEATURES 22
long  hModel = INVALID_HANDLE;
float g_features[NUM_FEATURES];

//--- Bar buffers ----------------------------------------------------
#define MAX_BARS 800
double buf_close[MAX_BARS];   // index 0 = oldest, MAX_BARS-1 = newest closed bar
double buf_high[MAX_BARS];
double buf_low[MAX_BARS];
double buf_open[MAX_BARS];
double buf_volume[MAX_BARS];
datetime buf_time[MAX_BARS];

//--- Cross-asset buffers (close only, latest first via series order; we re-index manually) ---
double buf_eurgbp_close[MAX_BARS];
double buf_xau_close[MAX_BARS];
double buf_usdjpy_close[MAX_BARS];
double buf_usdchf_close[MAX_BARS];
double buf_gbpusd_close[MAX_BARS];

string SYM_USDJPY = "USDJPY";
string SYM_USDCHF = "USDCHF";
string SYM_GBPUSD = "GBPUSD";
string SYM_XAUUSD = "XAUUSD";

//--- Trading state --------------------------------------------------
CTrade   g_trade;
datetime g_lastBarTime  = 0;
int      g_todayTrades  = 0;
double   g_todayPnL     = 0.0;
double   g_dayStartBal  = 0.0;
int      g_lastDay      = -1;
int      g_tickCount    = 0;
int      g_holdBars     = 0;       // bars elapsed since current position opened
ulong    g_openTicket   = 0;       // tracked position ticket

//+------------------------------------------------------------------+
int OnInit()
{
   g_trade.SetExpertMagicNumber(MagicNumber);
   g_trade.SetDeviationInPoints(30);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);

   //--- Validate primary symbol is EURUSD-like (don't hard-fail; log warning)
   if(StringFind(_Symbol, "EURUSD") < 0)
      PrintFormat("EBB_TB WARNING: chart symbol '%s' is not EURUSD. Features may misalign.", _Symbol);

   //--- Load ONNX from embedded resource
   hModel = OnnxCreateFromBuffer(OnnxModelData, ONNX_DEFAULT);
   if(hModel == INVALID_HANDLE)
   {
      PrintFormat("EBB_TB FATAL: OnnxCreateFromBuffer failed err=%d", GetLastError());
      return INIT_FAILED;
   }

   long inputShape[]  = {1, NUM_FEATURES};
   long outputShape[] = {1, 2};  // ONNX re-exported with zipmap=False → plain float tensor [batch, 2 classes]
   if(!OnnxSetInputShape(hModel, 0, inputShape))
   { PrintFormat("EBB_TB FATAL: OnnxSetInputShape err=%d", GetLastError()); return INIT_FAILED; }
   if(!OnnxSetOutputShape(hModel, 1, outputShape))
   { PrintFormat("EBB_TB FATAL: OnnxSetOutputShape err=%d", GetLastError()); return INIT_FAILED; }

   //--- Verify required cross symbols are accessible
   string need[] = {SYM_USDJPY, SYM_USDCHF, SYM_GBPUSD, SYM_XAUUSD};
   bool ok = true;
   for(int i = 0; i < ArraySize(need); i++)
   {
      if(!SymbolSelect(need[i], true))
      {
         PrintFormat("EBB_TB symbol '%s' not in MarketWatch (SymbolSelect failed)", need[i]);
         ok = false;
      }
      else
      {
         double bid = SymbolInfoDouble(need[i], SYMBOL_BID);
         if(bid <= 0)
         {
            PrintFormat("EBB_TB symbol '%s' has no quote yet (bid=%.5f)", need[i], bid);
            ok = false;
         }
      }
   }
   if(!ok && RequireCrossSymbols)
   {
      Print("EBB_TB FATAL: cross-asset symbols required for DXY/EURGBP/XAU features. Subscribe USDJPY, USDCHF, GBPUSD, XAUUSD or set RequireCrossSymbols=false (features will use neutral fallbacks).");
      return INIT_FAILED;
   }

   PrintFormat("EBB_TB init OK. magic=%d thr=%.3f SL=%.2fxATR TP=%.2fxATR hold=%d risk=%.2f%% maxLot=%.2f",
               MagicNumber, ProbThreshold, SL_ATR_Mult, TP_ATR_Mult, MaxHoldBars, RiskPercent, MaxLotSize);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(hModel != INVALID_HANDLE)
   {
      OnnxRelease(hModel);
      hModel = INVALID_HANDLE;
   }
   PrintFormat("EBB_TB deinit reason=%d", reason);
}

//+------------------------------------------------------------------+
void OnTick()
{
   g_tickCount++;

   //--- New bar gate: only act once per closed M5 bar
   datetime currentBarTime = iTime(_Symbol, PERIOD_M5, 0);
   if(currentBarTime == 0) return;
   bool newBar = (currentBarTime != g_lastBarTime);

   //--- Manage open position every tick (price-based exits handled by SL/TP server-side; we add time-stop)
   if(HasOpenPosition())
   {
      if(newBar)
      {
         g_lastBarTime = currentBarTime;
         g_holdBars++;
         if(g_holdBars >= MaxHoldBars)
         {
            PrintFormat("EBB_TB time-stop: holdBars=%d >= MaxHoldBars=%d, closing", g_holdBars, MaxHoldBars);
            CloseMyPosition();
         }
      }
      DebugLogIfDue(-1.0f);
      return;
   }

   //--- Otherwise, only act on a new bar
   if(!newBar) { DebugLogIfDue(-1.0f); return; }
   g_lastBarTime = currentBarTime;

   //--- Reset daily counters at day flip (UTC server time of broker)
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   if(dt.day != g_lastDay)
   {
      g_lastDay = dt.day;
      g_todayTrades = 0;
      g_todayPnL = 0.0;
      g_dayStartBal = AccountInfoDouble(ACCOUNT_BALANCE);
   }

   if(g_todayTrades >= MaxTradesPerDay) return;
   if(g_todayPnL < -(g_dayStartBal * DailyLossCapPct / 100.0)) return;

   //--- Load bars
   if(!LoadBars()) return;
   if(!LoadCrossAssetBars()) return;

   //--- Compute features
   if(!ComputeFeatures())
   {
      DebugLogIfDue(-1.0f);
      return;
   }

   //--- Validate features (no NaN, no inf)
   for(int i = 0; i < NUM_FEATURES; i++)
   {
      if(!MathIsValidNumber(g_features[i]))
      {
         PrintFormat("EBB_TB feature[%d] invalid (%.6f), skipping bar", i, g_features[i]);
         return;
      }
   }

   //--- Run ONNX
   float prob = RunModel();
   DebugLogIfDue(prob);

   //--- DIAGNOSTIC: print features + prob + XAU raw + H1 slope raw for first 30 new bars
   static int diag_cnt = 0;
   if(diag_cnt < 30) {
      MqlDateTime mdt0;
      TimeToStruct(buf_time[MAX_BARS-1], mdt0);
      int idx0 = MAX_BARS - 1;
      string s = StringFormat("EBB_DIAG bar=%d t=%04d.%02d.%02d_%02d:%02d prob=%.6f xau_now=%.3f xau_5back=%.3f eurgbp_now=%.5f dxy_now=%.6f f=",
          diag_cnt, mdt0.year, mdt0.mon, mdt0.day, mdt0.hour, mdt0.min, prob,
          buf_xau_close[idx0], buf_xau_close[idx0-5],
          buf_eurgbp_close[idx0], g_features[19]);
      for(int i = 0; i < NUM_FEATURES; i++) s += StringFormat("%.6f,", g_features[i]);
      Print(s);
      diag_cnt++;
   }

   if(prob < 0.0f) return;

   if(prob < (float)ProbThreshold) return;

   //--- Compute ATR(14) for sizing/SL/TP
   double atr14 = ComputeATR_AtBar(14, 1);
   if(atr14 < _Point) return;

   double sl_dist = atr14 * SL_ATR_Mult;
   double tp_dist = atr14 * TP_ATR_Mult;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0 || bid <= 0) return;

   // Broker stops-level check (FIX v2 21:13: skip low-ATR bars entirely rather than stretch stops
   // unevenly, which destroys the TP/SL ratio and the edge along with it)
   int    stops_lvl = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double min_stop = (stops_lvl + 2) * _Point;  // +2 points safety margin
   if(sl_dist < min_stop || tp_dist < min_stop) return;  // don't trade if ATR too low for proper TB ratio

   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double sl = NormalizeDouble(ask - sl_dist, digits);
   double tp = NormalizeDouble(ask + tp_dist, digits);

   double lot = ComputeLotSize(sl_dist);
   if(lot <= 0) return;

   string note = StringFormat("EBB_TB p=%.3f", prob);
   if(g_trade.Buy(lot, _Symbol, ask, sl, tp, note))
   {
      g_openTicket = g_trade.ResultOrder();
      g_holdBars = 0;
      g_todayTrades++;
      PrintFormat("EBB_TB BUY @%.5f SL=%.5f TP=%.5f lot=%.2f prob=%.3f atr=%.5f",
                  ask, sl, tp, lot, prob, atr14);
   }
   else
   {
      PrintFormat("EBB_TB Buy FAILED: %s (retcode=%d)",
                  g_trade.ResultRetcodeDescription(), g_trade.ResultRetcode());
   }
}

//+------------------------------------------------------------------+
void DebugLogIfDue(float lastProb)
{
   if(DebugEveryNTicks <= 0) return;
   if(g_tickCount % DebugEveryNTicks != 0) return;
   PrintFormat("EBB_TB debug tick=%d lastProb=%.4f thr=%.3f tradesToday=%d holdBars=%d hasPos=%s",
               g_tickCount, lastProb, ProbThreshold, g_todayTrades, g_holdBars,
               HasOpenPosition() ? "yes" : "no");
}

//+------------------------------------------------------------------+
//| Position helpers                                                 |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
         PositionGetString(POSITION_SYMBOL) == _Symbol)
         return true;
   }
   return false;
}

void CloseMyPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
         PositionGetString(POSITION_SYMBOL) == _Symbol)
      {
         g_trade.PositionClose(tk);
      }
   }
}

void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD || trans.deal == 0) return;
   if(!HistoryDealSelect(trans.deal)) return;
   if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != MagicNumber) return;

   long entry = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT)
                 + HistoryDealGetDouble(trans.deal, DEAL_COMMISSION)
                 + HistoryDealGetDouble(trans.deal, DEAL_SWAP);
   g_todayPnL += profit;

   if(entry == DEAL_ENTRY_OUT)
   {
      // Position fully closed
      g_openTicket = 0;
      g_holdBars = 0;
   }
}

//+------------------------------------------------------------------+
//| Lot sizing: risk-based                                           |
//+------------------------------------------------------------------+
double ComputeLotSize(double sl_dist_price)
{
   double riskUsd = AccountInfoDouble(ACCOUNT_BALANCE) * RiskPercent / 100.0;
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double lotStep   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(tickValue <= 0 || tickSize <= 0 || lotStep <= 0) return 0.0;
   double slTicks = sl_dist_price / tickSize;
   if(slTicks <= 0) return 0.0;
   double lot = riskUsd / (slTicks * tickValue);
   lot = MathFloor(lot / lotStep) * lotStep;
   lot = MathMax(lot, minLot);
   lot = MathMin(lot, MaxLotSize);
   return lot;
}

//+------------------------------------------------------------------+
//| Bar loaders                                                      |
//| buf_*[i]: i=0 oldest in window, i=MAX_BARS-1 most recent CLOSED  |
//| (we copy from index 1 = last closed bar going back)              |
//+------------------------------------------------------------------+
bool LoadBars()
{
   MqlRates rates[];
   int copied = CopyRates(_Symbol, PERIOD_M5, 1, MAX_BARS, rates);  // skip current incomplete bar
   if(copied < MAX_BARS) return false;
   for(int i = 0; i < MAX_BARS; i++)
   {
      // rates is series order: rates[0] = oldest, rates[copied-1] = newest
      buf_close[i]  = rates[i].close;
      buf_high[i]   = rates[i].high;
      buf_low[i]    = rates[i].low;
      buf_open[i]   = rates[i].open;
      buf_volume[i] = (double)rates[i].tick_volume;
      buf_time[i]   = rates[i].time;
   }
   return true;
}

bool CopyCloseSafe(string sym, double &dst[])
{
   double tmp[];
   int copied = CopyClose(sym, PERIOD_M5, 1, MAX_BARS, tmp);
   if(copied < MAX_BARS) return false;
   for(int i = 0; i < MAX_BARS; i++) dst[i] = tmp[i];
   return true;
}

bool LoadCrossAssetBars()
{
   // EURGBP synthetic = EURUSD / GBPUSD; we need GBPUSD closes aligned to our bars.
   if(!CopyCloseSafe(SYM_GBPUSD, buf_gbpusd_close)) { ZeroOrFallback(buf_gbpusd_close); }
   if(!CopyCloseSafe(SYM_USDJPY, buf_usdjpy_close)) { ZeroOrFallback(buf_usdjpy_close); }
   if(!CopyCloseSafe(SYM_USDCHF, buf_usdchf_close)) { ZeroOrFallback(buf_usdchf_close); }
   if(!CopyCloseSafe(SYM_XAUUSD, buf_xau_close))    { ZeroOrFallback(buf_xau_close);    }

   // Build EURGBP synthetic
   for(int i = 0; i < MAX_BARS; i++)
   {
      double g = buf_gbpusd_close[i];
      buf_eurgbp_close[i] = (g > 0.0) ? buf_close[i] / g : 0.0;
   }
   return true;
}

void ZeroOrFallback(double &arr[])
{
   for(int i = 0; i < MAX_BARS; i++) arr[i] = 0.0;
}

//+------------------------------------------------------------------+
//| Helpers                                                          |
//+------------------------------------------------------------------+
double SafeDiv(double a, double b)
{ return MathAbs(b) < 1e-12 ? 0.0 : a / b; }

double SafeLog(double x)
{ return x > 1e-12 ? MathLog(x) : 0.0; }

//+------------------------------------------------------------------+
//| Wilder ATR(period) ending at bar `at_bar` (0 = oldest end)       |
//| Bars are stored series-order: index MAX_BARS-1 = most recent.    |
//| at_bar=0 means "oldest bar" so use at_bar=MAX_BARS-1 for latest. |
//| For convenience: ComputeATR_AtBar(period, k) returns ATR at the  |
//| bar k positions back from the most recent closed bar.            |
//+------------------------------------------------------------------+
double ComputeATR_AtBar(int period, int k_back)
{
   // FIX (auditor #1): Python uses simple rolling mean of TR, not Wilder smoothing.
   // SMA over the last `period` TR values ending at target index.
   int target = MAX_BARS - 1 - k_back;
   if(target - period < 1) return 0.0;
   double sum = 0.0;
   for(int i = target - period + 1; i <= target; i++)
   {
      double tr = MathMax(buf_high[i] - buf_low[i],
                  MathMax(MathAbs(buf_high[i] - buf_close[i-1]),
                          MathAbs(buf_low[i]  - buf_close[i-1])));
      sum += tr;
   }
   return sum / period;
}

//+------------------------------------------------------------------+
//| pandas-style EWM (alpha = 2/(span+1)), evaluated at index `idx`. |
//| Matches pandas .ewm(span=N).mean() with adjust=True default.     |
//| For simplicity we use the simple exponential recursion           |
//| (adjust=False). This is the SAME convention used in GBB EAs;     |
//| training-time pandas uses adjust=True so there is a small        |
//| transient mismatch in the first ~3*span bars but converges.      |
//+------------------------------------------------------------------+
double ComputeEMA_AtIdx(const double &arr[], int span, int idx)
{
   // FIX (auditor #2): pandas default is adjust=True (weighted-sum formula).
   // adjust=True: EMA[t] = sum(w_i * x_i) / sum(w_i), w_i = (1-alpha)^(t-i)
   if(idx < 0) return 0.0;
   double mult = 2.0 / (span + 1.0);
   double one_minus = 1.0 - mult;
   double num = 0.0, den = 0.0;
   double w = 1.0;
   for(int i = idx; i >= 0; i--)
   {
      num += w * arr[i];
      den += w;
      w *= one_minus;
      if(w < 1e-15) break;
   }
   return num / den;
}

//+------------------------------------------------------------------+
//| RSI(period) at bar idx (Wilder smoothing, matches pandas         |
//| rolling().mean() approximation used in feature script).          |
//+------------------------------------------------------------------+
double ComputeRSI_AtIdx(const double &arr[], int period, int idx)
{
   if(idx < period) return 50.0;
   // Use simple rolling mean of gains/losses over last `period` diffs ending at idx.
   double gain = 0.0, loss = 0.0;
   for(int i = idx - period + 1; i <= idx; i++)
   {
      double d = arr[i] - arr[i-1];
      if(d > 0) gain += d; else loss -= d;
   }
   gain /= period;
   loss /= period;
   if(loss < 1e-12) return 100.0;
   double rs = gain / loss;
   return 100.0 - 100.0 / (1.0 + rs);
}

//+------------------------------------------------------------------+
//| Rolling mean / std of an arbitrary array slice                   |
//+------------------------------------------------------------------+
double RollingMean(const double &arr[], int idx, int window)
{
   if(idx - window + 1 < 0) return 0.0;
   double s = 0.0;
   for(int i = idx - window + 1; i <= idx; i++) s += arr[i];
   return s / window;
}

double RollingStd(const double &arr[], int idx, int window)
{
   if(idx - window + 1 < 0) return 0.0;
   double m = RollingMean(arr, idx, window);
   double v = 0.0;
   for(int i = idx - window + 1; i <= idx; i++)
   {
      double d = arr[i] - m;
      v += d * d;
   }
   // pandas default rolling().std() uses ddof=1
   return MathSqrt(v / (window - 1));
}

double RollingMedian(const double &arr[], int idx, int window)
{
   if(idx - window + 1 < 0) return 0.0;
   double tmp[];
   ArrayResize(tmp, window);
   for(int i = 0; i < window; i++) tmp[i] = arr[idx - window + 1 + i];
   ArraySort(tmp);
   if(window % 2 == 1) return tmp[window/2];
   return 0.5 * (tmp[window/2 - 1] + tmp[window/2]);
}

//+------------------------------------------------------------------+
//| Compute the 23 features in the EXACT order required by ONNX:     |
//|  0  session_asia                                                 |
//|  1  session_london                                               |
//|  2  session_overlap                                              |
//|  3  session_ny                                                   |
//|  4  min_since_london_open                                        |
//|  5  min_to_ny_fix                                                |
//|  6  near_ny_fix                                                  |
//|  7  dow                                                          |
//|  8  atr14_norm                                                   |
//|  9  vol_of_vol                                                   |
//| 10  rsi14                                                        |
//| 11  rsi7                                                         |
//| 12  ema20_zdist                                                  |
//| 13  ema50_zdist                                                  |
//| 14  bar_of_sess_mom                                              |
//| 15  range_exp                                                    |
//| 16  h1_ema50_slope (sign of slope, 1/0/-1)                       |
//| 17  tickvol_z                                                    |
//| 18  eurgbp_ret_5                                                 |
//| 19  xau_ret_5                                                    |
//| 20  dxy_proxy                                                    |
//| 21  dxy_ret_5                                                    |
//| 22  dxy_z50                                                      |
//+------------------------------------------------------------------+
bool ComputeFeatures()
{
   int idx = MAX_BARS - 1;   // most recent CLOSED bar

   //--- Time-of-day (use the bar's open time, treat as broker UTC; matches python which uses index hour/minute)
   MqlDateTime mdt;
   TimeToStruct(buf_time[idx], mdt);
   int h = mdt.hour;
   int m = mdt.min;
   int dow = (mdt.day_of_week + 6) % 7;   // python Monday=0 ... Sunday=6 vs MT5 Sunday=0...Saturday=6

   g_features[0] = ((h >= 22) || (h < 7))  ? 1.0f : 0.0f;
   g_features[1] = ((h >= 7)  && (h < 12)) ? 1.0f : 0.0f;
   g_features[2] = ((h >= 12) && (h < 16)) ? 1.0f : 0.0f;
   g_features[3] = ((h >= 16) && (h < 22)) ? 1.0f : 0.0f;
   g_features[4] = (float)(((h - 7 + 24) % 24) * 60 + m);
   g_features[5] = (float)(((16 - h + 24) % 24) * 60 - m);
   g_features[6] = (((h == 15) && (m >= 55)) || ((h == 16) && (m <= 5))) ? 1.0f : 0.0f;
   g_features[7] = (float)dow;

   //--- ATR family
   double atr14 = ComputeATR_AtBar(14, 0);   // ATR at most recent closed bar
   double c     = buf_close[idx];
   double atr14_norm = SafeDiv(atr14, c);
   g_features[8] = (float)atr14_norm;

   // vol_of_vol: rolling 20 std / mean of atr14 series
   // Build atr14 series for the last 40 bars
   double atrSeries[];
   ArrayResize(atrSeries, 40);
   for(int i = 0; i < 40; i++)
   {
      // ATR at bar k_back = i, k_back=0 => idx, k_back=39 => idx-39
      atrSeries[39 - i] = ComputeATR_AtBar(14, i);
   }
   double atrMean = RollingMean(atrSeries, 39, 20);
   double atrStd  = RollingStd(atrSeries, 39, 20);
   g_features[9] = (float)SafeDiv(atrStd, atrMean);

   //--- RSI(14), RSI(7) on close
   g_features[10] = (float)ComputeRSI_AtIdx(buf_close, 14, idx);
   g_features[11] = (float)ComputeRSI_AtIdx(buf_close, 7,  idx);

   //--- EMA20 / EMA50 z-distance: (close - ema)/atr14
   double ema20 = ComputeEMA_AtIdx(buf_close, 20, idx);
   double ema50 = ComputeEMA_AtIdx(buf_close, 50, idx);
   g_features[12] = (float)SafeDiv(c - ema20, atr14);
   g_features[13] = (float)SafeDiv(c - ema50, atr14);

   //--- bar_of_sess_mom: (close - session_open) / atr14
   //    session_id grouping in python uses (hour AS STRING + '_' + date).
   //    That groups all bars in the same hour-of-day on the same date together,
   //    so "session open" = the OPEN of the first bar in the current hour.
   //    Walk back until we find a different (date,hour).
   double sess_open = buf_open[idx];
   for(int i = idx - 1; i >= 0; i--)
   {
      MqlDateTime tt;
      TimeToStruct(buf_time[i], tt);
      if(tt.hour == h && tt.day == mdt.day && tt.mon == mdt.mon && tt.year == mdt.year)
         sess_open = buf_open[i];
      else
         break;
   }
   g_features[14] = (float)SafeDiv(c - sess_open, atr14);

   //--- range_exp: (high-low) / rolling-20 median of (high-low)
   double rngSeries[];
   ArrayResize(rngSeries, 20);
   for(int i = 0; i < 20; i++) rngSeries[i] = buf_high[idx - 19 + i] - buf_low[idx - 19 + i];
   double rngMed = RollingMedian(rngSeries, 19, 20);
   double rng_now = buf_high[idx] - buf_low[idx];
   g_features[15] = (float)SafeDiv(rng_now, rngMed);

   //--- h1_ema50_slope: FIX (auditor #3) — Python uses .resample('1H').last() which
   // CLOCK-ALIGNS to hour boundaries (00:00, 01:00, ...) and ffills the same slope
   // value to all M5 bars within the hour. We must do the same — find the most
   // recent M5 bar with minute==0 and snap our hourly grid to it.
   int h1_anchor = -1;
   for(int i = idx; i >= 0; i--)
   {
      MqlDateTime t;
      TimeToStruct(buf_time[i], t);
      if(t.min == 0) { h1_anchor = i; break; }
   }
   int need_h1 = 12 * 60;   // ~60 hourly samples for stable EMA50
   bool h1_ok = false;
   if(h1_anchor >= 0 && h1_anchor >= need_h1 - 12)
   {
      int n_h1 = need_h1 / 12;
      double h1_arr[];
      ArrayResize(h1_arr, n_h1);
      bool valid = true;
      for(int i = 0; i < n_h1 && valid; i++)
      {
         int bar_idx = h1_anchor - (n_h1 - 1 - i) * 12;
         if(bar_idx < 0) { valid = false; break; }
         h1_arr[i] = buf_close[bar_idx];
      }
      if(valid)
      {
         double h1_ema_now  = ComputeEMA_AtIdx(h1_arr, 50, n_h1 - 1);
         double h1_ema_prev = ComputeEMA_AtIdx(h1_arr, 50, n_h1 - 4);
         double slope = h1_ema_now - h1_ema_prev;
         g_features[16] = (slope > 0) ? 1.0f : (slope < 0 ? -1.0f : 0.0f);
         h1_ok = true;
      }
   }
   if(!h1_ok) g_features[16] = 0.0f;

   //--- tickvol_z: (vol - rolling20 mean) / rolling20 std
   double volMean = RollingMean(buf_volume, idx, 20);
   double volStd  = RollingStd(buf_volume, idx, 20);
   g_features[17] = (float)SafeDiv(buf_volume[idx] - volMean, volStd);

   //--- eurgbp_ret_5 = log(EURGBP[t] / EURGBP[t-5])
   double eurgbp_now  = buf_eurgbp_close[idx];
   double eurgbp_prev = buf_eurgbp_close[idx - 5];
   if(eurgbp_now > 0 && eurgbp_prev > 0)
      g_features[18] = (float)(SafeLog(eurgbp_now) - SafeLog(eurgbp_prev));
   else
      g_features[18] = 0.0f;

   //--- xau_ret_5 REMOVED: XAU buffer frozen in tester; feature was causing ONNX prediction collapse.
   //    22-feat model retrained without xau_ret_5 (AUC loss -0.002, null edge 16.46x, gate 5x).
   //    Downstream feature indices shifted: was [20..22], now [19..21].

   //--- DXY proxy = log(USDJPY) + log(USDCHF) - log(GBPUSD)
   double jpy = buf_usdjpy_close[idx];
   double chf = buf_usdchf_close[idx];
   double gbp = buf_gbpusd_close[idx];
   if(jpy > 0 && chf > 0 && gbp > 0)
   {
      double dxy_now = SafeLog(jpy) + SafeLog(chf) - SafeLog(gbp);
      g_features[19] = (float)dxy_now;

      // dxy_ret_5 = dxy[t] - dxy[t-5]
      double jpy5 = buf_usdjpy_close[idx-5];
      double chf5 = buf_usdchf_close[idx-5];
      double gbp5 = buf_gbpusd_close[idx-5];
      double dxy_prev = (jpy5 > 0 && chf5 > 0 && gbp5 > 0)
                        ? (SafeLog(jpy5) + SafeLog(chf5) - SafeLog(gbp5))
                        : dxy_now;
      g_features[20] = (float)(dxy_now - dxy_prev);

      // dxy_z50 = (dxy_now - rolling50_mean) / rolling50_std
      double dxySeries[];
      ArrayResize(dxySeries, 50);
      for(int i = 0; i < 50; i++)
      {
         int b = idx - 49 + i;
         double j2 = buf_usdjpy_close[b];
         double c2 = buf_usdchf_close[b];
         double g2 = buf_gbpusd_close[b];
         dxySeries[i] = (j2 > 0 && c2 > 0 && g2 > 0)
                        ? (SafeLog(j2) + SafeLog(c2) - SafeLog(g2))
                        : dxy_now;
      }
      double dxyMean = RollingMean(dxySeries, 49, 50);
      double dxyStd  = RollingStd(dxySeries, 49, 50);
      g_features[21] = (float)SafeDiv(dxy_now - dxyMean, dxyStd);
   }
   else
   {
      g_features[19] = 0.0f;
      g_features[20] = 0.0f;
      g_features[21] = 0.0f;
   }

   return true;
}

//+------------------------------------------------------------------+
float RunModel()
{
   float input_data[];
   ArrayResize(input_data, NUM_FEATURES);
   for(int i = 0; i < NUM_FEATURES; i++) input_data[i] = g_features[i];

   long  labels[];
   float probas[];
   ArrayResize(labels, 1);
   ArrayResize(probas, 2);

   if(!OnnxRun(hModel, ONNX_NO_CONVERSION, input_data, labels, probas))
   {
      PrintFormat("EBB_TB OnnxRun FAILED err=%d", GetLastError());
      return -1.0f;
   }
   return probas[1];   // P(class=1, i.e. up move). Inversion tested 21:25 — PF collapsed to 0.90, confirming this index is correct.
}
//+------------------------------------------------------------------+
