//+------------------------------------------------------------------+
//| EBB_Core.mq5 - symbol-agnostic orchestrator EA (heartbeat skel)  |
//|                                                                  |
//| W1 step 1 of docs/design_ebb_core_refactor.md: compilable shell. |
//| No strategy logic ported yet (W1 step 3). No byte-identical      |
//| regression vs GBB_Core yet (W1 step 4). OnTick is heartbeat only.|
//+------------------------------------------------------------------+
#property copyright "EuroBigBrain"
#property version   "0.01"
#property strict

#include <Trade/Trade.mqh>
#include "EBB/SymbolHash.mqh"

//--- Strategy entry-mode enum (mirrors GBB for later port; unused in skeleton)
enum ENUM_ENTRY_MODE
{
   MODE_ATR_BRACKET      = 0,
   MODE_ASIAN_RANGE      = 1,
   MODE_MOMENTUM_LONG    = 2,
   MODE_MOMENTUM_SHORT   = 3,
   MODE_FADE_LONG        = 4,
   MODE_FADE_SHORT       = 5,
   MODE_EMA_CROSS_LONG   = 6,
   MODE_EMA_CROSS_SHORT  = 7,
   MODE_BREAKOUT_RANGE   = 8,
   MODE_VOL_SPIKE_BRACKET= 9,
   MODE_NULL_BRACKET     = 10,
   MODE_ML_ENTRY         = 11,
   MODE_OR_BREAKOUT      = 12,
   MODE_PIVOT_BOUNCE     = 13,
   MODE_ENGULFING        = 14
};

//--- Core inputs (kept 1:1 with GBB_Core for eventual regression parity)
input ENUM_ENTRY_MODE EntryMode = MODE_FADE_LONG;
input double   RiskPercent      = 0.8;
input double   SL_ATR_Mult      = 1.0;
input double   TP_ATR_Mult      = 3.0;
input double   BracketOffset    = 0.3;
input int      BracketBars      = 3;
input int      MaxTradesPerDay  = 20;
input double   DailyLossCapPct  = 5.0;
input int      SessionStart     = 7;
input int      SessionEnd       = 20;
input int      MagicNumber      = 26100000;   // base; per-symbol magic = +hash%1000
input double   MaxLotSize       = 0.10;

//--- Trade management
input bool     EnableBreakEven  = true;
input double   BE_ATR_Mult      = 0.3;
input bool     EnableTrailing   = true;
input double   Trail_ATR_Mult   = 0.5;
input bool     EnableTimeStop   = true;
input int      MaxHoldBars      = 8;

//--- ONNX vol gate (0 = disabled; auto-disabled if per-symbol model missing)
input double   VolThreshold     = 0.0;

//--- Fade RSI thresholds
input double   FadeLongRSI      = 35.0;
input double   FadeShortRSI     = 65.0;

//--- EMA cross periods
input int      EmaFastPeriod    = 8;
input int      EmaSlowPeriod    = 21;

//--- Globals
CTrade   g_trade;
long     g_hVolModel      = INVALID_HANDLE;
bool     g_volGateEnabled = false;          // false when model missing
int      g_magic          = 0;
datetime g_lastBarTime    = 0;
datetime g_lastHeartbeat  = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   // Derive per-symbol magic (design doc section 2.2: isolation rule)
   g_magic = MagicForSymbol(MagicNumber, _Symbol);
   g_trade.SetExpertMagicNumber(g_magic);
   g_trade.SetDeviationInPoints(30);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);

   // Runtime-load symbol-suffixed ONNX model (design doc section 2.3).
   // NOTE: pre-EUR-training, files will be absent for non-XAU symbols.
   // Per user: missing model -> log + continue with gate DISABLED (not INIT_FAILED).
   string modelRel = StringFormat("vol_model_%s.onnx", _Symbol);
   if(FileIsExist(modelRel, FILE_COMMON) || FileIsExist(modelRel))
   {
      g_hVolModel = OnnxCreate(modelRel, ONNX_DEFAULT);
      if(g_hVolModel == INVALID_HANDLE)
      {
         PrintFormat("EBB_Core: OnnxCreate failed for %s err=%d - vol gate DISABLED",
                     modelRel, GetLastError());
         g_volGateEnabled = false;
      }
      else
      {
         g_volGateEnabled = (VolThreshold > 0.0);
         PrintFormat("EBB_Core: loaded %s; vol gate %s (threshold=%.3f)",
                     modelRel, g_volGateEnabled ? "ENABLED" : "disabled (VT=0)",
                     VolThreshold);
      }
   }
   else
   {
      PrintFormat("EBB_Core: model %s not found - vol gate DISABLED (ok pre-training)",
                  modelRel);
      g_volGateEnabled = false;
   }

   PrintFormat("EBB_Core init symbol=%s magic=%d mode=%s VT=%.3f SL=%.2f TP=%.2f",
               _Symbol, g_magic, EnumToString(EntryMode),
               VolThreshold, SL_ATR_Mult, TP_ATR_Mult);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_hVolModel != INVALID_HANDLE)
   {
      OnnxRelease(g_hVolModel);
      g_hVolModel = INVALID_HANDLE;
   }
   PrintFormat("EBB_Core deinit reason=%d symbol=%s", reason, _Symbol);
}

//+------------------------------------------------------------------+
//| OnTick: heartbeat-only stub. Proves event loop runs and per-bar  |
//| edge-trigger works. No signals, no orders. Strategy dispatch is  |
//| W1 step 3 (see design doc section 5 migration plan).             |
//+------------------------------------------------------------------+
void OnTick()
{
   datetime currentBarTime = iTime(_Symbol, PERIOD_M5, 0);
   if(currentBarTime == g_lastBarTime) return;
   g_lastBarTime = currentBarTime;

   // Log one heartbeat per bar (cheap; Tester log-friendly).
   if(currentBarTime - g_lastHeartbeat >= 300)  // at least 5 min between logs
   {
      g_lastHeartbeat = currentBarTime;
      PrintFormat("EBB_Core heartbeat %s bar=%s vol_gate=%s magic=%d",
                  _Symbol, TimeToString(currentBarTime, TIME_DATE|TIME_MINUTES),
                  g_volGateEnabled ? "on" : "off", g_magic);
   }
}

//+------------------------------------------------------------------+
