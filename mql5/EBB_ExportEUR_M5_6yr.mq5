//+------------------------------------------------------------------+
//| EBB_ExportEUR_M5_6yr.mq5 — exports EURUSD M5 bars 2020-2026       |
//| Attach to any chart in MT5; runs OnInit, dumps CSV, exits.        |
//+------------------------------------------------------------------+
#property copyright "EuroBigBrain"
#property version   "1.00"
#property strict

input string   InpSymbol     = "EURUSD";
input datetime InpFromDate   = D'2020.01.01';
input datetime InpToDate     = D'2026.04.13';
input string   InpBarsFile   = "EURUSD_M5_6yr.csv";
input bool     InpShutdownTerminal = true;

int OnInit()
{
   if(!SymbolSelect(InpSymbol, true))
   { Print("Failed to select ", InpSymbol); return INIT_FAILED; }

   ExportBarsM5();

   Print("EBB_EXPORT_DONE");
   if(InpShutdownTerminal) TerminalClose(0);
   ExpertRemove();
   return INIT_SUCCEEDED;
}

void ExportBarsM5()
{
   int h = FileOpen(InpBarsFile, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(h == INVALID_HANDLE) { Print("bars open fail ", GetLastError()); return; }

   FileWrite(h, "time", "open", "high", "low", "close", "tick_volume", "real_volume", "spread");

   // Use datetime-range overload — pulls all bars between start and stop
   MqlRates rates[];
   datetime cursor = InpFromDate;
   datetime stop_time = InpToDate;
   long total = 0;

   // Try one big call first
   int copied = CopyRates(InpSymbol, PERIOD_M5, cursor, stop_time, rates);
   PrintFormat("CopyRates(%s..%s) returned %d", TimeToString(cursor), TimeToString(stop_time), copied);

   if(copied <= 0)
   {
      // Walk in 30-day chunks instead
      while(cursor < stop_time)
      {
         datetime chunk_end = cursor + 30*86400;
         if(chunk_end > stop_time) chunk_end = stop_time;
         ArrayFree(rates);
         copied = CopyRates(InpSymbol, PERIOD_M5, cursor, chunk_end, rates);
         if(copied > 0)
         {
            for(int i = 0; i < copied; i++)
            {
               FileWrite(h,
                         TimeToString(rates[i].time, TIME_DATE|TIME_SECONDS),
                         DoubleToString(rates[i].open, 5),
                         DoubleToString(rates[i].high, 5),
                         DoubleToString(rates[i].low, 5),
                         DoubleToString(rates[i].close, 5),
                         (long)rates[i].tick_volume,
                         (long)rates[i].real_volume,
                         (int)rates[i].spread);
               total++;
            }
         }
         else
         {
            PrintFormat("chunk %s..%s: 0 bars err=%d", TimeToString(cursor), TimeToString(chunk_end), GetLastError());
         }
         cursor = chunk_end;
      }
   }
   else
   {
      // Single call worked
      for(int i = 0; i < copied; i++)
      {
         FileWrite(h,
                   TimeToString(rates[i].time, TIME_DATE|TIME_SECONDS),
                   DoubleToString(rates[i].open, 5),
                   DoubleToString(rates[i].high, 5),
                   DoubleToString(rates[i].low, 5),
                   DoubleToString(rates[i].close, 5),
                   (long)rates[i].tick_volume,
                   (long)rates[i].real_volume,
                   (int)rates[i].spread);
         total++;
      }
   }
   FileClose(h);
   PrintFormat("EBB_EXPORT bars=%d file=%s", total, InpBarsFile);
}
//+------------------------------------------------------------------+
