//+------------------------------------------------------------------+
//| EBB/SymbolHash.mqh - stable magic derivation per symbol          |
//+------------------------------------------------------------------+
//| Goal: two EBB_Core instances on different symbols must NEVER     |
//| share a magic number. Per design doc section 2.2, magic =        |
//| MagicBase + StringHash(symbol) % 1000, giving ~1000-wide buckets |
//| per base. Uses FNV-1a 32-bit hash for determinism across build/  |
//| platforms (MQL5 has no built-in string hash).                    |
//+------------------------------------------------------------------+
#ifndef __EBB_SYMBOL_HASH_MQH__
#define __EBB_SYMBOL_HASH_MQH__

// FNV-1a 32-bit: deterministic, well-distributed, no deps.
uint StringHashFNV1a(const string s)
{
   uint h = 2166136261;            // FNV offset basis
   int n = StringLen(s);
   for(int i = 0; i < n; i++)
   {
      h ^= (uint)StringGetCharacter(s, i);
      h *= 16777619;                // FNV prime (32-bit wraparound ok)
   }
   return h;
}

int MagicForSymbol(const int magicBase, const string sym)
{
   return magicBase + (int)(StringHashFNV1a(sym) % 1000);
}

#endif // __EBB_SYMBOL_HASH_MQH__
