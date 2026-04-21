"""Temp inspector for MT5 htm structure (delete after)."""
import sys
path = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\kahad\IdeaProjects\beast-drawdowns\runs\RC3_OnLeash_Mac.htm'
with open(path, 'rb') as f:
    raw = f.read()
text = raw.decode('utf-16-le', errors='ignore')
if text and text[0] == '﻿':
    text = text[1:]
keys = ['Symbol:', 'Period:', 'Inputs:', 'Total Net Profit', 'Profit Factor',
        'Balance Drawdown Maximal', 'Sharpe Ratio', 'Recovery Factor',
        'Total Trades', 'Profit Trades', 'Average position holding',
        'Largest profit', 'Largest loss']
for kw in keys:
    idx = text.find(kw)
    if idx >= 0:
        snippet = text[idx:idx+400].replace('\n', ' ').replace('\r', ' ')
        print('---', kw, '---')
        print(snippet[:400])
        print()
