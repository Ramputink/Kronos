"""Random-filter control: keep each new entry with the same probability Kronos-confirm kept it."""
import json, sys, zlib
import numpy as np
import pandas as pd
sys.path.insert(0, '/Users/matveypro/kronos_lab/marketresearcher')
import run_mr_kronos as h
from config import BacktestConfig, RiskConfig
from engine.backtester import Backtester
import importlib

D = h.OUT
SEEDS = 20
sym = sys.argv[1]
r = pd.DataFrame([json.loads(l) for l in open(f'{D}/results.jsonl')]).drop_duplicates(
    ['strategy', 'symbol', 'variant'], keep='last')
conf = r[(r.variant == 'confirm') & (r.symbol == sym)].set_index('strategy')
df = h.load_symbol(sym)
bt_cfg = BacktestConfig(commission_rate=0.00055, slippage_bps=2.0)
out = open(f'{D}/random_{sym}.jsonl', 'w')
for name in h.STRATS:
    h.set_genome(name)
    fn = getattr(importlib.import_module(f'strategies.{name}'), f'{name}_strategy')
    p_keep = conf.loc[name, 'kronos_kept'] / max(conf.loc[name, 'kronos_seen'], 1)
    rets = []
    for s in range(SEEDS):
        rng = np.random.default_rng(zlib.crc32(f'{name}{sym}{s}'.encode()))

        def f(d, i, pos):
            sig = fn(d, i, pos, regime=d['_regime'].iat[i])
            if sig is None or pos is not None or getattr(sig, 'side', None) not in ('long', 'short'):
                return sig
            return sig if rng.random() < p_keep else None
        _, eq, _ = Backtester(bt_cfg, RiskConfig()).run(df, f, name)
        rets.append(float(eq.iloc[-1] / bt_cfg.initial_capital * 100 - 100))
    out.write(json.dumps({'strategy': name, 'symbol': sym, 'p_keep': p_keep, 'rets': rets}) + '\n')
    out.flush()
    print(name, round(p_keep, 2), np.round(np.percentile(rets, [5, 50, 95]), 2), flush=True)
