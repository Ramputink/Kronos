"""
MarketReasercher strategies (evolved genomes) on Bybit 1h data, with Bybit fees,
alone and filtered by Kronos-small forecasts.

Variants per strategy x symbol:
  base     strategy as-is
  confirm  new entry only if Kronos 24h forecast agrees with the trade side
  veto     new entry only if Kronos 24h forecast DISAGREES (contrarian finding)
Exits are never filtered. Kronos forecasts use only candles <= signal bar.

Run with the Kronos venv:
  ~/.venvs/kronos/bin/python run_mr_kronos.py [--strategies a,b] [--symbols BTCUSDT]
"""
import argparse, copy, importlib, json, os, pickle, sys, time
import numpy as np
import pandas as pd

MR = '/Users/matveypro/MarketReasercher'
KR = '/Users/matveypro/Kronos/.claude/worktrees/bybit-poc'
CACHE = '/Users/matveypro/kronos_lab/cache'
OUT = '/Users/matveypro/kronos_lab/marketresearcher'
sys.path.insert(0, MR)
sys.path.insert(0, KR)

from config import BacktestConfig, RiskConfig  # noqa: E402
from engine.backtester import Backtester  # noqa: E402
from engine.features import build_all_features  # noqa: E402
from mirofish.scenario_engine import classify_regime_quantitative  # noqa: E402

STRATS = ['chaos_trend', 'connors_rsi2', 'donchian_breakout', 'dual_ma', 'fisher_transform',
          'heikin_ashi_ema', 'ichimoku_kumo', 'kama_trend', 'keltner_breakout', 'momentum',
          'supertrend', 'trend_following', 'vol_regime_arb', 'volatility_squeeze', 'williams_cci']
COLS = ['open', 'high', 'low', 'close', 'volume', 'amount']
LOOKBACK, HORIZON, SAMPLES = 256, 24, 4


def load_symbol(sym):
    pk = f'{OUT}/feat_{sym}.pkl'
    if os.path.exists(pk):
        return pd.read_pickle(pk)
    raw = pd.read_csv(f'{CACHE}/{sym}_60_ohlcv.csv', parse_dates=['timestamps'])
    raw['timestamp'] = raw.timestamps.astype('int64') // 10**6
    df = build_all_features(raw).reset_index(drop=True)
    df['_regime'] = ['unknown' if i < 50 else classify_regime_quantitative(df, bar_idx=i).regime.value
                     for i in range(len(df))]
    df.to_pickle(pk)
    return df


class KronosOracle:
    """Lazy, disk-cached Kronos 24h return forecast at a given bar."""

    def __init__(self, sym, df):
        self.sym, self.df = sym, df
        self.path = f'{OUT}/kronos_{sym}.pkl'
        self.memo = pickle.load(open(self.path, 'rb')) if os.path.exists(self.path) else {}
        self.predictor = None
        self.new = 0

    def _load(self):
        from model import Kronos, KronosTokenizer, KronosPredictor
        tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base')
        mdl = Kronos.from_pretrained('NeoQuasar/Kronos-small')
        self.predictor = KronosPredictor(mdl, tok, max_context=512)

    def ret(self, i):
        if i in self.memo:
            return self.memo[i]
        if i < LOOKBACK:
            return 0.0
        if self.predictor is None:
            self._load()
        h = self.df.iloc[i - LOOKBACK + 1:i + 1]
        step = pd.Timedelta(hours=1)
        y_ts = pd.Series([h.timestamps.iloc[-1] + step * (k + 1) for k in range(HORIZON)])
        p = self.predictor.predict(h[COLS], h.timestamps.reset_index(drop=True), y_ts, pred_len=HORIZON,
                                   T=1.0, top_p=0.9, sample_count=SAMPLES, verbose=False)
        r = float(p.close.iloc[-1] / h.close.iloc[-1] - 1)
        self.memo[i] = r
        self.new += 1
        if self.new % 50 == 0:
            self.save()
        return r

    def save(self):
        pickle.dump(self.memo, open(self.path, 'wb'))


def make_fn(name, variant, oracle, stats):
    mod = importlib.import_module(f'strategies.{name}')
    fn = getattr(mod, f'{name}_strategy')

    def strategy_fn(d, i, pos):
        sig = fn(d, i, pos, regime=d['_regime'].iat[i])
        if sig is None or pos is not None or variant == 'base':
            return sig
        side = getattr(sig, 'side', None)
        if side not in ('long', 'short'):
            return sig
        r = oracle.ret(i)
        agree = (r > 0) == (side == 'long')
        stats['seen'] += 1
        keep = agree if variant == 'confirm' else not agree
        stats['kept'] += keep
        return sig if keep else None
    return strategy_fn


def set_genome(name):
    mod = importlib.import_module(f'strategies.{name}')
    g = json.load(open(f'{MR}/reports/best_genome_{name}.json'))
    params = copy.deepcopy(mod.PARAMS)
    params.update(g['params'])
    mod.PARAMS = params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategies', default=','.join(STRATS))
    ap.add_argument('--symbols', default='BTCUSDT,ETHUSDT,SOLUSDT')
    ap.add_argument('--variants', default='base,confirm,veto')
    ap.add_argument('--fee', type=float, default=0.00055, help='per side (Bybit perp taker)')
    ap.add_argument('--slip-bps', type=float, default=2.0)
    a = ap.parse_args()

    bt_cfg = BacktestConfig(commission_rate=a.fee, slippage_bps=a.slip_bps)
    rows_path = f'{OUT}/results.jsonl'
    done = set()
    if os.path.exists(rows_path):
        for line in open(rows_path):
            r = json.loads(line)
            done.add((r['strategy'], r['symbol'], r['variant']))

    for sym in a.symbols.split(','):
        t0 = time.time()
        df = load_symbol(sym)
        print(f'{sym}: {len(df)} bars features+regime ready ({time.time() - t0:.0f}s)', flush=True)
        oracle = KronosOracle(sym, df)
        for name in a.strategies.split(','):
            set_genome(name)
            for variant in a.variants.split(','):
                if (name, sym, variant) in done:
                    continue
                stats = {'seen': 0, 'kept': 0}
                t0 = time.time()
                bt = Backtester(bt_cfg, RiskConfig())
                try:
                    trades, eq, m = bt.run(df, make_fn(name, variant, oracle, stats), name)
                except Exception as e:  # keep the sweep going
                    print(f'  {name} {variant} FAILED: {e!r}', flush=True)
                    continue
                pnl = trades['pnl_net'].sum() if len(trades) else 0.0
                row = {'strategy': name, 'symbol': sym, 'variant': variant,
                       'trades': int(len(trades)),
                       'ret_pct': float(eq.iloc[-1] / bt_cfg.initial_capital * 100 - 100) if len(eq) else 0.0,
                       'pnl_net': float(pnl),
                       'fees': float(trades['commission'].sum()) if len(trades) else 0.0,
                       'win_rate': float((trades['pnl_net'] > 0).mean()) if len(trades) else 0.0,
                       'max_dd_pct': float((eq / eq.cummax() - 1).min() * 100) if len(eq) else 0.0,
                       'sharpe': float(getattr(m, 'sharpe_ratio', np.nan)),
                       'kronos_seen': stats['seen'], 'kronos_kept': stats['kept'],
                       'secs': round(time.time() - t0, 1)}
                open(rows_path, 'a').write(json.dumps(row) + '\n')
                print(f"  {name:18} {variant:7} trades {row['trades']:4} ret {row['ret_pct']:+7.2f}% "
                      f"dd {row['max_dd_pct']:6.2f}% ({row['secs']}s)", flush=True)
        oracle.save()
    bh = {}
    for sym in a.symbols.split(','):
        c = pd.read_csv(f'{CACHE}/{sym}_60_ohlcv.csv').close
        bh[sym] = float(c.iloc[-1] / c.iloc[0] * 100 - 100)
    json.dump(bh, open(f'{OUT}/buy_hold.json', 'w'))
    print('buy&hold', bh)


if __name__ == '__main__':
    main()
