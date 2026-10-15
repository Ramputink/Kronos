"""Economic test of Kronos path-vol vs HAR/Parkinson baseline vol forecaster.
Walk-forward expanding OLS on log realised 24h vol. Three pre-registered uses:
 (a) vol-targeted long, non-overlapping 24h windows
 (b) regime gate: low forecast vol -> Bollinger(20,2) MR, else Donchian48 (or flat); 12h segments
 (c) 24h momentum entry with stop at 1.5 * forecast 24h vol
All net of 0.055%/side. No lookahead: forecaster at origin t trains on origins j with O[j]+24 <= O[t].
"""
import numpy as np, pandas as pd, json, sys
C = '/Users/matveypro/kronos_lab/cache/'
FEE = 0.00055; K_STOP = 1.5; MIN_TRAIN = 60; NSHUF = 20; B = 2000
rng = np.random.default_rng(7)


def sharpe(x, ppy):
    return x.mean() / x.std() * np.sqrt(ppy) if x.std() > 0 else 0.0


def mdd(x):
    eq = np.cumsum(x); return float((np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max())


def metrics(x, ppy):
    return dict(sh=sharpe(x, ppy), mdd=mdd(x), net=float(x.sum()))


def boot_diff(a, b, ppy, blk):
    n = len(a); out = {'sh': [], 'mdd': [], 'net': []}
    for _ in range(B):
        st = rng.integers(0, n - blk + 1, n // blk + 1); ii = np.concatenate([np.arange(s, s + blk) for s in st])[:n]
        ma, mb = metrics(a[ii], ppy), metrics(b[ii], ppy)
        for k in out: out[k].append(ma[k] - mb[k])
    return {k: (float(np.mean(a_ := np.array(v))), float(np.percentile(a_, 5)), float(np.percentile(a_, 95))) for k, v in out.items()}


def state_bb(c, n=20, k=2.0):
    s = pd.Series(c); m = s.rolling(n).mean().values; sd = s.rolling(n).std().values
    pos = np.zeros(len(c)); p = 0
    for t in range(len(c)):
        if np.isnan(m[t]): pos[t] = 0; continue
        if p == 0:
            if c[t] < m[t] - k * sd[t]: p = 1
            elif c[t] > m[t] + k * sd[t]: p = -1
        elif p == 1 and c[t] >= m[t]: p = 0
        elif p == -1 and c[t] <= m[t]: p = 0
        pos[t] = p
    return pos


def state_don(c, h, l, n=48):
    hh = pd.Series(h).rolling(n).max().shift(1).values; ll = pd.Series(l).rolling(n).min().shift(1).values
    pos = np.zeros(len(c)); p = 0
    for t in range(len(c)):
        if not np.isnan(hh[t]):
            if c[t] > hh[t]: p = 1
            elif c[t] < ll[t]: p = -1
        pos[t] = p
    return pos


def run(sym):
    df = pd.read_csv(C + f'{sym}_60_ohlcv.csv'); z = np.load(C + f'{sym}_60_paths.npz'); O, P = z['origins'], z['paths']
    c, h, l, op = df.close.values, df.high.values, df.low.values, df.open.values
    ts = pd.to_datetime(df.timestamps)
    lr = np.r_[np.nan, np.diff(np.log(c))]; pk = np.log(h / l) ** 2 / (4 * np.log(2))
    keep = O + 24 < len(c); O, P = O[keep], P[keep]; n = len(O)
    tr_ = lambda x, o, w: np.sqrt(np.nansum(x[o - w + 1:o + 1]) * 24 / w)
    rv = np.array([np.sqrt(np.sum(lr[o + 1:o + 25] ** 2)) for o in O])
    kr = []
    for i, o in enumerate(O):
        pc = P[i][:, :, 3]; pl = np.diff(np.log(np.c_[np.full(6, c[o]), pc]), axis=1)
        kr.append(np.sqrt((pl ** 2).sum(1)).mean())
    kr = np.log(np.array(kr))
    wk = np.array([(ts.iloc[o + 1:o + 25].dt.dayofweek >= 5).mean() for o in O])  # known calendar
    X0 = np.column_stack([np.ones(n)] + [np.log([tr_(pk, o, w) for o in O]) for w in (24, 168)]
                         + [np.log([tr_(lr ** 2, o, w) for o in O]) for w in (24, 168)] + [wk])
    y = np.log(rv)

    def wf(X):
        f = np.full(n, np.nan)
        for t in range(MIN_TRAIN, n):
            tr = np.where(O + 24 <= O[t])[0]
            b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; f[t] = np.exp(X[t] @ b)
        return f
    F = {'base': wf(X0), 'kron': wf(np.c_[X0, kr])}
    shuf = [wf(np.c_[X0, rng.permutation(kr)]) for _ in range(NSHUF)]
    ev = np.arange(MIN_TRAIN, n)
    yt = y[ev]; ss = ((yt - yt.mean()) ** 2).sum()
    r2 = {k: float(1 - ((yt - np.log(v[ev])) ** 2).sum() / ss) for k, v in F.items()}
    # trailing realised vol for target / reference
    rv24 = np.array([tr_(lr ** 2, o, 24) for o in O])

    pos_bb = state_bb(c); pos_don = state_don(c, h, l)
    hret = np.r_[np.diff(c) / c[:-1], 0]  # hret[t] = return from t to t+1

    def strat_a(f, idx):
        out = []
        for t in idx:
            o = O[t]
            if f is None: w = 1.0
            else:
                past = rv[np.where(O + 24 <= o)[0]]; w = np.clip(np.median(past) / f[t], 0, 3)
            g = w * (c[o + 24] / c[o] - 1); out.append(g - 2 * FEE * w)
        return np.array(out), None

    def strat_b(f, idx, hi='don'):
        out = []; prev = 0.0
        for t in idx:
            o = O[t]
            if f is None: low = (hi == 'bb')  # no gate: always bb or always don
            else:
                past = f[ev[0]:t + 1]; low = f[t] < np.median(past)
            seg = 0.0
            for s in range(o, o + 12):
                if f is None: p = pos_bb[s] if hi == 'bb' else pos_don[s]
                else: p = pos_bb[s] if low else (pos_don[s] if hi == 'don' else 0.0)
                seg += p * hret[s] - FEE * abs(p - prev); prev = p
            out.append(seg)
        return np.array(out), None

    def strat_c(f, idx):
        out = []; stops = 0
        for t in idx:
            o = O[t]; side = np.sign(c[o] / c[o - 24] - 1) or 1.0; e = c[o]; r = None
            if f is not None:
                d = K_STOP * f[t]; sp = e * np.exp(-side * d)
                for s in range(o + 1, o + 25):
                    if side > 0 and l[s] <= sp: r = min(op[s], sp) / e - 1; break
                    if side < 0 and h[s] >= sp: r = -(max(op[s], sp) / e - 1); break
            if r is None: r = side * (c[o + 24] / e - 1)
            else: stops += 1
            out.append(r - 2 * FEE)
        return np.array(out), stops / len(idx)

    res = {'sym': sym, 'n_origins': n, 'r2_walkforward': r2,
           'bh_full': float(c[O[ev[-1]] + 24] / c[O[ev[0]]] - 1)}
    cut = MIN_TRAIN + 2 * (n - MIN_TRAIN) // 3  # not used; last third of all origins below
    last3 = int(n * 2 / 3)
    for period, start in [('full', ev[0]), ('last3', max(last3, ev[0]))]:
        res[period] = {}
        idx12 = np.arange(start, n); idx24 = idx12[::2]
        res[period]['bh'] = float(c[O[idx12[-1]] + 24] / c[O[idx12[0]]] - 1)
        for name, fn, idx, ppy, blk, extra in [
                ('a_voltarget', strat_a, idx24, 365, 5, {}),
                ('b_gate_don', strat_b, idx12, 730, 10, {'hi': 'don'}),
                ('b_gate_flat', strat_b, idx12, 730, 10, {'hi': 'flat'}),
                ('c_stop', strat_c, idx24, 365, 5, {})]:
            rk, sk = fn(F['kron'], idx, **extra); rb, sb = fn(F['base'], idx, **extra)
            d = {'n': len(idx), 'kron': metrics(rk, ppy), 'base': metrics(rb, ppy),
                 'diff_kron_minus_base_CI90': boot_diff(rk, rb, ppy, blk)}
            sh = [fn(s, idx, **extra) for s in shuf]
            d['shuffled'] = {m: float(np.mean([metrics(x[0], ppy)[m] for x in sh])) for m in ('sh', 'mdd', 'net')}
            d['shuffled_kron_pctile_sh'] = float(np.mean([metrics(x[0], ppy)['sh'] < d['kron']['sh'] for x in sh]))
            if name == 'a_voltarget': d['nogate_const1'] = metrics(fn(None, idx)[0], ppy)
            if name.startswith('b'):
                d['nogate_bb'] = metrics(fn(None, idx, hi='bb')[0], ppy)
                d['nogate_don'] = metrics(fn(None, idx, hi='don')[0], ppy)
            if name == 'c_stop':
                d['stoprate_kron'] = sk; d['stoprate_base'] = sb
                d['stoprate_shuf'] = float(np.mean([x[1] for x in sh]))
                d['nostop'] = metrics(fn(None, idx)[0], ppy)
                rr = strat_c(np.where(np.isnan(F['base']), np.nan, rv24), idx)  # trailing rv24 stop
                d['trail_rv24_stop'] = metrics(rr[0], ppy); d['stoprate_trail'] = rr[1]
            res[period][name] = d
    return res


if __name__ == '__main__':
    allr = [run(s) for s in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']]
    json.dump(allr, open('/Users/matveypro/kronos_lab/agents/r2_vol_gate_economics/results.json', 'w'), indent=1)
    for r in allr:
        print(f"\n===== {r['sym']} n={r['n_origins']} WF R2 {r['r2_walkforward']}")
        for p in ('full', 'last3'):
            print(f" -- {p}  B&H {r[p]['bh']:+.3f}")
            for k, d in r[p].items():
                if k == 'bh': continue
                f = lambda m: f"sh {m['sh']:+.2f} mdd {m['mdd']:.3f} net {m['net']:+.3f}"
                ci = d['diff_kron_minus_base_CI90']
                line = f"  {k:12s} n={d['n']:3d} KRON[{f(d['kron'])}] BASE[{f(d['base'])}] SHUF[{f(d['shuffled'])}] kron>shuf {d['shuffled_kron_pctile_sh']:.2f}"
                print(line)
                print("     diff K-B: " + "  ".join(f"{m} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]" for m, v in ci.items()))
                for x in ('nogate_const1', 'nogate_bb', 'nogate_don', 'nostop', 'trail_rv24_stop'):
                    if x in d: print(f"     {x}: {f(d[x])}")
                if 'stoprate_kron' in d:
                    print(f"     stoprate kron {d['stoprate_kron']:.2f} base {d['stoprate_base']:.2f} shuf {d['stoprate_shuf']:.2f} trail {d['stoprate_trail']:.2f}")
