"""Kronos envelope as 24h range for TP/SL and grid bot vs ATR / realised-vol ranges.
No lookahead: everything at origin o uses csv rows <= o and paths[o]."""
import numpy as np, pandas as pd, os, sys, json
C = '/Users/matveypro/kronos_lab/cache'
FEE = 0.00055
H = 24
rng = np.random.default_rng(0)

def atr(d, n=24):
    pc = d.close.shift(1)
    tr = np.maximum(d.high - d.low, np.maximum((d.high - pc).abs(), (d.low - pc).abs()))
    return tr.rolling(n).mean()

def bar_path(o, h, l, c):
    # intrabar ordering heuristic
    return [o, l, h, c] if c >= o else [o, h, l, c]

def grid_sim(prices, c0, L, Hh, n=10):
    """Neutral long-biased grid: levels between L and H. Start: buy inventory for each level above c0
    (to be sold one level up). Buy orders at levels below c0. If price exits [L,H] -> close all (stop).
    Returns net and gross return as fraction of capital (capital = n units of c0 notional)."""
    lv = np.linspace(L, Hh, n + 1)
    # each 'slot' i between lv[i], lv[i+1]: holds 1 unit if bought, sells at lv[i+1]
    hold = np.zeros(n, bool); cost = np.zeros(n)
    hold[lv[:-1] >= c0] = True; cost[hold] = c0  # initial inventory bought at market
    # slots whose buy level < c0 wait for buy at lv[i]; slots with lv[i]<c0<lv[i+1] wait too
    fees = hold.sum() * c0 * FEE
    realised = 0.0
    last = c0; stopped = False
    for p in prices:
        # move from last to p, fill crossings
        if p < last:
            for i in range(n):
                if not hold[i] and p <= lv[i] < last:
                    hold[i] = True; cost[i] = lv[i]; fees += lv[i] * FEE
        else:
            for i in range(n):
                if hold[i] and last < lv[i + 1] <= p:
                    realised += lv[i + 1] - cost[i]; fees += lv[i + 1] * FEE; hold[i] = False
        last = p
        if p < L or p > Hh:
            stopped = True
            break
    # close remaining at last price (stop at break or end of 24h)
    realised += (last - cost[hold]).sum(); fees += hold.sum() * last * FEE
    cap = n * c0
    return realised / cap, (realised - fees) / cap, stopped

def run(sym):
    d = pd.read_csv(f'{C}/{sym}_60_ohlcv.csv')
    z = np.load(f'{C}/{sym}_60_paths.npz')
    org, P = z['origins'], z['paths']
    A = atr(d).values
    lr = np.log(d.close).diff()
    vol24 = (lr.rolling(168).std() * np.sqrt(24)).values  # realised 1w vol scaled to 24h
    rows = []
    for k, o in enumerate(org):
        if o + H >= len(d) or np.isnan(vol24[o]): continue
        c0 = d.close.iloc[o]
        fut = d.iloc[o + 1:o + 1 + H]
        rh, rl = fut.high.max(), fut.low.min()
        ph, pl = P[k][:, :, 1], P[k][:, :, 2]
        rows.append(dict(o=o, c0=c0, rh=rh, rl=rl,
                         kh=ph.max(), kl=pl.min(),                       # full envelope
                         kmh=np.median(ph.max(1)), kml=np.median(pl.min(1)),  # median path envelope
                         atr=A[o], v24=vol24[o],
                         prices=sum((bar_path(*r) for r in fut[['open', 'high', 'low', 'close']].values), [])))
    return pd.DataFrame(rows)

def width(df, m):
    if m == 'kronos_env': return np.log(df.kh / df.c0), np.log(df.c0 / df.kl)
    if m == 'kronos_med': return np.log(df.kmh / df.c0), np.log(df.c0 / df.kml)
    if m == 'kronos_sym': w = np.log(df.kh / df.kl) / 2; return w, w
    if m == 'atr': w = df.atr / df.c0 * np.sqrt(24); return w, w
    if m == 'vol': return df.v24, df.v24
    if m == 'const': w = pd.Series(1.0, index=df.index); return w, w

out = {}
syms = [s for s in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'] if os.path.exists(f'{C}/{s}_60_paths.npz')]
lines = []
for sym in syms:
    df = run(sym); n = len(df); split = int(n * 2 / 3)
    tr, te = df.iloc[:split], df.iloc[split:]
    rup = np.log(df.rh / df.c0); rdn = np.log(df.c0 / df.rl); rrng = np.log(df.rh / df.rl)
    lines.append(f'\n=== {sym} n={n} (train {split}, test {n-split}) ===')
    # 1) raw containment and width
    for m in ['kronos_env', 'kronos_med']:
        u, dn = width(df, m)
        cont = ((rup <= u) & (rdn <= dn)).mean()
        lines.append(f'{m:11s} raw: mean width {100*(u+dn).mean():.2f}% | realised range {100*rrng.mean():.2f}% | '
                     f'both inside {100*cont:.1f}% | high inside {100*(rup<=u).mean():.1f}% | low inside {100*(rdn<=dn).mean():.1f}%')
    # 2) informativeness: corr of predicted width with realised range (log), and partial vs ATR
    lr_ = np.log(rrng)
    for m in ['kronos_env', 'kronos_sym', 'kronos_med', 'atr', 'vol']:
        u, dn = width(df, m)
        lines.append(f'corr(log width {m}, log realised range) = {np.corrcoef(np.log(u+dn), lr_)[0,1]:.3f}')
    # regression: log range ~ log atr + log kronos
    X = np.column_stack([np.ones(n), np.log(2*width(df,'atr')[0]), np.log(sum(width(df,'kronos_env')))])
    for cols, name in [([0,1],'atr only'), ([0,2],'kronos only'), ([0,1,2],'atr+kronos')]:
        b = np.linalg.lstsq(X[:split][:, cols], lr_.values[:split], rcond=None)[0]
        res = lr_.values[split:] - X[split:][:, cols] @ b
        r2 = 1 - (res**2).mean() / ((lr_.values[split:] - lr_.values[:split].mean())**2).mean()
        lines.append(f'OOS R2 log-range [{name}] = {r2:.3f}  coefs {np.round(b,3)}')
    # asymmetry: does kronos skew (up width - down width) predict realised skew?
    u, dn = width(df, 'kronos_env')
    lines.append(f'corr(kronos up-down width skew, realised up-down) = {np.corrcoef(u-dn, rup-rdn)[0,1]:.3f}')
    # 3) calibrated ranges: scale k on train to hit 70% two-sided containment, evaluate on test
    for target in [0.5, 0.7]:
        for m in ['kronos_env', 'kronos_sym', 'kronos_med', 'atr', 'vol', 'const']:
            u, dn = width(df, m)
            ks = np.linspace(0.001, 30, 30000) if m == 'const' else np.linspace(0.05, 6, 6000)
            if m == 'const': u = u * 0.001; dn = dn * 0.001; ks = np.linspace(1, 300, 3000)
            ctr = np.array([((rup[:split] <= k*u[:split]) & (rdn[:split] <= k*dn[:split])).mean() for k in ks])
            k = ks[np.argmin(np.abs(ctr - target))]
            cte = ((rup[split:] <= k*u[split:]) & (rdn[split:] <= k*dn[split:])).mean()
            wte = 100*(k*(u+dn))[split:].mean()
            lines.append(f'target {target:.0%} {m:11s} k={k:.2f} test containment {100*cte:.1f}% mean width {wte:.2f}%')
    # 4) grid bot on test with calibrated 70% ranges, plus TP/SL bracket
    res = {}
    for m in ['kronos_env', 'kronos_sym', 'kronos_med', 'atr', 'vol', 'const']:
        u, dn = width(df, m)
        if m == 'const': u = u*0.001; dn = dn*0.001; ks = np.linspace(1, 300, 3000)
        else: ks = np.linspace(0.05, 6, 6000)
        ctr = np.array([((rup[:split] <= k*u[:split]) & (rdn[:split] <= k*dn[:split])).mean() for k in ks])
        k = ks[np.argmin(np.abs(ctr - 0.7))]
        for part, sub in [('all', df.index), ('test', df.index[split:])]:
            g = []; br = []
            for i in sub:
                r = df.loc[i]; Hh = r.c0*np.exp(k*u[i]); L = r.c0*np.exp(-k*dn[i])
                gg, gn, st = grid_sim(r.prices, r.c0, L, Hh)
                g.append((gg, gn, st))
                # long bracket: TP at H, SL at L, else exit at 24h close
                ex = r.prices[-1]
                for p in r.prices:
                    if p >= Hh: ex = Hh; break
                    if p <= L: ex = L; break
                br.append(ex / r.c0 - 1)
            g = np.array(g, float); br = np.array(br)
            res[(m, part)] = dict(grid_gross=g[:,0].mean(), grid_net=g[:,1].mean(), stop_rate=g[:,2].mean(),
                                  brk_gross=br.mean(), brk_net=br.mean()-2*FEE, n=len(sub))
    bh_all = (df.prices.apply(lambda x: x[-1]) / df.c0 - 1)
    for part, sub in [('all', df.index), ('test', df.index[split:])]:
        lines.append(f'-- {part} (n={len(sub)}): buy&hold 24h mean {100*bh_all[sub].mean():.3f}%  (gross)')
        for m in ['kronos_env', 'kronos_sym', 'kronos_med', 'atr', 'vol', 'const']:
            r = res[(m, part)]
            lines.append(f'   {m:11s} GRID gross {100*r["grid_gross"]:+.3f}% net {100*r["grid_net"]:+.3f}% stopped {100*r["stop_rate"]:.0f}% | '
                         f'LONG BRACKET gross {100*r["brk_gross"]:+.3f}% net {100*r["brk_net"]:+.3f}%')
    # random baseline: kronos widths shuffled across windows (same distribution, no timing info)
    u, dn = width(df, 'kronos_env')
    rs = []
    for rep in range(20):
        perm = rng.permutation(n); us, ds = u.values[perm], dn.values[perm]
        ks = np.linspace(0.05, 6, 1200)
        ctr = np.array([((rup[:split] <= k*us[:split]) & (rdn[:split] <= k*ds[:split])).mean() for k in ks])
        k = ks[np.argmin(np.abs(ctr - 0.7))]
        rs.append([((rup[split:] <= k*us[split:]) & (rdn[split:] <= k*ds[split:])).mean(), 100*(k*(us+ds))[split:].mean(),
                   np.mean([grid_sim(df.prices[i], df.c0[i], df.c0[i]*np.exp(-k*ds[i]), df.c0[i]*np.exp(k*us[i]))[1] for i in df.index[split:]])])
    rs = np.array(rs)
    lines.append(f'shuffled-kronos-width baseline (20 reps, test): containment {100*rs[:,0].mean():.1f}% width {rs[:,1].mean():.2f}% grid net {100*rs[:,2].mean():+.3f}% (sd {100*rs[:,2].std():.3f})')
print('\n'.join(lines))
open('/Users/matveypro/kronos_lab/agents/range_tp_sl/results.txt', 'w').write('\n'.join(lines))
