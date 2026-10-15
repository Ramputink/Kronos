import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '/Users/matveypro/kronos_lab/agents/pitchfork')
from pitchfork import pivots, fork_at
C = '/Users/matveypro/kronos_lab/cache'; FEE = 0.0011; H = 24
OUT = '/Users/matveypro/kronos_lab/agents/r2_disagree_preregistered/'
rng = np.random.default_rng(0)
def ema(x, n): return x.ewm(span=n, adjust=False).mean()
def rsi(close, n=14):
    d = np.diff(close, prepend=close[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1/n, adjust=False).mean()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1/n, adjust=False).mean()
    return (100 - 100/(1 + up/dn.replace(0, 1e-12))).values
recs = []; bh = []
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    df = pd.read_csv(f'{C}/{sym}_60_ohlcv.csv'); z = np.load(f'{C}/{sym}_60_paths.npz')
    O = z['origins']; P = z['paths']; ok = O + H < len(df); O = O[ok].astype(int); P = P[ok]
    T = len(O); split = 2*T//3
    c = df.close; cv = c.values; hi = df.high; lo = df.low
    sig = {}
    sig['ema12_48'] = np.sign(ema(c, 12) - ema(c, 48)).values
    macd = ema(c, 12) - ema(c, 26); sig['macd'] = np.sign(macd - ema(macd, 9)).values
    up = hi.rolling(48).max().shift(1); dn = lo.rolling(48).min().shift(1)
    st = np.zeros(len(c)); s = 0
    for i in range(len(c)):
        if cv[i] > up.iloc[i]: s = 1
        elif cv[i] < dn.iloc[i]: s = -1
        st[i] = s
    sig['donchian48'] = st
    r = rsi(cv, 14); sig['rsi14_35_65'] = np.where(r < 35, 1, np.where(r > 65, -1, 0))
    mid = pd.Series(cv).rolling(20).mean().values; sd = pd.Series(cv).rolling(20).std(ddof=0).values
    pb = (cv - (mid - 2*sd)) / (4*sd)
    mr = np.where(pb < 0.2, 1, np.where(pb > 0.8, -1, 0)); sig['bb_relaxed_MR'] = mr; sig['bb_relaxed_BO'] = -mr
    piv = pivots(hi.values, lo.values, 6)
    pf = np.zeros(len(cv))
    for o in O:
        f = fork_at(piv, o, 6)
        if f is None: continue
        med, slope, u, l = f
        zz = (cv[o] - med(o)) / max((u - l)/2, 1e-9)
        pf[o] = -np.sign(zz) if abs(zz) > 0.5 else 0
    sig['pitchfork_rev_k6'] = pf
    for w in range(split, T):
        o = O[w]; c0 = cv[o]; ret = cv[o+H]/c0 - 1
        pc = P[w, :, -1, 3]
        kmed = np.sign(np.median(pc) - c0); kmean = np.sign(pc.mean() - c0)
        fu = (pc > c0).mean(); kcons = 1 if fu >= 4/6 else (-1 if fu <= 2/6 else 0)
        kret = np.median(pc)/c0 - 1
        mom = np.sign(c0/cv[o-24] - 1)
        bh.append(dict(sym=sym, w=w, ret=ret))
        for nm, sg in sig.items():
            sd_ = sg[o]
            if sd_ == 0 or np.isnan(sd_): continue
            recs.append(dict(sym=sym, w=w, rule=nm, side=int(sd_), ret=ret, g=sd_*ret, n=sd_*ret - FEE,
                             kmed=kmed, kmean=kmean, kcons=kcons, kret=kret, mom=mom))
d = pd.DataFrame(recs); B = pd.DataFrame(bh)
d['kag'] = d.kmed == d.side; d['kdis'] = d.kmed == -d.side; d['magree'] = d.mom == d.side
d.to_csv(OUT + 'holdout_trades.csv', index=False)

def boot_ci(sub):
    if len(sub) < 3: return (np.nan, np.nan)
    sub = sub.assign(blk=sub.sym + '_' + (sub.w // 4).astype(str))
    g = sub.groupby('blk').n.agg(['sum', 'count']); S = g['sum'].values; N = g['count'].values
    ms = []
    for _ in range(2000):
        i = rng.integers(0, len(S), len(S)); ms.append(S[i].sum()/N[i].sum())
    return tuple(np.percentile(ms, [2.5, 97.5]) * 1e4)

def sharpe(sub):
    # equal-weight per (sym,origin) slot pnl (mean over rules trading in slot, 0 if flat); 24h book -> sqrt(365)
    agg = sub.groupby(['sym', 'w']).n.mean().reset_index()
    m = B[['sym', 'w']].merge(agg, on=['sym', 'w'], how='left').fillna(0)
    x = m.n.values
    return float(x.mean()/x.std()*np.sqrt(365)) if x.std() > 0 else np.nan

def summ(sub, base=None):
    if len(sub) == 0: return dict(n=0)
    L = sub[sub.side == 1]; S = sub[sub.side == -1]
    out = dict(n=len(sub), pct_long=round(100*len(L)/len(sub), 1),
               gross_bp=round(1e4*sub.g.mean(), 1), net_bp=round(1e4*sub.n.mean(), 1),
               hit=round(float((sub.g > 0).mean()), 3),
               t_eff=round(float(sub.n.mean()/sub.n.std()*np.sqrt(len(sub)/2)), 2) if len(sub) > 2 else np.nan,
               ci95_net_bp=[round(v, 1) for v in boot_ci(sub)], sharpe_ann=round(sharpe(sub), 2),
               long_n=len(L), long_net_bp=round(1e4*L.n.mean(), 1) if len(L) else None,
               short_n=len(S), short_net_bp=round(1e4*S.n.mean(), 1) if len(S) else None)
    if base is not None and 0 < len(sub) < len(base):
        sims = np.array([rng.choice(base.n.values, len(sub), replace=False).mean() for _ in range(2000)])
        out['rand_pctile'] = round(float((sims < sub.n.mean()).mean()*100), 1)
    return out

res = {'buyhold': dict(n=len(B), mean_24h_bp=round(1e4*B.ret.mean(), 1),
                       by_sym={s: round(1e4*g.ret.mean(), 1) for s, g in B.groupby('sym')})}
def variants(x):
    return {'alone': summ(x), 'kronos_agree': summ(x[x.kag], x), 'kronos_disagree': summ(x[x.kdis], x),
            'mom_agree': summ(x[x.magree], x), 'mom_agree&kdis': summ(x[x.magree & x.kdis], x),
            'mom_disagree&kdis': summ(x[~x.magree & x.kdis], x)}
for nm, x in d.groupby('rule'): res[nm] = variants(x)
res['ALL_RULES_STACKED'] = variants(d)

def ols_cluster(y, X, cl):
    X = np.column_stack([np.ones(len(y))] + X); b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b
    XtXi = np.linalg.pinv(X.T @ X); meat = np.zeros((X.shape[1],)*2)
    for g in np.unique(cl):
        m = cl == g; sg = X[m].T @ e[m]; meat += np.outer(sg, sg)
    G = len(np.unique(cl)); adj = G/(G-1)
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi)*adj); return b, b/se
cl = (d.sym + '_' + (d.w//2).astype(str)).values
reg = {}
for lab, kcol in [('primary_kmed', 'kmed'), ('sec_kmean', 'kmean'), ('sec_kcons', 'kcons')]:
    kd = (d[kcol] == -d.side).astype(float).values
    b, t = ols_cluster(d.n.values*1e4, [kd, d.magree.astype(float).values, (d.side == 1).astype(float).values], cl)
    reg[lab] = dict(coef_bp=dict(const=round(b[0], 1), kdis=round(b[1], 1), mom_agree=round(b[2], 1), long=round(b[3], 1)),
                    t=dict(kdis=round(t[1], 2), mom_agree=round(t[2], 2), long=round(t[3], 2)))
per_rule_t = {}
for nm, x in d.groupby('rule'):
    clr = (x.sym + '_' + (x.w//2).astype(str)).values
    b, t = ols_cluster(x.n.values*1e4, [x.kdis.astype(float).values, x.magree.astype(float).values, (x.side == 1).astype(float).values], clr)
    per_rule_t[nm] = dict(kdis_bp=round(b[1], 1), t_kdis=round(t[1], 2), t_mom=round(t[2], 2), t_long=round(t[3], 2), n=len(x))
reg['per_rule'] = per_rule_t
ic = pd.Series(-d.side*d.kret).rank().corr(d.g.rank())
reg['IC_disagree_strength_vs_tradeRet'] = round(float(ic), 3)
reg['IC_per_rule'] = {nm: round(float(pd.Series(-x.side*x.kret).rank().corr(x.g.rank())), 3) for nm, x in d.groupby('rule')}
reg['IC_kronos_raw_vs_ret_holdout'] = round(float(d.drop_duplicates(['sym', 'w']).pipe(lambda q: q.kret.rank().corr(q.ret.rank()))), 3)
reg['corr_kdis_magree'] = round(float(np.corrcoef(d.kdis, d.magree)[0, 1]), 3)
res['regression'] = reg
json.dump(res, open(OUT + 'results.json', 'w'), indent=1, default=float)
print(json.dumps(res['buyhold']))
for k, v in res.items():
    if k in ('buyhold', 'regression'): continue
    print('==', k)
    for vn, s in v.items():
        if s['n'] == 0: print(f'  {vn:18s} n=0'); continue
        print(f"  {vn:18s} n={s['n']:4d} L%={s['pct_long']:5.1f} g={s['gross_bp']:+7.1f} net={s['net_bp']:+7.1f} hit={s['hit']:.2f} t={s['t_eff']:+.2f} CI={s['ci95_net_bp']} SR={s['sharpe_ann']:+.2f} long={s['long_n']}/{s['long_net_bp']} short={s['short_n']}/{s['short_net_bp']} rp={s.get('rand_pctile')}")
print(json.dumps(reg, indent=1))
