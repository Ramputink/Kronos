import numpy as np, pandas as pd
C="/Users/matveypro/kronos_lab/cache/"; FEE=0.00055
rng=np.random.default_rng(0)
def sp(a,b): return np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1]
for sym in ["BTCUSDT","ETHUSDT","SOLUSDT"]:
    df=pd.read_csv(C+f"{sym}_60_ohlcv.csv"); z=np.load(C+f"{sym}_60_paths.npz"); O,P=z["origins"],z["paths"]
    c=df.close.values; h=df.high.values; l=df.low.values
    lr=np.diff(np.log(c),prepend=np.nan); s=pd.Series(lr)
    pk=pd.Series(np.log(h/l)**2/(4*np.log(2)))  # Parkinson per-bar variance
    feats={}
    for hl in [6,12,24,48,96]: feats[f"ewma_hl{hl}"]=s.pow(2).ewm(halflife=hl,adjust=False).mean().pow(.5).values
    for w in [24,72,168]: feats[f"hist{w}"]=s.rolling(w).std().values
    for w in [24,72,168]: feats[f"park{w}"]=pk.rolling(w).mean().pow(.5).values
    rows=[]
    for k,o in enumerate(O):
        if o+24>=len(c) or o<200: continue
        pc=P[k][:,:,3]; pl=np.diff(np.log(np.concatenate([np.full((6,1),c[o]),pc],1)),axis=1)
        r=dict(o=o,kron=pl.std(1).mean(),real=lr[o+1:o+25].std(),ret24=c[o+24]/c[o]-1)
        for f,v in feats.items(): r[f]=v[o]
        rows.append(r)
    d=pd.DataFrame(rows).dropna().reset_index(drop=True); n=len(d); cut=int(n*2/3)
    cols=[k for k in d.columns if k not in("o","real","ret24")]
    spd={f:sp(d[f],d.real) for f in cols}
    best=max([f for f in cols if f!="kron"],key=spd.get)
    def bb(fa,fb,B=2000,blk=10):
        diffs=[]
        for _ in range(B):
            st=rng.integers(0,n-blk,size=n//blk+1); ii=np.concatenate([np.arange(a,a+blk) for a in st])[:n]
            x=d.iloc[ii]; diffs.append(sp(x[fa],x.real)-sp(x[fb],x.real))
        diffs=np.array(diffs); return diffs.mean(), (diffs<=0).mean()
    print(f"\n=== {sym} n={n} (overlapping 24h windows, eff ~{n//2})")
    print(" spearman:",{k:round(v,3) for k,v in sorted(spd.items(),key=lambda x:-x[1])})
    for b in ["ewma_hl12",best]:
        m,p=bb("kron",b); print(f" kron - {b}: diff={m:.3f} boot P(diff<=0)={p:.2f}")
    y=np.log(d.real.values)
    def fit(cs):
        X=np.column_stack([np.ones(n)]+[np.log(d[cc].values) for cc in cs]); b=np.linalg.lstsq(X[:cut],y[:cut],rcond=None)[0]
        return ((y[cut:]-X[cut:]@b)**2).mean()
    for cs in [["ewma_hl12"],["kron"],["park24"],["park72"],["park24","park72","park168"],["hist24","hist72","hist168"],["park24","park72","park168","kron"],["ewma_hl12","kron"]]:
        print(f" OOS mse {'+'.join(cs):35s} {fit(cs):.4f}")
    t=d.iloc[cut:]; print(" last-third spearman:",{f:round(sp(t[f],t.real),3) for f in ["kron","ewma_hl12","park24","park72",best]})
    dd=d.iloc[::2].reset_index(drop=True)
    def strat(f):
        if f=="const": w=np.ones(len(dd))
        else:
            v=dd[f].values; med=pd.Series(v).expanding().median().shift(1).bfill().values
            w=np.clip(med/v,0,3)
        g=w*dd.ret24.values; net=g-2*FEE*w; return g,net
    res={}
    for f in ["kron","ewma_hl12","park24","park72","const"]:
        g,net=strat(f); res[f]=net
        print(f" voltgt {f:10s} gross_sh={g.mean()/g.std()*np.sqrt(365):.2f} net_sh={net.mean()/net.std()*np.sqrt(365):.2f} net_sum={net.sum():.3f} n={len(net)}")
    a,b=res["kron"],res["ewma_hl12"]; nn=len(a); ds=[]
    for _ in range(2000):
        ii=rng.integers(0,nn,nn); ds.append(a[ii].mean()/a[ii].std()-b[ii].mean()/b[ii].std())
    ds=np.array(ds)*np.sqrt(365); print(f" sharpe diff kron-ewma12: {ds.mean():.2f}, 90%CI [{np.percentile(ds,5):.2f},{np.percentile(ds,95):.2f}]")
    print(f" buy&hold total: {c[d.o.iloc[-1]+24]/c[d.o.iloc[0]]-1:.3f}")
