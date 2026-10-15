"""Adversarial check of 'Kronos path spread adds vol info beyond trailing RV'.
Stronger baselines: Parkinson/GK range vol (24h,168h), EWMA, 12h RV, weekday/time-of-origin dummies.
No lookahead: features use rows<=o, target rows o+1..o+24."""
import numpy as np, pandas as pd
C='/Users/matveypro/kronos_lab/cache/'
rng=np.random.default_rng(0)
def rk(x): return pd.Series(x).rank().values
def resid(y,X):
    X=np.column_stack([np.ones(len(y))]+X); return y-X@np.linalg.lstsq(X,y,rcond=None)[0]
def pcorr(y,k,ctrl):
    return np.corrcoef(resid(rk(k),[rk(c) for c in ctrl]),resid(rk(y),[rk(c) for c in ctrl]))[0,1]
for s in ['BTCUSDT','ETHUSDT','SOLUSDT']:
    df=pd.read_csv(C+f'{s}_60_ohlcv.csv'); z=np.load(C+f'{s}_60_paths.npz')
    c=df.close.values; h=df.high.values; l=df.low.values
    lr=np.r_[np.nan,np.diff(np.log(c))]; pk=(np.log(h/l))**2/(4*np.log(2))
    ts=pd.to_datetime(df.timestamps)
    ew=pd.Series(lr**2).ewm(halflife=12).mean().values
    R=[]
    for i,oi in enumerate(z['origins']):
        if oi+24>=len(c): continue
        pc=z['paths'][i,:,:,3]
        plr=np.diff(np.log(np.c_[np.full(6,c[oi]),pc]),axis=1)
        t=ts[oi]
        R.append(dict(rv=np.sqrt((lr[oi+1:oi+25]**2).sum()),
            rv12=np.sqrt((lr[oi-11:oi+1]**2).sum()), rv24=np.sqrt((lr[oi-23:oi+1]**2).sum()),
            rv168=np.sqrt((lr[oi-167:oi+1]**2).sum()/7),
            pk24=np.sqrt(pk[oi-23:oi+1].sum()), pk168=np.sqrt(pk[oi-167:oi+1].sum()/7),
            ew=np.sqrt(ew[oi]), wkend=float(((t+pd.Timedelta(hours=1)).dayofweek>=5) or ((t+pd.Timedelta(hours=24)).dayofweek>=5)),
            # fraction of next-24h hours falling on weekend
            wkfrac=np.mean([(t+pd.Timedelta(hours=k)).dayofweek>=5 for k in range(1,25)]),
            hr=float(t.hour), k=np.sqrt((plr**2).sum(1)).mean()))
    d=pd.DataFrame(R); n=len(d); y=d.rv.values
    base=[d.rv24,d.rv168]; strong=[d.rv24,d.rv168,d.pk24,d.pk168,d.ew,d.rv12,d.wkfrac,d.hr]
    print(f'\n{s} n={n} (overlapping 24h targets every 12h => ~{n//2} independent)')
    for nm,col in [('k',d.k),('pk24',d.pk24),('pk168',d.pk168),('ew',d.ew),('wkfrac',d.wkfrac)]:
        print(f'  spearman {nm:7s} vs RV: {np.corrcoef(rk(col),rk(y))[0,1]:.3f}')
    print(f'  rho(k, wkfrac) = {np.corrcoef(rk(d.k),rk(d.wkfrac))[0,1]:.3f}')
    for lab,ctrl in [('HAR(rv24,rv168)',base),('strong',strong)]:
        p_all=pcorr(y,d.k,ctrl)
        # non-overlapping: even origins and odd origins separately
        p_ev=pcorr(y[::2],d.k.values[::2],[np.asarray(c_)[::2] for c_ in ctrl])
        p_od=pcorr(y[1::2],d.k.values[1::2],[np.asarray(c_)[1::2] for c_ in ctrl])
        t3=slice(2*n//3,None)
        p_h=pcorr(y[t3],d.k.values[t3],[np.asarray(c_)[t3] for c_ in ctrl])
        # moving block bootstrap (block=10 origins = 5 days)
        bs=[]
        for _ in range(1000):
            st=rng.integers(0,n-10,n//10+1); idx=np.concatenate([np.arange(a,a+10) for a in st])[:n]
            bs.append(pcorr(y[idx],d.k.values[idx],[np.asarray(c_)[idx] for c_ in ctrl]))
        lo,hi=np.percentile(bs,[2.5,97.5])
        print(f'  partial rho | {lab:16s}: all {p_all:.3f} CI[{lo:.3f},{hi:.3f}]  even {p_ev:.3f} odd {p_od:.3f}  last-third {p_h:.3f} (n={n-2*n//3})')
    # OOS log-RV regression, train first 2/3, test last third
    cut=2*n//3; ly=np.log(y)
    def oos(cols):
        X=np.column_stack([np.ones(n)]+[np.log(np.asarray(c_)+1e-12) if c_ is not d.wkfrac and c_ is not d.hr else np.asarray(c_) for c_ in cols])
        b=np.linalg.lstsq(X[:cut],ly[:cut],rcond=None)[0]; e=ly[cut:]-X[cut:]@b
        ae=ly[cut:]-ly[:cut].mean()
        return 1-(e**2).sum()/(ae**2).sum(), np.mean(np.abs(e))
    for lab,cols in [('HAR',base),('HAR+K',base+[d.k]),('PK+wk',[d.pk24,d.pk168,d.wkfrac]),('PK+wk+K',[d.pk24,d.pk168,d.wkfrac,d.k]),('K only',[d.k])]:
        r2,mae=oos(cols); print(f'  OOS {lab:9s} R2={r2:+.3f} MAE(logRV)={mae:.3f}')
