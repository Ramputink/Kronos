# Zero-shot baseline metrics a fine-tuned model must beat (computed from cached paths; no lookahead in features)
import numpy as np, pandas as pd
C='/Users/matveypro/kronos_lab/cache/'
for s in ['BTCUSDT','ETHUSDT','SOLUSDT']:
    df=pd.read_csv(C+s+'_60_ohlcv.csv'); z=np.load(C+s+'_60_paths.npz'); O=z['origins']; P=z['paths']
    c=df.close.values; h=df.high.values; l=df.low.values
    rows=[]
    for i,o in enumerate(O):
        if o+24>=len(c): continue
        c0=c[o]; pr=np.log(P[i,:,-1,3]/c0)          # 6 sampled 24h log returns
        real=np.log(c[o+24]/c0)
        # predicted vol: mean path realized hourly vol; realized: actual hourly vol
        pc=np.concatenate([np.full((6,1),c0),P[i,:,:,3]],1); pv=np.log(pc[:,1:]/pc[:,:-1]).std(1).mean()
        rv=np.log(c[o+1:o+25]/c[o:o+24]).std()
        hv=np.log(c[o-23:o+1]/c[o-24:o]).std()   # trailing 24h vol (indicator-only baseline)
        prange=np.log(P[i,:,:,1].max(1)/P[i,:,:,2].min(1)).mean(); rrange=np.log(h[o+1:o+25].max()/l[o+1:o+25].min())
        crps=np.abs(pr-real).mean()-0.5*np.abs(pr[:,None]-pr[None,:]).mean()
        rows.append(dict(pm=pr.mean(),real=real,inside=pr.min()<=real<=pr.max(),pv=pv,rv=rv,hv=hv,prange=prange,rrange=rrange,crps=crps,
                         crps0=np.abs(real)))  # degenerate zero-return forecast CRPS
    r=pd.DataFrame(rows); n=len(r)
    print(f"{s} n={n} IC(pred mean,real)={r.pm.rank().corr(r.real.rank()):+.3f} hit={(np.sign(r.pm)==np.sign(r.real)).mean():.3f} "
          f"6path-range coverage={r.inside.mean():.2f} (ideal 5/7=0.71) | vol corr kronos={r.pv.corr(r.rv):.3f} trailing24h={r.hv.corr(r.rv):.3f} "
          f"| range corr={r.prange.corr(r.rrange):.3f} pred/real range ratio={r.prange.mean()/r.rrange.mean():.2f} pv/rv={r.pv.mean()/r.rv.mean():.2f} "
          f"| CRPS kronos={r.crps.mean()*100:.3f}% vs point-zero={r.crps0.mean()*100:.3f}%")

print("--- fair ensemble CRPS (6 members, diag excluded) Kronos vs Gaussian(0, trailing-24h vol*sqrt24) with 6 draws ---")
rng=np.random.default_rng(0)
def fcrps(x,y):
    m=len(x); return np.abs(x-y).mean()-np.abs(x[:,None]-x[None,:]).sum()/(2*m*(m-1))
for s in ['BTCUSDT','ETHUSDT','SOLUSDT']:
    df=pd.read_csv(C+s+'_60_ohlcv.csv'); z=np.load(C+s+'_60_paths.npz'); O=z['origins']; P=z['paths']; c=df.close.values
    k=[];g=[];spr=[]
    for i,o in enumerate(O):
        if o+24>=len(c): continue
        real=np.log(c[o+24]/c[o]); pr=np.log(P[i,:,-1,3]/c[o])
        hv=np.log(c[o-167:o+1]/c[o-168:o]).std()*np.sqrt(24)
        k.append(fcrps(pr,real)); g.append(fcrps(rng.normal(0,hv,6),real)); spr.append(pr.std()/hv)
    print(f"{s} n={len(k)} CRPS kronos={np.mean(k)*100:.3f}%  gauss-trailing7d={np.mean(g)*100:.3f}%  kronos endpoint spread / trailing vol={np.mean(spr):.2f}")
