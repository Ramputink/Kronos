import numpy as np, pandas as pd
C='/Users/matveypro/kronos_lab/cache/'; rng=np.random.default_rng(1)
def rank(x): return pd.Series(x).rank().values
def sp(a,b): return np.corrcoef(rank(a),rank(b))[0,1]
def resid(y,X):
    X=np.column_stack([np.ones(len(y))]+[rank(x) for x in X]); yr=rank(y)
    return yr-X@np.linalg.lstsq(X,yr,rcond=None)[0]
def partial(a,b,cs): return np.corrcoef(resid(a,cs),resid(b,cs))[0,1]
def mbb(n,L=10):
    st=rng.integers(0,n-L+1,int(np.ceil(n/L))); return np.concatenate([np.arange(s,s+L) for s in st])[:n]
def qlike(f,r): return np.mean(r/f-np.log(r/f)-1)
for sym in ['BTCUSDT','ETHUSDT','SOLUSDT']:
    df=pd.read_csv(C+f'{sym}_60_ohlcv.csv'); z=np.load(C+f'{sym}_60_paths.npz')
    o=z['origins']; P=z['paths']; cl=df.close.values; hi=df.high.values; lo=df.low.values
    keep=o+24<len(cl); o=o[keep]; P=P[keep]; c0=cl[o]; n=len(o)
    lr=np.log(cl); r1=np.diff(lr,prepend=lr[0])
    lp=np.log(np.concatenate([np.repeat(c0[:,None,None],6,1),P[:,:,:,3]],2))
    kv=np.median(np.std(np.diff(lp,axis=2),axis=2),1)
    kv_orig=np.median(np.std(np.diff(np.log(P[:,:,:,3]),axis=2),axis=2),1)
    kpk=np.median(np.sqrt(np.mean(np.log(P[:,:,:,1]/P[:,:,:,2])**2,2)/(4*np.log(2))),1)
    t24=np.array([r1[i-23:i+1].std() for i in o]); t72=np.array([r1[i-71:i+1].std() for i in o])
    t168=np.array([r1[i-167:i+1].std() for i in o])
    pk=np.sqrt((np.log(hi/lo)**2)/(4*np.log(2)))
    pk24=np.array([np.sqrt(np.mean(pk[i-23:i+1]**2)) for i in o])
    pk72=np.array([np.sqrt(np.mean(pk[i-71:i+1]**2)) for i in o])
    rv=np.array([r1[i+1:i+25].std() for i in o])
    base=[t24,t72,t168,pk24,pk72]
    def exp_fit(feats):
        f=np.full(n,np.nan); X=np.column_stack([np.ones(n)]+[np.log(x) for x in feats]); y=np.log(rv)
        for j in range(n):
            tr=np.where(o+24<=o[j])[0]
            if len(tr)>=60: f[j]=np.exp(X[j]@np.linalg.lstsq(X[tr],y[tr],rcond=None)[0])
        return f
    fh=exp_fit(base); fhk=exp_fit(base+[kv_orig]); fhk2=exp_fit(base+[kv_orig,kpk])
    m=~np.isnan(fh)
    print(f'\n{sym} n={n} IC kv_orig={sp(kv_orig,rv):.3f} kv_incl_step0={sp(kv,rv):.3f} kronosPK={sp(kpk,rv):.3f} | t24={sp(t24,rv):.3f} t72={sp(t72,rv):.3f} t168={sp(t168,rv):.3f} pk24={sp(pk24,rv):.3f} pk72={sp(pk72,rv):.3f}')
    for name,cs in [('t24',[t24]),('t72',[t72]),('pk24',[pk24]),('all5',base)]:
        p=partial(kv_orig,rv,cs)
        bs=[partial(kv_orig[i],rv[i],[c[i] for c in cs]) for i in (mbb(n) for _ in range(500))]
        print(f'  partial IC kronos|{name}={p:.3f} CI[{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]')
    h=np.arange(n)>=2*n//3
    print(f'  last-third n={h.sum()}: IC kronos={sp(kv_orig[h],rv[h]):.3f} pk24={sp(pk24[h],rv[h]):.3f} partial|all5={partial(kv_orig[h],rv[h],[c[h] for c in base]):.3f}')
    for nm,f in [('HAR(base)',fh),('HAR+kronos',fhk),('HAR+kronos+kPK',fhk2)]:
        print(f'  OOS n={m.sum()} {nm}: QLIKE={qlike(f[m]**2,rv[m]**2):.4f} logMSE={np.mean((np.log(f[m])-np.log(rv[m]))**2):.4f} IC={sp(f[m],rv[m]):.3f}')
    d=(rv[m]**2/fhk[m]**2-np.log(rv[m]**2/fhk[m]**2))-(rv[m]**2/fh[m]**2-np.log(rv[m]**2/fh[m]**2))
    bs=[d[i].mean() for i in (mbb(len(d)) for _ in range(2000))]
    print(f'  QLIKE diff (HAR+kronos - HAR) mean={d.mean():.4f} CI[{np.percentile(bs,2.5):.4f},{np.percentile(bs,97.5):.4f}] (neg=kronos better)')
