import numpy as np, pandas as pd
C='/Users/matveypro/kronos_lab/cache/'
W='/Users/matveypro/kronos_lab/agents/r2_short_horizon_maker/'
rng=np.random.default_rng(0)
rows=[]
for s in ['BTCUSDT','ETHUSDT','SOLUSDT']:
    df=pd.read_csv(C+f'{s}_60_ohlcv.csv'); z=np.load(C+f'{s}_60_paths.npz')
    O=z['origins']; P=z['paths']; c=df.close.values; lo=df.low.values; hi=df.high.values
    for i,o in enumerate(O):
        if o+2>=len(c) or o<4: continue
        r={'sym':s,'o':o,'c0':c[o]}
        r['k1']=np.median(P[i,:,0,3])/c[o]-1
        r['k2']=np.median(P[i,:,1,3])/c[o]-1
        r['rev4']=-(c[o]/c[o-4]-1)
        r['y1']=c[o+1]/c[o]-1; r['y2']=c[o+2]/c[o]-1
        r['lo1']=lo[o+1]; r['hi1']=hi[o+1]
        rows.append(r)
d=pd.DataFrame(rows)
# expanding z-score per symbol using only PAST origins (no lookahead), need >20 history
def ez(x):
    m=x.expanding().mean().shift(1); sd=x.expanding().std().shift(1)
    z=(x-m)/sd; z[x.expanding().count()<=20]=np.nan; return z
d['zk1']=d.groupby('sym').k1.transform(ez); d['zr']=d.groupby('sym').rev4.transform(ez)
d['ens']=0.5*d.zk1+0.5*d.zr
d=d.dropna(subset=['ens']).reset_index(drop=True)
d['t']=d.groupby('sym').cumcount()
T=d.t.max()+1; cut=int(T*2/3)
print('n per sym',d.groupby('sym').size().to_dict(),'pooled',len(d))

def ic(x,y):
    return pd.Series(x).rank().corr(pd.Series(y).rank())
idx_by_t={t:np.where(d.t.values==t)[0] for t in range(T)}
def boot(fn, B=2000, blk=10):
    nb=int(np.ceil(T/blk)); out=[]
    for _ in range(B):
        st=rng.integers(0,T-blk+1,nb); ts=np.concatenate([np.arange(a,a+blk) for a in st])[:T]
        ii=np.concatenate([idx_by_t[t] for t in ts]); out.append(fn(ii))
    return np.percentile(out,[2.5,97.5])
groups=[np.where(d.sym.values==s)[0] for s in d.sym.unique()]
def shiftnull(sig,y,obs,B=1000):
    cnt=0
    for _ in range(B):
        yy=d[y].values.copy()
        for gi in groups:
            k=rng.integers(5,len(gi)-5); yy[gi]=np.roll(d[y].values[gi],k)
        if abs(ic(d[sig].values,yy))>=abs(obs): cnt+=1
    return (cnt+1)/(B+1)
tests=[('k1','y1'),('k2','y2'),('rev4','y1'),('rev4','y2'),('ens','y1'),('ens','y2'),('k1','y2')]
out=[]
a=(d.t<cut).values
for sig,y in tests:
    x=d[sig].values; yv=d[y].values
    obs=ic(x,yv); ci=boot(lambda ii: ic(x[ii],yv[ii]))
    p=shiftnull(sig,y,obs)
    persym={s:round(ic(x[d.sym==s],yv[d.sym==s]),3) for s in d.sym.unique()}
    out.append(dict(signal=sig,target=y,IC=round(obs,3),ci_lo=round(ci[0],3),ci_hi=round(ci[1],3),p_shift=round(p,3),
        IC_first2_3=round(ic(x[a],yv[a]),3),IC_last1_3=round(ic(x[~a],yv[~a]),3),**persym))
icdf=pd.DataFrame(out); print(icdf.to_string())
print('corr(k1,rev4) pooled spearman', round(ic(d.k1.values,d.rev4.values),3))
for y in ['y1','y2']:
    e=d.ens.values; r=d.rev4.values; yv=d[y].values
    dl=ic(e,yv)-ic(r,yv); ci=boot(lambda ii: ic(e[ii],yv[ii])-ic(r[ii],yv[ii]))
    print(f'DeltaIC(ens-rev4) {y}: {dl:.3f} CI [{ci[0]:.3f},{ci[1]:.3f}]  first2/3 {ic(e[a],yv[a])-ic(r[a],yv[a]):.3f} last1/3 {ic(e[~a],yv[~a])-ic(r[~a],yv[~a]):.3f}')
fees={'taker':11e-4,'maker':4e-4}
tr=[]
for sig in ['k1','k2','rev4','ens','buyhold','random']:
    for h,y in [(1,'y1'),(2,'y2')]:
        if sig=='buyhold': sg=np.ones(len(d))
        elif sig=='random': sg=rng.choice([-1,1],len(d))
        else: sg=np.sign(d[sig].values)
        g=sg*d[y].values
        fill=np.where(sg>0, d.lo1.values<d.c0.values, d.hi1.values>d.c0.values)
        for regime in ['taker','maker','maker_fill']:
            if regime=='maker_fill': gg=g[fill]; f=fees['maker']; last=(~a)[fill]
            else: gg=g; f=fees[regime]; last=~a
            n=len(gg); m=gg.mean()*1e4; se=gg.std()/np.sqrt(n)*1e4
            tr.append(dict(sig=sig,h=h,regime=regime,n=n,fill_rate=round(fill.mean(),2) if regime=='maker_fill' else 1.0,
                gross_bp=round(m,2),se_bp=round(se,2),net_bp=round(m-f*1e4,2),
                net_t=round((m-f*1e4)/se,2),net_last1_3=round(gg[last].mean()*1e4-f*1e4,2),
                n_req_net=int(np.ceil((2.8*gg.std()*1e4/max(abs(m-f*1e4),1e-9))**2))))
trd=pd.DataFrame(tr); print(trd.to_string())
for v in [0.03,0.05,0.08,0.10,0.16]: print('IC',v,'n_req(80%pow,5%)',int(7.85/v**2+3))
icdf.to_csv(W+'ic.csv',index=False); trd.to_csv(W+'trading.csv',index=False)
