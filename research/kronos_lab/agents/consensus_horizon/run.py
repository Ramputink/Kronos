import numpy as np, pandas as pd, os, sys
C='/Users/matveypro/kronos_lab/cache'; FEE=0.0011
rng=np.random.default_rng(0)
def spear(a,b):
    a=pd.Series(a).rank(); b=pd.Series(b).rank(); return np.corrcoef(a,b)[0,1]
def stats(sig, r24, name):
    m=sig!=0; n=m.sum()
    if n==0: return dict(name=name,n=0)
    g=sig[m]*r24[m]; net=g-FEE
    return dict(name=name,n=int(n),hit=round((g>0).mean(),3),gross_bp=round(g.mean()*1e4,1),net_bp=round(net.mean()*1e4,1),
                t_net=round(net.mean()/(net.std(ddof=1)/np.sqrt(n)+1e-12),2),sum_net=round(net.sum()*0.5,4))  # *0.5: 12h spacing, 24h hold overlap
out=[]
for sym in ['BTCUSDT','ETHUSDT','SOLUSDT']:
    f=f'{C}/{sym}_60_paths.npz'
    if not os.path.exists(f): print('missing',sym); continue
    df=pd.read_csv(f'{C}/{sym}_60_ohlcv.csv'); cl=df.close.values
    z=np.load(f); O=z['origins']; P=z['paths']
    keep=O+24<len(cl); O=O[keep]; P=P[keep]; W=len(O)
    c0=cl[O]
    pr=P[:,:,:,3]/c0[:,None,None]-1          # (W,6,24)
    real=np.stack([cl[O+k]/c0-1 for k in range(1,25)],1)  # (W,24)
    med=np.median(pr,1)                       # (W,24)
    r24=real[:,23]
    print(f'\n===== {sym} W={W}  period {df.timestamps.iloc[O[0]]} -> {df.timestamps.iloc[O[-1]+24]}')
    # IC per step
    ic_step=[spear(med[:,k],real[:,k]) for k in range(24)]
    ic_to24=[spear(med[:,k],r24) for k in range(24)]
    # incremental: predicted step k->k+1 change vs realised
    print('IC(med pred step k, realised step k):',' '.join(f'{k+1}:{v:+.2f}' for k,v in enumerate(ic_step)))
    print('IC(med pred step k, realised 24h):  ',' '.join(f'{k+1}:{v:+.2f}' for k,v in enumerate(ic_to24)))
    slope=np.array([np.polyfit(np.arange(24),med[i],1)[0] for i in range(W)])
    print('IC slope(median path) vs r24: %+.3f' % spear(slope,r24))
    npos=(pr[:,:,23]>0).sum(1)
    disp=pr[:,:,23].std(1)
    print('IC frac paths up vs r24: %+.3f' % spear(npos,r24))
    print('consensus distribution (#paths up):',np.bincount(npos,minlength=7))
    # strategies
    mom=np.sign(c0/cl[O-24]-1)   # indicator alone: 24h momentum
    sma=np.sign(c0-pd.Series(cl).rolling(48).mean().values[O])
    ksign=np.sign(med[:,23])
    cons=np.where(npos>=5,1,np.where(npos<=1,-1,0))
    cons6=np.where(npos==6,1,np.where(npos==0,-1,0))
    rows=[stats(np.ones(W),r24,'buy&hold(long every origin)'),
          stats(ksign,r24,'K median sign'),
          stats(np.sign(slope),r24,'K median slope sign'),
          stats(cons,r24,'K consensus>=5/6'),
          stats(cons6,r24,'K consensus 6/6'),
          stats(-cons,r24,'K consensus>=5/6 CONTRARIAN'),
          stats(mom,r24,'MOM24 alone'),
          stats(np.where(mom==cons,mom,0),r24,'MOM24 & K cons agree'),
          stats(np.where((mom==ksign),mom,0),r24,'MOM24 & K sign agree'),
          stats(sma,r24,'SMA48 alone'),
          stats(np.where(sma==ksign,sma,0),r24,'SMA48 & K sign agree'),
          stats(np.where(npos>=5,1,0),r24,'K long-only cons>=5'),]
    # random baseline: shuffled consensus signal, 1000 draws
    rn=[ (rng.permutation(cons)*r24)[cons!=0].mean()*1e4 for _ in range(1000)]
    for r in rows: r['sym']=sym; print(r)
    print('random shuffled-cons gross_bp: mean %.1f, 5-95%% [%.1f, %.1f]'%(np.mean(rn),*np.percentile(rn,[5,95])))
    # holdout: first 2/3 vs last 1/3 for consensus + IC
    s=int(W*2/3)
    for nm,sl in [('first2/3',slice(0,s)),('last1/3',slice(s,W))]:
        print(f'  {nm}: IC med24 {spear(med[sl,23],r24[sl]):+.3f}  IC slope {spear(slope[sl],r24[sl]):+.3f}  IC npos {spear(npos[sl],r24[sl]):+.3f}', stats(cons[sl],r24[sl],'cons'))
    # non-overlapping subsample (every other origin)
    print('  non-overlap IC med24 even/odd: %+.3f %+.3f'%(spear(med[::2,23],r24[::2]),spear(med[1::2,23],r24[1::2])))
    # does consensus/dispersion predict |r24| (volatility)?
    print('  IC dispersion vs |r24|: %+.3f ; IC |med24| vs |r24|: %+.3f'%(spear(disp,np.abs(r24)),spear(np.abs(med[:,23]),np.abs(r24))))
    out+= [dict(sym=sym,step=k+1,ic_step=ic_step[k],ic_to24=ic_to24[k]) for k in range(24)]
pd.DataFrame(out).to_csv('/Users/matveypro/kronos_lab/agents/consensus_horizon/ic_by_step.csv',index=False)
