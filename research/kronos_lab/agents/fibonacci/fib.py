import numpy as np, pandas as pd, sys, os, json
C='/Users/matveypro/kronos_lab/cache/'
FEE=0.0011; LB=72; H=24
rng=np.random.default_rng(0)

def swing(df,o):
    w=df.iloc[o-LB+1:o+1]
    ih=w.high.values.argmax(); il=w.low.values.argmin()
    return w.high.values[ih], w.low.values[il], ih>il   # up=True if high after low

def sim(df,o,side,tgt,stop):
    """side +1/-1, entry at close[o], exits intrabar, conservative (stop first). returns gross ret"""
    e=df.close.values[o]
    for k in range(1,H+1):
        hi,lo=df.high.values[o+k],df.low.values[o+k]
        if side>0:
            if lo<=stop: return stop/e-1
            if hi>=tgt: return tgt/e-1
        else:
            if hi>=stop: return 1-stop/e
            if lo<=tgt: return 1-tgt/e
    x=df.close.values[o+H]; return side*(x/e-1)

def signal(df,o):
    hh,ll,up=swing(df,o); R=hh-ll; c=df.close.values[o]
    if R<=0: return None
    if up:  # retracement from high, look for long in 38.2-61.8 zone
        z_hi=hh-0.382*R; z_lo=hh-0.618*R
        if z_lo<=c<=z_hi: return dict(side=1,tgt=hh,stop=hh-0.786*R,ll=ll,hh=hh,up=up)
    else:
        z_lo=ll+0.382*R; z_hi=ll+0.618*R
        if z_lo<=c<=z_hi: return dict(side=-1,tgt=ll,stop=ll+0.786*R,ll=ll,hh=hh,up=up)
    return None

def stats(r):
    r=np.asarray(r)
    if len(r)==0: return dict(n=0)
    net=r-FEE
    return dict(n=len(r),gross_mean=round(r.mean()*100,3),net_mean=round(net.mean()*100,3),
                net_sum=round(net.sum()*100,2),win=round((net>0).mean(),3),
                t=round(net.mean()/(net.std(ddof=1)/np.sqrt(len(r))+1e-12),2) if len(r)>2 else None)

out={}
for sym in ['BTCUSDT','ETHUSDT','SOLUSDT']:
    f=C+f'{sym}_60_paths.npz'
    if not os.path.exists(f): continue
    df=pd.read_csv(C+f'{sym}_60_ohlcv.csv'); z=np.load(f); O=z['origins']; P=z['paths']
    ok=[(i,o) for i,o in enumerate(O) if o>=LB and o+H<len(df)]
    res=dict(n_windows=len(ok))
    # --- A: magnet test: are Kronos predicted / realised 24h extremes near fib levels more than chance?
    lv=np.array([0,0.236,0.382,0.5,0.618,0.786,1,1.272,1.618,-0.272,-0.618])
    def nd(x,ll,R,shift=0): # distance (in range units) to nearest fib level
        u=(x-ll)/R+shift; return np.min(np.abs(u-lv))
    dk,dr,dshuf=[],[],[]
    kh_err,kl_err=[],[]
    for i,o in ok:
        hh,ll,up=swing(df,o); R=hh-ll
        kh=np.median(P[i,:,:,1].max(1)); kl=np.median(P[i,:,:,2].min(1))
        rh=df.high.values[o+1:o+H+1].max(); rl=df.low.values[o+1:o+H+1].min()
        dk+= [nd(kh,ll,R),nd(kl,ll,R)]; dr+=[nd(rh,ll,R),nd(rl,ll,R)]
        s=rng.uniform(-0.1,0.1); dshuf+=[nd(rh,ll,R,s),nd(rl,ll,R,s)]
        # does snapping kronos extreme to nearest fib level improve its forecast of realised extreme?
        def snap(x):
            u=(x-ll)/R; return ll+R*lv[np.argmin(np.abs(u-lv))]
        kh_err.append((abs(kh-rh)/R, abs(snap(kh)-rh)/R)); kl_err.append((abs(kl-rl)/R, abs(snap(kl)-rl)/R))
    kh_err=np.array(kh_err); kl_err=np.array(kl_err)
    res['magnet']=dict(kronos_extreme_dist=round(np.mean(dk),4),realised_extreme_dist=round(np.mean(dr),4),
        realised_dist_with_random_shifted_grid=round(np.mean(dshuf),4),
        kronos_high_err_raw=round(kh_err[:,0].mean(),4),kronos_high_err_snapped=round(kh_err[:,1].mean(),4),
        kronos_low_err_raw=round(kl_err[:,0].mean(),4),kronos_low_err_snapped=round(kl_err[:,1].mean(),4))
    # --- B: trading
    fib_r,conf_r,conf2_r,rej_r,rand_r,shuf=[],[],[],[],[],[]
    sig_list=[]
    for i,o in ok:
        s=signal(df,o)
        if s is None: continue
        r=sim(df,o,s['side'],s['tgt'],s['stop']); fib_r.append(r)
        c=df.close.values[o]
        med_ret=np.median(P[i,:,-1,3])/c-1
        if s['side']>0: hit=(P[i,:,:,1].max(1)>=s['tgt']).mean(); stp=(P[i,:,:,2].min(1)<=s['stop']).mean()
        else: hit=(P[i,:,:,2].min(1)<=s['tgt']).mean(); stp=(P[i,:,:,1].max(1)>=s['stop']).mean()
        c1= s['side']*med_ret>0          # median path closes in trade direction
        c2= hit>stp                      # more paths reach target than stop
        sig_list.append((r,c1,c2,s['side'],o))
        (conf_r if c1 else rej_r).append(r)
        if c2: conf2_r.append(r)
    # random direction baseline at same times w/ symmetric 24h hold
    rd=[]
    for (r,c1,c2,side,o) in sig_list:
        sd=rng.choice([-1,1]); rd.append(sd*(df.close.values[o+H]/df.close.values[o]-1))
    # shuffled confirmation: random subset same size as conf_r, 2000 draws
    allr=np.array([x[0] for x in sig_list]); k=len(conf_r)
    sh=[np.mean(rng.choice(allr,k,replace=False))-FEE for _ in range(2000)] if 0<k<len(allr) else [np.nan]
    conf_mean=np.mean(conf_r)-FEE if k else np.nan
    # kronos-only (no fib) on all windows: median path direction, 24h hold
    ko=[np.sign(np.median(P[i,:,-1,3])-df.close.values[o])*(df.close.values[o+H]/df.close.values[o]-1) for i,o in ok]
    bh=df.close.values[ok[-1][1]+H]/df.close.values[ok[0][1]]-1
    bh24=[df.close.values[o+H]/df.close.values[o]-1 for i,o in ok]
    # fib-alone on ALL hourly origins in the same period (bigger sample, overlapping)
    allh=[]
    for o in range(ok[0][1],ok[-1][1]+1):
        s=signal(df,o)
        if s: allh.append(sim(df,o,s['side'],s['tgt'],s['stop']))
    res['trade']=dict(fib_alone=stats(fib_r),fib_plus_kronos_medianDir=stats(conf_r),
        fib_rejected_by_kronos=stats(rej_r),fib_plus_kronos_targetVsStop=stats(conf2_r),
        random_dir_same_times_24h=stats(rd),kronos_alone_24h=stats(ko),
        shuffled_conf_p=round(float(np.mean(np.array(sh)>=conf_mean)),3) if k else None,
        long_every_window_24h=stats(bh24),buyhold_period_pct=round(bh*100,2),
        fib_alone_all_hourly_origins=stats(allh))
    # holdout last third
    cut=ok[int(len(ok)*2/3)][1]
    te=[x for x in sig_list if x[4]>=cut]
    res['holdout_last_third']=dict(fib_alone=stats([x[0] for x in te]),fib_plus_kronos_medianDir=stats([x[0] for x in te if x[1]]),
        fib_plus_kronos_targetVsStop=stats([x[0] for x in te if x[2]]))
    out[sym]=res
print(json.dumps(out,indent=1))
json.dump(out,open('/Users/matveypro/kronos_lab/agents/fibonacci/results.json','w'),indent=1)
