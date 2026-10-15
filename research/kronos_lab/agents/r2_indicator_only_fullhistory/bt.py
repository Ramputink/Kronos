import numpy as np, pandas as pd, json, itertools
C='/Users/matveypro/kronos_lab/cache/'; F='/Users/matveypro/kronos_lab/agents/bybit_features/'
W='/Users/matveypro/kronos_lab/agents/r2_indicator_only_fullhistory/'
SY=['BTCUSDT','ETHUSDT','SOLUSDT']; FEE=0.00055; AN=np.sqrt(8760)

def load(s):
    d=pd.read_csv(C+f'{s}_60_ohlcv.csv',parse_dates=['timestamps']).set_index('timestamps')
    f=pd.read_csv(F+f'{s}_funding.csv',parse_dates=['ts']).set_index('ts')['funding']
    d['fund']=f.reindex(d.index).fillna(0.0)  # funding charged to positions held into this timestamp
    return d

def ind(d):
    c,h,l=d.close,d.high,d.low
    m=c.rolling(20).mean(); sd=c.rolling(20).std()
    d['mid']=m; d['up']=m+2*sd; d['lo']=m-2*sd
    bw=4*sd/m; d['bwexp']=(bw>bw.rolling(100).median())&(bw>bw.shift(1))
    dl=c.diff(); g=dl.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); ls=(-dl.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    d['rsi']=100-100/(1+g/ls)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False).mean(); d['atr']=atr
    up=h.diff(); dn=-l.diff()
    pdm=pd.Series(np.where((up>dn)&(up>0),up,0.),index=d.index); mdm=pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=d.index)
    pdi=100*pdm.ewm(alpha=1/14,adjust=False).mean()/atr; mdi=100*mdm.ewm(alpha=1/14,adjust=False).mean()/atr
    d['adx']=(100*(pdi-mdi).abs()/(pdi+mdi)).ewm(alpha=1/14,adjust=False).mean()
    d['dhi']=h.rolling(48).max().shift(1); d['dlo']=l.rolling(48).min().shift(1)
    return d

def sig(k,r):
    if k=='bbmr': return 1 if r.close<r.lo else (-1 if r.close>r.up else 0)
    if k=='rsimr': return 1 if r.rsi<35 else (-1 if r.rsi>65 else 0)
    if k=='don': return 1 if r.close>r.dhi else (-1 if r.close<r.dlo else 0)
    if k=='bbbo': return 1 if r.close>r.up else (-1 if r.close<r.lo else 0)

def exit_rule(k,p,r):
    if k=='bbmr': return (p>0 and r.close>=r.mid) or (p<0 and r.close<=r.mid)
    if k=='rsimr': return (p>0 and r.rsi>=50) or (p<0 and r.rsi<=50)
    if k=='bbbo': return (p>0 and r.close<r.mid) or (p<0 and r.close>r.mid)
    if k=='don': return False  # exits only by stop / max hold
    return False

def finish(d,pos,stopret,start,end):
    n=len(d); c=d.close.values; ret=np.zeros(n); ret[1:]=c[1:]/c[:-1]-1
    gross=np.zeros(n); gross[1:]=pos[:-1]*ret[1:]
    if stopret is not None:
        m=~np.isnan(stopret); gross[m]=stopret[m]
    fee=np.zeros(n); fee[1:]=np.abs(np.diff(np.r_[0,pos]))[:-1]*FEE
    fee[end-1]+=abs(pos[end-2])*FEE if end>=2 else 0  # close at end
    fund=np.zeros(n); fund[1:]=pos[:-1]*d.fund.values[1:]
    sl=slice(start,end)
    return dict(gross=gross[sl],net=(gross-fee-fund)[sl],nofund=(gross-fee)[sl],pos=pos[sl],fee=fee[sl].sum(),fund=fund[sl].sum())

def run(d,mode,adxth=25,maxhold=24,stopk=2.0,start=0,end=None):
    """Decision at close of bar i, earns bar i+1. Stop checked intrabar on later bars."""
    end=end or len(d); n=len(d); rec=d.to_records()
    pos=np.zeros(n); stopret=np.full(n,np.nan); p=0; ent=0; stop=0.; kind=None; trades=0
    for i in range(max(start,120),end-1):
        r=rec[i]; stopped=False
        if p!=0:
            if (p>0 and r.low<=stop) or (p<0 and r.high>=stop):
                px=min(stop,r.open) if p>0 else max(stop,r.open)  # gap-through fills at open
                stopret[i]=p*(px/rec[i-1].close-1); p=0; stopped=True
            elif i-ent>=maxhold or exit_rule(kind,p,r): p=0
        if p==0 and not stopped:
            if isinstance(mode,tuple): k=mode[1] if ((r.adx>adxth) or r.bwexp) else mode[2]
            else: k=mode
            s=sig(k,r)
            if s and not np.isnan(r.atr):
                p=s; ent=i; kind=k; stop=r.close-s*stopk*r.atr; trades+=1
        pos[i]=p
    o=finish(d,pos,stopret,start,end); o['trades']=trades; return o

def mom(d,start,end):
    n=len(d); pos=np.zeros(n); c=d.close.values
    for i in range(max(start,24),end-1):
        pos[i]=pos[i-1] if (i-start)%24 else np.sign(c[i]/c[i-24]-1)
    o=finish(d,pos,None,start,end); o['trades']=int((np.diff(o['pos'])!=0).sum()); return o

def bh(d,start,end):
    pos=np.zeros(len(d)); pos[start:end-1]=1
    o=finish(d,pos,None,start,end); o['trades']=1; return o

def randbase(d,pos_ref,start,end,rng):
    p=pos_ref; segs=[]; i=0
    while i<len(p):
        if p[i]!=0:
            j=i
            while j<len(p) and p[j]==p[i]: j+=1
            segs.append(j-i); i=j
        else: i+=1
    L=end-start; pos=np.zeros(len(d))
    for s in segs:
        a=start+rng.integers(0,max(1,L-s-1)); pos[a:a+s]=rng.choice([-1,1])
    return finish(d,pos,None,start,end)['nofund']

def sh(x): x=np.asarray(x); return float(x.mean()/x.std()*AN) if x.std()>0 else 0.
def mdd(x): e=np.cumsum(x); return float((e-np.maximum.accumulate(e)).min())
def boot(x,B=1000,blk=168,seed=0):
    rng=np.random.default_rng(seed); x=np.asarray(x); n=len(x); nb=n//blk+1; out=[]
    for _ in range(B):
        st=rng.integers(0,n-blk,nb); idx=(st[:,None]+np.arange(blk)).ravel()[:n]; out.append(sh(x[idx]))
    return [float(v) for v in np.percentile(out,[2.5,97.5])]

D={s:ind(load(s)) for s in SY}; N=len(D['BTCUSDT']); SPLIT=int(N*0.6)
STRATS={'bbmr':'bbmr','rsimr':'rsimr','don48':'don','bbbo':'bbbo',
 'REG_don_bbmr*':('regime','don','bbmr'),'reg_don_rsimr':('regime','don','rsimr'),
 'reg_bbbo_bbmr':('regime','bbbo','bbmr'),'reg_bbbo_rsimr':('regime','bbbo','rsimr')}

# tuning: 2 params (ADX threshold, max hold) on pre-registered REG_don_bbmr, first 60%, pooled net Sharpe
tune=[]
for th,mh in itertools.product([20,25,30],[12,24,48]):
    rs=[run(D[s],STRATS['REG_don_bbmr*'],adxth=th,maxhold=mh,start=0,end=SPLIT)['net'] for s in SY]
    tune.append((th,mh,sh(np.mean(rs,axis=0))))
tune.sort(key=lambda t:-t[2]); bth,bmh=tune[0][:2]
print('tune (adx,maxhold,trainSharpe):',[(a,b,round(c,2)) for a,b,c in tune])

rows=[]; ci={}
for per,(a,b) in {'train':(0,SPLIT),'test':(SPLIT,N),'full':(0,N)}.items():
    allres={}
    for name,m in STRATS.items():
        kw=dict(adxth=bth,maxhold=bmh) if name=='REG_don_bbmr*' else {}
        allres[name]={s:run(D[s],m,start=a,end=b,**kw) for s in SY}
    allres['buyhold']={s:bh(D[s],a,b) for s in SY}
    allres['mom24']={s:mom(D[s],a,b) for s in SY}
    for name,res in allres.items():
        for s in SY:
            r=res[s]; rows.append(dict(per=per,strat=name,sym=s,trades=r['trades'],gross_sh=sh(r['gross']),net_sh=sh(r['net']),
                net_ret_pct=100*r['net'].sum(),mdd_pct=100*mdd(r['net']),fees_pct=100*r['fee'],fund_pct=100*r['fund'],expo=float(np.mean(r['pos']!=0))))
        pn=np.mean([res[s]['net'] for s in SY],axis=0); pg=np.mean([res[s]['gross'] for s in SY],axis=0)
        rows.append(dict(per=per,strat=name,sym='POOLED',trades=sum(res[s]['trades'] for s in SY),gross_sh=sh(pg),net_sh=sh(pn),net_ret_pct=100*pn.sum(),mdd_pct=100*mdd(pn)))
        if per=='test':
            e=dict(net_sh=sh(pn),ci95=boot(pn))
            if name not in('buyhold',):
                rng=np.random.default_rng(1)
                rsh=[sh(np.mean([randbase(D[s],np.r_[np.zeros(a),res[s]['pos']],a,b,rng) for s in SY],axis=0)) for _ in range(200)]
                e.update(rand_mean=float(np.mean(rsh)),rand_p95=float(np.percentile(rsh,95)),p_vs_rand=float(np.mean(np.array(rsh)>=sh(pn))))
            ci[name]=e
R=pd.DataFrame(rows); R.to_csv(W+'results.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
print(R[R.sym=='POOLED'].drop(columns=['fees_pct','fund_pct','expo']).round(2).to_string())
print(R[(R.per=='test')&(R.sym!='POOLED')].round(2).to_string())
for k,v in ci.items(): print(k,{x:(np.round(y,2) if not isinstance(y,list) else [round(z,2) for z in y]) for x,y in v.items()})
we=[]
for s in SY:
    d=D[s]; r=np.log(d.close).diff().dropna(); w=r.index.dayofweek>=5
    t=(r[w].mean()-r[~w].mean())/np.sqrt(r[w].var()/w.sum()+r[~w].var()/(~w).sum())
    we.append(dict(sym=s,wkend_mean_bp_h=r[w].mean()*1e4,wkday_mean_bp_h=r[~w].mean()*1e4,t_diff=t,wkend_vol_ann=r[w].std()*AN,wkday_vol_ann=r[~w].std()*AN,
      vol_ratio=r[w].std()/r[~w].std(),fund_mean_8h_bp=d.fund[d.fund!=0].mean()*1e4,fund_ann_pct_long=d.fund.sum()/len(d)*8760*100))
print(pd.DataFrame(we).round(3).to_string())
print('range',D['BTCUSDT'].index[0],'split',D['BTCUSDT'].index[SPLIT],'end',D['BTCUSDT'].index[-1])
json.dump(dict(tune=tune,best=[bth,bmh],test=ci,weekend=we,split=str(D['BTCUSDT'].index[SPLIT])),open(W+'summary.json','w'),default=float,indent=1)
