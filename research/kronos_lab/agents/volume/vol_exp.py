import numpy as np, pandas as pd
C='/Users/matveypro/kronos_lab/cache/'
W='/Users/matveypro/kronos_lab/agents/volume/'
rng=np.random.default_rng(0)
FEE=0.0011
def spearman(a,b): return pd.Series(a).rank().corr(pd.Series(b).rank())
def auc(score,label):
    r=pd.Series(score).rank().values; n1=label.sum(); n0=len(label)-n1
    return (r[label].sum()-n1*(n1+1)/2)/(n1*n0)
def ols_oos(X,y,split):
    X=np.column_stack([np.ones(len(y)),X]); b=np.linalg.lstsq(X[:split],y[:split],rcond=None)[0]
    e=y[split:]-X[split:]@b; return 1-(e**2).sum()/((y[split:]-y[:split].mean())**2).sum()
allres={}
for S in ['BTCUSDT','ETHUSDT','SOLUSDT']:
    df=pd.read_csv(C+f'{S}_60_ohlcv.csv'); d=np.load(C+f'{S}_60_paths.npz'); O=d['origins']; P=d['paths']
    ok=O+24<len(df); O=O[ok]; P=P[ok]
    c=df.close.values; v=df.volume.values; h=df.high.values; l=df.low.values
    lr=np.diff(np.log(c),prepend=np.nan)
    dup=np.mean([len(np.unique(P[i,:,:,3].round(4),axis=0)) for i in range(len(O))])
    R=[]
    for i,o in enumerate(O):
        pp=P[i]
        pc=np.concatenate([np.full((6,1),c[o]),pp[:,:,3]],1); plr=np.diff(np.log(pc),axis=1)
        R.append(dict(
          fut_v=np.log(v[o+1:o+25].sum()), fut_rv=np.log(np.std(lr[o+1:o+25])+1e-9),
          fut_rng=np.log(np.log(h[o+1:o+25].max()/l[o+1:o+25].min())),
          fut_ret=np.log(c[o+24]/c[o]),
          tr_v24=np.log(v[o-23:o+1].sum()), tr_v168=np.log(v[o-167:o+1].sum()/7),
          tr_rv24=np.log(np.std(lr[o-23:o+1])), tr_rv168=np.log(np.std(lr[o-167:o+1])),
          tr_rng24=np.log(np.log(h[o-23:o+1].max()/l[o-23:o+1].min())),
          kv=np.log(pp[:,:,4].sum(1).mean()), krv=np.log(plr.std(1).mean()),
          krng=np.log(np.log(pp[:,:,1].max(1)/pp[:,:,2].min(1)).mean()),
          kdisp=np.log(np.log(pp[:,-1,3]/c[o]).std()+1e-6),
          mom24=np.log(c[o]/c[o-24]),
          donch=(1 if c[o]>h[o-48:o].max() else (-1 if c[o]<l[o-48:o].min() else 0))))
    R=pd.DataFrame(R); n=len(R); sp=int(n*2/3)
    res={'n':n,'distinct_paths_avg':dup}
    for tgt,trs,k in [('fut_v',['tr_v24','tr_v168'],'kv'),('fut_rv',['tr_rv24','tr_rv168'],'krv'),('fut_rng',['tr_rng24','tr_rv168'],'krng')]:
        res[f'{tgt}_sp_trail24']=spearman(R[trs[0]],R[tgt]); res[f'{tgt}_sp_kronos']=spearman(R[k],R[tgt])
        res[f'{tgt}_R2oos_trail']=ols_oos(R[trs].values,R[tgt].values,sp)
        res[f'{tgt}_R2oos_kronos']=ols_oos(R[[k]].values,R[tgt].values,sp)
        res[f'{tgt}_R2oos_both']=ols_oos(R[trs+[k]].values,R[tgt].values,sp)
    res['fut_rv_sp_kdisp']=spearman(R.kdisp,R.fut_rv)
    lab=(R.fut_v-R.tr_v168).values>np.log(1.5)
    res['spike_rate']=lab.mean()
    res['spike_auc_trail']=auc((R.tr_v24-R.tr_v168).values,lab)
    res['spike_auc_kronos']=auc((R.kv-R.tr_v168).values,lab)
    labv=(R.fut_rv-R.tr_rv168).values>np.log(1.5)
    res['volburst_rate']=labv.mean()
    res['volburst_auc_trail']=auc((R.tr_rv24-R.tr_rv168).values,labv)
    res['volburst_auc_kronos']=auc((R.krv-R.tr_rv168).values,labv)
    kexp=(R.krv-R.tr_rv24).values
    texp=(R.tr_rv24-R.tr_rv168).values
    fr=R.fut_ret.values
    for name,base in [('mom',np.sign(R.mom24.values)),('donch',R.donch.values.astype(float))]:
        for fname,mask in [('none',np.ones(n,bool)),('kronos_exp',kexp>np.median(kexp)),('trail_squeeze',texp<np.median(texp))]:
            pos=base*mask; g=pos*fr; nt=g-np.abs(pos)*FEE
            res[f'{name}_{fname}_ntr']=int((pos!=0).sum()); res[f'{name}_{fname}_gross']=g.sum(); res[f'{name}_{fname}_net']=nt.sum()
            res[f'{name}_{fname}_net_test']=nt[sp:].sum()
        keep=kexp>np.median(kexp); rs=[]
        for _ in range(2000):
            m=rng.permutation(keep); p2=base*m; rs.append((p2*fr-np.abs(p2)*FEE).sum())
        rs=np.array(rs)
        res[f'{name}_rand_filter_net_mean']=rs.mean(); res[f'{name}_kronos_filter_pctile']=(rs<res[f'{name}_kronos_exp_net']).mean()
    thr=np.median(kexp[:sp]); pos=np.sign(R.mom24.values)*(kexp>thr)
    res['mom_kronos_exp_trainthr_net_test']=(pos*fr-np.abs(pos)*FEE)[sp:].sum()
    res['buyhold_period_logret']=np.log(c[O[-1]+24]/c[O[0]])
    allres[S]=res; R.to_csv(W+f'{S}_feat.csv',index=False)
T=pd.DataFrame(allres); pd.set_option('display.max_rows',200); pd.set_option('display.width',200)
print(T.round(4)); T.to_csv(W+'results.csv')
