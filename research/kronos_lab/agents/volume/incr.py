import numpy as np, pandas as pd
W='/Users/matveypro/kronos_lab/agents/volume/'
rng=np.random.default_rng(1)
def resid(y,X):
    X=np.column_stack([np.ones(len(y)),X]); b=np.linalg.lstsq(X,y,rcond=None)[0]; return y-X@b
def rk(x): return pd.Series(x).rank().values
for S in ['BTCUSDT','ETHUSDT','SOLUSDT']:
    R=pd.read_csv(W+f'{S}_feat.csv'); n=len(R)
    for tgt,trs,k in [('fut_v',['tr_v24','tr_v168'],'kv'),('fut_rv',['tr_rv24','tr_rv168'],'krv')]:
        y=rk(R[tgt]); X=np.column_stack([rk(R[t]) for t in trs]); kk=rk(R[k])
        pc=np.corrcoef(resid(y,X),resid(kk,X))[0,1]
        # block bootstrap (blocks of 10 origins = 5 days) for spearman(kronos)-spearman(trail24)
        diffs=[]
        for _ in range(1000):
            st=rng.integers(0,n-10,n//10); idx=np.concatenate([np.arange(s,s+10) for s in st])
            a=pd.Series(R[k].values[idx]).rank().corr(pd.Series(R[tgt].values[idx]).rank())
            b=pd.Series(R[trs[0]].values[idx]).rank().corr(pd.Series(R[tgt].values[idx]).rank())
            diffs.append(a-b)
        diffs=np.array(diffs)
        print(S,tgt,f'partial_spearman(kronos|trailing)={pc:.3f}',f'diff_sp CI90=[{np.percentile(diffs,5):.3f},{np.percentile(diffs,95):.3f}]')
