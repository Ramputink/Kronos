import numpy as np, pandas as pd, os
C='/Users/matveypro/kronos_lab/cache'
def rk(a): return pd.Series(a).rank().values
for sym in ['BTCUSDT','ETHUSDT','SOLUSDT']:
    f=f'{C}/{sym}_60_paths.npz'
    if not os.path.exists(f): print('missing',sym); continue
    cl=pd.read_csv(f'{C}/{sym}_60_ohlcv.csv').close.values
    z=np.load(f); O=z['origins']; P=z['paths']; k=O+24<len(cl); O=O[k]; P=P[k]; c0=cl[O]
    med=np.median(P[:,:,:,3]/c0[:,None,None]-1,1)
    rev=-(c0/cl[O-1]-1)
    for h in [1,2,3]:
        rh=cl[O+h]/c0-1
        y,k1,x=rk(rh),rk(med[:,h-1]),rk(rev)
        X=np.column_stack([np.ones_like(x),x]); res_k=k1-X@np.linalg.lstsq(X,k1,rcond=None)[0]; res_y=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
        print(f'{sym} h={h}: corr(K,rev1h)={np.corrcoef(k1,x)[0,1]:+.2f} IC_K={np.corrcoef(k1,y)[0,1]:+.3f} IC_rev={np.corrcoef(x,y)[0,1]:+.3f} partial IC_K|rev={np.corrcoef(res_k,res_y)[0,1]:+.3f}')
