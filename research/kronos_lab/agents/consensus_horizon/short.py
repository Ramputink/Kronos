import numpy as np, pandas as pd, os
C='/Users/matveypro/kronos_lab/cache'; FEE=0.0011
def spear(a,b): return np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1]
for sym in ['BTCUSDT','ETHUSDT','SOLUSDT']:
    f=f'{C}/{sym}_60_paths.npz'
    if not os.path.exists(f): print('missing',sym); continue
    cl=pd.read_csv(f'{C}/{sym}_60_ohlcv.csv').close.values
    z=np.load(f); O=z['origins']; P=z['paths']; k=O+24<len(cl); O=O[k]; P=P[k]; W=len(O); c0=cl[O]
    pr=P[:,:,:,3]/c0[:,None,None]-1; med=np.median(pr,1)
    s=int(W*2/3)
    print(f'== {sym} W={W}')
    for h in [1,2,3,4,6]:
        rh=cl[O+h]/c0-1
        sig=np.sign(med[:,h-1]); npos=(pr[:,:,h-1]>0).sum(1); cons=np.where(npos>=5,1,np.where(npos<=1,-1,0))
        rev=-np.sign(c0/cl[O-h]-1)   # indicator: short-term reversal over same h
        momh=-rev
        def st(sg,sl=slice(None)):
            sg=sg[sl]; r=rh[sl]; m=sg!=0; g=sg[m]*r[m]; return f'n={m.sum()} gross={g.mean()*1e4:+.1f}bp net={(g-FEE).mean()*1e4:+.1f}bp hit={(g>0).mean():.2f}'
        print(f' h={h}: IC={spear(med[:,h-1],rh):+.3f} (1st2/3 {spear(med[:s,h-1],rh[:s]):+.3f}, last1/3 {spear(med[s:,h-1],rh[s:]):+.3f})  IC(past {h}h ret)={spear(momh,rh):+.3f} IC(last-1h ret)={spear(c0/cl[O-1]-1,rh):+.3f}')
        print(f'    K sign: {st(sig)} | K cons>=5: {st(cons)} | K sign last1/3: {st(sig,slice(s,None))}')
        print(f'    mom_h: {st(momh)} | K&mom agree: {st(np.where(sig==momh,sig,0))} | mean|r_h|={np.abs(rh).mean()*1e4:.0f}bp')
