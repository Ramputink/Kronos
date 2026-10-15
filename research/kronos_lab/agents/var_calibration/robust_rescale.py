import numpy as np, pandas as pd
D='/Users/matveypro/kronos_lab/agents/var_calibration/'
def pb(y,q,a): d=y-q; return 1e4*np.mean(np.maximum(a*d,(a-1)*d))
for s in ['BTCUSDT','ETHUSDT']:
    d=pd.read_csv(D+f'{s}_windows.csv'); n=len(d); cut=int(n*2/3); tr,te=d[:cut],d[cut:]
    out=[]
    for lab,sd in [('Kronos sd',d.k_sd),('EWMA sd',d.ew_sd),('geo-mean EWMA*Kronos',np.sqrt(d.k_sd*d.ew_sd))]:
        z=(tr.r/sd[:cut]); ql,qh=np.quantile(z,.05),np.quantile(z,.95)  # empirical quantile-of-z rescale (robust)
        lo,hi=ql*sd[cut:],qh*sd[cut:]; y=te.r.values
        out.append(f'{lab:22s} test lo_breach={np.mean(y<lo):.2f} hi_breach={np.mean(y>hi):.2f} pinball05={pb(y,lo,.05):.1f} pinball95={pb(y,hi,.95):.1f} width={1e4*np.mean(hi-lo):.0f}bp')
    print(s); print('\n'.join(out))
