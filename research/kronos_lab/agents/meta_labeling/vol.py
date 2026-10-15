import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, '.')
import numpy as np, pandas as pd, json
from meta import walk, spearman, SYMS
d = pd.read_csv('features.csv')
for f in ['atr', 'rv24', 'k_range', 'k_pathvol', 'k_disp', 'rr', 'absy']:
    d['l_'+f] = np.log(d[f]+1e-6)
sets = {'ind_vol': ['l_atr', 'l_rv24'], 'kro_vol': ['l_k_range', 'l_k_pathvol', 'l_k_disp'],
        'both': ['l_atr', 'l_rv24', 'l_k_range', 'l_k_pathvol', 'l_k_disp']}
out = {}
for n, F in sets.items():
    p = walk(d, F, 'l_rr'); m = ~np.isnan(p)
    e = d.l_rr[m].values-p[m]
    r = dict(n=int(m.sum()), ic=float(spearman(p[m], d.l_rr[m].values)), mse=float((e**2).mean()),
             r2=float(1-(e**2).mean()/d.l_rr[m].var()))
    pa = walk(d, F, 'l_absy'); r['absret_ic'] = float(spearman(pa[m], d.absy[m].values))
    for s in SYMS:
        mm = m & (d.sym == s).values; r[s+'_ic'] = float(spearman(p[mm], d.l_rr[mm].values))
    out[n] = r
m = (d.t >= 100).values
out['naive_atr_rank_ic'] = float(spearman(d.l_atr[m].values, d.l_rr[m].values))
out['naive_kpathvol_rank_ic'] = float(spearman(d.l_k_pathvol[m].values, d.l_rr[m].values))
for s in SYMS:
    mm = m & (d.sym == s).values
    out[s+'_within_ic_vs_range'] = {f: float(spearman(d[f][mm].values, d.rr[mm].values)) for f in ['atr', 'rv24', 'k_range', 'k_pathvol', 'k_disp']}
json.dump(out, open('vol_results.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
