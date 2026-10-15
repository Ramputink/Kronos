# width-matched containment: for each method scale so mean train width = W; compare test containment
import numpy as np, os
os.chdir('/Users/matveypro/kronos_lab/agents/range_tp_sl')
exec(open('run.py').read().split("out = {}")[0])
for sym in [s for s in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'] if os.path.exists(f'{C}/{s}_60_paths.npz')]:
    df = run(sym); n = len(df); sp = int(n * 2 / 3)
    rup = np.log(df.rh / df.c0); rdn = np.log(df.c0 / df.rl)
    base = (rup + rdn)[:sp].mean()
    print(f'{sym} n={n}')
    for mult in [1.0, 1.5, 2.0]:
        W = base * mult; s = []
        for m in ['kronos_env', 'kronos_sym', 'kronos_med', 'atr', 'vol', 'const']:
            u, dn = width(df, m); k = W / (u + dn)[:sp].mean()
            s.append(f'{m}={100*((rup[sp:]<=k*u[sp:])&(rdn[sp:]<=k*dn[sp:])).mean():.0f}%')
        print(f'  width={100*W:.2f}% test containment: ' + ' '.join(s))
