"""Ex-ante portfolio selections (ADDENDUM_PREREG.md). Uses only TRAIN/VAL metrics; never touches TEST."""
import collections, glob, json

OUT = '/Users/matveypro/kronos_lab/overnight/out'
FILES = sorted(glob.glob(f'{OUT}/candidates_*.jsonl'))


def lines():
    for f in FILES:
        with open(f) as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


best = {'SPOT': {}, 'PERP': {}}            # coin -> (score, record)
spot_count = collections.Counter()         # cfg_hash -> number of coins (SPOT candidates)
seen = set()
n = 0
for m in lines():
    n += 1
    key = m['cfg_hash'] + m['coin']
    if key in seen:
        continue
    seen.add(key)
    mode = m['cfg']['mode']
    score = min(m['sh_tr'], m['sh_va'])
    b = best[mode].get(m['coin'])
    if b is None or score > b[0]:
        best[mode][m['coin']] = (score, m)
    if mode == 'SPOT':
        spot_count[m['cfg_hash']] += 1
print(f'lines {n:,}, unique pairs {len(seen):,}')
del seen

r3_hashes = {h for h, c in spot_count.items() if c >= 5}
print(f'SPOT configs candidate in >=5 coins: {len(r3_hashes):,}')
r3, seen = [], set()
for m in lines():
    if m['cfg_hash'] in r3_hashes and m['cfg']['mode'] == 'SPOT':
        k = m['cfg_hash'] + m['coin']
        if k not in seen:
            seen.add(k); r3.append(m)

pick = lambda d: [dict(cfg=r['cfg'], coin=c, score=round(s, 3), cagr_tr=r['cagr_tr'], cagr_va=r['cagr_va'],
                       sh_tr=r['sh_tr'], sh_va=r['sh_va'], entries=r['entries'], mdd=r['mdd'], cfg_hash=r['cfg_hash'])
                  for c, (s, r) in sorted(d.items(), key=lambda kv: -kv[1][0])]
sel = {'R2_best_per_coin_SPOT': pick(best['SPOT']),
       'R4_top10_SPOT': pick(best['SPOT'])[:10],
       'R2_best_per_coin_PERP_ref': pick(best['PERP']),
       'R4_top10_PERP_ref': pick(best['PERP'])[:10],
       'R3_cross_coin_SPOT': [dict(cfg=m['cfg'], coin=m['coin'], cfg_hash=m['cfg_hash'],
                                   score=round(min(m['sh_tr'], m['sh_va']), 3)) for m in r3]}
json.dump(sel, open(f'{OUT}/selections_exante.json', 'w'), indent=1)
for k, v in sel.items():
    print(k, len(v))
