"""Discovery-only summary of candidates (TRAIN+VAL metrics; never touches TEST). Streams all files."""
import collections, glob, json
import pandas as pd

OUT = '/Users/matveypro/kronos_lab/overnight/out'
seen = set()
by = {k: collections.Counter() for k in ('coin', 'tf', 'mode', 'family', 'ncomp', 'adx', 'vt', 'stop')}
coins_per_cfg = collections.defaultdict(set)
cfg_of = {}
best = []
n_lines = 0
for f in glob.glob(f'{OUT}/candidates_*.jsonl'):
    with open(f) as fh:
        for line in fh:
            n_lines += 1
            try:
                m = json.loads(line)
            except Exception:
                continue
            key = (m['cfg_hash'], m['coin'])
            if key in seen:
                continue
            seen.add(key)
            c = m['cfg']
            fam = '+'.join(sorted(x['t'] for x in c['comps']))
            for k, v in (('coin', m['coin']), ('tf', c['tf']), ('mode', c['mode']), ('family', fam),
                         ('ncomp', len(c['comps'])), ('adx', c['adx']), ('vt', c['vt']), ('stop', c['stop'])):
                by[k][v] += 1
            coins_per_cfg[m['cfg_hash']].add(m['coin'])
            cfg_of[m['cfg_hash']] = c
            best.append((min(m['sh_tr'], m['sh_va']), m['coin'], m['cfg_hash'], m['cagr_tr'], m['cagr_va'], m['entries']))

n = len(seen)
lines = [f'# Candidatos (solo TRAIN+VAL) — {pd.Timestamp.now():%Y-%m-%d %H:%M}',
         f'Líneas {n_lines:,} · pares (config, moneda) únicos {n:,} · configs únicas {len(coins_per_cfg):,}', '']
for k, cnt in by.items():
    top = ', '.join(f'{a}: {b / n * 100:.1f}%' for a, b in cnt.most_common(12))
    lines.append(f'- **{k}**: {top}')
multi = collections.Counter(min(len(s), 10) for s in coins_per_cfg.values())
lines += ['', '## Robustez entre monedas (nº de monedas en que la MISMA config es candidata)',
          ', '.join(f'{k}{"+" if k == 10 else ""}: {v:,}' for k, v in sorted(multi.items()))]
robust = sorted(coins_per_cfg.items(), key=lambda kv: -len(kv[1]))[:15]
lines += ['', '## Configs más transversales']
for h, cs in robust:
    c = cfg_of[h]
    desc = ' & '.join(f"{x['t']}{tuple(x['p'].values())}" for x in c['comps'])
    lines.append(f"- {len(cs)} monedas · {c['tf']} {c['mode']} · {desc} · comb={c['comb']} adx={c['adx']} "
                 f"vt={c['vt']} stop={c['stop']} · {sorted(cs)}")
best.sort(reverse=True)
lines += ['', '## Top 15 por Sharpe mínimo(TRAIN, VAL)']
for s, coin, h, ct, cv, e in best[:15]:
    c = cfg_of[h]
    desc = ' & '.join(f"{x['t']}{tuple(x['p'].values())}" for x in c['comps'])
    lines.append(f"- {coin} · Sharpe≥{s:.2f} · CAGR train {ct * 100:.0f}% val {cv * 100:.0f}% · {e} entradas · "
                 f"{c['tf']} {c['mode']} · {desc} · adx={c['adx']} vt={c['vt']} stop={c['stop']}")
open(f'{OUT}/candidates_summary.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
