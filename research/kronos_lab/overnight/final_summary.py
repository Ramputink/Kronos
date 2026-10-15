"""Run after all workers finished: builds out/RESUMEN_NOCHE.md (discovery only; TEST still sealed)."""
import glob, json, subprocess

OUT = '/Users/matveypro/kronos_lab/overnight/out'
PY = '/Users/matveypro/.venvs/kronos/bin/python'
base = '/Users/matveypro/kronos_lab/overnight'
status = subprocess.run([PY, f'{base}/status.py'], capture_output=True, text=True).stdout.strip()
subprocess.run([PY, f'{base}/summarize_candidates.py'], capture_output=True, text=True)
summary = open(f'{OUT}/candidates_summary.md').read()
tot = dict(configs=0, trials=0, candidates=0)
for f in glob.glob(f'{OUT}/stats_*.json'):
    s = json.load(open(f))
    for k in tot:
        tot[k] += s.get(k, 0)
progress = open(f'{OUT}/progress.log').read()
md = f"""# Resumen de la búsqueda nocturna (TEST todavía sellado)

Pre-registro: ../PREREGISTRATION.md (sha256 1df9d2df…). Datos < 2025-07-01. CPU: 10 workers; GPU: rejillas densas.

**Totales:** {tot['configs']:,} configuraciones · {tot['trials']:,} pruebas (config × moneda) · {tot['candidates']:,} candidatos
(CAGR > 20 % y Sharpe > 0.8 en TRAIN y en VAL, ≥ 8 entradas, maxDD > −45 %).

Con este número de pruebas, por puro azar aparecen cientos de miles de "candidatos". La pregunta real se responde
mañana con `morning_test.py`: ¿qué fracción se confirma en el TEST (>20 % y > buy&hold) frente a configuraciones
aleatorias, y cuánto rinde la cartera de TODOS los candidatos?

Comando de mañana (abre el TEST una sola vez):
    cd ~/kronos_lab/overnight && ~/.venvs/kronos/bin/python -W ignore morning_test.py

## Último estado
{status}

{summary}

## Progreso por hora
```
{progress}
```
"""
open(f'{OUT}/RESUMEN_NOCHE.md', 'w').write(md)
print(f'{OUT}/RESUMEN_NOCHE.md written')
