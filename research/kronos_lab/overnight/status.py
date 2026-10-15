"""Aggregate worker stats, check liveness and disk; append a line to out/progress.log."""
import glob, json, os, shutil, subprocess, time
from datetime import datetime

OUT = '/Users/matveypro/kronos_lab/overnight/out'
launched = datetime.strptime(open(f'{OUT}/launched_at.txt').read().strip(), '%Y-%m-%d %H:%M:%S')
elapsed_h = (datetime.now() - launched).total_seconds() / 3600
alive = sorted(int(l.split()[-2]) for l in subprocess.run(['pgrep', '-fl', 'worker.py'], capture_output=True,
                                                            text=True).stdout.splitlines() if l.split()[-2].isdigit())
tot = dict(configs=0, trials=0, candidates=0, errors=0)
mom = {}
for f in glob.glob(f'{OUT}/stats_*.json'):
    s = json.load(open(f))
    for k in tot:
        tot[k] += s.get(k, 0)
    for tf, (n, a, b) in s['sr_moments'].items():
        m = mom.setdefault(tf, [0, 0.0, 0.0]); m[0] += n; m[1] += a; m[2] += b
cand_bytes = sum(os.path.getsize(f) for f in glob.glob(f'{OUT}/candidates_*.jsonl'))
free_gb = shutil.disk_usage('/Users/matveypro').free / 1e9
sr = {tf: (round(a / n, 3), round(((b / n) - (a / n) ** 2) ** 0.5, 3)) for tf, (n, a, b) in mom.items() if n}
line = (f"{time.strftime('%Y-%m-%d %H:%M')} | {elapsed_h:4.1f} h | workers vivos {len(alive)} {alive} | "
        f"configs {tot['configs']:,} | pruebas {tot['trials']:,} | candidatos {tot['candidates']:,} | "
        f"errores {tot['errors']} | candidatos {cand_bytes / 1e6:,.0f} MB | disco libre {free_gb:.1f} GB | "
        f"Sharpe medio/std por tf {sr}")
print(line)
open(f'{OUT}/progress.log', 'a').write(line + '\n')
if free_gb < 10:
    subprocess.run(['pkill', '-f', 'worker.py'])
    print('DISK GUARD: workers stopped (free < 10 GB)')
