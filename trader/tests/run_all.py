"""Run every trader/tests/test_*.py in its own process and print a one-line verdict per file."""
import glob, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
failed = 0
for path in sorted(glob.glob(os.path.join(HERE, 'test_*.py'))):
    t0 = time.time()
    p = subprocess.run([sys.executable, path], cwd=ROOT, capture_output=True, text=True, timeout=600)
    tail = (p.stdout + p.stderr).strip().splitlines()[-1:] or ['']
    ok = p.returncode == 0
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'} {os.path.basename(path):22} {time.time() - t0:5.0f}s  {tail[0][:140]}", flush=True)
print(f'{failed} failing file(s)')
sys.exit(1 if failed else 0)
