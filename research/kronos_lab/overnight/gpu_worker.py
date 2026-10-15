"""GPU (Apple MPS) dense-grid search. Same sealed data, same candidate criterion, CPU-verified candidates.

Component banks are built with worker.py's own component functions (exact same signals), then the GPU
evaluates huge batches of (component A AND component B) x ADX x vol-target x mode combinations.
usage: python gpu_worker.py <hours>
"""
import json, math, random, sys, time, hashlib
import numpy as np
import torch
import worker as W

OUT = W.OUT
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
BATCH = 4096


def bank(d):
    specs = [('ma', dict(kind='sma', n=n)) for n in range(5, 301, 1)]
    specs += [('roc', dict(n=n)) for n in range(2, 251)]
    specs += [('donch', dict(N=N, M=M)) for N in range(10, 201, 5) for M in range(5, N, 5)]
    specs += [('bb_break', dict(n=n, k=round(k, 2))) for n in range(10, 61, 2) for k in np.arange(1.0, 3.01, 0.25)]
    S = np.vstack([W.CATALOG[t][0](d, **p) for t, p in specs]).astype(np.float16)
    return specs, torch.from_numpy(S).to(dev)


def run_batch(Sb, ia, ib, ret, fund, adxv, volv, m_tr, m_va, m_ev, bpy, mode, adx_thr, vt):
    A, B = Sb[ia].float(), Sb[ib].float()
    pos = torch.where((A > 0) & (B > 0), 1.0, torch.where((A < 0) & (B < 0), -1.0, 0.0))
    if mode == 'SPOT':
        pos = pos.clamp(min=0)
    if adx_thr:
        pos = pos * (adxv > adx_thr)
    if vt:
        pos = pos * torch.clamp(vt / volv, max=1.0 if mode == 'SPOT' else 2.0).nan_to_num(0.0)
    wx = torch.cat([torch.zeros_like(pos[:, :1]), pos[:, :-1]], 1)
    turn = torch.abs(torch.diff(pos, dim=1, prepend=torch.zeros_like(pos[:, :1])))
    r = wx * ret - torch.cat([torch.zeros_like(turn[:, :1]), turn[:, :-1]], 1) * W.COST[mode]
    if mode == 'PERP':
        r = r - wx * fund
    out = {}
    for name, m in (('tr', m_tr), ('va', m_va), ('ev', m_ev)):
        rr = r[:, m]
        n = rr.shape[1]
        lg = torch.log1p(rr.clamp(min=-0.999)).sum(1)
        out['cagr_' + name] = torch.exp(lg * bpy / n) - 1
        sd = rr.std(1)
        out['sh_' + name] = torch.where(sd > 0, rr.mean(1) / sd * math.sqrt(bpy), torch.zeros_like(sd))
    le = torch.cumsum(torch.log1p(r[:, m_ev].clamp(min=-0.999)), 1)
    out['mdd'] = torch.exp(le - torch.cummax(le, 1).values).min(1).values - 1
    we = wx[:, m_ev]
    out['entries'] = ((we[:, 1:] != 0) & (torch.sign(we[:, 1:]) != torch.sign(we[:, :-1]))).sum(1)
    return {k: v.cpu().numpy() for k, v in out.items()}


def main():
    hours = float(sys.argv[1])
    deadline = time.time() + hours * 3600
    rng = random.Random(int(time.time()))
    data = W.load_all()
    keys = [(c, tf) for tf in ('1d', '4h') for c in W.COINS]
    stats = {'trials': 0, 'configs': 0, 'candidates': 0, 'errors': 0, 'gpu_hits': 0,
             'sr_moments': {tf: [0, 0.0, 0.0] for tf in W.BPY}}
    try:  # resume counters after a restart (needed for the total-trials count / DSR)
        stats.update(json.load(open(f'{OUT}/stats_gpu.json')))
        stats.pop('finished', None)
    except Exception:
        pass
    cand_f = open(f'{OUT}/candidates_gpu.jsonl', 'a')
    last = time.time()
    while time.time() < deadline:
        coin, tf = rng.choice(keys)
        d = data[(coin, tf)]
        if d['m_train'].sum() < 180 * W.BPY[tf] / 365:
            continue
        specs, Sb = bank(d)
        c = d['c']
        ret = torch.tensor(np.r_[0.0, c[1:] / c[:-1] - 1], dtype=torch.float32, device=dev)
        fund = torch.tensor(d['fund'], dtype=torch.float32, device=dev)
        adxv = torch.tensor(np.nan_to_num(W.aux(d, (coin, tf), 'adx')), dtype=torch.float32, device=dev)
        volv = torch.tensor(W.aux(d, (coin, tf), 'vol'), dtype=torch.float32, device=dev)
        m_tr, m_va, m_ev = (torch.tensor(np.flatnonzero(d[k]), device=dev) for k in ('m_train', 'm_val', 'm_eval'))
        t_visit = time.time()
        while time.time() - t_visit < 120 and time.time() < deadline:  # ~2 min per coin/tf, then rotate
            mode, adx_thr, vt = rng.choice(['SPOT', 'PERP']), rng.choice(W.FILTERS['adx']), rng.choice(W.FILTERS['vt'])
            bsz = BATCH if tf == '1d' else BATCH // 4
            ia = torch.randint(0, len(specs), (bsz,), device=dev)
            ib = torch.randint(0, len(specs), (bsz,), device=dev)
            try:
                res = run_batch(Sb, ia, ib, ret, fund, adxv, volv, m_tr, m_va, m_ev, W.BPY[tf], mode, adx_thr, vt)
            except Exception:
                stats['errors'] += 1
                continue
            stats['configs'] += bsz; stats['trials'] += bsz
            mom = stats['sr_moments'][tf]
            sh = res['sh_ev'][np.isfinite(res['sh_ev'])]
            mom[0] += len(sh); mom[1] += float(sh.sum()); mom[2] += float((sh ** 2).sum())
            hit = ((res['cagr_tr'] > 0.2) & (res['cagr_va'] > 0.2) & (res['sh_tr'] > 0.8) & (res['sh_va'] > 0.8)
                   & (res['entries'] >= 8) & (res['mdd'] > -0.45))
            ia_np, ib_np = ia.cpu().numpy(), ib.cpu().numpy()
            for j in np.flatnonzero(hit):
                stats['gpu_hits'] += 1
                ta, pa = specs[ia_np[j]]; tb, pb = specs[ib_np[j]]
                comps = [dict(t=ta, p=pa)] if ia_np[j] == ib_np[j] else [dict(t=ta, p=pa), dict(t=tb, p=pb)]
                cfg = dict(tf=tf, mode=mode, comps=comps, comb='and', k=len(comps), adx=adx_thr, vt=vt, stop=0)
                W.COINS_ALL = W.COINS
                saved, W.COINS = W.COINS, [coin]
                try:
                    ms = W.evaluate(cfg, data)  # CPU re-check with the exact same code as the CPU workers
                finally:
                    W.COINS = saved
                if ms and W.is_candidate(ms[0]):
                    h = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
                    stats['candidates'] += 1
                    cand_f.write(json.dumps(dict(cfg=cfg, cfg_hash=h, src='gpu', **ms[0])) + '\n')
            if time.time() - last > 60:
                cand_f.flush()
                stats['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
                json.dump(stats, open(f'{OUT}/stats_gpu.json', 'w'))
                last = time.time()
        del Sb, ret, fund, adxv, volv, m_tr, m_va, m_ev
        W.CACHE.clear()
        if dev.type == 'mps':
            torch.mps.empty_cache()
    cand_f.close()
    stats['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(stats, open(f'{OUT}/stats_gpu.json', 'w'))


if __name__ == '__main__':
    main()
