"""Andrews' Pitchfork (last 3 confirmed fractal pivots) + Kronos path confirmation.
No lookahead: pivot at index i with half-window k is only known at i+k <= o.
"""
import sys, os, json
import numpy as np, pandas as pd

CACHE = "/Users/matveypro/kronos_lab/cache"
FEE = 0.0011  # round trip
H = 24
rng = np.random.default_rng(0)


def pivots(high, low, k):
    """Return list of (idx, price, type) fractal pivots, type +1 high, -1 low."""
    n = len(high); out = []
    for i in range(k, n - k):
        w_h = high[i - k:i + k + 1]; w_l = low[i - k:i + k + 1]
        if high[i] == w_h.max():
            out.append((i, high[i], 1))
        elif low[i] == w_l.min():
            out.append((i, low[i], -1))
    # enforce alternation: keep the more extreme of consecutive same-type
    alt = []
    for p in out:
        if alt and alt[-1][2] == p[2]:
            if (p[2] == 1 and p[1] >= alt[-1][1]) or (p[2] == -1 and p[1] <= alt[-1][1]):
                alt[-1] = p
        else:
            alt.append(p)
    return alt


def fork_at(piv, o, k):
    known = [p for p in piv if p[0] + k <= o]
    if len(known) < 3:
        return None
    (i1, p1, _), (i2, p2, _), (i3, p3, _) = known[-3:]
    mi, mp = (i2 + i3) / 2, (p2 + p3) / 2
    if mi == i1:
        return None
    slope = (mp - p1) / (mi - i1)
    med = lambda t: p1 + slope * (t - i1)
    # tines: parallel through P2 and P3; half-width at time t is constant offset
    off2 = p2 - med(i2); off3 = p3 - med(i3)
    up, lo = max(off2, off3), min(off2, off3)
    return med, slope, up, lo


def stats(r):
    r = np.asarray(r, float)
    if len(r) == 0:
        return dict(n=0, mean=np.nan, hit=np.nan, total=0.0, t=np.nan)
    sd = r.std(ddof=1) if len(r) > 1 else np.nan
    return dict(n=int(len(r)), mean=float(r.mean() * 100), hit=float((r > 0).mean()),
                total=float(r.sum() * 100), t=float(r.mean() / sd * np.sqrt(len(r) / 2)) if sd and sd > 0 else np.nan)
    # t uses n/2 effective samples (24h holds, 12h spacing -> ~50% overlap)


def run(sym, k):
    df = pd.read_csv(f"{CACHE}/{sym}_60_ohlcv.csv")
    z = np.load(f"{CACHE}/{sym}_60_paths.npz")
    origins, paths = z["origins"], z["paths"]
    h, l, c = df.high.values, df.low.values, df.close.values
    piv = pivots(h, l, k)
    rows = []
    for w, o in enumerate(origins):
        o = int(o)
        if o + H >= len(c):
            continue
        f = fork_at(piv, o, k)
        if f is None:
            continue
        med, slope, up, lo = f
        c0 = c[o]; m0 = med(o); m24 = med(o + H)
        dev = c0 - m0
        half = max((up - lo) / 2, 1e-9)
        z_dev = dev / half  # >0 above median, ~1 = at a tine
        realized = c[o + H] / c0 - 1
        pc = paths[w][:, :, 3]  # (6,24) closes
        kr_ret = pc[:, -1].mean() / c0 - 1
        kr_up = (pc[:, -1] > c0).mean()
        # fraction of Kronos paths that touch the (projected) median line within 24h
        medline = np.array([med(o + j + 1) for j in range(H)])
        ph, pl = paths[w][:, :, 1], paths[w][:, :, 2]
        touch = ((ph >= medline) & (pl <= medline)).any(axis=1) | \
                ((np.sign(pc - medline) != np.sign(dev))).any(axis=1)
        kr_touch = touch.mean()
        # Kronos predicted move toward median? (end distance smaller than now)
        kr_toward = np.abs(pc[:, -1].mean() - m24) < abs(dev)
        rows.append(dict(o=o, z=z_dev, realized=realized, kr_ret=kr_ret, kr_up=kr_up,
                         kr_touch=kr_touch, kr_toward=kr_toward, slope_sign=np.sign(slope)))
    d = pd.DataFrame(rows)
    res = {}
    n = len(d)
    bh = d.realized.values
    res["buy_hold_24h_windows"] = stats(bh)
    res["buy_hold_total_period_%"] = float((c[int(d.o.iloc[-1]) + H] / c[int(d.o.iloc[0])] - 1) * 100)

    def add(name, pos):
        pos = np.asarray(pos, float)
        m = pos != 0
        g = pos[m] * d.realized.values[m]
        res[name] = dict(gross=stats(g), net=stats(g - FEE))

    # 1. Median-line reversion: fade toward median when |z| > 0.5
    rev = np.where(np.abs(d.z) > 0.5, -np.sign(d.z), 0)
    add("reversion", rev)
    # reversion + Kronos agrees on direction
    add("reversion+kronos_dir", np.where(np.sign(d.kr_ret) == rev, rev, 0))
    # reversion + majority of Kronos paths touch median
    add("reversion+kronos_touch>=0.5", np.where(d.kr_touch >= 0.5, rev, 0))
    add("reversion+kronos_toward", np.where(d.kr_toward, rev, 0))
    # 2. Tine breakout: momentum when price beyond a tine (|z|>1)
    brk = np.where(np.abs(d.z) > 1.0, np.sign(d.z), 0)
    add("breakout", brk)
    add("breakout+kronos_dir", np.where(np.sign(d.kr_ret) == brk, brk, 0))
    # 3. Trend-of-fork: trade in direction of median slope
    add("fork_slope", d.slope_sign.values)
    add("fork_slope+kronos_dir", np.where(np.sign(d.kr_ret) == d.slope_sign, d.slope_sign, 0))
    # Kronos alone
    add("kronos_alone", np.sign(d.kr_ret))
    # Random baseline: shuffle reversion positions (same trade count), 500 draws
    rnd = []
    for _ in range(500):
        p = rng.permutation(rev)
        g = p[p != 0] * d.realized.values[p != 0]
        rnd.append(g.mean() * 100)
    rnd = np.array(rnd)
    res["random_shuffle_reversion_gross_mean_%"] = dict(mean=float(rnd.mean()), p95=float(np.percentile(rnd, 95)),
                                                        p_value_rev=float((rnd >= res["reversion"]["gross"]["mean"]).mean()))
    # Kronos filter value: within reversion trades, does kronos agreement split outcomes?
    m = rev != 0
    g = rev[m] * d.realized.values[m]
    agree = (np.sign(d.kr_ret.values[m]) == rev[m])
    res["rev_split_by_kronos"] = dict(agree=stats(g[agree]), disagree=stats(g[~agree]))
    # Holdout: last third
    cut = int(n * 2 / 3)
    for nm, pos in [("reversion", rev), ("reversion+kronos_dir", np.where(np.sign(d.kr_ret) == rev, rev, 0)),
                    ("breakout", brk), ("breakout+kronos_dir", np.where(np.sign(d.kr_ret) == brk, brk, 0))]:
        pos = np.asarray(pos, float)[cut:]
        rr = d.realized.values[cut:]
        mm = pos != 0
        res[f"HOLDOUT_{nm}_net"] = stats(pos[mm] * rr[mm] - FEE)
    res["n_windows"] = n
    res["corr_z_vs_realized"] = float(np.corrcoef(d.z, d.realized)[0, 1])
    res["corr_kr_touch_vs_revpnl"] = float(np.corrcoef(d.kr_touch[m], g)[0, 1]) if m.sum() > 3 else None
    return res


if __name__ == "__main__":
    out = {}
    for sym in sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        if not os.path.exists(f"{CACHE}/{sym}_60_paths.npz"):
            print("missing", sym); continue
        for k in (6, 12):
            out[f"{sym}_k{k}"] = run(sym, k)
    with open("/Users/matveypro/kronos_lab/agents/pitchfork/results.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    for key, r in out.items():
        print("=====", key, "n=", r["n_windows"], "B&H total%", round(r["buy_hold_total_period_%"], 1),
              "corr_z", round(r["corr_z_vs_realized"], 3), "corr_touch", r["corr_kr_touch_vs_revpnl"])
        for nm, v in r.items():
            if isinstance(v, dict) and "net" in v:
                print(f"  {nm:30s} n={v['net']['n']:3d} gross={v['gross']['mean']:+.3f}% net={v['net']['mean']:+.3f}% hit={v['net']['hit']:.2f} t={v['net']['t']:+.2f} totNet={v['net']['total']:+.1f}%")
            elif nm.startswith("HOLDOUT"):
                print(f"  {nm:30s} n={v['n']:3d} net={v['mean']:+.3f}% t={v['t']:+.2f}")
        print("  random:", r["random_shuffle_reversion_gross_mean_%"])
        s = r["rev_split_by_kronos"]
        print(f"  rev split agree n={s['agree']['n']} {s['agree']['mean']:+.3f}%  disagree n={s['disagree']['n']} {s['disagree']['mean']:+.3f}%")
        print("  buyhold windows:", {k2: round(v2, 3) for k2, v2 in r["buy_hold_24h_windows"].items()})
