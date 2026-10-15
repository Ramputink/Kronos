import requests, pandas as pd, time, os
OUT = os.path.dirname(os.path.abspath(__file__))
B = "https://api.bybit.com/v5/market/"
start = int(pd.Timestamp("2025-09-20", tz="UTC").timestamp() * 1000)
now = int(time.time() * 1000)
for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    rows = []; end = now
    while True:
        r = requests.get(B + "funding/history", params=dict(category="linear", symbol=sym, endTime=end, limit=200), timeout=20).json()
        L = r["result"]["list"]
        if not L: break
        rows += L
        mn = min(int(x["fundingRateTimestamp"]) for x in L)
        if mn <= start or len(L) < 200: break
        end = mn - 1
    f = pd.DataFrame(rows)
    f["ts"] = pd.to_datetime(f.fundingRateTimestamp.astype("int64"), unit="ms")
    f["funding"] = f.fundingRate.astype(float)
    f[["ts", "funding"]].drop_duplicates("ts").sort_values("ts").to_csv(f"{OUT}/{sym}_funding.csv", index=False)
    # OI: paginate backwards by endTime
    rows = []; end = now
    for i in range(100):
        r = requests.get(B + "open-interest", params=dict(category="linear", symbol=sym, intervalTime="1h", limit=200, endTime=end), timeout=20).json()
        L = r["result"]["list"]
        if not L: break
        rows += L
        mn = min(int(x["timestamp"]) for x in L)
        if mn <= start: break
        end = mn - 1
    o = pd.DataFrame(rows)
    o["ts"] = pd.to_datetime(o.timestamp.astype("int64"), unit="ms")
    o["oi"] = o.openInterest.astype(float)
    o[["ts", "oi"]].drop_duplicates("ts").sort_values("ts").to_csv(f"{OUT}/{sym}_oi.csv", index=False)
    print(sym, len(f), f.ts.min(), f.ts.max(), len(o), o.ts.min(), o.ts.max())
