import requests, time, statistics as st
U = "https://api.bybit.com/v5/market/kline"
lat = []
for i in range(10):
    t = time.time()
    r = requests.get(U, params=dict(category="linear", symbol="BTCUSDT", interval="60", limit=257), timeout=10)
    lat.append((time.time() - t) * 1000)
j = r.json(); L = j["result"]["list"]
now = int(time.time() * 1000)
print("retCode", j["retCode"], "n", len(L), "newest_start", L[0][0], "age_min %.1f" % ((now - int(L[0][0])) / 60000),
      "newest_first:", int(L[0][0]) > int(L[-1][0]))
print("REST kline(257) latency ms median %.0f p90 %.0f" % (st.median(lat), sorted(lat)[8]))
t = time.time()
r = requests.get("https://api.bybit.com/v5/market/instruments-info", params=dict(category="linear", symbol="BTCUSDT"), timeout=10)
f = r.json()["result"]["list"][0]
print("instr ms %.0f" % ((time.time() - t) * 1000), f["lotSizeFilter"], f["priceFilter"]["tickSize"])
t = time.time()
r = requests.get("https://api-testnet.bybit.com/v5/market/time", timeout=10)
print("testnet time ms %.0f" % ((time.time() - t) * 1000), r.json().get("retCode"))
