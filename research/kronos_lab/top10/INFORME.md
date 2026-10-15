# 10 estrategias con rentabilidad superior por cripto (Bybit spot, 2026-10-14)

Pre-registro PREREGISTRATION.md (sha256 1fbfa997…); selecciones congeladas antes del test en
selections.json (sha256 077dda73…). Selección por moneda en TRAIN (≤2023) + VAL (2024→2025-06) entre 240
configuraciones; TEST 2025-07-01→2026-10-13 abierto una vez. "Superior" = CAGR TEST > 0 y > buy&hold.

Resultado: 13 de 29 monedas cumplen; 4 de ellas > 15 % anual. Las 10 con mayor CAGR en TEST:

| # | Moneda | Estrategia | CAGR TEST | Buy&hold | MaxDD | Entradas |
|---|---|---|---|---|---|---|
| 1 | UNI | Supertrend(10,3) + vol-target | +33.9 % | +4.5 % | −27.8 % | 5 |
| 2 | EGLD | Voto 5/5 + ADX>25 + vol-target | +18.5 % | −63.1 % | −7.2 % | 1 |
| 3 | CRV | Bollinger breakout + ADX>20 + vol-target | +17.7 % | −28.5 % | −27.6 % | 5 |
| 4 | ALGO | ROC 90 + ADX>20 + vol-target | +15.2 % | −29.0 % | −19.5 % | 9 |
| 5 | BNB | RSI(2) reversión + ADX>25 | +14.5 % | +9.8 % | −13.1 % | 7 |
| 6 | AVAX | Voto 5/5 + ADX>25 + vol-target | +9.9 % | −35.4 % | −16.4 % | 3 |
| 7 | LTC | Bollinger reversión + ADX>20 + vol-target | +5.7 % | −23.3 % | −1.3 % | 2 |
| 8 | XRP | Donchian 100/10 + ADX>25 + vol-target | +4.8 % | −31.8 % | −15.5 % | 1 |
| 9 | ETH | MACD con filtro SMA200 + ADX>25 + vol-target | +4.6 % | +0.4 % | −17.8 % | 5 |
| 10 | ETC | Voto 5/5 + ADX>20 + vol-target | +3.2 % | −42.6 % | −13.8 % | 4 |
(También cumplen VET, DOGE, THETA.)

Advertencia clave: elegir estas 10 DESPUÉS de ver el test es sesgo retrospectivo. Ex ante, operar las 29
selecciones juntas dio CAGR −0.4 % (Sharpe 0.01, maxDD −16 %) frente a −11.1 % del buy&hold (maxDD −63 %):
el valor real es proteger capital en un mercado bajista, no generar >15 %.
