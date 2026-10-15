# Combinación de indicadores para large caps en Bybit (2026-10-14)

Pre-registro: PREREGISTRATION.md (sha256 d26e29d1…). 960 configuraciones (40 reglas × ADX{0,20,25} × vol-targeting{no,sí}
× {1d,4h} × {SPOT,PERP}), mismos parámetros para BTC ETH SOL XRP BNB DOGE ADA LINK AVAX, datos Bybit + funding real.
TRAIN ≤2023 · VAL 2024-01→2025-06 · TEST 2025-07→2026-10-13 (abierto una vez).

## Ganadora: RSI(14) régimen con histéresis + vol-targeting, diario, spot solo largos
- Entrar largo cuando RSI(14) diario > 55; salir a USDT cuando < 50.
- Tamaño = min(1, 50 % / volatilidad anualizada de 30 días) del capital asignado a esa moneda; el resto en USDT.
- TEST (fuera de muestra, mercado bajista; buy&hold BTC −18 %, SOL −25 % anual):
  ETH **+26.8 %** anual (DD −32 %), SOL +12.1 %, BNB +38 %, cartera 9 monedas +12.1 % (DD −25 %).
  Con comisión Bybit EU 0.25 %: ETH +23.0 %, SOL +9.0 %, cartera +9.1 %.
- Deflated Sharpe (N=960, train+val) = 0.55.
- Criterio estricto pre-registrado (también >15 % en la cartera): NO cumple (12.1 %).

## Lo que no funcionó
- Votaciones de 5 indicadores (EMA, SMA, Donchian, Supertrend, MACD): buenas en train, mediana por activo floja; SOL negativo en VAL.
- 4h: peor que diario (costes). PERP con cortos: peor que spot (funding + cortos perdedores).
- Donchian 100/50 y EMA10/50 con ADX: +1.7 % y +4.1 % en TEST.

## Bybit (ver brief del agente): en España → Bybit EU: spot sí, perps no; API propia posiblemente limitada
a apps de broker; taker spot posiblemente 0.25 %. Verificar en la cuenta antes de construir el bot.
