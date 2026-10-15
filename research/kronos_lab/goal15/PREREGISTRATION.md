# Pre-registro: objetivo >15 % anual (escrito ANTES de ver resultados)

Fecha: 2026-10-13. Autor: Claude (sesión con Matvey).

## Pregunta
¿Existe una estrategia implementable en Bybit que supere un 15 % anual (CAGR) neto de costes,
fuera de muestra, sin sobreajuste?

## Datos
- Bybit v5, perpetuos lineales USDT, velas diarias (UTC) y funding histórico.
- Universo fijo = las 30 monedas que MarketReasercher/phase2 ya definió (no lo elijo yo):
  BTC ETH SOL XRP ADA AVAX LINK DOGE BNB LTC DOT POL ATOM NEAR APT BCH TRX ETC XLM DASH ZEC VET THETA
  ALGO COMP CRV RUNE SUSHI EGLD UNI. Un activo entra cuando tiene historia suficiente (NaN antes).
  Sesgo de supervivencia reconocido: la lista se fijó en 2026 con monedas que siguen listadas.
- Periodos:
  - IS  (in-sample):        inicio datos Bybit -> 2024-06-07
  - OOS-A (lockbox del repo): 2024-06-07 -> 2026-06-07  (OJO: el repo ya lo leyó para elegir "breakout")
  - OOS-B (forward virgen):   2026-06-07 -> último cierre diario disponible. Nadie lo ha visto.

## Costes
- Variante SPOT: 0.10 % comisión + 5 bp slippage por lado sobre el peso negociado; sin funding.
- Variante PERP: 0.055 % comisión + 5 bp slippage por lado + funding real pagado/cobrado sobre la posición.

## Candidatos (fijos, sin ajustar parámetros; son los de phase3_five_strategies.py tal cual)
- C1 breakout: nuevo máximo de 50d y > EMA200, mantener 15d, top-10 por mom 60d, 1/10 por slot.
- C2 trend_ts: > EMA200, top-10 por mom 60d.
- C3 ensemble 3 patas (trend_ts + breakout + capitulation), pesos iguales.
- C4 C1 + filtro de régimen BTC (BTC > EMA200 o se va a liquidez).
- C5 C1 con vol-targeting: escala la exposición a 30 % vol anual objetivo (vol realizada 30d de la
     cartera, decidida con datos hasta t), con apalancamiento máximo 2x (solo variante PERP).
- C6 BTC+ETH trend simple: 50/50, cada uno largo si cierre > SMA 100d, si no liquidez.

Total de candidatos = 6 (x2 variantes de coste). Se reporta TODO, también lo que falle.

## Criterio de éxito (decidido ahora)
Un candidato "cumple" solo si:
1. CAGR neto > 15 % en OOS-A **y** Sharpe OOS-A > 0.8, y
2. no pierde más de lo que pierde buy&hold de BTC en OOS-B (comprobación de que sigue vivo), y
3. Deflated Sharpe (con N = 12 pruebas) > 0.5 en OOS-A.
Las cifras IS solo se muestran como contexto; no cuentan para el criterio.
Si ninguno cumple, la respuesta honesta es "ninguno".
