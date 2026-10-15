# Pre-registro v2: combinación de indicadores técnicos >15 % anual en large caps (Bybit)

Fecha: 2026-10-14, escrito ANTES de ejecutar ninguna búsqueda.

## Universo (fijo)
Large caps con perpetuo lineal USDT en Bybit e historia desde 2021:
BTC, ETH, SOL, XRP, BNB, DOGE, ADA, LINK, AVAX (9). Foco de reporte: ETH y SOL.
Los MISMOS parámetros para todos los activos (prohibido ajustar por activo).

## Datos y periodos
Bybit v5 velas 1d y 4h (perp lineal; el precio spot sigue al perp en pocos bp) + funding real.
- TRAIN:      inicio -> 2023-12-31      (selección)
- VALIDATION: 2024-01-01 -> 2025-06-30  (confirmación; descarta lo que no aguante)
- TEST:       2025-07-01 -> último cierre (sellado; se abre UNA vez con los finalistas)
Los indicadores se calculan sobre la serie completa con ventanas hacia atrás (sin lookahead);
la posición decidida al cierre de la vela t se aplica de t a t+1.

## Costes Bybit
- SPOT (solo largos): 0.10 % comisión + 5 bp slippage por lado.
- PERP (largos y cortos): 0.055 % comisión + 5 bp slippage por lado + funding real.

## Espacio de búsqueda (rejilla completa, todas las pruebas cuentan para el Deflated Sharpe)
Señal = posición objetivo en [-1, 1] (en SPOT se recorta a [0, 1]).
Familias base:
- EMA cross: fast {10, 20, 50} x slow {50, 100, 200} (fast<slow)
- Precio vs SMA: {50, 100, 150, 200}
- Donchian: entrada máximo N {20, 50, 100}, salida mínimo M {10, 20, 50} (M<N)
- Supertrend: ATR {10, 14} x mult {2, 3, 4}
- MACD (12,26,9): histograma >0 ; con/sin filtro SMA200
- RSI régimen (14): long > {50, 55}, flat/short < {45, 50}
- Bollinger(20,2) breakout: entra > banda sup., sale < media
- Reversión: RSI(2) < 10 con precio > SMA200, salida RSI(2) > 70 ; Bollinger inferior con SMA200
- Momentum ROC: ROC {30, 60, 90} > 0
Conjuntos ("combinación de indicadores"):
- VOTO: media de las señales de 5 indicadores base estándar (EMA20/100, SMA100, Donchian50/20,
  Supertrend14x3, MACD) -> posición continua; y versión umbral (>= k de 5, k ∈ {3,4,5}).
- Cada familia y voto con/sin filtro ADX(14) > {20, 25}, y con/sin vol-targeting (objetivo 50 % vol
  anual por activo, máx 1x en SPOT, máx 2x en PERP).
Timeframes: 1d y 4h. Modos de coste: SPOT y PERP.

## Regla de selección (decidida ahora)
1. En TRAIN: cartera equiponderada de los 9 activos; ordenar por Sharpe de cartera; exigir además que
   la mediana de Sharpe por activo > 0.5.
2. Top-10 de TRAIN -> VALIDATION: pasan los que tengan Sharpe de cartera > 0.5 y CAGR > 0 en VAL.
3. Finalistas (máx. 3) -> TEST una sola vez.
Éxito = algún finalista con CAGR > 15 % en TEST en ETH o SOL **y** en la cartera, con Deflated Sharpe
(N = número total de configuraciones evaluadas en TRAIN) > 0.5 sobre TRAIN+VAL.
Si no hay éxito se reporta así, junto con lo mejor encontrado y por qué falla.
