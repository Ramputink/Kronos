# Pre-registro v4: búsqueda nocturna de estrategias >20 % anual (2026-10-14, antes de lanzar)

## Datos (congelados)
Bybit perp lineal USDT, velas 1d y 4h de las 29 monedas de top10 (sin POL), descargadas 2026-10-14.
Los workers cargan SOLO datos < 2025-07-01 (TRAIN ≤ 2023-12-31, VAL 2024-01-01→2025-06-30).
El TEST (2025-07-01→2026-10-13/14) no existe en memoria de los workers; se abre UNA vez mañana.

## Espacio de búsqueda (muestreo aleatorio + mutación local de candidatos)
- Timeframe {1d, 4h}; modo {SPOT solo largos (0.10 %+5 bp/lado), PERP largos y cortos (0.055 %+5 bp + funding real)}.
- 1 a 3 componentes combinados con AND (o voto k-de-n): cruces EMA, precio vs SMA/EMA, Donchian, Supertrend,
  MACD, régimen RSI, ROC, ruptura Bollinger, ruptura Keltner, Aroon, CCI, Ichimoku (nube), tendencia OBV,
  reversión RSI corto / Bollinger con filtro de tendencia. Parámetros de conjuntos discretos.
- Filtros: ADX {0,15,20,25,30}; vol-targeting {no, 30 %, 50 %, 80 %}; trailing stop ATR {no, 2, 3, 4}.
- Cada configuración se evalúa en las 29 monedas (evaluación por moneda).

## Candidato (criterio fijado ahora)
(config, moneda) con: CAGR TRAIN > 20 % y CAGR VAL > 20 %, Sharpe TRAIN > 0.8 y Sharpe VAL > 0.8,
≥ 8 entradas en TRAIN+VAL, max drawdown TRAIN+VAL > −45 %.

## Registro de pruebas
Se cuenta TODA prueba (config × moneda) y se guardan momentos del Sharpe por (timeframe) para el
Deflated Sharpe con N = número total de pruebas.

## Mañana (TEST, una sola vez)
Para cada candidato: CAGR TEST, Sharpe, maxDD, vs buy&hold. "Confirmado" = CAGR TEST > 20 % y > buy&hold.
Se reporta también la tasa de confirmación (confirmados / candidatos) frente a lo esperable por azar,
y la cartera equiponderada de TODOS los candidatos (lo que se habría operado sin mirar el test).
