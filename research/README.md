# Investigación: Kronos + estrategias de trading cripto en Bybit (13–15 oct 2026)

Todo el trabajo sigue el mismo método: pre-registro (con hash) antes de mirar resultados, separación temporal
TRAIN / VALIDACIÓN / TEST, costes reales de Bybit (comisión, slippage y funding) y corrección por número de pruebas
(Deflated Sharpe). Nada de esto es consejo financiero.

## 1. Kronos (modelo fundacional de velas) en Bybit
- `examples/bybit_forecast.py`: descarga velas públicas de Bybit, predicción en vivo y walk-forward.
- Bugs corregidos en este fork: el predictor no ponía el modelo en `eval()` (dropout activo en inferencia) y
  `finetune_csv` normalizaba cada ventana con datos futuros (los objetivos).
- **Resultado:** Kronos-small zero-shot no tiene ventaja direccional. Siguiendo su señal a 24h con comisiones de
  Bybit: BTC −21 %, ETH −47 %, SOL −62 % (frente a +4.5/+12/+20 % de buy&hold, mayo–oct 2026).
- Laboratorio multi-agente (34 agentes): `kronos_bybit_lab_report.md`. Como filtro de Bollinger, RSI, Fibonacci,
  Pitchfork, etc. empeora o iguala a un filtro aleatorio. Solo aporta algo como estimador de volatilidad, sin
  traducirse en dinero.

## 2. MarketReasercher + Kronos (`kronos_lab/marketresearcher/`)
15 estrategias evolucionadas del repo MarketReasercher en Bybit 1h (BTC/ETH/SOL, 2026): media −1.2 % sobre el capital;
con Kronos como filtro, −0.4 %, igual que un filtro aleatorio (−0.3 %). Las comisiones se comen el edge bruto.

## 3. Objetivo >15 % anual (`kronos_lab/goal15/`)
Breakout 50d + EMA200 con filtro BTC (de la fase 3 de MarketReasercher), replicado en Bybit:
+33.5 % anual OOS 2024-26 y +20.3 % anual 2021-26 (BTC +19.4 %), caída máxima −54 %. No pasa el criterio estricto
pre-registrado: en el forward de 4 meses, +15 % frente a +29 % de BTC.

## 4. Combinación de indicadores en large caps (`kronos_lab/indicators/`)
960 configuraciones con los mismos parámetros en 9 large caps. Mejor: RSI(14) diario como régimen (>55 entrar,
<50 salir) con tamaño por volatilidad, spot. TEST (jul-2025→oct-2026, mercado bajista): ETH +26.8 %/año,
SOL +12.1 %, cartera +12.1 % (Deflated Sharpe 0.55). Brief de Bybit para España: Bybit EU solo spot, sin perps.

## 5. Mejor estrategia por moneda (`kronos_lab/top10/`)
240 configuraciones por moneda, elegidas sin mirar el TEST. 13 de 29 monedas superan a buy&hold en el TEST (las
mejores: UNI +33.9 %, EGLD +18.5 %, CRV +17.7 %, ALGO +15.2 %, BNB +14.5 %). Pero operar las 29 selecciones juntas
(lo único que se sabía de antemano) dio −0.4 %/año frente a −11.1 % de buy&hold: protegen capital, no generan >15 %.

## 6. Búsqueda nocturna masiva (`kronos_lab/overnight/`)
10 procesos de CPU + búsqueda densa en GPU (MPS) durante 8.9 h: **2.048 millones de configuraciones, 2.155 millones
de pruebas (config × moneda), 7,89 millones de candidatos** (CAGR > 20 % y Sharpe > 0.8 en TRAIN y en VALIDACIÓN).
- El TEST (2025-07-01 → 2026-10) **sigue sellado**. Los candidatos (3,9 GB) se quedan en local, en
  `~/kronos_lab/overnight/out/`. Aquí solo se suben el código, el pre-registro, las estadísticas y el resumen.
- Siguiente paso, para abrir el TEST una sola vez:
  `cd ~/kronos_lab/overnight && ~/.venvs/kronos/bin/python -W ignore morning_test.py`
  Compara la tasa de confirmación de los candidatos con la de configuraciones aleatorias y calcula la cartera de
  todos los candidatos.
- Con este número de pruebas, la gran mayoría de candidatos serán suerte. Lo relevante será si la tasa de
  confirmación supera a la del azar.

## Conclusión transversal
La palanca más fiable no ha sido la señal sino: pocas operaciones (diario, no 1h), salir a liquidez en tendencias
bajistas (filtros ADX/EMA200) y dimensionar por volatilidad. Ninguna combinación ha demostrado un >15–20 % anual
robusto fuera de muestra en todas las monedas.
