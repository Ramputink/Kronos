# Kronos para trading cripto en Bybit: informe final

**Fecha:** 2026-10-13 · **Equipo:** 25 agentes en 2 rondas (19 exploradores y 4 verificadores adversariales en la ronda 1, 6 agentes en la ronda 2) · **Laboratorio:** `/Users/matveypro/kronos_lab/`

**Datos usados en todas las pruebas:** Kronos-small zero-shot, velas de 1h de Bybit para BTC, ETH y SOL, horizonte de 24h, 6 trayectorias muestreadas por origen (top_p=0.9, contexto de 256 velas). Hay ~300 orígenes por símbolo cada 12h, del 2026-05-15 al 2026-10-12. Como las ventanas de 24h se solapan, la n efectiva es de unas 150 por símbolo. Todo el periodo es un único régimen de ~5 meses, mayormente alcista. La comisión es taker de 0.055% por lado (0.11% ida y vuelta).

---

## 1) Veredicto en 5 líneas

1. **La dirección de Kronos no es un edge.** Operar el signo de su previsión a 24h pierde entre -0.3% y -0.6% netos por operación en los 3 símbolos, y el IC a 24h está entre 0 y -0.09.
2. **Usarlo como filtro de confirmación de indicadores es perjudicial.** Lo probamos con Bollinger, RSI, Fibonacci, Pitchfork, EMA/MACD/Donchian, VWAP/pivots/Ichimoku, funding/OI y volumen, y en casi todos los casos empeora el resultado y queda por debajo de un filtro aleatorio del mismo tamaño.
3. **El motivo es mecánico.** `predict()` normaliza sobre la ventana de contexto y las trayectorias se encogen hacia la media. En la práctica Kronos-small es una señal de reversión a 24h amortiguada: su correlación con el retorno pasado de 24h es de -0.7.
4. **Lo único con algo de señal es la volatilidad, y no da dinero.** Su dispersión de trayectorias predice la vol de 24h igual que ATR o Parkinson, con un pequeño extra en BTC. Ese extra no se traduce en Sharpe en vol-targeting, gates ni stops (3 de 4 verificaciones adversariales lo tumbaron).
5. **Hoy su valor es de infraestructura:** la caché, el loop de Bybit con la latencia medida y un plan de caché v2 y de fine-tuning. Antes de arriesgar dinero hay que recalibrar o reentrenar y medir CRPS. MiroFish no es backtestable y no merece la pena para este objetivo.

---

## 2) Ranking de usos de Kronos

Leyenda de la verificación adversarial: ✅ sobrevive · ⚠️ sobrevive debilitado · ❌ no sobrevive · — sin verificación dedicada (sí replicado por varios agentes independientes).

### Lo que (como mucho) funciona un poco

| # | Uso | Números clave | n | Verificación | Carpeta |
|---|---|---|---|---|---|
| 1 | **Previsión de volatilidad a 24h (dispersión de trayectorias, `k_rv`/`k_path`)** | Rank IC con la vol realizada: 0.41 / 0.38 / 0.51 (BTC/ETH/SOL), similar a ATR (0.43/0.42/0.56). El nivel sale casi correcto sin ajustar nada (ratio 0.86-0.93). Con controles fuertes (Parkinson, EWMA, HAR, fin de semana), el rho parcial queda en BTC 0.135 [IC 0.03-0.27], ETH 0.07 (n.s.) y SOL 0.04 (n.s.). En el holdout, el MSE de log-vol es mejor solo en BTC y ETH. | 300/símbolo (~150 efectivas) | ❌ vol_forecast · ❌ bybit_arch · ⚠️ literature ("débil") · ❌ stats_baseline | `agents/vol_forecast/`, `agents/verify_vol_forecast/`, `agents/literature/`, `agents/verify_literature/`, `agents/bybit_arch/`, `agents/verify_bybit_arch/`, `agents/volume/`, `agents/meta_labeling/` |
| 2 | **Vol de Kronos convertida en dinero** (vol-targeting, gate de régimen, ancho de stop) | Vol-target long: Sharpe neto con Kronos frente a sin Kronos = BTC 1.29 vs 1.31, ETH 2.05 vs 2.03, SOL 1.92 vs 1.95. Todos los ΔSh tienen IC que incluye 0. El gate BB/Donchian da ETH −1.01 y SOL +1.08, con signos opuestos, así que es ruido. En stops: misma tasa de stop-out y mismo P&L. | 120 días no solapados/símbolo | Pre-especificado en la ronda 2, **no añade valor** | `agents/r2_vol_gate_economics/` |
| 3 | **IC a 1-2 pasos (1-2h)** | IC de k1 a 1h agregado = 0.10 [0.02, 0.17]. La reversión de 4h sola da 0.21. Añadir Kronos aporta −0.016 (1h) y +0.033 (2h), con IC que incluye 0 en ambos casos. Gross de +2 a +7 bp, frente a 11 bp de taker. Con maker (4 bp) y el modelo de fill, todo sale negativo. | 840 agregadas | Prueba pre-registrada en la ronda 2: **no** | `agents/consensus_horizon/`, `agents/r2_short_horizon_maker/`, `agents/stats_baseline/` |

### Lo que no funciona

| # | Uso | Números clave | n | Verificación | Carpeta |
|---|---|---|---|---|---|
| 4 | **Dirección a 24h (signo de la mediana o la media)** | Netos por operación: BTC −0.29/−0.31%, ETH −0.35/−0.48%, SOL −0.55/−0.62% (t de −1.8 a −2.5; agregado −0.47%, t −3.7). El consenso de ≥5/6 trayectorias es igual de malo (−35 a −48 bp). | 300/símbolo | — (replicado por más de 8 agentes) | `agents/consensus_horizon/`, `agents/regime/`, `agents/stats_baseline/`, `agents/bybit_features/` |
| 5 | **Filtro de confirmación sobre indicadores** | Empeora en ~18/18 celdas de pitchfork, en 6/6 de tendencia y en todas las de VWAP/Ichimoku y funding/OI. Las operaciones que conserva quedan en el percentil 0.1-15 frente a filtros aleatorios. | 15-300 por celda | — (replicado por 9 agentes) | ver la sección 3 |
| 6 | **"Fade" de Kronos / operar solo cuando Kronos discrepa** | Positivo post-hoc en más de 6 agentes, pero en el holdout pre-registrado de la ronda 2 (7 reglas apiladas, 1424 operaciones): discrepa −11.4 bp, regla sola −20.1 bp, buy&hold +16.6 bp. El término kdis tiene t=0.82 y el IC es −0.054. Casi toda la "ganancia" es beta long (61% de las operaciones son largas). corr(pred, real) pasa de −0.2 en los 2/3 iniciales a ≈0 en el último tercio. | 1424 operaciones en el holdout | ❌ (r2_disagree_preregistered) | `agents/contrarian/`, `agents/r2_disagree_preregistered/` |
| 7 | **Trayectorias recalibradas (drift + escala de spread)** | El CRPS baja un 25-35% tras recalibrar, pero solo empata con una gaussiana de vol de 7d en BTC/SOL y pierde en ETH (+11.9 bp, IC [+1, +20]). El IC direccional es igual sin Kronos. Operar el signo da −38 / −35 / +22 bp (SOL p=0.17 frente a aleatorio). | 100 test/símbolo | Pre-especificado en la ronda 2: **no** | `agents/r2_recalibrated_paths/` |
| 8 | **Motor de riesgo, VaR o escenarios (abanico de 6 trayectorias)** | Cobertura del rango min-max de 34-39% frente a ~71% nominal. Las violaciones de VaR al 5% ocurren el 17-30% de las veces. El CRPS es ~70% peor que el de una gaussiana con vol trailing. Hay saltos artefacto de un solo paso (curtosis horaria de 178 frente a 15 real). | 300/símbolo | — | `agents/var_calibration/`, `agents/montecarlo_stress/`, `agents/finetune_plan/` |
| 9 | **Rangos para TP/SL y grid bots (envolvente high/low)** | La envolvente contiene el rango real solo el 40-44% de las veces. A igual ancho, contiene menos que ATR, que la vol e incluso que una banda constante. El grid de 10 niveles pierde en todos los casos, y Kronos con anchos barajados rinde igual. La asimetría prevista tiene el signo contrario (−0.15 a −0.22). | 300 / 100 test | — | `agents/range_tp_sl/` |
| 10 | **Relative value (pares ETH/BTC, SOL/BTC, SOL/ETH, ranking de 3 activos)** | IC del spread ≈ 0 (+0.01 / −0.05 / −0.01). Netos de −27 a −35 bp por par, percentil 15-34 frente a señales barajadas. | 300 | — | `agents/cross_asset/` |
| 11 | **Detección de picos de volumen o vol** | AUC de Kronos 0.64-0.72 frente a 0.68-0.84 del simple ratio 24h/7d. | 300 | — | `agents/volume/` |
| 12 | **Meta-labeling (logística o ridge walk-forward)** | Acierto direccional: Kronos 44.0%, indicadores 47.8%, ambos 45.2%, frente a 53.5% de siempre largo. En vol: IC de 0.446 a 0.461 al añadir Kronos (ruido). | 600 OOS | — | `agents/meta_labeling/` |
| 13 | **Por régimen (ADX, vol, fin de semana, acuerdo)** | Ninguna celda tiene p<0.10. Todos los efectos cambian de signo entre train y test (ADX>25: 0.43 → 0.64). | 600 | — | `agents/regime/` |

---

## 3) Indicadores técnicos: ¿les añade algo Kronos?

**No.** En todos los casos el filtro "Kronos confirma" empeora el indicador. La razón es que Kronos ya es un predictor de reversión: correlaciona −0.56 con el RSI14 y −0.7 con el retorno de 24h. Por eso duplica las señales de mean-reversion (RSI, BB) y va en contra de las de tendencia (EMA, Donchian).

| Indicador | Indicador solo (neto/op) | + Kronos confirma | Comentario | Carpeta |
|---|---|---|---|---|
| **Bollinger(20,2)**, MR y breakout, estricto y relajado | p. ej. BTC MR relajado −0.24%; SOL BO relajado +0.13% | SOL BO: −0.23% (percentil 1 frente a aleatorio); SOL MR: −0.75% (percentil 2) | En BTC/ETH los resultados quedan dentro del ruido (36 tests). En SOL, las operaciones que Kronos rechazó ganaban +1.46%. | `agents/bollinger/` |
| **RSI 7/14/21**, umbrales 30/70 a 40/60, y divergencias | SOL RSI14 40/60: −0.26% | −0.72% (p=0.99 frente a un subconjunto aleatorio) | Kronos nunca bate a un subconjunto aleatorio (p de 0.28 a 0.99). | `agents/rsi/` |
| **Fibonacci** (swing de 72h, zona 38.2-61.8, stop en 78.6) | BTC −0.40%, ETH −0.13%, SOL −0.10% | BTC −0.82%, ETH −0.13%, SOL −0.49% | Los extremos reales no se agrupan en niveles Fibo más que en una rejilla desplazada al azar. | `agents/fibonacci/` |
| **Andrews' Pitchfork** (pivots fractales k=6/12) | Sin edge propio (todos con \|t\|<1.4) | Peor en 18/18 comparaciones | El signo del pitchfork solo cambia con k. | `agents/pitchfork/` |
| **EMA12/48, MACD, Donchian48** | BTC Donchian +15 bp gross; ETH MACD +13 bp | Confirmado: de −4 a −45 bp gross, percentil 5-13 | En el holdout de la ronda 2, "Kronos confirma" en tendencia es significativamente malo (EMA −104 bp en el percentil 1.5, Donchian −87 bp en el percentil 3). Es el único efecto consistente, y solo sirve como veto. | `agents/trend_ma/`, `agents/r2_disagree_preregistered/` |
| **VWAP 24/168, pivots, Donchian, Ichimoku** | SOL VWAP168 +10.9 bp | −52.9 bp; Ichimoku SOL de −18 a −108 bp | Las probabilidades de toque o ruptura (AUC 0.46-0.72) son peores que la simple distancia/ATR (0.59-0.86). | `agents/levels_vwap/` |
| **Funding y OI de Bybit** (contrarian, OI-surge fade) | Combinado agregado −0.24% | −0.49% (el 99.6% de los filtros aleatorios lo baten) | El funding y el OI tampoco predicen nada solos (\|rho\|<0.05). | `agents/bybit_features/` |

**Hallazgo sin Kronos, más útil que todo lo anterior** (`agents/r2_indicator_only_fullhistory/`). Lo probamos con 1 año de datos, un 40% de holdout intacto, comisiones, funding y stop ATR:

- **Ningún sistema de indicadores bate a buy&hold** (Sharpe neto 0.95 en el test).
- El mejor fue BB-breakout/RSI-MR con switch de régimen: Sharpe 0.74, IC de −2.1 a 3.8.
- La mean-reversion de BB y RSI pierde fuerte (Sharpe −3.0 y −2.1), porque las comisiones se comen entre un 17% y un 20% en 5 meses.
- **La palanca principal son las comisiones y la rotación, no la señal.** Lo que hay que probar es usar timeframes de 4h o diarios, holds mínimos de 24h o más y órdenes maker.
- El régimen domina: el año incluye un bear del −60% y luego un bull.

---

## 4) Riesgo y volatilidad

- **Calibración del abanico (crudo):** el abanico es demasiado estrecho, entre 2.1 y 2.7 veces.
  - El std entre trayectorias es solo 0.37 veces el std realizado.
  - El histograma de rangos tiene forma de U (29% y 32% en los extremos).
  - El rango min-max cubre el 34-39% de los casos frente a un ~71% esperado.
  - Fuentes: `agents/r2_cache_audit_v2_spec/`, `agents/montecarlo_stress/`, `agents/var_calibration/`, `agents/regime/`.
- **VaR y CVaR:** con Kronos en bruto, el VaR al 5% se rompe un 17-25% de las veces en el test.
  - EWMA y HistSim de 30 días ya están cerca del nominal (6-9%).
  - Reescalado, Kronos sigue peor en Winkler y pinball. **Para riesgo hay que usar EWMA o HistSim, no Kronos.** (`agents/var_calibration/`)
- **Simulación de Monte Carlo:** las trayectorias de Kronos no reproducen los hechos estilizados.
  - La curtosis es de 178 frente a 15 real, por saltos artefacto.
  - Un bootstrap por bloques del histórico calibra mucho mejor (cobertura 0.70-0.74). (`agents/montecarlo_stress/`)
- **Volatilidad como nivel:** es razonable sin ajuste, pero la información incremental es pequeña y depende del símbolo (solo BTC es robusto, ver el ranking #1 y #2).
- **Lo que sí funciona, gratis y sin modelo:**
  - **Vol-targeting con HAR/Parkinson:** mejora el Sharpe frente a peso constante (BTC 1.31 vs 1.00, ETH 2.03 vs 1.58).
  - **Efecto fin de semana:** la vol del fin de semana es 0.64-0.79 veces la de entre semana, con una correlación de −0.3 a −0.58 con la vol realizada.
  - Ambos son útiles para dimensionar posiciones y ajustar el ancho de los stops. (`agents/r2_vol_gate_economics/`, `agents/r2_indicator_only_fullhistory/`)
- **Umbral de significancia** (`agents/stats_baseline/`): con esta muestra, para creerse un resultado hace falta lo siguiente.
  - En un solo símbolo: IC > ~0.26 o Sharpe anualizado > ~5.
  - Agregando los 3 símbolos: IC > ~0.15 o Sharpe > ~2.8.
  - Ningún uso de Kronos se acerca a ese listón.

---

## 5) MiroFish: qué es y si merece la pena

**Qué es** (`agents/mirofish/FINDINGS.md`). Es un motor de simulación social con un enjambre de LLMs (github.com/666ghj/MiroFish, licencia AGPL-3.0, ~78k estrellas, sin releases; la demo financiera sigue "coming soon").

- Convierte un texto semilla en un grafo GraphRAG alojado en Zep Cloud.
- Genera personas como agentes LLM y las simula en plataformas tipo Twitter/Reddit con CAMEL-AI OASIS.
- Un ReportAgent redacta al final un informe en prosa.

**¿Merece la pena para trading? No.**

- No produce previsiones numéricas ni ingiere precios.
- Cuesta miles de llamadas a LLM por ejecución.
- **No se puede backtestar** sobre nuestras 900 ventanas: necesitaría noticias point-in-time, y los LLMs probablemente ya "conocen" lo que pasó (lookahead).
- La única evaluación práctica que encontramos (WPP Research, n=1 día) hizo la misma predicción errónea del S&P que una sola llamada a un LLM.
- **No se ejecutó ningún experimento** (no hay claves de API y cuesta dinero), así que la evidencia es débil y de escritorio.

**Único uso defendible:** generar escenarios narrativos discretos y traducirlos a shocks de drift y vol aplicados a trayectorias simuladas, para stress-testing y no para alpha.

- La línea base que tendría que batir es **una única llamada a un LLM con salida JSON** `{bias, conviction, tail_risk_flag}`, validada en paper trading hacia delante.

---

## 6) Arquitectura recomendada para Bybit y plan de fine-tuning

### Arquitectura (`agents/bybit_arch/`)

- **Latencia medida:** de cierre de vela a orden pasan unos 2 s.
  - REST de kline: mediana de 254 ms, p90 de 373 ms.
  - Inferencia de Kronos: ~1.4 s.
  - Orden: ~0.3 s.
  - Con velas de 1h la latencia es irrelevante; **lo que manda es la comisión**.
- **Trampa crítica:** las klines de Bybit v5 vienen de la más nueva a la más vieja e **incluyen la vela en curso sin confirmar**. Hay que descartarla, o el sistema en vivo verá algo distinto al backtest.
- **Diseño del sistema:**
  - Scheduler a HH:00:05, o al evento de kline confirmada por websocket.
  - pybit HTTP más reconciliación por websocket privado.
  - **Primero demo trading o testnet.**
  - `orderLinkId` idempotente.
  - SL y TP en el servidor, y salida reduceOnly.
- **Kill-switches:**
  - Pérdida diaria o drawdown.
  - Datos obsoletos (más de 65 min sin vela confirmada).
  - 3 errores seguidos.
  - Desajuste de posición.
  - Fichero HALT manual.
  - Al arrancar, cancelar todas las órdenes.
- **Seguridad:**
  - Subcuenta con API key solo de trading, sin retiros y con IP en whitelist.
  - Logs en JSONL y alertas por Telegram.
- **Pendiente de comprobar:** pybit y ccxt no están en el venv, así que los nombres de parámetros (`demo=`, `tpslMode`) hay que verificarlos en testnet.
- **Papel de Kronos en producción hoy:** como mucho, una capa de logging o investigación (registrar previsión frente a realidad). No debe generar señales ni dimensionar posiciones. Para dimensionar, HAR/Parkinson con ajuste de fin de semana es gratis y rinde igual.

### Bugs y auditoría (`agents/finetune_plan/`, `agents/r2_cache_audit_v2_spec/`)

- **Bug real de lookahead:** `finetune_csv/finetune_base_model.py` (`CustomKlineDataset.__getitem__`, líneas ~118-127) normaliza cada ventana con media y std calculadas incluyendo las velas futuras (los targets). El pipeline Qlib (`finetune/dataset.py`) lo hace bien. **Hay que corregirlo antes de cualquier fine-tune.**
- **Otros problemas del pipeline:**
  - Las ventanas de la plantilla miden 561 tokens, por encima del `max_context` de 512.
  - Se elige checkpoint por cross-entropy y no por una métrica de previsión muestreada.
  - El learning rate del predictor es de 1e-6, prácticamente nada.
  - Concatenar símbolos en un solo CSV crea ventanas que cruzan de un símbolo a otro.
  - No hay embargo en el split temporal.
  - No hay soporte de MPS ni de AMP.
- **Auditoría de la caché v1:**
  - Se confirma lookback 256: la afirmación de "400" era incorrecta según el npz.
  - Se confirman top_p=0.9 y 6 trayectorias independientes.
  - Promediar no explica la sub-dispersión, así que los sospechosos son el truncado de top_p y el propio modelo.

### Plan, por orden y con go/no-go

1. **Caché v2, gratis y solo con GPU.** El script `agents/r2_cache_audit_v2_spec/build_kronos_cache_v2.py` ya está probado en dry-run.
   - Configuración: top_p=1.0, T=1.0, 20-30 trayectorias, origen cada 4-8h, periodo 2024-01 a 2025-06 (disjunto del v1). Descarta velas sin confirmar, comprueba huecos y es reanudable.
   - Tiempo: ~4.4 h con `--paths 20 --stride 8` y ~13.3 h en la versión completa.
   - Comando: `cd /Users/matveypro/Kronos && /Users/matveypro/.venvs/kronos/bin/python /Users/matveypro/kronos_lab/agents/r2_cache_audit_v2_spec/build_kronos_cache_v2.py --out /Users/matveypro/kronos_lab/cache_v2`
2. **Medir CRPS y cobertura en v2** frente a una gaussiana con vol trailing de 7 días. Si top_p=1.0 arregla la calibración, se reevalúa el uso para vol y riesgo.
3. **Fine-tune solo si Kronos sigue perdiendo frente a la gaussiana.**
   - Qué se entrena: solo el predictor, con el tokenizer congelado.
   - Datos: perps lineales de Bybit (top 30 en 1h desde 2021 más top 10 en 15m), ~3.5 M velas, ~400k ventanas.
   - Split: train 2021-2024, val 2025H1, test 2025H2 hasta hoy, con embargos de 1 semana y 3 folds walk-forward.
   - Hiperparámetros: LR de 1e-5 a 3e-5, bf16.
   - Selección de checkpoint por CRPS y cobertura en validación.
   - Coste: ~2.5 h y 2-5 $ por run de Kronos-small en una 4090 o A100 alquilada, unos 20-45 $ por un sweep de 9 runs. En el M2 Pro local serían 6-10 h tras parchear MPS. Todo son estimaciones por FLOPs, no mediciones.
   - **No-go** si el modelo fine-tuneado no bate a la gaussiana por al menos un 5% de CRPS en validación.

---

## 7) Próximos pasos concretos (de mayor valor/esfuerzo a menor)

1. **Bajar la rotación y las comisiones de tus bots actuales** (no requiere Kronos y es lo de mayor impacto). Retesta tus estrategias en 4h o diario, con un hold mínimo de 24h o más y entradas maker (~0.02% por lado frente a 0.055%). Las comisiones se comieron el edge bruto de todas las estrategias de indicadores. (`agents/r2_indicator_only_fullhistory/`)
2. **Dimensionar con vol HAR/Parkinson y ajuste de fin de semana** en los bots de Bybit y Binance. Es la única pieza robusta que encontramos y es gratis. (`agents/r2_vol_gate_economics/`)
3. **Corregir la fuga de normalización** en `finetune_csv/finetune_base_model.py`, que es una línea de código, antes de cualquier fine-tune.
4. **Lanzar la caché v2** (unas 4.4 h de GPU desatendida) y medir CRPS y cobertura. Así se separa el efecto del ajuste top_p=0.9 del del propio modelo.
5. **Reprobar sin retocar en el periodo disjunto 2024-01 a 2025-06** las dos únicas pistas: (a) el veto "Kronos confirma una entrada de tendencia → no entrar" y (b) el extra de vol en BTC. Ambas deben estar pre-registradas.
6. **Paper trading en Bybit demo durante 60-90 días**, registrando previsión frente a realidad para tres variantes: indicador solo, indicador con sizing HAR, y lo mismo más el log de Kronos. Debe descartar la vela sin confirmar.
7. **Fine-tune de Kronos-small (20-45 $)** solo si el paso 4 sigue perdiendo frente a la gaussiana, y con el go/no-go de la sección 6.
8. **(Opcional, investigación)** Ranking cross-sectional de más de 20 perps de Bybit, que es el uso real del paper de Kronos (top-k sobre acciones A chinas), no el timing de un solo activo. (`agents/literature/`)
9. **(Baja prioridad)** MiroFish o un LLM solo como veto de eventos y generador de escenarios de estrés, y únicamente después de que una sola llamada JSON a un LLM demuestre valor en forward test.

---

## 8) Advertencias

- **Muestra pequeña:** ~300 ventanas solapadas por símbolo (~150 independientes) en **un solo régimen de 5 meses**, y BTC, ETH y SOL están muy correlacionados. Los intervalos de confianza son anchos (en Sharpe, ±3 o más).
- **Multiple testing:** entre todos los agentes se probaron cientos de celdas, así que es esperable que alguna salga "significativa" por azar. Por eso **no damos por buena ninguna mejora encontrada post-hoc**, como el fade de Kronos, el +154 bp de RSI con n=8, el MACD con doble desacuerdo o el bracket de SOL. Las pruebas pre-registradas de la ronda 2 no confirmaron ninguna.
- **Alcance del modelo:** solo se probó Kronos-small zero-shot, con 6 trayectorias y top_p=0.9. Kronos-base o large, una caché recalibrada o un fine-tune podrían comportarse distinto, aunque nada de lo visto lo sugiere para la dirección.
- **Ejecución simplificada:** fills al cierre, sin slippage. El funding solo se modela en parte, y los fills maker con un modelo simple.
- **Dos verificaciones no se hicieron como experimento:** MiroFish es investigación de escritorio sin ejecución, y los costes de fine-tuning son estimaciones.
- **Esto no es consejo financiero.** Es un estudio estadístico interno. Ninguna estrategia de este informe ha demostrado un edge neto robusto, y operar con apalancamiento en perps puede suponer la pérdida total del capital.

---

*Índice de carpetas:* `/Users/matveypro/kronos_lab/agents/` contiene bollinger, rsi, fibonacci, pitchfork, trend_ma, levels_vwap, bybit_features, volume, meta_labeling, regime, contrarian, consensus_horizon, cross_asset, range_tp_sl, var_calibration, montecarlo_stress, vol_forecast, literature, bybit_arch, finetune_plan, mirofish, stats_baseline, las carpetas de verificación verify_* (vol_forecast, literature, bybit_arch, stats_baseline) y las de la ronda 2 r2_* (recalibrated_paths, disagree_preregistered, vol_gate_economics, indicator_only_fullhistory, cache_audit_v2_spec, short_horizon_maker).