# Pre-registro v3: 10 estrategias con rentabilidad superior en criptos distintas

Fecha: 2026-10-14, escrito ANTES de ejecutar la búsqueda por moneda.

## Universo (fijo, el de MarketReasercher/phase2)
BTC ETH SOL XRP ADA AVAX LINK DOGE BNB LTC DOT POL ATOM NEAR APT BCH TRX ETC XLM DASH ZEC VET THETA
ALGO COMP CRV RUNE SUSHI EGLD UNI. Requisito: velas diarias en Bybit antes de 2022-12-31 (si no, se excluye).
Nota de contaminación: ya vi el TEST de 9 monedas (BTC ETH SOL XRP BNB DOGE ADA LINK AVAX) para 3
configuraciones concretas; se señalará en el informe.

## Espacio de búsqueda por moneda
La rejilla de ~/kronos_lab/indicators/engine.py (40 reglas base × ADX {0,20,25} × vol-targeting {no,sí}),
velas 1d, costes SPOT Bybit (0.10 % + 5 bp por lado), solo largos. 240 configuraciones por moneda.

## Periodos
TRAIN ≤ 2023-12-31 · VAL 2024-01-01 → 2025-06-30 · TEST 2025-07-01 → último cierre (se abre una vez).

## Selección por moneda (sin mirar TEST)
Candidata = configuración con Sharpe > 0 tanto en TRAIN como en VAL; de ellas, la de mayor
Sharpe en TRAIN+VAL combinado. Si ninguna tiene Sharpe > 0 en ambos, la moneda no tiene estrategia.

## Definición de "rentabilidad superior" (decidida ahora)
En TEST: CAGR de la estrategia > 0 **y** > CAGR de buy&hold de esa moneda.
Nivel exigente adicional: CAGR TEST > 15 %.
También se reporta el Deflated Sharpe (N = 240) de cada selección sobre TRAIN+VAL.

## Objetivo
Contar cuántas monedas cumplen. Si son ≥ 10, se listan las 10 con mayor CAGR en TEST
(y se dice cuántas cumplen también el nivel del 15 %). Si son < 10, se reporta el número real.
