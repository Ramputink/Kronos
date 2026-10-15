# Apertura del TEST y plan para 1.000–2.000 € (2026-10-15)

TEST = 2025-07-01 → 2026-10-13 (470 días, mercado mayormente bajista: BTC −23 %, ETH 0 %, media de altcoins ≈ −50 %).
Pre-registro v4 (1df9d2df…) + addendum (ebbce68e…); selecciones congeladas antes del TEST (a5231a02…).

## 1. ¿Sirvieron los 7,9 M de candidatos de la noche? No.
- Muestra aleatoria del 10 % (596.560 candidatos): **6,3 %** se confirman (CAGR TEST > 20 % y > buy&hold),
  frente a **9,4 %** de configuraciones aleatorias → los candidatos lo hacen PEOR que el azar (sobreajuste).
- Cartera equiponderada de todos los candidatos: CAGR +1,0 %, Sharpe 0,14.
- Carteras ex-ante (spot, operables en Bybit EU), resultado total en 15 meses:
  R4 top-10 +1,7 % (1.000 € → 1.017 €), R2 mejor-por-moneda +3,2 % (1.032 €), R3 transversales +5,5 % (1.055 €),
  con caídas de −170 a −240 € por cada 1.000 €.
- La puntuación TRAIN/VAL no predijo el TEST (RUNE, la mejor puntuada: +0,3 %; AVAX −38 %, BTC −27 %).
- Única regularidad: en ETH y BNB ~68 % de los candidatos se confirman (coincide con el estudio de indicadores).

## 2. Opciones simples (regla RSI(14) régimen + vol-target, pre-registrada en el estudio de indicadores)
| Opción (spot, comisión Bybit EU 0,25 %) | Total 15 meses | 1.000 € → | Peor caída | 2.000 € → | Entradas |
|---|---|---|---|---|---|
| A. Solo ETH | +30,6 % | 1.306 € | −517 € | 2.612 € | 13 |
| B. ETH + BNB (*elegida a posteriori*) | +38,5 % | 1.385 € | −457 € | 2.771 € | 24 |
| C. BTC + ETH + SOL + BNB | +22,1 % | 1.221 € | −322 € | 2.442 € | 49 |
| D. 9 large caps (universo fijado antes) | +10,5 % | 1.105 € | −360 € | 2.211 € | 117 |
| Buy&hold BTC | −22,6 % | 774 € | −615 € | 1.548 € | 1 |
| Buy&hold ETH | +0,2 % | 1.002 € | −1.310 €* | 2.004 € | 1 |
*sobre 2.000 € la caída de ETH sería −1.310 €; sobre 1.000 €, ≈ −655 €.

Advertencias: D y A son las más limpias (universo/moneda fijados antes de ver el TEST); B y C eligen monedas
sabiendo cómo fue cada una en este TEST → optimistas. Es un único periodo bajista de 15 meses.

## 3. Qué hacer con 1.000–2.000 € (propuesta)
1. No operar los candidatos de la búsqueda nocturna.
2. Si se quiere un sistema: regla RSI(14) diario (>55 entra, <50 sale) con tamaño por volatilidad, spot,
   en BTC+ETH+SOL+BNB (o las 9 large caps), revisión una vez al día. Expectativa honesta: 10–25 %/año en
   buen escenario, con caídas de −25 a −35 % y años negativos posibles (2022: −17 % a −25 %).
3. Arrancar en Bybit demo 2–3 meses; luego con 1.000 € reales, sin apalancamiento; ampliar a 2.000 € solo si
   la ejecución real coincide con la simulada.
4. Mínimos de orden (5 USDT) OK con este capital; comprobar comisión real ("My Fee Rate") y disponibilidad de
   API en Bybit EU. Fiscalidad España: las ganancias tributan en la base del ahorro (19–28 %).
No es asesoramiento financiero.
