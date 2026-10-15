# Addendum al pre-registro v4 (2026-10-15, escrito ANTES de abrir el TEST)

Objetivo: traducir los candidatos a una cartera implementable con 1.000–2.000 € en Bybit EU (España → solo SPOT).
Reglas de selección decididas ahora, usando SOLO métricas de TRAIN/VAL:

- R1 (pre-registro original): todos los candidatos, equiponderados. Se estima con una muestra aleatoria por hash
  (determinista) si el total es demasiado grande para evaluar en tiempo razonable.
- R2 "mejor por moneda": para cada moneda, el candidato SPOT con mayor min(Sharpe TRAIN, Sharpe VAL);
  cartera equiponderada de esas estrategias (una por moneda).
- R3 "transversales": configuraciones SPOT que son candidatas en ≥ 5 monedas; se aplican a esas monedas;
  cartera equiponderada de todos esos pares.
- R4 "top-10 distintas": los 10 candidatos SPOT con mayor min(Sharpe TRAIN, Sharpe VAL), con monedas distintas;
  cartera equiponderada.
- Variantes PERP de R2/R4 solo como referencia (no operables desde Bybit EU).

Simulación de capital: 1.000 € y 2.000 € repartidos a partes iguales entre las estrategias de la cartera,
sin rebalanceo entre estrategias, costes ya incluidos en los retornos. Se reporta valor final, CAGR,
peor caída en €, y número de operaciones; y se comprueba el mínimo de orden de Bybit (5 USDT).
Comparativas: buy&hold equiponderado de las mismas monedas y la tasa de confirmación frente al azar.
