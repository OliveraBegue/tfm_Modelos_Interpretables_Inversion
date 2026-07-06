# Bloque 3 — Segunda iteración: filtro de régimen de mercado (HMM)

**Notebook:** `Iteracion_2_Final_TFM.ipynb`

Se incorpora un Hidden Markov Model para detectar regímenes de mercado (alcista/bajista) y filtrar las señales de la Iteración 1, activándolas solo en condiciones favorables. Incluye backtesting fuera de muestra (2020–2024) comparando Iteración 1 vs Iteración 2 vs Buy & Hold.

**Modelos entrenados:** `models/iteracion_2/` (HMM, 6 folds).
**Resultados:** `reports/iteracion_2/` — regímenes detectados, matrices de transición, métricas consolidadas y curvas de equity comparativas.
