# Bloque 2 — Primera iteración: EBM / LightGBM / Student / RuleFit

**Notebook:** `Iteracion_1_Final_TFM.ipynb`

Entrenamiento y comparación de modelos con validación Walk-Forward (6 folds) y optimización de hiperparámetros con Optuna:
- **EBM** (Explainable Boosting Machine) — modelo principal interpretable.
- **LightGBM** — referencia de alto rendimiento.
- **Student** (destilación de LightGBM) y **RuleFit / Decision Tree** — extracción de reglas comprensibles.

**Modelos entrenados:** `models/iteracion_1/` (6 folds × 5 tipos de modelo).
**Resultados:** `reports/iteracion_1/` — métricas por fold, importancia SHAP, análisis de umbral, estabilidad de reglas y backtesting inicial.
