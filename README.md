# Modelos Interpretables para Inversión Financiera — S&P 500

**Generación de señales de inversión mediante aprendizaje supervisado interpretable y detección de regímenes económicos**

Trabajo Fin de Máster · MSc Data Science · Universitat Oberta de Catalunya (UOC)

---

## 📌 Resumen

Este proyecto desarrolla un sistema de inteligencia artificial aplicado a mercados financieros para generar señales de inversión sobre el **S&P 500**, combinando modelos de Machine Learning interpretables con técnicas de detección de regímenes de mercado.

El objetivo central es responder a una pregunta clave en ML aplicado a finanzas: **¿es posible mantener un buen rendimiento predictivo sin renunciar a la transparencia del modelo?** Para ello se comparan modelos "caja negra" de alto rendimiento (LightGBM) frente a modelos intrínsecamente interpretables (Explainable Boosting Machine), y se incorpora un filtro de régimen de mercado basado en Hidden Markov Models para robustecer las señales generadas.

El desarrollo sigue la metodología **CRISP-DM** con un enfoque iterativo tipo Agile, estructurado en tres bloques principales que se corresponden con la organización de este repositorio.

---

## 🧩 Estructura del proyecto

El proyecto se organiza en **tres bloques secuenciales**, cada uno reflejado en su propia carpeta de notebooks:

### 🔹 Bloque 1 — Preparación de datos (`notebooks/01_preparacion_datos`)
Ingesta y construcción del dataset base: datos diarios del S&P 500 (2004–2024) junto con más de 40 variables técnicas, de volatilidad, crédito, macroeconómicas (Reserva Federal), VIX, materias primas y comportamiento sectorial. Incluye:
- Construcción de etiquetas mediante **Triple Barrier Labeling** (López de Prado).
- Ingeniería de variables (tendencia, momento, volatilidad, estrés de mercado, ciclo económico, rotación sectorial).
- Eliminación de multicolinealidad (correlación + VIF).

### 🔹 Bloque 2 — Primera iteración: modelos supervisados interpretables (`notebooks/02_iteracion_1_ebm_lightgbm`)
Entrenamiento y comparación de tres aproximaciones con distinto nivel de interpretabilidad, bajo validación **Walk-Forward con ventana expandible** y optimización de hiperparámetros con **Optuna** (TPE/Bayesiana):
- **Explainable Boosting Machine (EBM)** — modelo principal, interpretable por diseño.
- **LightGBM** — referencia de alto rendimiento ("caja negra").
- **Modelo Student** (destilación) — árboles de decisión / RuleFit entrenados para imitar al LightGBM y extraer reglas de decisión comprensibles.

Evaluación mediante F1-Score, AUC-ROC y estabilidad temporal de la importancia de variables (valores SHAP).

### 🔹 Bloque 3 — Segunda iteración: filtro de régimen de mercado (`notebooks/03_iteracion_2_regimenes_hmm`)
Incorporación de un componente de aprendizaje **no supervisado** mediante **Hidden Markov Models (HMM)** para detectar regímenes de mercado (alcista/bajista) y filtrar las señales del Bloque 2, activándolas únicamente en condiciones de mercado favorables. Incluye el **backtesting fuera de muestra** (2020–2024, datos bloqueados durante el desarrollo) frente a Buy & Hold, con costes de transacción incorporados.

---

## 📁 Estructura del repositorio

```
tfm-repo/
├── data/
│   ├── raw/                     # Datos originales sin procesar
│   └── processed/               # Datos tras limpieza, features y labeling
├── notebooks/
│   ├── 01_preparacion_datos/            # Bloque de datos (bloque6)
│   ├── 02_iteracion_1_ebm_lightgbm/     # 1ª iteración: EBM / LightGBM / Student
│   └── 03_iteracion_2_regimenes_hmm/    # 2ª iteración: filtro HMM + backtesting
├── src/                          # Funciones y módulos reutilizables (.py / .r)
├── models/
│   ├── iteracion_1/              # Modelos entrenados (EBM, LightGBM, Student)
│   └── iteracion_2/              # Modelos HMM y pipeline combinado
├── reports/
│   ├── figures/                  # Gráficas (SHAP, curvas, importancia de variables)
│   └── backtesting/              # Resultados de backtesting y métricas
├── memoria/                       # Memoria del TFM (PDF/Word) y anexos
├── docs/                          # Documentación adicional
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Metodología técnica

| Aspecto | Técnica utilizada |
|---|---|
| Etiquetado | Triple Barrier Labeling (López de Prado) |
| Validación | Walk-Forward con ventana expandible |
| Optimización de hiperparámetros | Optuna (TPE / Bayesiana) |
| Modelos supervisados | EBM, LightGBM, Student (destilación) |
| Selección de variables | Correlación + VIF |
| Detección de régimen | Hidden Markov Models (HMM) |
| Interpretabilidad | SHAP values, funciones aditivas (EBM) |
| Evaluación | F1-Score, AUC-ROC, Sharpe Ratio, Calmar Ratio, Max Drawdown |
| Validación económica | Backtesting fuera de muestra 2020–2024 vs Buy & Hold |

---

## 📊 Resultados principales

- Los modelos generaron estrategias con **mejores métricas ajustadas al riesgo** que la inversión pasiva (Sharpe Ratio, Calmar Ratio y reducción de drawdown máximo superiores al Buy & Hold).
- El retorno acumulado quedó por debajo del benchmark, ya que el sistema permanecía en liquidez en algunos tramos alcistas — evidenciando el *trade-off* entre rentabilidad absoluta y control de riesgo.
- **EBM alcanzó un rendimiento muy cercano a LightGBM**, con alta consistencia en las variables más relevantes según SHAP, reforzando que las señales aprendidas tienen una base económica coherente y no responden a sobreajuste.

*(Resultados detallados, gráficas y tablas disponibles en `reports/` y en la memoria completa en `memoria/`.)*

---

## 📄 Memoria del TFM

La memoria completa del trabajo se encuentra en [`memoria/`](./memoria), incluyendo la fundamentación teórica, el desarrollo metodológico completo y las conclusiones.

---

## 🚀 Reproducibilidad

```bash
git clone https://github.com/<usuario>/<nombre-repo>.git
cd <nombre-repo>
pip install -r requirements.txt
```

Ejecutar los notebooks en orden: `01_preparacion_datos` → `02_iteracion_1_ebm_lightgbm` → `03_iteracion_2_regimenes_hmm`.

---

## 🛠️ Tecnologías

`Python` · `LightGBM` · `InterpretML (EBM)` · `Optuna` · `hmmlearn` · `SHAP` · `pandas` / `numpy` · `scikit-learn`

---

## ✍️ Autor

**Víctor Olivera Begué**
MSc Data Science — Universitat Oberta de Catalunya (UOC)

---

## 📜 Licencia

Este proyecto se distribuye con fines académicos. Ver [LICENSE](./LICENSE) para más detalles.
