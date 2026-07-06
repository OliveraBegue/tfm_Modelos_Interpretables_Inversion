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
│   ├── raw/                                 # Datos originales sin procesar
│   └── processed/                           # Datos tras limpieza, features y labeling
├── notebooks/
│   ├── 01_preparacion_datos/
│   │   └── Bloque_0_v6.ipynb                # Ingesta, features, Triple Barrier Labeling
│   ├── 02_iteracion_1_ebm_lightgbm/
│   │   └── Iteracion_1_Final_TFM.ipynb       # EBM / LightGBM / Student / RuleFit
│   └── 03_iteracion_2_regimenes_hmm/
│       └── Iteracion_2_Final_TFM.ipynb       # Filtro de régimen HMM + backtesting
├── src/                                      # Funciones y módulos reutilizables
├── models/
│   ├── iteracion_1/                          # EBM, LightGBM, Student, RuleFit, DT (6 folds walk-forward)
│   └── iteracion_2/                          # Modelos HMM (6 folds)
├── reports/
│   ├── iteracion_1/                          # Métricas por fold, SHAP, importancia de variables, backtesting
│   └── iteracion_2/                          # Regímenes HMM, comparativa iter1 vs iter2, backtesting condicionado
├── memoria/                                   # Memoria del TFM (PDF/Word) y anexos
├── docs/                                      # Documentación adicional
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

**Validación (media walk-forward, 6 folds):**

| Modelo | F1 media | AUC media |
|---|---|---|
| EBM | 0.507 | 0.549 |
| LightGBM | 0.511 | 0.599 |
| RuleFit | 0.477 | 0.598 |

**Backtesting fuera de muestra (2020–2024) — comparativa por iteración:**

| Versión | Modelo | Sharpe | Max Drawdown | Calmar | Retorno total | Nº señales |
|---|---|---|---|---|---|---|
| — | Buy & Hold | 0.56 | -0.361 | 1.55 | 0.622 | — |
| Iteración 1 | EBM | 0.339 | -0.129 | 2.62 | 0.144 | 110 |
| Iteración 1 | LightGBM | 0.459 | -0.353 | 1.30 | 0.414 | 763 |
| Iteración 1 | Student | 0.441 | -0.337 | 1.31 | 0.362 | 664 |
| **Iteración 2 (+HMM)** | **EBM** | **0.614** | **-0.015** | **41.36** | 0.051 | 24 |
| Iteración 2 (+HMM) | LightGBM | 0.536 | -0.084 | 6.37 | 0.177 | 323 |
| Iteración 2 (+HMM) | Student | 0.450 | -0.093 | 4.82 | 0.138 | 297 |

**Lecturas clave:**
- El filtro de régimen (HMM) de la **2ª iteración mejora drásticamente el control de riesgo** en los tres modelos: el Max Drawdown de EBM pasa de -12.9% a -1.5%, y el Sharpe supera al del Buy & Hold (0.61 vs 0.56).
- El Calmar Ratio de EBM en la 2ª iteración (41.4) es un orden de magnitud superior al de la 1ª iteración y al benchmark, a costa de un número mucho menor de señales activas (24 vs 110) y por tanto de un retorno acumulado más contenido — el clásico *trade-off* riesgo/rentabilidad.
- **EBM mantiene un rendimiento predictivo muy cercano a LightGBM** (F1/AUC en validación) siendo un modelo intrínsecamente interpretable, y es precisamente el modelo que mejor aprovecha el filtro de régimen en el backtesting.

*(Métricas y gráficas completas —SHAP, curvas de equity, matrices de transición HMM, estabilidad de reglas, etc.— en `reports/iteracion_1/` y `reports/iteracion_2/`, y desarrollo completo en `memoria/`.)*

---

## 📄 Memoria del TFM

La memoria completa del trabajo se encuentra en [`memoria/`](./memoria), incluyendo la fundamentación teórica, el desarrollo metodológico completo y las conclusiones.

---

## 🚀 Reproducibilidad

```bash
git clone https://github.com/OliveraBegue/tfm_Modelos_Interpretables_Inversion.git
cd <tfm_Modelos_Interpretables_Inversion>
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
