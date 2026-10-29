# AirCast — Air Quality Trend Predictor

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-submission%20ready-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-21%20passed-success.svg)]()

AirCast is a machine learning air quality forecasting project developed for the **Singularity Research Lab AI Division**. The project predicts short-term fine particulate matter ($\text{PM}_{2.5}$) concentrations one hour ahead ($y_t = \text{PM}_{2.5, t+1}$) using historical air quality telemetry and meteorological observations from the Beijing Multi-Site Air Quality dataset.

---

## 1. Project Objective & Scope

- **Prediction Target**: Hourly $\text{PM}_{2.5}$ concentration at hour $t+1$ ($\mu\text{g}/\text{m}^3$), formulated as a supervised regression task given information available up to hour $t$.
- **Monitoring Station**: **Aotizhongxin** (Olympic Sports Center, Chaoyang District, Beijing), a central urban monitoring station.
- **Historical Temporal Span**: March 1, 2013, 00:00:00 to February 28, 2017, 23:00:00 (4 full meteorological years, 35,064 continuous hourly timesteps).
- **Primary Operational Benchmark**: 1-hour naive persistence ($\hat{y}_{t+1} = \text{PM}_{2.5, t}$), the standard operational baseline for continuous atmospheric telemetry.
- **Evaluation Criteria**: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE), evaluated under strict chronological out-of-sample protocol.

---

## 2. Dataset Source & Attribution

- **Dataset Name**: Beijing Multi-Site Air Quality Data
- **Host Repository**: [UCI Machine Learning Repository (Dataset ID: 501)](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)
- **Primary Scientific Citation**:
  > Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., & Chen, S. X. (2017). Cautionary tales on air-quality improvement in Beijing. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 473(2205), 20170457. [DOI: 10.1098/rspa.2017.0457](https://doi.org/10.1098/rspa.2017.0457)
- **Data Acquisition & Expected Layout**:
  Raw dataset files are stored locally under:
  ```
  data/raw/beijing+multi+site+air+quality+data/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228/
  ```
  This archive contains 12 individual station CSV files covering urban, suburban, and traffic stations across the Beijing municipal monitoring network. AirCast focuses initial development and deep diagnostic validation on the central urban station file: `PRSA_Data_Aotizhongxin_20130301-20170228.csv`.

---

## 3. Directory Structure & Organization

The repository is structured to maintain strict separation of concerns, data integrity, and reproducibility:

```
AirCast/
├── data/
│   ├── raw/                 # Immutable Beijing multi-site raw CSV telemetry (12 stations)
│   └── processed/           # Cleaned and feature-engineered datasets
│       ├── aotizhongxin_cleaned.csv  # Cleaned telemetry with quality audit flags
│       ├── train_features.csv        # Train split (2013-03-02 to 2016-02-29, 23,001 rows)
│       ├── val_features.csv          # Validation split (2016-03-01 to 2016-10-31, 5,275 rows)
│       └── test_features.csv         # Held-out test split (2016-11-01 to 2017-02-28, 2,736 rows)
├── models/                  # Serialized model artifacts and experiment metadata
│   ├── best_model.joblib             # Top-performing candidate (Random Forest Full)
│   ├── best_model_metadata.json      # Hyperparameters, features, and validation metrics
│   ├── persistence_baseline_metadata.json
│   └── ...                           # Other candidate model weights and metadata
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Executed Phase 2 exploratory data analysis
├── reports/                 # Scientific reports, metric exports, and figures
│   ├── figures/             # Diagnostic plots (actual vs. pred, residuals, diurnal, bins)
│   ├── final_test_evaluation.md      # Out-of-sample test results and error analysis
│   ├── model_comparison.md           # Validation model selection and comparison report
│   ├── submission_readiness_audit.md # Phase 6 reproducibility and Git hygiene audit
│   ├── validation_metrics.csv        # Machine-readable validation metrics
│   ├── validation_metrics.json
│   ├── test_evaluation_metrics.csv   # Machine-readable test evaluation metrics
│   └── test_evaluation_metrics.json
├── src/                     # Modular Python pipeline implementation
│   ├── __init__.py
│   ├── data_preprocessing.py # Schema verification, chronological alignment, conservative cleaning
│   ├── features.py           # Target construction, lag features, wind vectors, temporal splits
│   ├── train.py              # Persistence baseline, Linear Regression, Random Forest training
│   └── evaluate.py           # Out-of-sample test evaluation, residual diagnostics, figures
├── tests/                   # Automated regression test suite (21 unit tests)
│   ├── test_features.py     # Leakage prevention and feature alignment assertions
│   ├── test_train.py        # Baseline correctness and model reloadability tests
│   └── test_evaluate.py     # Evaluation metrics and residual calculation tests
├── .gitignore               # Strict exclusion of .venv, checkpoints, large caches, and joblib weights
└── requirements.txt         # Pinned virtual environment dependencies
```

---

## 4. Preprocessing & Missing Data Policy

Atmospheric telemetry often experiences sensor dropouts and maintenance blackouts. AirCast implements a strict, conservative data cleaning policy:

1. **Target Integrity Guarantee**:
   The forecasting target $\text{PM}_{2.5}$ is **never** synthetically imputed or forward-filled. Rows lacking an authentic ground-truth observation at $t+1$ are excluded from model training and evaluation.
2. **Sensor Outage Preservation**:
   A prominent 343-consecutive-hour sensor blackout (December 17–31, 2014) affecting all six co-pollutants is strictly preserved as `NaN` values rather than artificially interpolated, avoiding false atmospheric patterns.
3. **Conservative Short-Gap Imputation**:
   For input predictors only, missing value runs of $\le 2$ consecutive hours are forward-filled using past valid measurements. A binary indicator column (`is_imputed_short_gap`) is added to maintain full auditing transparency.
4. **Chronological Regularity**:
   All 35,064 expected hourly timesteps are validated for continuous monotonic ordering, zero timestamp duplicates, and zero timestamp gaps.

---

## 5. Feature Engineering & Chronological Splitting Strategy

### A. Chronological Splitting Protocol
To simulate true operational forecasting and eliminate lookahead data leakage, datasets are split strictly by target timestamp ($t+1$):
- **Training Set** (`2013-03-02 01:00` to `2016-02-29 23:00`, 36 months): **23,001 examples (74.2%)**. Used exclusively for model weight fitting and feature scaling.
- **Validation Set** (`2016-03-01 00:00` to `2016-10-31 23:00`, 8 months): **5,275 examples (17.0%)**. Used exclusively for hyperparameter tuning and model selection.
- **Test Set** (`2016-11-01 00:00` to `2017-02-28 23:00`, 4 peak winter months): **2,736 examples (8.8%)**. Kept completely untouched until final evaluation.

### B. Feature Group Hierarchy (38 Total Features)
Predictors are organized into three cumulative tiers:
1. **Autoregressive (15 features)**:
   - Current observation: $\text{PM}_{2.5}(t)$.
   - Historical lags: $\text{PM}_{2.5}$ at $t-1, t-2, t-3, t-6, t-12, t-24$.
   - Rolling window summary statistics: 6-hour and 24-hour mean, standard deviation, minimum, and maximum.
2. **Expanded (28 features)**:
   - Autoregressive group (15) +
   - Meteorological observations: temperature (`TEMP`), atmospheric pressure (`PRES`), dew point (`DEWP`), precipitation (`RAIN`), wind speed (`WSPM`), dew point depression, and 3-hour pressure tendency (7).
   - Cyclical calendar encodings: sine and cosine transformations of hour of day, day of year, and month (6).
3. **Full (38 features)**:
   - Expanded group (28) +
   - Co-pollutant telemetry: $\text{PM}_{10}$, $\text{SO}_2$, $\text{NO}_2$, $\text{CO}$, $\text{O}_3$ (5).
   - Wind direction and aerodynamic velocity vectors: unit azimuth vectors ($\sin \theta, \cos \theta$), meteorological wind velocity components ($u = -w \cdot \sin \theta$, $v = -w \cdot \cos \theta$), and wind direction missingness indicator (5).

---

## 6. Model Development & Validation Results

### Candidate Architectures
1. **Persistence Baseline**: Naive reference forecasting $\hat{y}_{t+1} = y_t$ (zero learned parameters).
2. **Linear Regression**: Ordinary Least Squares with `StandardScaler` fitted strictly on training data inside a `Pipeline`.
3. **Random Forest Regressor**: 100 decision trees, `max_depth=16`, `min_samples_split=8`, `min_samples_leaf=4`, `random_state=42`.

### Validation Performance Comparison ($N = 5{,}275$, March – October 2016)

| Model Architecture | Feature Group | Features | Training Time | Validation MAE ($\mu\text{g}/\text{m}^3$) | MAE vs. Baseline | Validation RMSE ($\mu\text{g}/\text{m}^3$) | RMSE vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | Autoregressive ($t$) | 1 | 0.00s | **8.8662** | *Baseline* | **15.2144** | *Baseline* |
| **Linear Regression** | Autoregressive | 15 | 0.01s | **8.7009** | +1.86% | **14.5652** | +4.27% |
| **Random Forest** | Autoregressive | 15 | 1.58s | **8.8240** | +0.47% | **14.7093** | +3.32% |
| **Linear Regression** | Expanded | 28 | 0.02s | **8.7372** | +1.45% | **14.4180** | +5.23% |
| **Random Forest** | Expanded | 28 | 2.67s | **8.5835** | +3.19% | **14.2214** | +6.53% |
| **Linear Regression** | Full | 38 | 0.02s | **8.9056** | -0.45% *(degraded)* | **14.4665** | +4.92% |
| **Random Forest (Selected)** | **Full** | **38** | **3.48s** | **8.4270** | **+4.95%** | **14.0050** | **+7.95%** |

### Rationale for Selecting Random Forest (Full)
- **Multicollinearity Sensitivity in Linear Models**: Atmospheric co-pollutants (`CO`, `PM10`, `NO2`) exhibit high mutual correlation (Spearman $\rho \approx 0.88$). In Linear Regression, multicollinearity increases coefficient variance and degraded MAE to $8.9056\,\mu\text{g}/\text{m}^3$ (performing worse than persistence).
- **Non-Linear Interactions**: Random Forest handles collinear features robustly via orthogonal tree splits and captures non-linear meteorological dynamics (e.g., wind transport thresholds where wind speeds above $2\,\text{m}/\text{s}$ rapidly clear accumulation).
- **Selection Decision**: `random_forest_full` achieved the lowest validation RMSE ($14.0050\,\mu\text{g}/\text{m}^3$, $+7.95\%$ improvement) and lowest MAE ($8.4270\,\mu\text{g}/\text{m}^3$, $+4.95\%$ improvement).

---

## 7. Final Out-of-Sample Test Evaluation

The selected model (`random_forest_full`) was evaluated once on the held-out winter test set ($N = 2,736$ contiguous hours, November 1, 2016 – February 28, 2017):

| Evaluation Set | Architecture | Inputs | Test MAE ($\mu\text{g}/\text{m}^3$) | MAE Improvement | Test RMSE ($\mu\text{g}/\text{m}^3$) | RMSE Improvement |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Held-Out Test Set** | **Persistence Baseline** | 1 | **13.3077** | *Baseline* | **25.3060** | *Baseline* |
| **Held-Out Test Set** | **Random Forest (Full)** | **38** | **12.5759** | **+5.50% (+0.73 μg/m³)** | **23.6652** | **+6.48% (+1.64 μg/m³)** |

### Error Analysis & Key Insights

1. **Seasonal Robustness**:
   Winter heating season exhibits more than double the particulate concentration and volatility compared to summer/autumn ($105.7\,\mu\text{g}/\text{m}^3$ test mean vs. $58.0\,\mu\text{g}/\text{m}^3$ annual median). Despite the elevated baseline MAE ($13.31$ vs. $8.87\,\mu\text{g}/\text{m}^3$), Random Forest achieved an even higher relative MAE improvement in winter (**+5.50%** vs. **+4.95%**), confirming robust generalizability across seasonal regimes.
2. **Performance Across Concentration Bins (HJ 633-2012 Standards)**:
   - Across **89.0% of all test conditions** ($0$ to $250\,\mu\text{g}/\text{m}^3$, covering Good through Heavily Polluted tiers under China's Technical Regulation on Ambient Air Quality Index HJ 633-2012), Random Forest **decisively outperforms persistence** by margins between **+6.5% and +12.9%**.
   - **Extreme-Pollution Limitation**: In the extreme severe pollution tier ($>250\,\mu\text{g}/\text{m}^3$, 11.0% of winter hours, 302 samples), Random Forest trails persistence (MAE $29.05$ vs. $27.03\,\mu\text{g}/\text{m}^3$, $-7.48\%$). Because decision tree ensembles predict using training leaf averages, they conservatively bound extreme peak predictions during unprecedented pollution episodes.
3. **Diurnal Pattern**:
   Random Forest outperforms persistence across 19 of the 24 daily hours. The largest outperformance occurs during evening traffic accumulation and nocturnal boundary layer collapse (19:00 to 23:00, **+7.1% to +17.3% MAE reduction**), where aerodynamic wind vector features provide vital early signals of stagnation.
4. **Exogenous Emission Spike (January 28, 2017)**:
   The maximum prediction residual ($+300.6\,\mu\text{g}/\text{m}^3$) occurred at 01:00 on January 28, 2017 (Chinese New Year's Eve midnight). This spike was driven by celebratory fireworks emissions—an unmodeled anthropogenic event exogenous to atmospheric weather dynamics.

---

## 8. Verification & Reproduction Guide

### Environment Setup
The project runs on Python 3.14 (or $\ge 3.10$). Ensure the virtual environment is active:

```bash
cd AirCast
source .venv/bin/activate
```

Verify dependencies and check for package conflicts:
```bash
python -m pip check
```

### Reproducibility from a Clean Clone

> [!IMPORTANT]
> **Data and Model Artifact Availability**:
> - Raw CSV telemetry (~39 MB) and binary serialized models (`models/*.joblib`, ~103 MB) are excluded from Git tracking to maintain a lightweight repository (<10 MB) and adhere to repository hygiene standards.
> - Consequently, running `python src/evaluate.py` immediately on a fresh clone is **not** possible until the dataset is acquired and the model is generated.
> - **Fast Training**: Model training is deterministic and computationally lightweight, requiring only **~3.5 seconds** to fit and serialize `best_model.joblib` locally.
> - **Release Asset Option**: For reviewers desiring instant zero-step evaluation, pre-trained `.joblib` model weights can be downloaded directly from the GitHub Release Assets and placed in `models/`.

### Complete Reproduction Pipeline (Step-by-Step)

```bash
# 1. Dataset Placement:
# Download Beijing Multi-Site Air Quality Data from the UCI ML Repository and place
# station CSVs into:
# data/raw/beijing+multi+site+air+quality+data/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228/

# 2. Clean raw telemetry with conservative missing-value policy (~1.5s)
python src/data_preprocessing.py

# 3. Build feature groups (autoregressive, weather, wind vectors) and chronological splits (~3.0s)
python src/features.py

# 4. Train models, evaluate validation split, and save models/best_model.joblib (~8.0s)
python src/train.py

# 5. Perform out-of-sample test evaluation and generate diagnostic figures (~4.0s)
python src/evaluate.py

# 6. Run the complete automated test suite (21 unit tests, <0.4s)
python -m unittest discover -s tests -p "test_*.py" -v
```

*Note: All scripts use dynamic project-relative path resolution and execute identically regardless of current working directory.*

---

## 9. Academic & Diagnostic Artifacts

### Evaluation Reports
- [reports/final_test_evaluation.md](reports/final_test_evaluation.md): Complete Phase 5 test evaluation, residual analysis, concentration breakdown, and diurnal diagnostics.
- [reports/model_comparison.md](reports/model_comparison.md): Complete Phase 4 validation comparison, multicollinearity analysis, and architecture selection rationale.
- [reports/submission_readiness_audit.md](reports/submission_readiness_audit.md): Phase 6 comprehensive reproducibility, environment, and Git hygiene audit.
- [notebooks/01_exploratory_data_analysis.ipynb](notebooks/01_exploratory_data_analysis.ipynb): Fully executed exploratory data analysis notebook.

### Generated Figures ([reports/figures/](reports/figures/))
- `test_actual_vs_predicted.png`: Scatter plot comparing observed vs. predicted $\text{PM}_{2.5}$ with 1:1 reference diagonal.
- `test_residual_distribution.png`: Residual histogram verifying zero-centered bias ($+1.01\,\mu\text{g}/\text{m}^3$) and near-zero median ($+0.01\,\mu\text{g}/\text{m}^3$).
- `test_residuals_over_time.png`: Chronological residual timeline across the 4-month test period with Chinese New Year annotation.
- `test_error_by_hour.png`: Diurnal MAE profile comparing Random Forest against persistence across 24 hours.
- `test_error_by_pm25_range.png`: Performance across 6 standardized HJ 633-2012 air quality concentration categories.
- `pm25_time_series.png`: 4-year historical time-series highlighting seasonal winter cycles and the December 2014 outage.
- `correlation_heatmap.png`: Spearman rank correlation heatmap among co-pollutants and meteorological factors.

---

## 10. Limitations & Scientific Disclaimers

1. **Single-Station Urban Scope**:
   This experiment is calibrated specifically on telemetry from the Aotizhongxin urban station. Model weights cannot be assumed to generalize directly to mountainous or suburban stations (e.g., Dingling, Huairou) without retraining or multi-station spatial modeling.
2. **Exogenous Emission Events**:
   Atmospheric regression models relying on local meteorology and co-pollutants cannot predict unannounced anthropogenic shocks, such as holiday fireworks or sudden industrial curtailment decrees.
3. **Upper Tail Saturation**:
   Decision tree regression models bound predictions to historical leaf averages, resulting in conservative underprediction during unprecedented, historic smoke episodes ($>250\,\mu\text{g}/\text{m}^3$).
4. **Predictive Association vs. Causality**:
   Strong feature contributions (e.g., for carbon monoxide `CO` or wind speed `WSPM`) represent statistical associations and physical advection correlations, not mechanistic causal atmospheric chemistry.
5. **No Claims of Pollution Prevention**:
   AirCast is a forecasting and diagnostic tool. It does not actively reduce air pollution, control industrial emissions, or provide clinical medical advice.
