# AirCast — Phase 6: Reproducibility, Documentation & Submission Readiness Audit

**Project:** AirCast — Machine Learning Air Quality Trend Predictor<br>
**Author:** Singularity Research Lab AI Division<br>
**Target Horizon:** 1-Hour-Ahead PM2.5 Concentration ($y_t = \text{PM}_{2.5, t+1}$)<br>
**Monitoring Station:** Aotizhongxin (Olympic Sports Center), Beijing<br>
**Audit Date:** October 30, 2026<br>
**Audit Status:** PASSED — Submission Ready

---

## 1. Project Structure & Directory Manifest

The AirCast repository is organized into a modular, leak-free, reproducible machine learning system. Every directory serves a distinct, audited role:

| Directory | Purpose & Contents | Tracked in Git |
| :--- | :--- | :---: |
| `src/` | Modular Python pipeline modules: `data_preprocessing.py`, `features.py`, `train.py`, `evaluate.py`. | **Yes** |
| `tests/` | Standard library `unittest` suite (21 unit tests covering leakage, baseline, models, evaluation). | **Yes** |
| `notebooks/` | Fully executed exploratory data analysis notebook (`01_exploratory_data_analysis.ipynb`). | **Yes** |
| `reports/` | Peer-review academic markdown reports, machine-readable metrics (`.csv`, `.json`), and figure assets. | **Yes** |
| `reports/figures/` | High-resolution (300 DPI) publication diagnostic figures (EDA, residual analysis, diurnal, bins). | **Yes** |
| `models/` | Model metadata JSONs, hyperparameter logs, and `.gitkeep`. Heavy `.joblib` weights are excluded. | **JSONs Yes / Joblib No** |
| `data/raw/` | Original immutable Beijing Multi-Site Air Quality telemetry (12 station CSVs). Excluded from Git. | **No (`.gitkeep` only)** |
| `data/processed/` | Cleaned and feature-engineered datasets (`train_features.csv`, etc.). Excluded from Git. | **No (`.gitkeep` only)** |
| `.venv/` | Python virtual environment. Excluded from Git. | **No** |

---

## 2. Environment & Dependency Status

The execution environment was audited using the local virtual environment `.venv`:

### A. Python Interpreter Verification
```bash
$ .venv/bin/python --version
Python 3.14.8
```
- **Result**: Clean, modern Python 3.14 interpreter without deprecated syntax or warnings.

### B. Dependency Integrity Check
```bash
$ .venv/bin/python -m pip check
No broken requirements found.
```
- **Result**: All package dependencies are fully satisfied with zero missing packages, broken dependencies, or package conflicts.

### C. Core Scientific Stack Inventory (Pinned in `requirements.txt`)
- `numpy`: **2.3.4**
- `pandas`: **2.3.3**
- `scipy`: **1.16.2**
- `scikit-learn`: **1.7.2**
- `matplotlib`: **3.10.7**
- `seaborn`: **0.13.2**
- `joblib`: **1.5.2**
- Total installed dependencies: **113 packages** pinned strictly with exact versions (`==`).

---

## 3. Automated Test Suite Verification

The automated test suite is implemented using Python's standard library `unittest` module, requiring zero external test runner dependencies.

### Command Executed:
```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

### Verified Test Results:
```
Ran 21 tests in 0.355s
OK
```

### Breakdown by Test Module:

#### 1. `tests/test_features.py` (10 tests — Leakage & Feature Pipeline)
- `test_01_target_alignment`: Verifies that $y_t = \text{PM}_{2.5, t+1}$ across every row.
- `test_02_lag_features_temporal_correctness`: Verifies that autoregressive lags ($t-1, \dots, t-24$) contain strictly past measurements.
- `test_03_rolling_features_no_future_leakage`: Verifies that rolling window statistics (6h and 24h) include only closed historical intervals $[t-w+1, t]$.
- `test_04_feature_columns_exclude_forbidden_targets_and_indexes`: Asserts that `target_pm25`, `target_datetime`, and database indices are never included in feature matrices.
- `test_05_chronological_boundaries`: Asserts that training, validation, and test splits strictly follow their target timestamp bounds.
- `test_06_no_target_timestamp_overlap`: Verifies pairwise mutual exclusivity across splits (0 overlapping timestamps).
- `test_07_missing_targets_excluded_from_splits`: Asserts that rows lacking ground-truth target observations are excluded.
- `test_08_outage_not_synthetically_filled`: Asserts that the 343-hour pollutant sensor blackout in December 2014 remains strictly `NaN`.
- `test_09_eligibility_enforcement`: Asserts that rows with un-imputed missing input predictors are properly omitted from modeling.
- `test_10_pipeline_determinism`: Asserts that re-running feature generation produces byte-for-byte identical datasets.

#### 2. `tests/test_train.py` (5 tests — Model Training & Baselines)
- `test_01_persistence_baseline`: Asserts that $\hat{y}_{t+1} = \text{PM}_{2.5, t}$ matches baseline evaluation metrics.
- `test_02_persistence_missing_handling`: Asserts that missing inputs raise a clear `ValueError`.
- `test_03_saved_artifacts_exist`: Asserts that model metadata JSONs and artifacts are present in `models/`.
- `test_04_best_model_reload_and_predict`: Verifies that `joblib.load('models/best_model.joblib')` deserializes cleanly and generates predictions identical to in-memory training.
- `test_05_reports_exist`: Asserts that validation metrics and markdown reports exist on disk.

#### 3. `tests/test_evaluate.py` (6 tests — Test Evaluation & Residuals)
- `test_01_metric_calculation_on_known_samples`: Validates MAE and RMSE math against hand-calculated vectors.
- `test_02_fair_comparison_on_identical_rows`: Validates that Random Forest and Persistence are evaluated on identical test rows.
- `test_03_residual_definition`: Enforces the sign convention `residual = actual - predicted`.
- `test_04_feature_list_excludes_target_and_metadata`: Asserts that test evaluations receive only the 38 engineered feature columns.
- `test_05_output_dimensions_and_validity`: Validates output shape ($2,736$,), absence of nulls/infs, and physical plausibility.
- `test_06_saved_figures_and_reports_exist`: Asserts that all 5 diagnostic test figures and reports are saved and populated.

---

## 4. End-to-End Workflow & Path Portability

The entire AirCast pipeline was verified for end-to-end execution order, reproducibility, and working-directory independence.

### Verified Pipeline Execution Order:
```bash
# Step 1: Preprocessing & Data Cleaning
.venv/bin/python src/data_preprocessing.py

# Step 2: Supervised Target & Feature Engineering
.venv/bin/python src/features.py

# Step 3: Model Training & Validation Selection
.venv/bin/python src/train.py

# Step 4: Out-of-Sample Test Evaluation & Error Diagnostics
.venv/bin/python src/evaluate.py

# Step 5: Test Suite Verification
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

### Clean Clone Reproducibility & Release Assets:
- **Exclusion Rationale**: Raw CSV telemetry (~39MB) and binary `.joblib` model weights (~103MB) are excluded from Git to keep the repository under 10MB and adhere to academic archive hygiene.
- **Order of Execution on Fresh Clone**: A user cloning the repository cannot run `python src/evaluate.py` immediately without the data and model artifacts. They must first place the raw station telemetry in `data/raw/...` and run preprocessing, feature engineering, and training (`src/train.py` takes only ~3.5s to fit and save `best_model.joblib`).
- **Release Asset Recommendation**: For reviewers desiring instant zero-training evaluation, the pre-trained `best_model.joblib` binary should be attached as a downloadable GitHub Release Asset rather than bloating the git commit history.

### Path Independence Audit:
All four modules (`src/data_preprocessing.py`, `src/features.py`, `src/train.py`, `src/evaluate.py`) dynamically resolve the project root using `Path(__file__).resolve().parent.parent` and insert it into `sys.path`.
- **Verification Test**: Executing `.venv/bin/python /home/manas/Projects/AirCast/src/evaluate.py` from `/tmp` succeeded with zero path errors.
- **Randomness Control**: Random Forest models use `random_state=42` and `n_jobs=-1`. Linear Regression uses analytical Ordinary Least Squares (deterministic). Feature engineering uses deterministic chronological order.

---

## 5. Scientific Consistency & Documentation Audit

### A. Separation of Validation and Test Splits
The project maintains strict segregation across experimental splits:
- **Training Set** (`2013-03-02 01:00` to `2016-02-29 23:00`): **23,001 hours** (74.2%). Exclusively used for model fitting.
- **Validation Set** (`2016-03-01 00:00` to `2016-10-31 23:00`): **5,275 hours** (17.0%). Exclusively used for model comparison and feature group selection.
- **Held-Out Test Set** (`2016-11-01 00:00` to `2017-02-28 23:00`): **2,736 hours** (8.8%). Peak winter heating season, untouched until final evaluation.

### B. Metric Synchronization Audit
All reported metrics across `README.md`, `reports/model_comparison.md`, `reports/final_test_evaluation.md`, and CSV/JSON data files were cross-referenced:

#### 1. Validation Performance ($N = 5{,}275$, Non-Heating Season)
| Model | Feature Group | Features | Validation MAE ($\mu\text{g}/\text{m}^3$) | MAE Improvement | Validation RMSE ($\mu\text{g}/\text{m}^3$) | RMSE Improvement |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Persistence Baseline | Current $t$ | 1 | **8.8662** | Baseline | **15.2144** | Baseline |
| Linear Regression | Autoregressive | 15 | **8.7009** | +1.86% | **14.5652** | +4.27% |
| Random Forest | Autoregressive | 15 | **8.8240** | +0.47% | **14.7093** | +3.32% |
| Linear Regression | Expanded | 28 | **8.7372** | +1.45% | **14.4180** | +5.23% |
| Random Forest | Expanded | 28 | **8.5835** | +3.19% | **14.2214** | +6.53% |
| Linear Regression | Full | 38 | **8.9056** | -0.45% *(degraded)* | **14.4665** | +4.92% |
| **Random Forest (Selected)** | **Full** | **38** | **8.4270** | **+4.95%** | **14.0050** | **+7.95%** |

#### 2. Final Out-of-Sample Test Performance ($N = 2{,}736$, Peak Winter Heating Season)
| Model | Features | Test MAE ($\mu\text{g}/\text{m}^3$) | MAE Improvement | Test RMSE ($\mu\text{g}/\text{m}^3$) | RMSE Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 1 | **13.3077** | Baseline | **25.3060** | Baseline |
| **Random Forest (Full)** | **38** | **12.5759** | **+5.50% (+0.73 μg/m³)** | **23.6652** | **+6.48% (+1.64 μg/m³)** |

### C. Mathematical Consistency
- **Relative Improvement Formula**:
  $$\text{Improvement (\%)} = \frac{\text{Metric}_{\text{baseline}} - \text{Metric}_{\text{model}}}{\text{Metric}_{\text{baseline}}} \times 100\%$$
  Positive percentage denotes an error reduction (superior accuracy).
- **Residual Definition**:
  $$\text{residual}_t = y_{t, \text{actual}} - \hat{y}_{t, \text{predicted}}$$
  - Mean Residual (Bias): **$+1.0089\,\mu\text{g}/\text{m}^3$** (slight conservative underprediction).
  - Median Residual: **$+0.0097\,\mu\text{g}/\text{m}^3$** (unbiased central symmetry).
  - Residual Std Dev: **$23.6436\,\mu\text{g}/\text{m}^3$**.

### D. Scientific Explanations & Domain Hypotheses
1. **Multicollinearity in Linear Regression**:
   Co-pollutants (`CO`, `PM10`, `NO2`) have Spearman rank correlations up to $0.88$ with `PM2.5`. In Linear Regression, co-pollutant collinearity inflated parameter variance and degraded validation MAE from $8.7009$ to $8.9056\,\mu\text{g}/\text{m}^3$ (worse than persistence). In contrast, Random Forest leverages greedy variance reduction and non-linear interactions (e.g. wind velocity vectors $u, v$ indicating dispersion only above velocity thresholds), achieving the best score.
2. **Standardized Air Quality Breakpoints (HJ 633-2012)**:
   Air quality categories are strictly aligned with China's National Standard *Technical Regulation on Ambient Air Quality Index (HJ 633-2012)* 24h PM2.5 breakpoints:
   - $0 - 35\,\mu\text{g}/\text{m}^3$: Good / Excellent (38.9% of test set, RF MAE $5.27$ vs Persistence $5.85$, **+9.93%**)
   - $35 - 75\,\mu\text{g}/\text{m}^3$: Moderate (13.5% of test set, RF MAE $9.66$ vs Persistence $11.09$, **+12.91%**)
   - $75 - 115\,\mu\text{g}/\text{m}^3$: Lightly Polluted (11.5% of test set, RF MAE $12.87$ vs Persistence $14.38$, **+10.46%**)
   - $115 - 150\,\mu\text{g}/\text{m}^3$: Moderately Polluted (7.3% of test set, RF MAE $17.85$ vs Persistence $19.10$, **+6.53%**)
   - $150 - 250\,\mu\text{g}/\text{m}^3$: Heavily Polluted (17.8% of test set, RF MAE $18.20$ vs Persistence $19.73$, **+7.75%**)
   - $> 250\,\mu\text{g}/\text{m}^3$: Severely Polluted (11.0% of test set, RF MAE $29.05$ vs Persistence $27.03$, **-7.48% trails persistence**)
3. **Extreme Pollution Tail Limitation**:
   Across 89.0% of winter hours ($0$ to $250\,\mu\text{g}/\text{m}^3$), Random Forest decisively outperforms persistence. In the extreme severe pollution tail ($>250\,\mu\text{g}/\text{m}^3$), Random Forest underpredicts peak spikes due to decision tree ensemble leaf averaging, where persistence performs better.
4. **January 28, 2017 Outlier Event**:
   The maximum test residual ($+300.6\,\mu\text{g}/\text{m}^3$) occurred at 01:00 on January 28, 2017 (Chinese New Year's Eve). This is explicitly labeled as a domain-expert hypothesis: celebratory fireworks emissions during midnight Spring Festival festivities represent an unmodeled anthropogenic emission shock outside local meteorological dynamics.

---

## 6. Git Hygiene & Submission Packaging Audit

### A. Git Repository State
```bash
$ git status --short --branch
## No commits yet on main
```
- **Branch**: `main`
- **Remotes**: Zero remote URLs configured (`git remote -v` is empty).
- **History**: Initial local state ready for a deliberate, clean initial commit. No automatic commits or pushes have been executed.

### B. Ignore Rules Verification (`.gitignore`)
The audited `.gitignore` rigorously isolates large binaries and local environment artifacts:
- **Virtual Environment**: `.venv/`, `venv/` excluded.
- **Python Bytecode & Caches**: `__pycache__/`, `*.py[cod]`, `.ipynb_checkpoints/` excluded.
- **Secrets & Credentials**: `.env`, `.env.*` excluded.
- **Raw & Processed Data**: `data/raw/*`, `data/processed/*` excluded (39MB raw and 19MB processed data avoided; directory structure preserved via `.gitkeep`).
- **Heavy Model Binaries**: `models/*.joblib` excluded (**103MB** of serialized decision trees excluded).
- **Retained Metadata**: `!models/*.json` allows tracking lightweight hyperparameter and experiment records (~1KB each).

### C. Recommended Staged Files for Submission
The following clean manifest of 38 files represents the entire academic submission:

```
.gitignore
README.md
requirements.txt
src/__init__.py
src/data_preprocessing.py
src/features.py
src/train.py
src/evaluate.py
tests/.gitkeep
tests/test_features.py
tests/test_train.py
tests/test_evaluate.py
notebooks/01_exploratory_data_analysis.ipynb
models/.gitkeep
models/best_model_metadata.json
models/persistence_baseline_metadata.json
models/linear_regression_autoregressive_metadata.json
models/linear_regression_expanded_metadata.json
models/linear_regression_full_metadata.json
models/random_forest_autoregressive_metadata.json
models/random_forest_expanded_metadata.json
models/random_forest_full_metadata.json
reports/.gitkeep
reports/model_comparison.md
reports/final_test_evaluation.md
reports/submission_readiness_audit.md
reports/validation_metrics.csv
reports/validation_metrics.json
reports/test_evaluation_metrics.csv
reports/test_evaluation_metrics.json
reports/figures/correlation_heatmap.png
reports/figures/missingness_overview.png
reports/figures/pm25_distribution.png
reports/figures/pm25_time_series.png
reports/figures/seasonal_diurnal_patterns.png
reports/figures/test_actual_vs_predicted.png
reports/figures/test_error_by_hour.png
reports/figures/test_error_by_pm25_range.png
reports/figures/test_residual_distribution.png
reports/figures/test_residuals_over_time.png
data/raw/.gitkeep
data/processed/.gitkeep
```
- **Total Payload Size**: $< 10\,\text{MB}$ (fully compliant with GitHub and academic conference archive constraints).

### D. Files Recommended for Exclusion from Git
1. `data/raw/` (39MB): Raw Beijing air quality telemetry and unrelated files (must be acquired separately via UCI repository).
2. `data/processed/` (19MB): Cleaned CSVs and feature splits (reproducible deterministically via `src/data_preprocessing.py` and `src/features.py`).
3. `models/*.joblib` (103MB): Binary decision tree ensembles (reproducible in $<4$ seconds via `src/train.py`).
4. `.venv/` (Local environment directory).

---

## 7. Known Limitations & Caveats

1. **Single-Station Urban Representation**:
   Evaluated strictly on Aotizhongxin (central urban Beijing). Spatial transferability across complex topography (e.g. mountainous stations like Dingling or Huairou) requires multi-station spatial graph or spatio-temporal modeling.
2. **Exogenous Emission Events**:
   Models trained on meteorological and air telemetry cannot predict sudden anthropogenic shocks, such as holiday fireworks or emergency industrial production curtailments.
3. **Upper Tail Saturation**:
   Decision tree regressors bound predictions within historical leaf averages, leading to conservative underprediction during unprecedented, historic smoke episodes ($>250\,\mu\text{g}/\text{m}^3$).
4. **Predictive Association vs. Causality**:
   Strong predictive importance for co-pollutants and wind components represents physical advection correlation and collinear emission sources, not mechanistic atmospheric chemical pathways.
5. **No Claims of Pollution Mitigation**:
   AirCast is a forecasting and diagnostic predictive tool; it does not actively reduce air pollution, control industrial emissions, or provide clinical medical advice.

---

## 8. Academic Submission Readiness Checklist

- [x] **Project Objective Clear**: 1-hour-ahead PM2.5 forecasting ($y_t = \text{PM}_{2.5, t+1}$) explicitly specified.
- [x] **Data Attribution Complete**: UCI Machine Learning Repository and Zhang et al. (2017) citation documented.
- [x] **Leakage-Free Splitting**: Chronological train, validation, and test splits with zero future leakage.
- [x] **Conservative Data Policy**: Target never imputed; December 2014 outage preserved as NaN; input imputation transparently flagged.
- [x] **Model Comparison Documented**: Persistence, Linear Regression, and Random Forest compared across 3 feature groups on validation data.
- [x] **Model Selection Justified**: Random Forest (Full) chosen based on lowest validation RMSE and MAE, with clear multicollinearity analysis.
- [x] **Validation vs. Test Results Segregated**: Tables and reports clearly distinguish validation from out-of-sample test results.
- [x] **Honest Error Analysis**: Extreme tail limitation ($>250\,\mu\text{g}/\text{m}^3$) and fireworks outlier acknowledged and explained.
- [x] **Standards Cited**: Air quality concentration bins referenced to China's official HJ 633-2012 regulation.
- [x] **Residual Sign Convention**: Defined and tested as $\text{actual} - \text{predicted}$.
- [x] **Test Suite Passing**: 21/21 automated unit tests passing in 0.355 seconds.
- [x] **Dependencies Verified**: `pip check` clean, requirements pinned, Python 3.14 verified.
- [x] **Path Independence**: Scripts run cleanly from any current working directory.
- [x] **Figures Publication-Ready**: High-resolution 300 DPI figures with proper titles, axis labels, and units.
- [x] **Git Clean**: Zero secrets, `.gitignore` excludes 103MB binaries and raw data, ready for deliberate initial commit.
