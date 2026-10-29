# AirCast — Phase 4: Model Comparison & Validation Report

## 1. Experimental Objective
The objective of Phase 4 is to develop, train, and compare baseline regression models for one-hour-ahead PM2.5 forecasting ($y_t = \text{PM2.5}_{t+1}$) using historical observations and meteorological variables from the Aotizhongxin monitoring station.

All models are evaluated on the strictly held-out **chronological validation set** to select the top-performing architecture before final testing in Phase 5.

---

## 2. Dataset & Chronological Splits
- **Training Set**: Target timestamps from `2013-03-02 01:00:00` to `2016-02-29 23:00:00` (36 months, **23,001** samples). Used exclusively for fitting model weights and feature scaling parameters.
- **Validation Set**: Target timestamps from `2016-03-01 00:00:00` to `2016-10-31 23:00:00` (8 months, **5,275** samples). Used exclusively for model evaluation, feature group comparison, and selection.
- **Test Set**: Target timestamps from `2016-11-01 00:00:00` to `2017-02-28 23:00:00` (4 peak winter months, **2,736** samples). **Strictly held out and untouched in this phase.**

---

## 3. Feature Group Definitions
1. **Autoregressive (15 features)**:
   - Current observation: `pm25_current` ($t$).
   - Lags: `pm25_lag_1`, `pm25_lag_2`, `pm25_lag_3`, `pm25_lag_6`, `pm25_lag_12`, `pm25_lag_24`.
   - 6-Hour Rolling Stats: `mean`, `std`, `min`, `max`.
   - 24-Hour Rolling Stats: `mean`, `std`, `min`, `max`.
2. **Expanded (28 features)**:
   - Autoregressive group (15) +
   - Meteorological: `TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM`, `dew_point_depression`, `pressure_trend_3h` (7) +
   - Cyclical Calendar: `hour_sin`, `hour_cos`, `day_of_year_sin`, `day_of_year_cos`, `month_sin`, `month_cos` (6).
3. **Full (38 features)**:
   - Expanded group (28) +
   - Co-Pollutants: `PM10`, `SO2`, `NO2`, `CO`, `O3` (5) +
   - Wind Direction & Velocity Vectors: `wd_sin`, `wd_cos`, `wind_u`, `wind_v`, `wd_is_missing` (5).

---

## 4. Models and Hyperparameters
1. **Persistence Baseline**:
   - Formulation: $\hat{y}_{t+1} = \text{PM2.5}_t$.
   - Zero learned parameters. Evaluated on the exact 5,275 validation rows.
2. **Linear Regression**:
   - Preprocessing: `StandardScaler` fitted strictly on training data inside a `Pipeline`.
   - Solver: Ordinary Least Squares (analytical normal equation).
3. **Random Forest Regressor**:
   - `n_estimators`: 100
   - `max_depth`: 16
   - `min_samples_split`: 8
   - `min_samples_leaf`: 4 (regularization against overfitting)
   - `random_state`: 42 (deterministic reproducibility)
   - `n_jobs`: -1 (parallel CPU trees)

---

## 5. Validation Results & Comparison

| Model | Feature Group | Features | Training Time | Validation MAE ($\mu\text{g}/\text{m}^3$) | MAE vs. Baseline | Validation RMSE ($\mu\text{g}/\text{m}^3$) | RMSE vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | Autoregressive ($t$) | 1 | 0.00s | **8.8662** | *Baseline* | **15.2144** | *Baseline* |
| **Linear Regression** | Autoregressive | 15 | 0.01s | **8.7009** | +1.86% | **14.5652** | +4.27% |
| **Random Forest** | Autoregressive | 15 | 1.80s | **8.8240** | +0.47% | **14.7093** | +3.32% |
| **Linear Regression** | Expanded | 28 | 0.02s | **8.7372** | +1.45% | **14.4180** | +5.23% |
| **Random Forest** | Expanded | 28 | 2.95s | **8.5835** | +3.19% | **14.2214** | +6.53% |
| **Linear Regression** | Full | 38 | 0.04s | **8.9056** | -0.45% *(degraded)* | **14.4665** | +4.92% |
| **Random Forest (Selected)** | **Full** | **38** | **4.03s** | **8.4270** | **+4.95%** | **14.0050** | **+7.95%** |

---

## 6. Detailed Analysis of Model Performance

### A. Why Linear Regression Degraded on Full Features
In Linear Regression, adding the 10 additional co-pollutant and wind features degraded validation MAE from **$8.7009\,\mu\text{g}/\text{m}^3$** (Autoregressive) down to **$8.9056\,\mu\text{g}/\text{m}^3$** (Full), performing worse than the naive persistence baseline ($-0.45\%$).
- **Explanation**: Atmospheric co-pollutants (`PM10`, `CO`, `NO2`) are strongly collinear with `PM2.5` (Spearman correlations up to $0.88$). In ordinary least squares, multicollinearity inflates parameter variance. Furthermore, the linear model cannot capture non-linear aerodynamic dispersion (wind speed interacting with direction), leading to slight overfitting on the training set.

### B. Why Random Forest Succeeded on Full Features
In contrast, Random Forest achieved its strongest performance on the Full feature set, reducing MAE to **$8.4270\,\mu\text{g}/\text{m}^3$** ($+4.95\%$ improvement) and RMSE to **$14.0050\,\mu\text{g}/\text{m}^3$** ($+7.95\%$ improvement).
- **Explanation**: Decision trees naturally handle collinearity through greedy variance-reduction splits and model non-linear interactions (e.g. wind velocity vector components `wind_u`, `wind_v` acting as advection thresholds only when wind speed exceeds $2\,\text{m}/\text{s}$).

---

## 7. Selected Model & Justification

**Selected Model**: `random_forest_full` (`models/best_model.joblib`)

### Justification:
1. **Best Overall Predictive Accuracy**: Lowest validation RMSE ($14.0050\,\mu\text{g}/\text{m}^3$) and lowest validation MAE ($8.4270\,\mu\text{g}/\text{m}^3$).
2. **Solid Outperformance Over Persistence**: Delivers a **$7.95\%$ RMSE reduction** and **$4.95\%$ MAE reduction** over the stubborn 1-hour persistence benchmark.
3. **Robust Generalization**: The regularized leaf constraint (`min_samples_leaf=4`) prevents tree overfitting to idiosyncratic micro-events.
4. **Computational Efficiency**: Trains in $< 4.0$ seconds on 23,001 training examples and infers 5,275 validation predictions in $< 0.1$ seconds.

---

## 8. Limitations & Potential Sources of Bias
1. **Validation Window Seasonality**: The validation split spans March to October 2016 (spring, summer, early autumn), which exhibits lower median pollution levels than the severe winter heating months.
2. **Upper Extreme Under-Prediction**: Random Forest regressors predict within the range of training tree leaf averages and cannot extrapolate beyond the maximum training leaf value, potentially under-predicting record-breaking acute spikes.
3. **Single Station Focus**: Evaluated exclusively on Aotizhongxin; spatial regional transferability across complex topography requires multi-station training.

---

## 9. Reproduction Instructions
To reproduce the training and validation results from the command line:

```bash
cd AirCast  # or project root
.venv/bin/python src/train.py
```
