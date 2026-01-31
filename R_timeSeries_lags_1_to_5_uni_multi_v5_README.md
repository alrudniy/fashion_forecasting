# Time Series Analysis - Univariate and Multivariate Models (V5)

## Overview

This script runs time series forecasting experiments comparing univariate (Amazon Rank only) and multivariate (Amazon Rank + Twitter Volume) approaches using three model types across lags 1-5.

## Changes from V4

### Error Fix

| Error | Symptom | Cause | Fix Applied |
|-------|---------|-------|-------------|
| Multivariate MLP | `Length of xreg must be longer that y + forecast horizon` | `mlp()` requires xreg to cover both training and forecast periods during fitting | Pass full xreg (train+test) to `mlp()`, don't pass xreg to `forecast()` |

### Fix Details

#### Multivariate MLP — Full xreg During Fitting

**Before (v4):**
```r
xreg_train <- as.matrix(twitter.train)
xreg_test <- as.matrix(twitter.test)
mod_mlp <- mlp(amazon.train, xreg = xreg_train, ...)
pred_mlp <- forecast(mod_mlp, h = hVal, xreg = xreg_test)
```

**After (v5):**
```r
xreg_full <- as.matrix(c(as.numeric(twitter.train), as.numeric(twitter.test)))
mod_mlp <- mlp(amazon.train, xreg = xreg_full, ...)
pred_mlp <- forecast(mod_mlp, h = hVal)  # no xreg needed
```

The `nnfor::mlp()` function requires `xreg` length ≥ `length(y) + h` during model fitting. It uses the full xreg internally and `forecast()` automatically accesses the appropriate portion for the forecast horizon.

## Dependencies

```r
library(readr)
library(dplyr)
library(tsDyn)
library(nnfor)
library(forecast)
```

## Method Summary

### Pass 1: Univariate (Amazon Rank only)

| Method | Function | Description |
|--------|----------|-------------|
| VECM | `VECM(amazon_with_lag, lag=k, r=1)` | Series + dynamically lagged self as pseudo-multivariate |
| LINEAR | `linear(amazon.train, m=3, d=lag)` | Autoregressive linear model (tsDyn) |
| MLP | `mlp(amazon.train, lags=1:k)` | Neural network without external regressors |

### Pass 2: Multivariate (Amazon Rank + Twitter Volume)

| Method | Function | Description |
|--------|----------|-------------|
| VECM | `VECM(ts_data_mv, lag=k, r=1)` | True multivariate cointegration model |
| VAR | `lineVar(ts_data_mv, lag=k)` | Vector autoregression on both series |
| MLP | `mlp(amazon.train, xreg=xreg_full)` | Neural network with full Twitter series as xreg |

## Output Files

| File | Description |
|------|-------------|
| `./results/R_logging_uni_multi_v5.txt` | Processing log |
| `./results/results_all_R_uni_multi_v5.txt` | Accuracy metrics for all models |

## Accuracy Metrics

Results include the following metrics for each model:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Theil's U

## Version History

| Version | Changes |
|---------|---------|
| V1 | Original script with LINEAR and NNET |
| V2 | Replaced LINEAR with `lineVar()`, NNET with `nnfor::mlp()`, added pseudo-multivariate VECM |
| V3 | Fixed univariate VECM (dynamic lag offset), univariate VAR (matrix wrapper), multivariate MLP (concatenated xreg) |
| V4 | Fixed univariate VAR (`linear()` instead of `lineVar()`), multivariate MLP (separate xreg for train/forecast) |
| V5 | Fixed multivariate MLP (pass full xreg to `mlp()`, no xreg to `forecast()`) |

## Notes

### Pseudo-Multivariate VECM for Univariate Data

VECM inherently requires multiple variables to test cointegration relationships. To enable VECM in the univariate pass, the script creates a pseudo-multivariate series with a dynamic lag offset:

```r
amazon_with_lag <- cbind(
  amazon[(lag + 1):n],    # y(t)
  amazon[1:(n - lag)]     # y(t-lag)
)
VECM(amazon_with_lag, lag = lag, r = 1)
```

### MLP Configuration

- Hidden nodes: `hd = 2`
- Repetitions: `reps = 5`
- Lags: `1:k` for each lag value k in 1-5
- Multivariate: Full Twitter series (train + test) passed as `xreg` during fitting

### Key Insight for nnfor::mlp with xreg

The `mlp()` function from the `nnfor` package handles external regressors differently than some other forecasting functions:

1. **During fitting:** `xreg` must have length ≥ `length(y) + forecast_horizon`
2. **During forecasting:** No need to pass `xreg` again — it uses the already-provided data
3. The function internally manages which portion of `xreg` corresponds to training vs. forecasting
