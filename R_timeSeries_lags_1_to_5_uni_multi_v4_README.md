# Time Series Analysis - Univariate and Multivariate Models (V4)

## Overview

This script runs time series forecasting experiments comparing univariate (Amazon Rank only) and multivariate (Amazon Rank + Twitter Volume) approaches using three model types across lags 1-5.

## Changes from V3

### Error Fixes

| Error | Symptom | Cause | Fix Applied |
|-------|---------|-------|-------------|
| Univariate VAR | `attempt to set 'colnames' on an object with less than two dimensions` | `lineVar()` requires multivariate input | Use `linear()` instead |
| Multivariate MLP | `subscript out of bounds` | Misaligned xreg indexing with concatenated train+test | Pass `xreg_train` to `mlp()` and `xreg_test` to `forecast()` separately |

### Fix Details

#### Fix 1: Univariate VAR → LINEAR

**Before (v3):**
```r
amazon_matrix <- as.matrix(amazon)
mod_var <- lineVar(amazon_matrix, lag = lag)
```

**After (v4):**
```r
if (lag == 1) {
  mod <- linear(amazon.train, m = 3)
} else {
  mod <- linear(amazon.train, m = 3, d = lag)
}
```

#### Fix 2: Multivariate MLP — Separate xreg

**Before (v3):**
```r
twitter_full <- as.matrix(c(twitter.train, twitter.test))
mod_mlp <- mlp(amazon.train, xreg = twitter_full, ...)
pred_mlp <- forecast(mod_mlp, h = hVal)
```

**After (v4):**
```r
xreg_train <- as.matrix(twitter.train)
xreg_test <- as.matrix(twitter.test)
mod_mlp <- mlp(amazon.train, xreg = xreg_train, ...)
pred_mlp <- forecast(mod_mlp, h = hVal, xreg = xreg_test)
```

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
| MLP | `mlp(..., xreg=xreg_train)` + `forecast(..., xreg=xreg_test)` | Neural network with Twitter as external regressor |

## Output Files

| File | Description |
|------|-------------|
| `./results/R_logging_uni_multi_v4.txt` | Processing log |
| `./results/results_all_R_uni_multi_v4.txt` | Accuracy metrics for all models |

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
- Multivariate: Uses separate `xreg_train` and `xreg_test` matrices
