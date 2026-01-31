# Time Series Analysis - Univariate and Multivariate Models (V7)

## Overview

This script runs time series forecasting experiments comparing univariate (Amazon Rank only) and multivariate (Amazon Rank + Twitter Volume) approaches using three model types across lags 1-5.

## Changes from V6

### Major Change

Replaced `nnfor::mlp()` with `forecast::nnetar()` for the neural network models. The `nnfor::mlp()` function had persistent issues with `xreg` handling that caused errors on various dataset sizes. The `nnetar()` function from the `forecast` package is more stable and properly supports external regressors.

### Function Replacement

| Parameter | nnfor::mlp() (v6) | forecast::nnetar() (v7) |
|-----------|-------------------|-------------------------|
| Lag specification | `lags = 1:lag` | `p = lag` |
| Hidden nodes | `hd = 2` | `size = 2` |
| Repetitions | `reps = 5` | `repeats = 5` |
| External regressors | `xreg` (problematic) | `xreg` (stable) |

### Code Changes

**Univariate:**
```r
# Before (v6)
mod_mlp <- mlp(amazon.train, lags = 1:lag, hd = 2, reps = 5)
pred_mlp <- forecast(mod_mlp, h = hVal)

# After (v7)
mod_nn <- nnetar(amazon.train, p = lag, size = 2, repeats = 5)
pred_nn <- forecast(mod_nn, h = hVal)
```

**Multivariate:**
```r
# Before (v6) - caused errors
mod_mlp <- mlp(amazon.train, lags = 1:lag, xreg = xreg_full, xreg.lags = 0, hd = 2, reps = 5)
pred_mlp <- forecast(mod_mlp, h = hVal)

# After (v7) - stable
mod_nn <- nnetar(amazon.train, p = lag, size = 2, repeats = 5, xreg = xreg_train)
pred_nn <- forecast(mod_nn, h = hVal, xreg = xreg_test)
```

## Dependencies

```r
library(readr)
library(dplyr)
library(tsDyn)
library(forecast)
```

**Note:** The `nnfor` library has been removed as it is no longer needed.

## Method Summary

### Pass 1: Univariate (Amazon Rank only)

| Method | Function | Description |
|--------|----------|-------------|
| VECM | `VECM(amazon_with_lag, lag=k, r=1)` | Series + dynamically lagged self as pseudo-multivariate |
| LINEAR | `linear(amazon.train, m=3, d=lag)` | Autoregressive linear model (tsDyn) |
| NNETAR | `nnetar(amazon.train, p=lag, size=2, repeats=5)` | Neural network autoregression (forecast) |

### Pass 2: Multivariate (Amazon Rank + Twitter Volume)

| Method | Function | Description |
|--------|----------|-------------|
| VECM | `VECM(ts_data_mv, lag=k, r=1)` | True multivariate cointegration model |
| VAR | `lineVar(ts_data_mv, lag=k)` | Vector autoregression on both series |
| NNETAR | `nnetar(..., xreg=xreg_train)` + `forecast(..., xreg=xreg_test)` | Neural network with Twitter as external regressor |

## Output Files

| File | Description |
|------|-------------|
| `./results/R_logging_uni_multi_v7.txt` | Processing log |
| `./results/results_all_R_uni_multi_v7.txt` | Accuracy metrics for all models |

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
| V6 | Fixed multivariate MLP (use `xreg.lags = 0` to disable lagged xreg features) |
| V7 | Replaced `nnfor::mlp()` with `forecast::nnetar()` for stable xreg support |

## Notes

### Why nnetar instead of mlp?

The `nnfor::mlp()` function had multiple issues with external regressors:
- `subscript out of bounds` errors on shorter time series
- `replacement has length zero` errors with `xreg.lags = 0`
- Inconsistent behavior across different dataset sizes

The `forecast::nnetar()` function:
- Is part of the well-maintained `forecast` package
- Has stable and reliable `xreg` support
- Uses standard train/test xreg pattern matching other forecasting functions

### NNETAR Configuration

- `p`: Number of lagged inputs (1-5 in this experiment)
- `size`: Number of hidden nodes (2)
- `repeats`: Number of networks to fit and average (5)

### Pseudo-Multivariate VECM for Univariate Data

VECM inherently requires multiple variables to test cointegration relationships. To enable VECM in the univariate pass, the script creates a pseudo-multivariate series with a dynamic lag offset:

```r
amazon_with_lag <- cbind(
  amazon[(lag + 1):n],    # y(t)
  amazon[1:(n - lag)]     # y(t-lag)
)
VECM(amazon_with_lag, lag = lag, r = 1)
```
