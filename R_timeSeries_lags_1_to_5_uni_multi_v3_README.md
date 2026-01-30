# Time Series Analysis - Univariate and Multivariate Models (V3)

## Overview

This script runs time series forecasting experiments comparing univariate (Amazon Rank only) and multivariate (Amazon Rank + Twitter Volume) approaches using three model types across lags 1-5.

## Changes from V2

### Error Fixes

| Error | Symptom | Cause | Fix Applied |
|-------|---------|-------|-------------|
| Univariate VECM | `NaN` for lags 2-5 | Collinearity in pseudo-multivariate with fixed lag offset | Dynamic lag offset matching model lag |
| Univariate VAR | `dim(X) must have a positive length` | `lineVar()` requires matrix input | Wrap series as matrix |
| Multivariate MLP | `Length of xreg must be longer than y + forecast horizon` | `xreg` too short for forecasting | Concatenate train+test xreg |

### Fix Details

#### Fix 1: Univariate VECM — Dynamic Lag Offset

**Before (v2):**
```r
# Fixed lag=1 offset for all model lags
amazon_with_lag <- cbind(amazon[2:n], amazon[1:(n-1)])
VECM(amazon_with_lag, lag = lag, r = 1)
```

**After (v3):**
```r
# Dynamic offset matching model lag
amazon_with_lag <- cbind(
  amazon[(lag + 1):n],    # y(t)
  amazon[1:(n - lag)]     # y(t-lag)
)
VECM(amazon_with_lag, lag = lag, r = 1)
```

#### Fix 2: Univariate VAR — Matrix Wrapper

**Before (v2):**
```r
mod_var <- lineVar(amazon, lag = lag)
```

**After (v3):**
```r
amazon_matrix <- as.matrix(amazon)
mod_var <- lineVar(amazon_matrix, lag = lag)
```

#### Fix 3: Multivariate MLP — Concatenated xreg

**Before (v2):**
```r
xreg_train <- as.matrix(twitter.train)
mod_mlp <- mlp(amazon.train, xreg = xreg_train, ...)
```

**After (v3):**
```r
twitter_full <- as.matrix(c(twitter.train, twitter.test))
mod_mlp <- mlp(amazon.train, xreg = twitter_full, ...)
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
| VAR | `lineVar(as.matrix(amazon), lag=k)` | Matrix-wrapped univariate (reduces to AR model) |
| MLP | `mlp(amazon.train, lags=1:k)` | Neural network without external regressors |

### Pass 2: Multivariate (Amazon Rank + Twitter Volume)

| Method | Function | Description |
|--------|----------|-------------|
| VECM | `VECM(ts_data_mv, lag=k, r=1)` | True multivariate cointegration model |
| VAR | `lineVar(ts_data_mv, lag=k)` | Vector autoregression on both series |
| MLP | `mlp(amazon.train, xreg=twitter_full)` | Neural network with full Twitter series as external regressor |

## Output Files

| File | Description |
|------|-------------|
| `./results/R_logging_uni_multi_v3.txt` | Processing log |
| `./results/results_all_R_uni_multi_v3.txt` | Accuracy metrics for all models |

## Accuracy Metrics

Results include the following metrics for each model:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Theil's U

## Notes

### Pseudo-Multivariate VECM for Univariate Data

VECM inherently requires multiple variables to test cointegration relationships. To enable VECM in the univariate pass, the script creates a pseudo-multivariate series with a **dynamic lag offset**:

```r
# For lag=k, create [y(t), y(t-k)] pairs
amazon_with_lag <- cbind(
  amazon[(lag + 1):n],    # y(t)
  amazon[1:(n - lag)]     # y(t-lag)
)
VECM(amazon_with_lag, lag = lag, r = 1)
```

This avoids the collinearity issues that occurred with a fixed lag=1 offset.

### MLP Configuration

- Hidden nodes: `hd = 2`
- Repetitions: `reps = 5`
- Lags: `1:k` for each lag value k in 1-5
- Multivariate: Uses full Twitter series (train + test) as `xreg`
