# Time Series Analysis - Univariate and Multivariate Models (V2)

## Overview

This script runs time series forecasting experiments comparing univariate (Amazon Rank only) and multivariate (Amazon Rank + Twitter Volume) approaches using three model types across lags 1-5.

## Changes from V1

| Original (V1) | New (V2) | Reason |
|---------------|----------|--------|
| `linear()` | `lineVar()` | Proper VAR implementation that works for both univariate and multivariate |
| `nnetTs()` | `nnfor::mlp()` | Supports `xreg` parameter for external regressors in multivariate case |
| Univariate VECM failed | Pseudo-multivariate `[y(t), y(t-1)]` | Allows VECM to run on univariate data using `r=1` cointegrating rank |

## Dependencies

```r
library(readr)
library(dplyr)
library(tsDyn)
library(nnfor)    # NEW
library(forecast) # NEW
```

## Method Summary

### Pass 1: Univariate (Amazon Rank only)

| Method | Function | Description |
|--------|----------|-------------|
| VECM | `VECM(amazon_with_lag, lag=k, r=1)` | Series + its own lag as pseudo-multivariate |
| VAR | `lineVar(amazon, lag=k)` | Reduces to AR model for single series |
| MLP | `mlp(amazon.train, lags=1:k)` | Neural network without external regressors |

### Pass 2: Multivariate (Amazon Rank + Twitter Volume)

| Method | Function | Description |
|--------|----------|-------------|
| VECM | `VECM(ts_data_mv, lag=k, r=1)` | True multivariate cointegration model |
| VAR | `lineVar(ts_data_mv, lag=k)` | Vector autoregression on both series |
| MLP | `mlp(amazon.train, xreg=twitter.train)` | Neural network with Twitter as external regressor |

## Output Files

| File | Description |
|------|-------------|
| `./results/R_logging_uni_multi_v2.txt` | Processing log |
| `./results/results_all_R_uni_multi_v2.txt` | Accuracy metrics for all models |

## Accuracy Metrics

Results include the following metrics for each model:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Theil's U

## Notes

### Pseudo-Multivariate VECM for Univariate Data

VECM inherently requires multiple variables to test cointegration relationships. To enable VECM in the univariate pass, the script creates a pseudo-multivariate series:

```r
amazon_with_lag <- cbind(
  amazon[2:n],        # y(t)
  amazon[1:(n-1)]     # y(t-1)
)
VECM(amazon_with_lag, lag = k, r = 1)
```

This tests cointegration between the series and its own lag, which is trivially true for I(1) processes but allows methodological completeness in the experiment.

### MLP Configuration

- Hidden nodes: `hd = 2`
- Repetitions: `reps = 5`
- Lags: `1:k` for each lag value k in 1-5
