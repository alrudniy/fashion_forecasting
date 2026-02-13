# Time Series Analysis - Univariate and Multivariate Models (V8)

## Overview

This script runs time series forecasting experiments comparing univariate and multivariate approaches using three model types across lags 1-5. V8 adds Google Trends as an additional exogenous variable.

## Changes from V7

| Feature | v7 | v8 |
|---------|----|----|
| Input folder | `./datasets/` | `./datasets_with_GT/` |
| Log file | `R_logging_uni_multi_v7.txt` | `R_logging_uni_multi_v8.txt` |
| Results file | `results_all_R_uni_multi_v7.txt` | `results_all_R_uni_multi_v8.txt` |
| Passes | 2 (Univariate, Twitter) | 4 (Univariate, Twitter, GT, Twitter+GT) |
| GT column | N/A | Dynamically extracted from filename |

## Four Passes

| Pass | Label | Variables | Description |
|------|-------|-----------|-------------|
| 1 | `Univariate Amazon` | Amazon Rank only | Baseline autoregressive models |
| 2 | `Multivariate Amzn+Twttr` | Amazon Rank + Twitter | Twitter as exogenous variable |
| 3 | `Multivariate Amzn+GT` | Amazon Rank + GT_{keyword} | Google Trends as exogenous variable |
| 4 | `Multivariate Amzn+Twttr+GT` | Amazon Rank + Twitter + GT_{keyword} | Both Twitter and GT as exogenous variables |

## Keyword Extraction

The script extracts the keyword from the filename to identify the correct GT column:

| Filename | Extracted Keyword | GT Column |
|----------|-------------------|-----------|
| `beige_0..1000.csv` | `beige` | `GT_beige` |
| `high-waist_300000+.csv` | `high-waist` | `GT_high-waist` |
| `crop-top_010001..20000.csv` | `crop-top` | `GT_crop-top` |
| `black_005001..10000.csv` | `black` | `GT_black` |

## Dependencies

```r
library(readr)
library(dplyr)
library(tsDyn)
library(forecast)
library(stringr)
```

**Note:** V8 adds `stringr` for keyword extraction from filenames.

## Method Summary

### Pass 1: Univariate (Amazon Rank only)

| Method | Function | Description |
|--------|----------|-------------|
| VECM | `VECM(amazon_with_lag, lag=k, r=1)` | Series + dynamically lagged self as pseudo-multivariate |
| LINEAR | `linear(amazon.train, m=3, d=lag)` | Autoregressive linear model (tsDyn) |
| NNETAR | `nnetar(amazon.train, p=lag, size=2, repeats=5)` | Neural network autoregression (forecast) |

### Pass 2: Multivariate with Twitter (Amazon Rank + Twitter Volume)

| Method | Function | Description |
|--------|----------|-------------|
| VECM | `VECM(ts_data_mv_twitter, lag=k, r=1)` | Multivariate cointegration model |
| VAR | `lineVar(ts_data_mv_twitter, lag=k)` | Vector autoregression |
| NNETAR | `nnetar(..., xreg=xreg_twitter_train)` | Neural network with Twitter as xreg |

### Pass 3: Multivariate with Google Trends (Amazon Rank + GT_{keyword})

| Method | Function | Description |
|--------|----------|-------------|
| VECM | `VECM(ts_data_mv_gt, lag=k, r=1)` | Multivariate cointegration model |
| VAR | `lineVar(ts_data_mv_gt, lag=k)` | Vector autoregression |
| NNETAR | `nnetar(..., xreg=xreg_gt_train)` | Neural network with GT as xreg |

### Pass 4: Multivariate with Twitter + GT (Amazon Rank + Twitter + GT_{keyword})

| Method | Function | Description |
|--------|----------|-------------|
| VECM | `VECM(ts_data_mv_both, lag=k, r=1)` | 3-variable cointegration model |
| VAR | `lineVar(ts_data_mv_both, lag=k)` | 3-variable vector autoregression |
| NNETAR | `nnetar(..., xreg=xreg_both_train)` | Neural network with Twitter + GT as xreg |

## Output Files

| File | Description |
|------|-------------|
| `./results/R_logging_uni_multi_v8.txt` | Processing log with warnings |
| `./results/results_all_R_uni_multi_v8.txt` | Accuracy metrics for all models |

## Results Format

Each line in the results file contains:

```
dataset_name    pass_label    method,lag=N    MAE    MSE    RMSE    MAPE    TheilU
```

Example:
```
beige_0..1000    Multivariate Amzn+GT    VECM, lag=1    8.84    28.21    22.89    2.17    7.14
```

## Error Handling

- **Missing GT column**: If `GT_{keyword}` column not found, Passes 3 & 4 are skipped with a warning logged
- **Insufficient data**: Files with fewer than 10 rows after filtering are skipped
- **Model errors**: Individual model failures are caught and logged as `ERROR` with the error message

## Accuracy Metrics

Results include the following metrics for each model:

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |
| Theil's U | Relative accuracy measure |

## Data Filtering

Before analysis, rows are filtered:
- Remove rows where `N_Tweets == 0`
- Remove rows where `GT_{keyword}` is `NA` (for Passes 3 & 4)

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
| V8 | Added Google Trends passes; reads from `datasets_with_GT/`; dynamic GT column extraction |

## Usage

```r
# Ensure datasets_with_GT folder exists with processed files
# Ensure results folder exists
dir.create("./results", showWarnings = FALSE)

# Run the script
source("R_timeSeries_lags_1_to_5_uni_multi_v8.R")
```

## Expected Output Count

For each dataset file with GT data:
- Pass 1: 15 results (3 methods × 5 lags)
- Pass 2: 15 results (3 methods × 5 lags)
- Pass 3: 15 results (3 methods × 5 lags)
- Pass 4: 15 results (3 methods × 5 lags)
- **Total per file: 60 results**

For 101 dataset files: **up to 6,060 results**

(Files without GT columns will have 30 results each)
