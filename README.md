# Fashion Sales Forecasting with Time Series Models

This script replicates the R-based analysis for forecasting Amazon product popularity (N_ASINs in each bucket) using Twitter mention intensity as an exogenous variable.

## Models

| Model | Type | Description |
|-------|------|-------------|
| **VECM** | Multivariate | Vector Error Correction Model (Amazon + Twitter) |
| **Linear AR** | Univariate | Autoregressive Model (Amazon only) |
| **NNET** | Univariate | Neural Network Autoregressive Model (Amazon only) |

## Parameters

- **Lags:** 1 to 5
- **Holdout:** 8 weeks (rolling forecast)

## Installation

```bash
pip install pandas numpy statsmodels scikit-learn
```

## Usage

### Process individual CSV files:
```bash
python fashion_forecasting.py
```

### Prepare datasets from master file first:
```bash
python fashion_forecasting.py --prepare-data master_file.csv
```

## Author

Converted from R code (`timeSeries_5_MULTI_*.Rmd`)
