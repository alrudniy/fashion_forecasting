# Time Series Analysis - Univariate and Multivariate Models (V9)

## Overview

This script runs time series forecasting experiments comparing univariate and multivariate approaches using three model types across lags 1-5. V9 restores the original 77 time series from the validated dataset while retaining V8's four-pass Google Trends design.

## Changes from V8

| Feature | v8 | v9 |
|---------|----|----|
| File list | 101 datasets | 77 datasets (original validated set) |
| Log file | `R_logging_uni_multi_v8.txt` | `R_logging_uni_multi_v9.txt` |
| Results file | `results_all_R_uni_multi_v8.txt` | `results_all_R_uni_multi_v9.txt` |
| Expected results | Up to 6,060 | Up to 4,620 (77 × 60) |

### Why 77 instead of 101?

The original study validated 77 multivariate time series (Figure 20 in the paper) that satisfy:
- ≥50 weekly Sales Rank data points
- ≥10 weeks with non-zero Twitter mentions
- ≥85% train / 8-week holdout protocol

The v8 expansion to 101 added 24 datasets comprising:
- **7 new keywords** not in the original study: burgundy, fade, high-waist, patch, polyester, satin, slip-on (21 files)
- **3 additional buckets** for existing keywords: lace-up (2 extra), nylon (1 extra), solid (2 extra) — but note one of the solid extras was actually lace-up

These 24 were excluded because:
1. They were never part of the original validated Twitter-only analysis
2. Several violate the ≥85% train requirement after GT+Twitter filtering (e.g., burgundy: 33 rows = 75.8% train; patch: 38 rows = 78.9%)
3. Short series risk underdetermined VECM estimation at higher lags (3-variable VECM lag=5 has ~16 parameters)
4. Consistency with the paper's stated experimental protocol

## Four Passes (unchanged from v8)

| Pass | Label | Variables | Description |
|------|-------|-----------|-------------|
| 1 | `Univariate Amazon` | Amazon Rank only | Baseline autoregressive models |
| 2 | `Multivariate Amzn+Twttr` | Amazon Rank + Twitter | Twitter as exogenous variable |
| 3 | `Multivariate Amzn+GT` | Amazon Rank + GT_{keyword} | Google Trends as exogenous variable |
| 4 | `Multivariate Amzn+Twttr+GT` | Amazon Rank + Twitter + GT_{keyword} | Both Twitter and GT as exogenous variables |

## Expected Output Count

For each dataset file with GT data:
- Pass 1: 15 results (3 methods × 5 lags)
- Pass 2: 15 results (3 methods × 5 lags)
- Pass 3: 15 results (3 methods × 5 lags)
- Pass 4: 15 results (3 methods × 5 lags)
- **Total per file: 60 results**

For 77 dataset files: **up to 4,620 results**

## The 77 Datasets

| N | Time Series | N | Time Series | N | Time Series |
|---|------------|---|------------|---|------------|
| 1 | beige_0..1000 | 27 | floral_1001..5000 | 53 | pink_300000+ |
| 2 | beige_300000+ | 28 | floral_5001..10000 | 54 | purple_0..1000 |
| 3 | black_0..1000 | 29 | floral_10001..20000 | 55 | purple_300000+ |
| 4 | black_1001..5000 | 30 | floral_300000+ | 56 | red_0..1000 |
| 5 | black_5001..10000 | 31 | green_10001..20000 | 57 | red_300000+ |
| 6 | black_10001..20000 | 32 | green_300000+ | 58 | retro_5001..10000 |
| 7 | black_20001..30000 | 33 | grey_10001..20000 | 59 | rose_0..1000 |
| 8 | black_30001..40000 | 34 | grey_20001..30000 | 60 | rose_300000+ |
| 9 | black_150001..200000 | 35 | grey_300000+ | 61 | solid_0..1000 |
| 10 | black_300000+ | 36 | lace-up_0..1000 | 62 | solid_5001..10000 |
| 11 | blue_0..1000 | 37 | lace-up_1001..5000 | 63 | solid_10001..20000 |
| 12 | blue_300000+ | 38 | long-sleeve_10001..20000 | 64 | solid_20001..30000 |
| 13 | bright_0..1000 | 39 | long-sleeve_20001..30000 | 65 | solid_300000+ |
| 14 | bright_1001..5000 | 40 | long-sleeve_300000+ | 66 | steel_0..1000 |
| 15 | bright_300000+ | 41 | navy_1001..5000 | 67 | tape_0..1000 |
| 16 | cotton_0..1000 | 42 | navy_300000+ | 68 | tape_1001..5000 |
| 17 | cotton_1001..5000 | 43 | nude_300000+ | 69 | tape_5001..10000 |
| 18 | cotton_5001..10000 | 44 | nylon_0..1000 | 70 | vintage_1001..5000 |
| 19 | cotton_10001..20000 | 45 | nylon_5001..10000 | 71 | vintage_5001..10000 |
| 20 | cotton_20001..30000 | 46 | nylon_10001..20000 | 72 | vintage_300000+ |
| 21 | cotton_30001..40000 | 47 | nylon_20001..30000 | 73 | white_0..1000 |
| 22 | cotton_150001..200000 | 48 | nylon_150001..200000 | 74 | white_1001..5000 |
| 23 | cotton_300000+ | 49 | nylon_300000+ | 75 | white_5001..10000 |
| 24 | crop-top_0..1000 | 50 | pink_5001..10000 | 76 | white_10001..20000 |
| 25 | crop-top_10001..20000 | 51 | pink_10001..20000 | 77 | white_20001..30000 |
| 26 | floral_0..1000 | 52 | pink_300000+ | | white_300000+ |

## Version History

| Version | Changes |
|---------|---------|
| V1 | Original script with LINEAR and NNET |
| V2 | Replaced LINEAR with `lineVar()`, NNET with `nnfor::mlp()`, added pseudo-multivariate VECM |
| V3 | Fixed univariate VECM, univariate VAR, multivariate MLP |
| V4 | Fixed univariate VAR, multivariate MLP |
| V5 | Fixed multivariate MLP xreg handling |
| V6 | Fixed multivariate MLP lagged xreg |
| V7 | Replaced `nnfor::mlp()` with `forecast::nnetar()` |
| V8 | Added Google Trends passes; reads from `datasets_with_GT/`; 101 datasets |
| V9 | Restored original 77 datasets; updated output filenames |

## Usage

```r
# Ensure datasets_with_GT folder exists with processed files
# Ensure results folder exists
dir.create("./results", showWarnings = FALSE)

# Run the script
source("R_timeSeries_lags_1_to_5_uni_multi_v9.R")
```
