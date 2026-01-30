# Time Series Analysis - Univariate and Multivariate Models (V4)
# Pass 1: All methods UNIVARIATE (Amazon Rank only)
# Pass 2: All methods MULTIVARIATE (Amazon Rank + Twitter Volume)
# Runs VECM, VAR/LINEAR, and MLP (nnfor) models for lags 1-5
#
# Changes from v3:
# - Fix 1: Univariate VAR uses linear() instead of lineVar() (lineVar requires multivariate)
# - Fix 2: Multivariate MLP passes xreg_train to mlp() and xreg_test to forecast() separately

library(readr)
library(dplyr)
library(tsDyn)
library(nnfor)
library(forecast)

# Helper function to calculate accuracy metrics manually for MLP forecasts
calc_accuracy <- function(pred, actual) {
  errors <- actual - pred
  mae <- mean(abs(errors))
  mse <- mean(errors^2)
  rmse <- sqrt(mse)
  mape <- mean(abs(errors / actual)) * 100
  # Theil's U
  n <- length(actual)
  num <- sqrt(mean(errors^2))
  denom <- sqrt(mean(actual^2)) + sqrt(mean(pred^2))
  theil_u <- num / denom
  
  return(c(MAE = mae, MSE = mse, RMSE = rmse, MAPE = mape, TheilU = theil_u))
}

processAllFiles <- function(fname) {
  fname <- paste0("./datasets/", fname)
  
  write(paste0("Starting ", fname), file = "./results/R_logging_uni_multi_v4.txt", append = TRUE, sep = "\t")
  
  input <- gsub("./datasets/", "", fname)
  input <- gsub(".csv", "", input)
  outfile <- "./results/results_all_R_uni_multi_v4.txt"
  
  # Read and prepare data
  df <- read_csv(fname, col_names = TRUE)
  df <- select(df, yearWeek, avgRnkWeek, N_Tweets)
  df <- df[df$N_Tweets != 0, ]  # remove rows where N_Tweets = 0
  
  # Create time series objects
  amazon <- ts(df[, c("avgRnkWeek")], frequency = 1)
  twitter <- ts(df[, c("N_Tweets")], frequency = 1)
  ts_data_mv <- ts(df[, c("avgRnkWeek", "N_Tweets")], frequency = 1)  # multivariate
  
  nrows <- nrow(df)
  split <- (nrows - 8)  # forecast for 8 weeks
  hVal <- 8
  
  # Train/test split
  amazon.train <- ts(amazon[1:split], frequency = 1)
  amazon.test <- amazon[(split + 1):nrows]
  
  twitter.train <- ts(twitter[1:split], frequency = 1)
  twitter.test <- twitter[(split + 1):nrows]
  
  # Multivariate train/test
  mvts.test <- ts(cbind(amazon.test, twitter.test))
  mvts.train <- ts(cbind(amazon.train, twitter.train))
  colnames(mvts.test) <- c("amazon", "twitter")
  colnames(mvts.train) <- c("amazon", "twitter")
  
  # xreg matrices for MLP (separate train and test)
  xreg_train <- as.matrix(twitter.train)
  xreg_test <- as.matrix(twitter.test)
  
  ##################################################################################
  # PASS 1: UNIVARIATE (Amazon Rank only)
  ##################################################################################
  
  # --------------------------------------------------------------------------
  # Univariate: VECM for lags 1-5 (using amazon + amazon_lag with dynamic offset)
  # --------------------------------------------------------------------------
  for (lag in 1:5) {
    tryCatch({
      # Create pseudo-multivariate with dynamic lag offset
      amazon_vec <- as.numeric(amazon)
      n <- length(amazon_vec)
      amazon_with_lag <- ts(cbind(
        amazon_vec[(lag + 1):n],        # y(t)
        amazon_vec[1:(n - lag)]         # y(t-lag)
      ), frequency = 1)
      colnames(amazon_with_lag) <- c("amazon", "amazon_lag")
      
      # Adjust hVal for shortened series
      hVal_adj <- min(hVal, nrow(amazon_with_lag) - lag - 2)
      
      mod_vec <- VECM(amazon_with_lag, lag = lag, r = 1)
      preds_roll <- predict_rolling(mod_vec, nroll = hVal_adj)
      acc <- accuracy_stat(object = preds_roll$pred, true = preds_roll$true)
      
      # Extract accuracy for first variable (amazon) - indices 1,4,7,10,13
      message <- paste(input, "Univariate Amazon", paste0("VECM, lag=", lag), 
                       paste(acc[c(1, 4, 7, 10, 13)], collapse = "\t"), sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    }, error = function(e) {
      message <- paste(input, "Univariate Amazon", paste0("VECM, lag=", lag), 
                       "ERROR", e$message, sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    })
  }
  
  # --------------------------------------------------------------------------
  # Univariate: LINEAR for lags 1-5
  # FIX 1: Use linear() instead of lineVar() - lineVar requires multivariate
  # --------------------------------------------------------------------------
  for (lag in 1:5) {
    tryCatch({
      if (lag == 1) {
        mod <- linear(amazon.train, m = 3)
      } else {
        mod <- linear(amazon.train, m = 3, d = lag)
      }
      preds_roll <- predict_rolling(mod, newdata = amazon.test)
      acc <- accuracy_stat(object = preds_roll$pred, true = preds_roll$true)
      
      message <- paste(input, "Univariate Amazon", paste0("LINEAR, lag=", lag), 
                       paste(acc[c(1:5)], collapse = "\t"), sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    }, error = function(e) {
      message <- paste(input, "Univariate Amazon", paste0("LINEAR, lag=", lag), 
                       "ERROR", e$message, sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    })
  }
  
  # --------------------------------------------------------------------------
  # Univariate: MLP (nnfor::mlp) for lags 1-5
  # No xreg for univariate - just the series itself
  # --------------------------------------------------------------------------
  for (lag in 1:5) {
    tryCatch({
      # mlp with specified lags
      mod_mlp <- mlp(amazon.train, lags = 1:lag, hd = 2, reps = 5)
      pred_mlp <- forecast(mod_mlp, h = hVal)
      acc <- calc_accuracy(as.numeric(pred_mlp$mean), as.numeric(amazon.test))
      
      message <- paste(input, "Univariate Amazon", paste0("MLP, lag=", lag), 
                       paste(acc, collapse = "\t"), sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    }, error = function(e) {
      message <- paste(input, "Univariate Amazon", paste0("MLP, lag=", lag), 
                       "ERROR", e$message, sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    })
  }
  
  ##################################################################################
  # PASS 2: MULTIVARIATE (Amazon Rank + Twitter Volume)
  ##################################################################################
  
  # --------------------------------------------------------------------------
  # Multivariate: VECM for lags 1-5
  # --------------------------------------------------------------------------
  for (lag in 1:5) {
    tryCatch({
      mod_vec <- VECM(ts_data_mv, lag = lag, r = 1)
      preds_roll <- predict_rolling(mod_vec, nroll = hVal)
      acc <- accuracy_stat(object = preds_roll$pred, true = preds_roll$true)
      
      # Extract accuracy for first variable (Amazon) - indices 1,4,7,10,13
      message <- paste(input, "Multivariate Amzn+Twttr", paste0("VECM, lag=", lag), 
                       paste(acc[c(1, 4, 7, 10, 13)], collapse = "\t"), sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    }, error = function(e) {
      message <- paste(input, "Multivariate Amzn+Twttr", paste0("VECM, lag=", lag), 
                       "ERROR", e$message, sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    })
  }
  
  # --------------------------------------------------------------------------
  # Multivariate: VAR (lineVar) for lags 1-5
  # --------------------------------------------------------------------------
  for (lag in 1:5) {
    tryCatch({
      mod_var <- lineVar(ts_data_mv, lag = lag)
      preds_roll <- predict_rolling(mod_var, nroll = hVal)
      acc <- accuracy_stat(object = preds_roll$pred, true = preds_roll$true)
      
      # Extract accuracy for first variable (Amazon) - indices 1,4,7,10,13
      message <- paste(input, "Multivariate Amzn+Twttr", paste0("VAR, lag=", lag), 
                       paste(acc[c(1, 4, 7, 10, 13)], collapse = "\t"), sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    }, error = function(e) {
      message <- paste(input, "Multivariate Amzn+Twttr", paste0("VAR, lag=", lag), 
                       "ERROR", e$message, sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    })
  }
  
  # --------------------------------------------------------------------------
  # Multivariate: MLP (nnfor::mlp) with xreg for lags 1-5
  # FIX 2: Pass xreg_train to mlp() and xreg_test to forecast() separately
  # --------------------------------------------------------------------------
  for (lag in 1:5) {
    tryCatch({
      # Pass xreg_train during fitting, xreg_test during forecasting
      mod_mlp <- mlp(amazon.train, lags = 1:lag, xreg = xreg_train, hd = 2, reps = 5)
      pred_mlp <- forecast(mod_mlp, h = hVal, xreg = xreg_test)
      acc <- calc_accuracy(as.numeric(pred_mlp$mean), as.numeric(amazon.test))
      
      message <- paste(input, "Multivariate Amzn+Twttr", paste0("MLP, lag=", lag), 
                       paste(acc, collapse = "\t"), sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    }, error = function(e) {
      message <- paste(input, "Multivariate Amzn+Twttr", paste0("MLP, lag=", lag), 
                       "ERROR", e$message, sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    })
  }
}

##################################################################################
# Process all files
##################################################################################

files <- c(
  'beige_0..1000.csv', 'beige_300000+.csv', 
  'black_0..1000.csv', 'black_001001..5000.csv', 'black_005001..10000.csv', 
  'black_010001..20000.csv', 'black_020001..30000.csv', 'black_030001..40000.csv', 
  'black_150001..200000.csv', 'black_300000+.csv',
  'blue_0..1000.csv', 'blue_300000+.csv', 
  'bright_0..1000.csv', 'bright_001001..5000.csv', 'bright_300000+.csv',
  'burgundy_001001..5000.csv',
  'cotton_0..1000.csv', 'cotton_001001..5000.csv', 'cotton_005001..10000.csv', 
  'cotton_010001..20000.csv', 'cotton_020001..30000.csv', 'cotton_030001..40000.csv', 
  'cotton_150001..200000.csv', 'cotton_300000+.csv', 
  'crop-top_0..1000.csv', 'crop-top_010001..20000.csv', 
  'fade_001001..5000.csv', 
  'floral_0..1000.csv', 'floral_001001..5000.csv', 'floral_005001..10000.csv', 
  'floral_010001..20000.csv', 'floral_300000+.csv', 
  'green_010001..20000.csv', 'green_300000+.csv', 
  'grey_010001..20000.csv', 'grey_020001..30000.csv', 'grey_300000+.csv', 
  'high-waist_001001..5000.csv', 'high-waist_005001..10000.csv', 
  'high-waist_010001..20000.csv', 'high-waist_300000+.csv',
  'lace-up_0..1000.csv', 'lace-up_001001..5000.csv', 'lace-up_005001..10000.csv', 
  'lace-up_300000+.csv', 
  'long-sleeve_010001..20000.csv', 'long-sleeve_020001..30000.csv', 'long-sleeve_300000+.csv', 
  'navy_001001..5000.csv', 'navy_300000+.csv', 
  'nude_300000+.csv', 
  'nylon_0..1000.csv', 'nylon_005001..10000.csv', 'nylon_010001..20000.csv', 
  'nylon_020001..30000.csv', 'nylon_150001..200000.csv', 'nylon_200001..250000.csv', 
  'nylon_300000+.csv', 
  'patch_300000+.csv', 
  'pink_005001..10000.csv', 'pink_010001..20000.csv', 'pink_300000+.csv', 
  'polyester_0..1000.csv', 'polyester_001001..5000.csv', 'polyester_005001..10000.csv', 
  'polyester_010001..20000.csv', 'polyester_020001..30000.csv', 'polyester_030001..40000.csv', 
  'polyester_150001..200000.csv', 'polyester_300000+.csv', 
  'purple_0..1000.csv', 'purple_300000+.csv', 
  'red_0..1000.csv', 'red_300000+.csv', 
  'retro_005001..10000.csv', 
  'rose_0..1000.csv', 'rose_300000+.csv', 
  'satin_0..1000.csv', 'satin_300000+.csv', 
  'slip-on_0..1000.csv', 'slip-on_300000+.csv', 
  'solid_0..1000.csv', 'solid_001001..5000.csv', 'solid_005001..10000.csv', 
  'solid_010001..20000.csv', 'solid_020001..30000.csv', 'solid_030001..40000.csv', 
  'solid_300000+.csv', 
  'steel_0..1000.csv', 
  'tape_0..1000.csv', 'tape_001001..5000.csv', 'tape_005001..10000.csv', 
  'vintage_001001..5000.csv', 'vintage_005001..10000.csv', 'vintage_300000+.csv', 
  'white_0..1000.csv', 'white_001001..5000.csv', 'white_005001..10000.csv', 
  'white_010001..20000.csv', 'white_020001..30000.csv', 'white_300000+.csv'
)

# Run analysis on all files
for (file in files) {
  processAllFiles(file)
}
