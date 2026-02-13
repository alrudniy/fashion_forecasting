# Time Series Analysis - Univariate and Multivariate Models (V9)
# Pass 1: All methods UNIVARIATE (Amazon Rank only)
# Pass 2: All methods MULTIVARIATE with Twitter (Amazon Rank + Twitter Volume)
# Pass 3: All methods MULTIVARIATE with Google Trends (Amazon Rank + GT_{keyword})
# Pass 4: All methods MULTIVARIATE with Twitter + GT (Amazon Rank + Twitter + GT_{keyword})
# Runs VECM, VAR/LINEAR, and NNETAR models for lags 1-5
#
# Changes from v8:
# - Reduced file list from 101 to original 77 time series (Figure 20)
#   to enforce >=85% train / 8-week holdout protocol consistently
# - Excluded 24 datasets: 7 new keywords (burgundy, fade, high-waist,
#   patch, polyester, satin, slip-on) + additional buckets for lace-up,
#   nylon, solid that were not in the original validated set
# - Updated output filenames to v9

library(readr)
library(dplyr)
library(tsDyn)
library(forecast)
library(stringr)

# Helper function to extract keyword from filename
# e.g., "beige_0..1000.csv" -> "beige", "high-waist_300000+.csv" -> "high-waist"
extract_keyword <- function(filename) {
  # Remove path and extension
  name <- basename(filename)
  name <- gsub(".csv", "", name)
  # Extract keyword (everything before underscore followed by digits)
  keyword <- str_extract(name, "^[a-zA-Z-]+")
  return(keyword)
}

# Helper function to calculate accuracy metrics manually for NNETAR forecasts
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
  # Read from datasets_with_GT subfolder
  fname <- paste0("./datasets_with_GT/", fname)
  
  write(paste0("Starting ", fname), file = "./results/R_logging_uni_multi_v9.txt", append = TRUE, sep = "\t")
  
  input <- gsub("./datasets_with_GT/", "", fname)
  input <- gsub(".csv", "", input)
  outfile <- "./results/results_all_R_uni_multi_v9.txt"
  
  # Extract keyword for GT column name
  keyword <- extract_keyword(fname)
  gt_col_name <- paste0("GT_", keyword)
  
  write(paste0("  Keyword: ", keyword, ", GT column: ", gt_col_name), 
        file = "./results/R_logging_uni_multi_v9.txt", append = TRUE, sep = "\t")
  
  # Read and prepare data
  df <- read_csv(fname, col_names = TRUE, show_col_types = FALSE)
  
  # Check if GT column exists
  has_gt <- gt_col_name %in% colnames(df)
  
  if (!has_gt) {
    write(paste0("  WARNING: GT column '", gt_col_name, "' not found. Skipping GT passes."), 
          file = "./results/R_logging_uni_multi_v9.txt", append = TRUE, sep = "\t")
  }
  
  # Select columns - include GT if available
  if (has_gt) {
    df <- select(df, yearWeek, avgRnkWeek, N_Tweets, all_of(gt_col_name))
    # Remove rows where GT is NA or N_Tweets is 0
    df <- df[!is.na(df[[gt_col_name]]) & df$N_Tweets != 0, ]
  } else {
    df <- select(df, yearWeek, avgRnkWeek, N_Tweets)
    df <- df[df$N_Tweets != 0, ]
  }
  
  # Create time series objects
  amazon <- ts(df[, c("avgRnkWeek")], frequency = 1)
  twitter <- ts(df[, c("N_Tweets")], frequency = 1)
  ts_data_mv_twitter <- ts(df[, c("avgRnkWeek", "N_Tweets")], frequency = 1)
  
  if (has_gt) {
    gtrends <- ts(df[[gt_col_name]], frequency = 1)
    ts_data_mv_gt <- ts(df[, c("avgRnkWeek", gt_col_name)], frequency = 1)
    ts_data_mv_both <- ts(df[, c("avgRnkWeek", "N_Tweets", gt_col_name)], frequency = 1)
  }
  
  nrows <- nrow(df)
  split <- (nrows - 8)  # forecast for 8 weeks
  hVal <- 8
  
  # Check if we have enough data
  if (split < 10) {
    write(paste0("  WARNING: Not enough data (", nrows, " rows). Skipping."), 
          file = "./results/R_logging_uni_multi_v9.txt", append = TRUE, sep = "\t")
    return()
  }
  
  # Train/test split
  amazon.train <- ts(amazon[1:split], frequency = 1)
  amazon.test <- amazon[(split + 1):nrows]
  
  twitter.train <- ts(twitter[1:split], frequency = 1)
  twitter.test <- twitter[(split + 1):nrows]
  
  if (has_gt) {
    gtrends.train <- ts(gtrends[1:split], frequency = 1)
    gtrends.test <- gtrends[(split + 1):nrows]
  }
  
  # Multivariate train/test for Twitter
  mvts_twitter.train <- ts(cbind(amazon.train, twitter.train))
  mvts_twitter.test <- ts(cbind(amazon.test, twitter.test))
  colnames(mvts_twitter.train) <- c("amazon", "twitter")
  colnames(mvts_twitter.test) <- c("amazon", "twitter")
  
  # xreg matrices for nnetar
  xreg_twitter_train <- as.matrix(twitter.train)
  xreg_twitter_test <- as.matrix(twitter.test)
  
  if (has_gt) {
    # Multivariate train/test for GT
    mvts_gt.train <- ts(cbind(amazon.train, gtrends.train))
    mvts_gt.test <- ts(cbind(amazon.test, gtrends.test))
    colnames(mvts_gt.train) <- c("amazon", "gtrends")
    colnames(mvts_gt.test) <- c("amazon", "gtrends")
    
    # Multivariate train/test for Twitter + GT
    mvts_both.train <- ts(cbind(amazon.train, twitter.train, gtrends.train))
    mvts_both.test <- ts(cbind(amazon.test, twitter.test, gtrends.test))
    colnames(mvts_both.train) <- c("amazon", "twitter", "gtrends")
    colnames(mvts_both.test) <- c("amazon", "twitter", "gtrends")
    
    # xreg matrices for GT
    xreg_gt_train <- as.matrix(gtrends.train)
    xreg_gt_test <- as.matrix(gtrends.test)
    
    # xreg matrices for Twitter + GT
    xreg_both_train <- cbind(as.matrix(twitter.train), as.matrix(gtrends.train))
    xreg_both_test <- cbind(as.matrix(twitter.test), as.matrix(gtrends.test))
  }
  
  ##################################################################################
  # PASS 1: UNIVARIATE (Amazon Rank only)
  ##################################################################################
  
  # --------------------------------------------------------------------------
  # Univariate: VECM for lags 1-5
  # --------------------------------------------------------------------------
  for (lag in 1:5) {
    tryCatch({
      amazon_vec <- as.numeric(amazon)
      n <- length(amazon_vec)
      amazon_with_lag <- ts(cbind(
        amazon_vec[(lag + 1):n],
        amazon_vec[1:(n - lag)]
      ), frequency = 1)
      colnames(amazon_with_lag) <- c("amazon", "amazon_lag")
      
      hVal_adj <- min(hVal, nrow(amazon_with_lag) - lag - 2)
      
      mod_vec <- VECM(amazon_with_lag, lag = lag, r = 1)
      preds_roll <- predict_rolling(mod_vec, nroll = hVal_adj)
      acc <- accuracy_stat(object = preds_roll$pred, true = preds_roll$true)
      
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
  # Univariate: NNETAR for lags 1-5
  # --------------------------------------------------------------------------
  for (lag in 1:5) {
    tryCatch({
      mod_nn <- nnetar(amazon.train, p = lag, size = 2, repeats = 5)
      pred_nn <- forecast(mod_nn, h = hVal)
      acc <- calc_accuracy(as.numeric(pred_nn$mean), as.numeric(amazon.test))
      
      message <- paste(input, "Univariate Amazon", paste0("NNETAR, lag=", lag), 
                       paste(acc, collapse = "\t"), sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    }, error = function(e) {
      message <- paste(input, "Univariate Amazon", paste0("NNETAR, lag=", lag), 
                       "ERROR", e$message, sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    })
  }
  
  ##################################################################################
  # PASS 2: MULTIVARIATE with Twitter (Amazon Rank + Twitter Volume)
  ##################################################################################
  
  # --------------------------------------------------------------------------
  # Multivariate Twitter: VECM for lags 1-5
  # --------------------------------------------------------------------------
  for (lag in 1:5) {
    tryCatch({
      mod_vec <- VECM(ts_data_mv_twitter, lag = lag, r = 1)
      preds_roll <- predict_rolling(mod_vec, nroll = hVal)
      acc <- accuracy_stat(object = preds_roll$pred, true = preds_roll$true)
      
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
  # Multivariate Twitter: VAR for lags 1-5
  # --------------------------------------------------------------------------
  for (lag in 1:5) {
    tryCatch({
      mod_var <- lineVar(ts_data_mv_twitter, lag = lag)
      preds_roll <- predict_rolling(mod_var, nroll = hVal)
      acc <- accuracy_stat(object = preds_roll$pred, true = preds_roll$true)
      
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
  # Multivariate Twitter: NNETAR with xreg for lags 1-5
  # --------------------------------------------------------------------------
  for (lag in 1:5) {
    tryCatch({
      mod_nn <- nnetar(amazon.train, p = lag, size = 2, repeats = 5, xreg = xreg_twitter_train)
      pred_nn <- forecast(mod_nn, h = hVal, xreg = xreg_twitter_test)
      acc <- calc_accuracy(as.numeric(pred_nn$mean), as.numeric(amazon.test))
      
      message <- paste(input, "Multivariate Amzn+Twttr", paste0("NNETAR, lag=", lag), 
                       paste(acc, collapse = "\t"), sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    }, error = function(e) {
      message <- paste(input, "Multivariate Amzn+Twttr", paste0("NNETAR, lag=", lag), 
                       "ERROR", e$message, sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    })
  }
  
  ##################################################################################
  # PASS 3: MULTIVARIATE with Google Trends (Amazon Rank + GT_{keyword})
  ##################################################################################
  
  if (has_gt) {
    
    # --------------------------------------------------------------------------
    # Multivariate GT: VECM for lags 1-5
    # --------------------------------------------------------------------------
    for (lag in 1:5) {
      tryCatch({
        mod_vec <- VECM(ts_data_mv_gt, lag = lag, r = 1)
        preds_roll <- predict_rolling(mod_vec, nroll = hVal)
        acc <- accuracy_stat(object = preds_roll$pred, true = preds_roll$true)
        
        message <- paste(input, "Multivariate Amzn+GT", paste0("VECM, lag=", lag), 
                         paste(acc[c(1, 4, 7, 10, 13)], collapse = "\t"), sep = "\t")
        write(message, file = outfile, append = TRUE, sep = "\t")
      }, error = function(e) {
        message <- paste(input, "Multivariate Amzn+GT", paste0("VECM, lag=", lag), 
                         "ERROR", e$message, sep = "\t")
        write(message, file = outfile, append = TRUE, sep = "\t")
      })
    }
    
    # --------------------------------------------------------------------------
    # Multivariate GT: VAR for lags 1-5
    # --------------------------------------------------------------------------
    for (lag in 1:5) {
      tryCatch({
        mod_var <- lineVar(ts_data_mv_gt, lag = lag)
        preds_roll <- predict_rolling(mod_var, nroll = hVal)
        acc <- accuracy_stat(object = preds_roll$pred, true = preds_roll$true)
        
        message <- paste(input, "Multivariate Amzn+GT", paste0("VAR, lag=", lag), 
                         paste(acc[c(1, 4, 7, 10, 13)], collapse = "\t"), sep = "\t")
        write(message, file = outfile, append = TRUE, sep = "\t")
      }, error = function(e) {
        message <- paste(input, "Multivariate Amzn+GT", paste0("VAR, lag=", lag), 
                         "ERROR", e$message, sep = "\t")
        write(message, file = outfile, append = TRUE, sep = "\t")
      })
    }
    
    # --------------------------------------------------------------------------
    # Multivariate GT: NNETAR with xreg for lags 1-5
    # --------------------------------------------------------------------------
    for (lag in 1:5) {
      tryCatch({
        mod_nn <- nnetar(amazon.train, p = lag, size = 2, repeats = 5, xreg = xreg_gt_train)
        pred_nn <- forecast(mod_nn, h = hVal, xreg = xreg_gt_test)
        acc <- calc_accuracy(as.numeric(pred_nn$mean), as.numeric(amazon.test))
        
        message <- paste(input, "Multivariate Amzn+GT", paste0("NNETAR, lag=", lag), 
                         paste(acc, collapse = "\t"), sep = "\t")
        write(message, file = outfile, append = TRUE, sep = "\t")
      }, error = function(e) {
        message <- paste(input, "Multivariate Amzn+GT", paste0("NNETAR, lag=", lag), 
                         "ERROR", e$message, sep = "\t")
        write(message, file = outfile, append = TRUE, sep = "\t")
      })
    }
    
    ##################################################################################
    # PASS 4: MULTIVARIATE with Twitter + GT (Amazon Rank + Twitter + GT_{keyword})
    ##################################################################################
    
    # --------------------------------------------------------------------------
    # Multivariate Twitter+GT: VECM for lags 1-5
    # --------------------------------------------------------------------------
    for (lag in 1:5) {
      tryCatch({
        mod_vec <- VECM(ts_data_mv_both, lag = lag, r = 1)
        preds_roll <- predict_rolling(mod_vec, nroll = hVal)
        acc <- accuracy_stat(object = preds_roll$pred, true = preds_roll$true)
        
        message <- paste(input, "Multivariate Amzn+Twttr+GT", paste0("VECM, lag=", lag), 
                         paste(acc[c(1, 4, 7, 10, 13)], collapse = "\t"), sep = "\t")
        write(message, file = outfile, append = TRUE, sep = "\t")
      }, error = function(e) {
        message <- paste(input, "Multivariate Amzn+Twttr+GT", paste0("VECM, lag=", lag), 
                         "ERROR", e$message, sep = "\t")
        write(message, file = outfile, append = TRUE, sep = "\t")
      })
    }
    
    # --------------------------------------------------------------------------
    # Multivariate Twitter+GT: VAR for lags 1-5
    # --------------------------------------------------------------------------
    for (lag in 1:5) {
      tryCatch({
        mod_var <- lineVar(ts_data_mv_both, lag = lag)
        preds_roll <- predict_rolling(mod_var, nroll = hVal)
        acc <- accuracy_stat(object = preds_roll$pred, true = preds_roll$true)
        
        message <- paste(input, "Multivariate Amzn+Twttr+GT", paste0("VAR, lag=", lag), 
                         paste(acc[c(1, 4, 7, 10, 13)], collapse = "\t"), sep = "\t")
        write(message, file = outfile, append = TRUE, sep = "\t")
      }, error = function(e) {
        message <- paste(input, "Multivariate Amzn+Twttr+GT", paste0("VAR, lag=", lag), 
                         "ERROR", e$message, sep = "\t")
        write(message, file = outfile, append = TRUE, sep = "\t")
      })
    }
    
    # --------------------------------------------------------------------------
    # Multivariate Twitter+GT: NNETAR with xreg for lags 1-5
    # --------------------------------------------------------------------------
    for (lag in 1:5) {
      tryCatch({
        mod_nn <- nnetar(amazon.train, p = lag, size = 2, repeats = 5, xreg = xreg_both_train)
        pred_nn <- forecast(mod_nn, h = hVal, xreg = xreg_both_test)
        acc <- calc_accuracy(as.numeric(pred_nn$mean), as.numeric(amazon.test))
        
        message <- paste(input, "Multivariate Amzn+Twttr+GT", paste0("NNETAR, lag=", lag), 
                         paste(acc, collapse = "\t"), sep = "\t")
        write(message, file = outfile, append = TRUE, sep = "\t")
      }, error = function(e) {
        message <- paste(input, "Multivariate Amzn+Twttr+GT", paste0("NNETAR, lag=", lag), 
                         "ERROR", e$message, sep = "\t")
        write(message, file = outfile, append = TRUE, sep = "\t")
      })
    }
    
  } # end if (has_gt)
}

##################################################################################
# Process all files
##################################################################################

files <- c(
  # Original 77 multivariate time series (Figure 20)
  # Excludes: burgundy, fade, high-waist, patch, polyester, satin, slip-on (new keywords)
  #           + additional buckets for lace-up, nylon, solid not in original set
  'beige_0..1000.csv', 'beige_300000+.csv', 
  'black_0..1000.csv', 'black_001001..5000.csv', 'black_005001..10000.csv', 
  'black_010001..20000.csv', 'black_020001..30000.csv', 'black_030001..40000.csv', 
  'black_150001..200000.csv', 'black_300000+.csv',
  'blue_0..1000.csv', 'blue_300000+.csv', 
  'bright_0..1000.csv', 'bright_001001..5000.csv', 'bright_300000+.csv',
  'cotton_0..1000.csv', 'cotton_001001..5000.csv', 'cotton_005001..10000.csv', 
  'cotton_010001..20000.csv', 'cotton_020001..30000.csv', 'cotton_030001..40000.csv', 
  'cotton_150001..200000.csv', 'cotton_300000+.csv', 
  'crop-top_0..1000.csv', 'crop-top_010001..20000.csv', 
  'floral_0..1000.csv', 'floral_001001..5000.csv', 'floral_005001..10000.csv', 
  'floral_010001..20000.csv', 'floral_300000+.csv', 
  'green_010001..20000.csv', 'green_300000+.csv', 
  'grey_010001..20000.csv', 'grey_020001..30000.csv', 'grey_300000+.csv', 
  'lace-up_0..1000.csv', 'lace-up_001001..5000.csv', 
  'long-sleeve_010001..20000.csv', 'long-sleeve_020001..30000.csv', 'long-sleeve_300000+.csv', 
  'navy_001001..5000.csv', 'navy_300000+.csv', 
  'nude_300000+.csv', 
  'nylon_0..1000.csv', 'nylon_005001..10000.csv', 'nylon_010001..20000.csv', 
  'nylon_020001..30000.csv', 'nylon_150001..200000.csv', 
  'nylon_300000+.csv', 
  'pink_005001..10000.csv', 'pink_010001..20000.csv', 'pink_300000+.csv', 
  'purple_0..1000.csv', 'purple_300000+.csv', 
  'red_0..1000.csv', 'red_300000+.csv', 
  'retro_005001..10000.csv', 
  'rose_0..1000.csv', 'rose_300000+.csv', 
  'solid_0..1000.csv', 'solid_005001..10000.csv', 
  'solid_010001..20000.csv', 'solid_020001..30000.csv', 
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
