# Time Series Analysis - Univariate and Multivariate Models
# Pass 1: All methods UNIVARIATE (Amazon Rank only)
# Pass 2: All methods MULTIVARIATE (Amazon Rank + Twitter Volume)
# Runs VECM, LINEAR, and NNET models for lags 1-5

library(readr)
library(dplyr)
library(tsDyn)

processAllFiles <- function(fname) {
  fname <- paste0("./datasets/", fname)
  
  write(paste0("Starting ", fname), file = "./results/R_logging.txt", append = TRUE, sep = "\t")
  
  input <- gsub("./datasets/", "", fname)
  input <- gsub(".csv", "", input)
  outfile <- "./results/results_all_R.txt"
  
  # Read and prepare data
  df <- read_csv(fname, col_names = TRUE)
  df <- select(df, yearWeek, avgRnkWeek, N_Tweets)
  df <- df[df$N_Tweets != 0, ]  # remove rows where N_Tweets = 0
  
  # Create time series objects
  amazon <- ts(df[, c("avgRnkWeek")], frequency = 1)
  twitter <- ts(df[, c("N_Tweets")], frequency = 1)
  ts_data_mv <- ts(df[, c("avgRnkWeek", "N_Tweets")], frequency = 1)  # multivariate
  ts_data_uv <- ts(df[, c("avgRnkWeek")], frequency = 1)              # univariate
  
  nrows <- nrow(df)
  split <- (nrows - 8)  # forecast for 8 weeks
  hVal <- 8
  
  # Train/test split
  amazon.train <- amazon[1:split]
  amazon.test <- amazon[(split + 1):nrows]
  
  twitter.train <- twitter[1:split]
  twitter.test <- twitter[(split + 1):nrows]
  
  # Multivariate train/test
  mvts.test <- ts(cbind(amazon.test, twitter.test))
  mvts.train <- ts(cbind(amazon.train, twitter.train))
  colnames(mvts.test) <- c("amazon", "twitter")
  colnames(mvts.train) <- c("amazon", "twitter")
  
  ##################################################################################
  # PASS 1: UNIVARIATE (Amazon Rank only)
  ##################################################################################
  
  # --------------------------------------------------------------------------
  # Univariate: VECM for lags 1-5 (single variable)
  # --------------------------------------------------------------------------
  for (lag in 1:5) {
    tryCatch({
      mod_vec <- VECM(ts_data_uv, lag = lag)
      preds_roll <- predict_rolling(mod_vec, nroll = hVal)
      acc <- accuracy_stat(object = preds_roll$pred, true = preds_roll$true)
      
      message <- paste(input, "Univariate Amazon", paste0("VECM, lag=", lag), 
                       paste(acc[c(1:5)], collapse = "\t"), sep = "\t")
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
  # Univariate: NNET for lags 1-5
  # --------------------------------------------------------------------------
  for (lag in 1:5) {
    tryCatch({
      if (lag == 1) {
        mod <- nnetTs(amazon.train, m = 2, size = 2)
      } else {
        mod <- nnetTs(amazon.train, m = 2, size = 2, d = lag)
      }
      preds_roll <- predict_rolling(mod, newdata = amazon.test)
      acc <- accuracy_stat(object = preds_roll$pred, true = preds_roll$true)
      
      message <- paste(input, "Univariate Amazon", paste0("NNET, lag=", lag), 
                       paste(acc[c(1:5)], collapse = "\t"), sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    }, error = function(e) {
      message <- paste(input, "Univariate Amazon", paste0("NNET, lag=", lag), 
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
      mod_vec <- VECM(ts_data_mv, lag = lag)
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
  # Multivariate: LINEAR (VAR-style) for lags 1-5
  # --------------------------------------------------------------------------
  for (lag in 1:5) {
    tryCatch({
      if (lag == 1) {
        mod <- linear(mvts.train, m = 3)
      } else {
        mod <- linear(mvts.train, m = 3, d = lag)
      }
      preds_roll <- predict_rolling(mod, newdata = mvts.test)
      acc <- accuracy_stat(object = preds_roll$pred, true = preds_roll$true)
      
      message <- paste(input, "Multivariate Amzn+Twttr", paste0("LINEAR, lag=", lag), 
                       paste(acc[c(1, 4, 7, 10, 13)], collapse = "\t"), sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    }, error = function(e) {
      message <- paste(input, "Multivariate Amzn+Twttr", paste0("LINEAR, lag=", lag), 
                       "ERROR", e$message, sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    })
  }
  
  # --------------------------------------------------------------------------
  # Multivariate: NNET for lags 1-5
  # --------------------------------------------------------------------------
  for (lag in 1:5) {
    tryCatch({
      if (lag == 1) {
        mod <- nnetTs(mvts.train, m = 2, size = 2)
      } else {
        mod <- nnetTs(mvts.train, m = 2, size = 2, d = lag)
      }
      preds_roll <- predict_rolling(mod, newdata = mvts.test)
      acc <- accuracy_stat(object = preds_roll$pred, true = preds_roll$true)
      
      message <- paste(input, "Multivariate Amzn+Twttr", paste0("NNET, lag=", lag), 
                       paste(acc[c(1, 4, 7, 10, 13)], collapse = "\t"), sep = "\t")
      write(message, file = outfile, append = TRUE, sep = "\t")
    }, error = function(e) {
      message <- paste(input, "Multivariate Amzn+Twttr", paste0("NNET, lag=", lag), 
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
