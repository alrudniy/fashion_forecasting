"""
Fashion Sales Forecasting with Time Series Models

# INSTALL DEPENDENCIES
# pip install pandas numpy statsmodels scikit-learn 

This script replicates the R-based analysis for forecasting Amazon product 
popularity (N_ASINs in each bucket) using Twitter mention intensity as an 
exogenous variable.

Models:
- VECM (Vector Error Correction Model) - Multivariate (Amazon + Twitter)
- Linear AR (Autoregressive) - Univariate (Amazon only)
- NNET (Neural Network Autoregressive) - Univariate (Amazon only)

Lags: 1 to 5
Holdout: 8 weeks (rolling forecast)

Usage:
    # Process individual CSV files:
    python fashion_forecasting.py
    
    # Or prepare datasets from master file first:
    python fashion_forecasting.py --prepare-data master_file.csv

Author: Converted from R code (timeSeries_5_MULTI_*.Rmd)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional

# Statistical/ML libraries
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.ar_model import AutoReg
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'datasets_dir': './datasets',
    'results_dir': './results',
    'holdout_size': 8,  # 8 weeks for testing
    'lags': [1, 2, 3, 4, 5],
    'ar_order': 3,  # m=3 in R code
    'nnet_hidden_size': 2,  # size=2 in R code
    'nnet_input_lags': 2,  # m=2 in R code
    'min_observations': 50,  # Minimum data points required
    'min_nonzero_tweets': 10,  # Minimum weeks with tweets > 0
    # Target variable: 'avgRnkWeek' for sales rank, 'N_ASINs' for product count
    'target_variable': 'N_ASINs',
}

# List of dataset files to process
FILES = [
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
]


# =============================================================================
# Accuracy Metrics (matching R's accuracy_stat)
# =============================================================================

def calculate_accuracy_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate accuracy metrics matching R's accuracy_stat function.
    
    Returns: ME, RMSE, MAE, MPE, MAPE
    """
    errors = y_true - y_pred
    
    # Mean Error (ME)
    me = np.mean(errors)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(errors))
    
    # Mean Percentage Error (MPE)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_errors = 100 * errors / y_true
        pct_errors = np.where(np.isfinite(pct_errors), pct_errors, 0)
    mpe = np.mean(pct_errors)
    
    # Mean Absolute Percentage Error (MAPE)
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_pct_errors = 100 * np.abs(errors) / np.abs(y_true)
        abs_pct_errors = np.where(np.isfinite(abs_pct_errors), abs_pct_errors, 0)
    mape = np.mean(abs_pct_errors)
    
    return {
        'ME': me,
        'RMSE': rmse,
        'MAE': mae,
        'MPE': mpe,
        'MAPE': mape
    }


# =============================================================================
# Rolling Forecast Functions
# =============================================================================

def rolling_forecast_vecm(data: pd.DataFrame, lag: int, n_roll: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform rolling one-step-ahead forecasts using VECM.
    
    Args:
        data: DataFrame with columns ['target', 'twitter']
        lag: Number of lags for VECM
        n_roll: Number of rolling forecasts
        
    Returns:
        predictions: Array of predictions for target variable
        actuals: Array of actual values
    """
    values = data[['target', 'twitter']].values
    n = len(values)
    
    predictions = []
    actuals = []
    
    for i in range(n_roll):
        # Training data up to current point
        train_end = n - n_roll + i
        train_data = values[:train_end]
        
        try:
            # Fit VECM model
            model = VECM(train_data, k_ar_diff=lag, coint_rank=1)
            fitted = model.fit()
            
            # One-step ahead forecast
            forecast = fitted.predict(steps=1)
            pred_value = forecast[0, 0]  # First variable (target)
            
        except Exception as e:
            # If VECM fails, use last value as prediction
            pred_value = train_data[-1, 0]
        
        predictions.append(pred_value)
        actuals.append(values[train_end, 0])
    
    return np.array(predictions), np.array(actuals)


def rolling_forecast_linear_ar(train: np.ndarray, test: np.ndarray, 
                                ar_order: int, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform rolling one-step-ahead forecasts using Linear AR model.
    
    Args:
        train: Training data array
        test: Test data array
        ar_order: AR order (m in R code)
        lag: Forecast delay (d in R code)
        
    Returns:
        predictions: Array of predictions
        actuals: Array of actual values
    """
    predictions = []
    actuals = []
    
    # Combine train and test for rolling
    full_data = np.concatenate([train, test])
    train_size = len(train)
    
    for i in range(len(test)):
        # Expanding window training
        current_train = full_data[:train_size + i]
        
        try:
            # Fit AR model
            model = AutoReg(current_train, lags=ar_order, old_names=False)
            fitted = model.fit()
            
            # Forecast 'lag' steps ahead, take the last one
            forecast = fitted.predict(start=len(current_train), end=len(current_train) + lag - 1)
            pred_value = forecast[-1] if len(forecast) > 0 else current_train[-1]
            
        except Exception as e:
            # Fallback to last value
            pred_value = current_train[-1]
        
        predictions.append(pred_value)
        actuals.append(test[i])
    
    return np.array(predictions), np.array(actuals)


def create_lagged_features(data: np.ndarray, n_lags: int, forecast_lag: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lagged features for neural network input.
    
    Args:
        data: Time series data
        n_lags: Number of lag features to create
        forecast_lag: Steps ahead to forecast
        
    Returns:
        X: Feature matrix
        y: Target vector
    """
    X, y = [], []
    
    for i in range(n_lags, len(data) - forecast_lag + 1):
        X.append(data[i-n_lags:i])
        y.append(data[i + forecast_lag - 1])
    
    return np.array(X), np.array(y)


def rolling_forecast_nnet(train: np.ndarray, test: np.ndarray, 
                          n_lags: int, hidden_size: int, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform rolling one-step-ahead forecasts using Neural Network.
    
    Args:
        train: Training data array
        test: Test data array
        n_lags: Number of input lags (m in R code)
        hidden_size: Hidden layer size
        lag: Forecast delay (d in R code)
        
    Returns:
        predictions: Array of predictions
        actuals: Array of actual values
    """
    predictions = []
    actuals = []
    
    # Combine for rolling
    full_data = np.concatenate([train, test])
    train_size = len(train)
    
    for i in range(len(test)):
        current_train = full_data[:train_size + i]
        
        try:
            # Create lagged features
            X_train, y_train = create_lagged_features(current_train, n_lags, lag)
            
            if len(X_train) < 5:  # Need minimum samples
                pred_value = current_train[-1]
            else:
                # Scale features
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                
                X_scaled = scaler_X.fit_transform(X_train)
                y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
                
                # Fit neural network
                model = MLPRegressor(
                    hidden_layer_sizes=(hidden_size,),
                    activation='logistic',  # Sigmoid, matching R's default
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=42
                )
                model.fit(X_scaled, y_scaled)
                
                # Prepare input for prediction
                X_pred = current_train[-n_lags:].reshape(1, -1)
                X_pred_scaled = scaler_X.transform(X_pred)
                
                # Predict
                y_pred_scaled = model.predict(X_pred_scaled)
                pred_value = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
                
        except Exception as e:
            pred_value = current_train[-1]
        
        predictions.append(pred_value)
        actuals.append(test[i])
    
    return np.array(predictions), np.array(actuals)


# =============================================================================
# Data Preparation Functions
# =============================================================================

def prepare_datasets_from_master(master_filepath: str, output_dir: str, 
                                  min_obs: int = 50, min_tweets: int = 10) -> List[str]:
    """
    Prepare individual feature×bucket CSV files from the master dataset.
    
    Args:
        master_filepath: Path to TwitterAmazonTimeSeries_Buckets.csv
        output_dir: Directory to save individual CSV files
        min_obs: Minimum number of observations required
        min_tweets: Minimum weeks with non-zero tweets
        
    Returns:
        List of created file paths
    """
    print(f"Loading master dataset: {master_filepath}")
    df = pd.read_csv(master_filepath)
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    created_files = []
    
    # Get unique feature×bucket combinations
    combinations = df.groupby(['keywords', 'rankBucket']).size().reset_index(name='count')
    
    print(f"Found {len(combinations)} unique feature×bucket combinations")
    
    for _, row in combinations.iterrows():
        keyword = row['keywords']
        bucket = row['rankBucket']
        
        # Filter data for this combination
        subset = df[(df['keywords'] == keyword) & (df['rankBucket'] == bucket)].copy()
        subset = subset.sort_values('yearWeek')
        
        # Check minimum requirements
        if len(subset) < min_obs:
            continue
        
        nonzero_tweets = (subset['N_Tweets'] > 0).sum()
        if nonzero_tweets < min_tweets:
            continue
        
        # Create filename (sanitize for filesystem)
        safe_keyword = keyword.replace(' ', '-').replace('/', '-')
        safe_bucket = bucket.replace('..', '..').replace('+', '+')
        filename = f"{safe_keyword}_{safe_bucket}.csv"
        filepath = Path(output_dir) / filename
        
        # Save to CSV
        subset.to_csv(filepath, index=False)
        created_files.append(str(filepath))
    
    print(f"Created {len(created_files)} dataset files in {output_dir}")
    return created_files


def get_available_files(datasets_dir: str) -> List[str]:
    """Get list of available CSV files in the datasets directory."""
    datasets_path = Path(datasets_dir)
    if not datasets_path.exists():
        return []
    return [f.name for f in datasets_path.glob('*.csv')]


# =============================================================================
# Main Processing Function
# =============================================================================

def process_file(filepath: str, config: Dict) -> List[Dict]:
    """
    Process a single dataset file and run all models with all lags.
    
    Args:
        filepath: Path to the CSV file
        config: Configuration dictionary
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Extract dataset name
    dataset_name = Path(filepath).stem
    
    # Load data
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return results
    
    # Determine target variable
    target_var = config.get('target_variable', 'avgRnkWeek')
    
    # Check for required columns - support both old and new formats
    if target_var == 'avgRnkWeek' and 'avgRnkWeek' in df.columns:
        required_cols = ['yearWeek', 'avgRnkWeek', 'N_Tweets']
    elif 'N_ASINs' in df.columns:
        target_var = 'N_ASINs'
        required_cols = ['yearWeek', 'N_ASINs', 'N_Tweets']
    else:
        print(f"Missing required columns in {filepath}. Available: {df.columns.tolist()}")
        return results
    
    # Select columns
    df = df[required_cols].copy()
    df.columns = ['yearWeek', 'target', 'N_Tweets']  # Standardize column names
    
    # Remove rows where N_Tweets = 0 (matching R code)
    df = df[df['N_Tweets'] != 0]
    df = df.dropna()
    df = df.sort_values('yearWeek').reset_index(drop=True)
    
    if len(df) < config['holdout_size'] + 10:  # Need minimum data
        print(f"Insufficient data in {filepath}: {len(df)} rows")
        return results
    
    # Split data
    n = len(df)
    split_idx = n - config['holdout_size']
    
    target_train = df['target'].values[:split_idx]
    target_test = df['target'].values[split_idx:]
    
    # Process each lag
    for lag in config['lags']:
        print(f"  Processing {dataset_name}, lag={lag}")
        
        # --- VECM (Multivariate) ---
        try:
            # Prepare multivariate data
            mv_df = df[['target', 'N_Tweets']].copy()
            mv_df.columns = ['target', 'twitter']
            preds, actuals = rolling_forecast_vecm(mv_df, lag, config['holdout_size'])
            metrics = calculate_accuracy_metrics(actuals, preds)
            results.append({
                'dataset': dataset_name,
                'model_type': 'Multivariate Target+Twitter',
                'model': f'VECM, lag={lag}',
                'lag': lag,
                **metrics
            })
        except Exception as e:
            print(f"    VECM failed for lag={lag}: {e}")
        
        # --- Linear AR (Univariate) ---
        try:
            preds, actuals = rolling_forecast_linear_ar(
                target_train, target_test, 
                config['ar_order'], lag
            )
            metrics = calculate_accuracy_metrics(actuals, preds)
            results.append({
                'dataset': dataset_name,
                'model_type': 'Univariate Target',
                'model': f'LINEAR, lag={lag}',
                'lag': lag,
                **metrics
            })
        except Exception as e:
            print(f"    LINEAR failed for lag={lag}: {e}")
        
        # --- NNET (Univariate) ---
        try:
            preds, actuals = rolling_forecast_nnet(
                target_train, target_test,
                config['nnet_input_lags'], config['nnet_hidden_size'], lag
            )
            metrics = calculate_accuracy_metrics(actuals, preds)
            results.append({
                'dataset': dataset_name,
                'model_type': 'Univariate Target',
                'model': f'NNET, lag={lag}',
                'lag': lag,
                **metrics
            })
        except Exception as e:
            print(f"    NNET failed for lag={lag}: {e}")
    
    return results


def main():
    """Main execution function."""
    
    # Check for command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--prepare-data' and len(sys.argv) > 2:
            # Prepare datasets from master file
            master_file = sys.argv[2]
            prepare_datasets_from_master(
                master_file, 
                CONFIG['datasets_dir'],
                CONFIG.get('min_observations', 50),
                CONFIG.get('min_nonzero_tweets', 10)
            )
            print("\nDatasets prepared. Run script again without arguments to process.")
            return []
        elif sys.argv[1] == '--help':
            print(__doc__)
            print("\nUsage:")
            print("  python fashion_forecasting.py                    # Run forecasting")
            print("  python fashion_forecasting.py --prepare-data FILE  # Prepare data from master")
            return []
    
    print("=" * 70)
    print("Fashion Sales Forecasting Analysis")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Lags: {CONFIG['lags']}")
    print(f"Holdout size: {CONFIG['holdout_size']} weeks")
    print(f"Target variable: {CONFIG['target_variable']}")
    print("=" * 70)
    
    # Create results directory
    results_dir = Path(CONFIG['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of files to process
    available_files = get_available_files(CONFIG['datasets_dir'])
    
    if not available_files:
        # Try to use the predefined FILES list
        files_to_process = [f for f in FILES if (Path(CONFIG['datasets_dir']) / f).exists()]
        if not files_to_process:
            print(f"No dataset files found in {CONFIG['datasets_dir']}")
            print("Use --prepare-data to create datasets from master file.")
            return []
    else:
        files_to_process = available_files
    
    print(f"Number of datasets to process: {len(files_to_process)}")
    print("=" * 70)
    
    # Process all files
    all_results = []
    
    for i, filename in enumerate(files_to_process, 1):
        filepath = Path(CONFIG['datasets_dir']) / filename
        print(f"\n[{i}/{len(files_to_process)}] Processing: {filename}")
        
        if not filepath.exists():
            print(f"  File not found: {filepath}")
            continue
        
        file_results = process_file(str(filepath), CONFIG)
        all_results.extend(file_results)
        
        # Log progress
        with open(results_dir / 'logging.txt', 'a') as f:
            f.write(f"{datetime.now()}: Completed {filename}\n")
    
    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save detailed results
        results_df.to_csv(results_dir / 'results_all.csv', index=False)
        
        # Save tab-separated format (matching R output)
        with open(results_dir / 'results_all.txt', 'w') as f:
            for _, row in results_df.iterrows():
                line = f"{row['dataset']}\t{row['model_type']}\t{row['model']}\t"
                line += f"{row['ME']:.6f}\t{row['RMSE']:.6f}\t{row['MAE']:.6f}\t"
                line += f"{row['MPE']:.6f}\t{row['MAPE']:.6f}\n"
                f.write(line)
        
        # Generate summary statistics
        summary = results_df.groupby(['model', 'lag']).agg({
            'RMSE': ['mean', 'std', 'min', 'max'],
            'MAE': ['mean', 'std'],
            'MAPE': ['mean', 'std']
        }).round(4)
        summary.to_csv(results_dir / 'results_summary.csv')
        
        # Count wins by method and lag
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY: Best Model Counts by Lag")
        print("=" * 70)
        
        for lag in CONFIG['lags']:
            lag_data = results_df[results_df['lag'] == lag]
            
            # Find best model (lowest RMSE) for each dataset
            best_models = lag_data.loc[lag_data.groupby('dataset')['RMSE'].idxmin()]
            
            # Count wins
            wins = best_models['model'].value_counts()
            
            print(f"\nLag {lag}:")
            for model, count in wins.items():
                pct = 100 * count / len(best_models)
                print(f"  {model}: {count} wins ({pct:.1f}%)")
        
        print("\n" + "=" * 70)
        print(f"Results saved to: {results_dir}")
        print(f"End time: {datetime.now()}")
        print("=" * 70)
    
    else:
        print("No results generated. Check data files and paths.")
    
    return all_results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    results = main()
