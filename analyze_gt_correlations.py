"""
Google Trends vs Twitter Correlation Analysis

Reads datasets with Google Trends columns and calculates correlations
between GT columns and Twitter volume (N_Tweets) and Amazon rank (avgRnkWeek).
"""

import pandas as pd
import numpy as np
import os
import re
from glob import glob

def extract_keyword_from_filename(filename):
    """Extract keyword from filename like 'beige_0..1000.csv' -> 'beige'"""
    name = os.path.basename(filename).replace('.csv', '')
    match = re.match(r'^([a-zA-Z-]+)_', name)
    if match:
        return match.group(1)
    return None

def extract_bucket_from_filename(filename):
    """Extract bucket from filename like 'beige_0..1000.csv' -> '0..1000'"""
    name = os.path.basename(filename).replace('.csv', '')
    match = re.search(r'_(\d+\.\.\.?\d+|\d+\+)$', name)
    if match:
        return match.group(1)
    return None

def calculate_correlations(df, gt_cols, target_cols=['N_Tweets', 'avgRnkWeek']):
    """
    Calculate correlations between GT columns and target columns.
    
    Returns dict of {(gt_col, target_col): correlation}
    """
    correlations = {}
    
    for gt_col in gt_cols:
        for target_col in target_cols:
            if gt_col in df.columns and target_col in df.columns:
                # Drop NaN values for this pair
                valid_data = df[[gt_col, target_col]].dropna()
                
                if len(valid_data) >= 10:  # Need minimum observations
                    corr = valid_data[gt_col].corr(valid_data[target_col])
                    correlations[(gt_col, target_col)] = corr
                else:
                    correlations[(gt_col, target_col)] = np.nan
    
    return correlations

def analyze_single_file(filepath):
    """
    Analyze a single dataset file.
    
    Returns dict with file info and correlations.
    """
    filename = os.path.basename(filepath)
    keyword = extract_keyword_from_filename(filename)
    bucket = extract_bucket_from_filename(filename)
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return {'filename': filename, 'error': str(e)}
    
    # Find GT columns
    gt_cols = [c for c in df.columns if c.startswith('GT_')]
    
    if not gt_cols:
        return {
            'filename': filename,
            'keyword': keyword,
            'bucket': bucket,
            'error': 'No GT columns found'
        }
    
    # Calculate correlations
    correlations = calculate_correlations(df, gt_cols)
    
    # Build result
    result = {
        'filename': filename,
        'keyword': keyword,
        'bucket': bucket,
        'n_rows': len(df),
        'gt_columns': gt_cols,
    }
    
    # Add individual correlations
    for (gt_col, target_col), corr in correlations.items():
        col_suffix = gt_col.replace(f'GT_{keyword}', '').lstrip('_') or 'general'
        result[f'corr_{col_suffix}_vs_{target_col}'] = corr
    
    return result

def main():
    # Configuration
    input_dir = './datasets_with_GT'
    
    print("=" * 80)
    print("GOOGLE TRENDS vs TWITTER CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Find all CSV files (exclude summary file)
    files = glob(os.path.join(input_dir, '*.csv'))
    files = [f for f in files if not os.path.basename(f).startswith('_')]
    
    print(f"\nInput directory: {input_dir}")
    print(f"Files found: {len(files)}")
    
    if not files:
        print("No files found! Make sure datasets_with_GT directory exists.")
        return
    
    # Analyze each file
    results = []
    
    print("\n" + "-" * 80)
    print("PROCESSING FILES")
    print("-" * 80)
    
    for i, filepath in enumerate(sorted(files)):
        filename = os.path.basename(filepath)
        print(f"[{i+1}/{len(files)}] {filename}...", end=' ')
        
        result = analyze_single_file(filepath)
        results.append(result)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
        else:
            print("OK")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Identify correlation columns
    corr_cols = [c for c in results_df.columns if c.startswith('corr_')]
    
    print("\n" + "=" * 80)
    print("CORRELATION SUMMARY")
    print("=" * 80)
    
    # Summary statistics for correlations
    print("\n### Overall Correlation Statistics ###\n")
    
    if corr_cols:
        corr_stats = results_df[corr_cols].describe().round(4)
        print(corr_stats.to_string())
    
    # Separate Twitter and Amazon rank correlations
    twitter_corr_cols = [c for c in corr_cols if 'N_Tweets' in c]
    rank_corr_cols = [c for c in corr_cols if 'avgRnkWeek' in c]
    
    print("\n" + "-" * 80)
    print("CORRELATIONS WITH TWITTER (N_Tweets)")
    print("-" * 80)
    
    if twitter_corr_cols:
        print("\nBy GT Column Type:")
        for col in twitter_corr_cols:
            valid_corrs = results_df[col].dropna()
            if len(valid_corrs) > 0:
                gt_type = col.replace('corr_', '').replace('_vs_N_Tweets', '')
                print(f"\n  {gt_type}:")
                print(f"    Mean:   {valid_corrs.mean():+.4f}")
                print(f"    Median: {valid_corrs.median():+.4f}")
                print(f"    Std:    {valid_corrs.std():.4f}")
                print(f"    Min:    {valid_corrs.min():+.4f}")
                print(f"    Max:    {valid_corrs.max():+.4f}")
                print(f"    N:      {len(valid_corrs)}")
    
    print("\n" + "-" * 80)
    print("CORRELATIONS WITH AMAZON RANK (avgRnkWeek)")
    print("-" * 80)
    print("(Negative = higher GT interest → lower rank → better sales)")
    
    if rank_corr_cols:
        print("\nBy GT Column Type:")
        for col in rank_corr_cols:
            valid_corrs = results_df[col].dropna()
            if len(valid_corrs) > 0:
                gt_type = col.replace('corr_', '').replace('_vs_avgRnkWeek', '')
                print(f"\n  {gt_type}:")
                print(f"    Mean:   {valid_corrs.mean():+.4f}")
                print(f"    Median: {valid_corrs.median():+.4f}")
                print(f"    Std:    {valid_corrs.std():.4f}")
                print(f"    Min:    {valid_corrs.min():+.4f}")
                print(f"    Max:    {valid_corrs.max():+.4f}")
                print(f"    N:      {len(valid_corrs)}")
    
    # Aggregate by keyword
    print("\n" + "-" * 80)
    print("AVERAGE CORRELATIONS BY KEYWORD")
    print("-" * 80)
    
    if 'keyword' in results_df.columns and twitter_corr_cols:
        # Find the "general" GT correlation column for Twitter
        general_twitter_col = [c for c in twitter_corr_cols if 'general' in c]
        if general_twitter_col:
            keyword_summary = results_df.groupby('keyword')[general_twitter_col[0]].mean().sort_values(ascending=False)
            print("\nGT (general) vs Twitter by Keyword:")
            print(keyword_summary.round(4).to_string())
    
    # Detailed results table
    print("\n" + "-" * 80)
    print("DETAILED RESULTS (sorted by Twitter correlation)")
    print("-" * 80)
    
    # Select columns for display
    display_cols = ['filename', 'keyword', 'bucket', 'n_rows'] + twitter_corr_cols[:2] + rank_corr_cols[:2]
    display_cols = [c for c in display_cols if c in results_df.columns]
    
    if twitter_corr_cols:
        sorted_df = results_df.sort_values(twitter_corr_cols[0], ascending=False)
    else:
        sorted_df = results_df
    
    print("\n" + sorted_df[display_cols].head(20).to_string())
    
    # Save full results
    output_file = os.path.join(input_dir, '_correlation_analysis.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n\nFull results saved to: {output_file}")
    
    # Create pivot table summary
    print("\n" + "=" * 80)
    print("PIVOT TABLE: GT vs Twitter Correlation by Keyword and Bucket")
    print("=" * 80)
    
    if 'keyword' in results_df.columns and 'bucket' in results_df.columns and twitter_corr_cols:
        pivot = results_df.pivot_table(
            index='keyword',
            columns='bucket',
            values=twitter_corr_cols[0],
            aggfunc='mean'
        ).round(4)
        print("\n" + pivot.to_string())
        
        pivot_file = os.path.join(input_dir, '_correlation_pivot.csv')
        pivot.to_csv(pivot_file)
        print(f"\nPivot table saved to: {pivot_file}")
    
    return results_df

if __name__ == "__main__":
    results = main()
