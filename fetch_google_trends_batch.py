"""
Google Trends Data Integration Script - Batch Processing
Processes all fashion datasets and adds Google Trends columns.

For each keyword (e.g., beige, black, cotton), adds:
- GT_{keyword}: General search interest
- GT_{keyword}_fashion_cat: Search interest in Fashion & Style category
- GT_{keyword}_dress: "{keyword} dress" search interest
- GT_{keyword}_shoes: "{keyword} shoes" search interest
"""

import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import time
import os
import re

def yearweek_to_date(yearweek):
    """Convert yearWeek format (e.g., 201711) to datetime"""
    year = int(str(yearweek)[:4])
    week = int(str(yearweek)[4:])
    return datetime.strptime(f'{year}-W{week:02d}-1', '%G-W%V-%u')

def date_to_yearweek(date):
    """Convert datetime to yearWeek format"""
    iso_cal = date.isocalendar()
    return int(f"{iso_cal[0]}{iso_cal[1]:02d}")

def extract_keyword_from_filename(filename):
    """Extract keyword from filename like 'beige_0..1000.csv' -> 'beige'"""
    # Remove .csv extension
    name = filename.replace('.csv', '')
    # Extract keyword (everything before the underscore followed by numbers)
    match = re.match(r'^([a-zA-Z-]+)_', name)
    if match:
        return match.group(1)
    return None

def fetch_google_trends_safe(pytrends, keywords, timeframe, geo='US', category=0):
    """
    Safely fetch Google Trends data with error handling.
    """
    try:
        pytrends.build_payload(
            kw_list=keywords,
            cat=category,
            timeframe=timeframe,
            geo=geo,
            gprop=''
        )
        
        trends_df = pytrends.interest_over_time()
        
        if trends_df.empty:
            return None
            
        if 'isPartial' in trends_df.columns:
            trends_df = trends_df.drop('isPartial', axis=1)
        
        trends_df = trends_df.reset_index()
        trends_df.rename(columns={'date': 'Date'}, inplace=True)
        trends_df['yearWeek'] = trends_df['Date'].apply(date_to_yearweek)
        
        return trends_df
        
    except Exception as e:
        print(f"    Error: {e}")
        return None

def get_keyword_variants(keyword):
    """
    Generate keyword variants for Google Trends search.
    Returns list of (search_term, category, column_name) tuples.
    """
    # Clean keyword (replace hyphens with spaces for search)
    search_keyword = keyword.replace('-', ' ')
    
    variants = [
        # General term (all categories)
        (search_keyword, 0, f'GT_{keyword}'),
        
        # Fashion & Style category (185)
        (search_keyword, 185, f'GT_{keyword}_fashion_cat'),
        
        # Keyword + dress
        (f'{search_keyword} dress', 0, f'GT_{keyword}_dress'),
        
        # Keyword + shoes
        (f'{search_keyword} shoes', 0, f'GT_{keyword}_shoes'),
    ]
    
    return variants

def fetch_trends_for_keyword(pytrends, keyword, timeframe, delay=2):
    """
    Fetch all Google Trends variants for a single keyword.
    Returns dict of {column_name: DataFrame with yearWeek and values}
    """
    variants = get_keyword_variants(keyword)
    results = {}
    
    for search_term, category, col_name in variants:
        print(f"    Fetching '{search_term}' (cat={category})...", end=' ')
        
        trends_df = fetch_google_trends_safe(
            pytrends, [search_term], timeframe,
            geo='US', category=category
        )
        
        if trends_df is not None:
            # Check data quality
            non_zero_pct = (trends_df[search_term] > 0).sum() / len(trends_df) * 100
            
            if non_zero_pct > 50:
                results[col_name] = trends_df[['yearWeek', search_term]].copy()
                results[col_name].rename(columns={search_term: col_name}, inplace=True)
                print(f"✓ ({non_zero_pct:.0f}% non-zero)")
            else:
                print(f"✗ (only {non_zero_pct:.0f}% non-zero, skipping)")
        else:
            print("✗ (no data)")
        
        time.sleep(delay)  # Rate limiting
    
    return results

def process_single_dataset(input_path, output_path, pytrends, trends_cache, delay=2):
    """
    Process a single dataset file and add Google Trends columns.
    
    Uses trends_cache to avoid re-fetching data for the same keyword.
    """
    filename = os.path.basename(input_path)
    keyword = extract_keyword_from_filename(filename)
    
    if not keyword:
        print(f"  Could not extract keyword from {filename}, skipping")
        return None
    
    print(f"\n  Processing: {filename} (keyword: '{keyword}')")
    
    # Load dataset
    df = pd.read_csv(input_path)
    
    # Determine timeframe
    min_date = yearweek_to_date(df['yearWeek'].min())
    max_date = yearweek_to_date(df['yearWeek'].max())
    start_date = (min_date - timedelta(days=7)).strftime('%Y-%m-%d')
    end_date = (max_date + timedelta(days=7)).strftime('%Y-%m-%d')
    timeframe = f'{start_date} {end_date}'
    
    # Check if we already have trends data for this keyword and timeframe
    cache_key = (keyword, timeframe)
    
    if cache_key not in trends_cache:
        print(f"  Fetching Google Trends for '{keyword}'...")
        trends_data = fetch_trends_for_keyword(pytrends, keyword, timeframe, delay)
        trends_cache[cache_key] = trends_data
    else:
        print(f"  Using cached Google Trends for '{keyword}'")
        trends_data = trends_cache[cache_key]
    
    # Merge trends data with dataset
    merged_df = df.copy()
    
    for col_name, trends_df in trends_data.items():
        merged_df = merged_df.merge(trends_df, on='yearWeek', how='left')
    
    # Save output
    merged_df.to_csv(output_path, index=False)
    
    # Return summary
    gt_cols = [c for c in merged_df.columns if c.startswith('GT_')]
    return {
        'filename': filename,
        'keyword': keyword,
        'rows': len(merged_df),
        'gt_columns': gt_cols,
        'missing_values': merged_df[gt_cols].isnull().sum().to_dict() if gt_cols else {}
    }

def main():
    # Configuration
    input_dir = './datasets'
    output_dir = './datasets_with_GT'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # List of all dataset files
    files = [
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
    
    print("=" * 70)
    print("GOOGLE TRENDS BATCH PROCESSING")
    print("=" * 70)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total files to process: {len(files)}")
    
    # Get unique keywords
    unique_keywords = set()
    for f in files:
        kw = extract_keyword_from_filename(f)
        if kw:
            unique_keywords.add(kw)
    
    print(f"Unique keywords: {len(unique_keywords)}")
    print(f"Keywords: {sorted(unique_keywords)}")
    
    # Initialize pytrends
    pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
    
    # Cache for trends data (to avoid re-fetching for same keyword)
    trends_cache = {}
    
    # Process each file
    results = []
    errors = []
    
    print("\n" + "-" * 70)
    print("PROCESSING FILES")
    print("-" * 70)
    
    for i, filename in enumerate(files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"\n[{i+1}/{len(files)}] {filename}")
        
        if not os.path.exists(input_path):
            print(f"  ✗ File not found: {input_path}")
            errors.append({'filename': filename, 'error': 'File not found'})
            continue
        
        try:
            result = process_single_dataset(
                input_path, output_path, pytrends, trends_cache, delay=1
            )
            if result:
                results.append(result)
                print(f"  ✓ Saved to {output_path}")
                print(f"    Added columns: {result['gt_columns']}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            errors.append({'filename': filename, 'error': str(e)})
    
    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"\nSuccessfully processed: {len(results)} files")
    print(f"Errors: {len(errors)} files")
    
    if errors:
        print("\nFiles with errors:")
        for err in errors:
            print(f"  - {err['filename']}: {err['error']}")
    
    # Save summary report
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(output_dir, '_processing_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    return results, errors

if __name__ == "__main__":
    results, errors = main()
