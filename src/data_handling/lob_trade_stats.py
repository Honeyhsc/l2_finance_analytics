import pandas as pd
import glob
import os
import ast
import numpy as np

# Define paths
lob_path = r"C:\Users\rashm\Documents\UoB\TB-2\DSMP\project\dsmp-2024-groupt33\data\processed\all_lobs"
trade_path = r"C:\Users\rashm\Documents\UoB\TB-2\DSMP\project\dsmp-2024-groupt33\data\processed\all_trades"
out_file = r"C:\Users\rashm\Documents\UoB\TB-2\DSMP\project\dsmp-2024-groupt33\data\processed\daily_summarised_stats_2.csv"

def read_trade_file(file):
    """Read trade file with enhanced parsing"""
    date_str = os.path.basename(file).split('_')[1][:10]
    df = pd.read_csv(file, names=['timestamp', 'price', 'vol'], 
                    dtype={'timestamp': 'float64', 'price': 'float64', 'vol': 'float64'},
                    low_memory=False)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'price'])
    df['date'] = date_str
    return df

def read_lob_file(file):
    """Read and parse LOB data with robust error handling"""
    try:
        date_str = os.path.basename(file).split('_')[1][:10]
        df = pd.read_csv(file, usecols=['timestamp', 'bid', 'ask'], 
                        dtype={'timestamp': 'float64', 'bid': 'str', 'ask': 'str'})
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df['date'] = date_str
        
        # Safely parse bid/ask lists
        def safe_parse(x):
            try:
                return ast.literal_eval(x.strip())
            except:
                return []
        
        df['bid'] = df['bid'].apply(safe_parse)
        df['ask'] = df['ask'].apply(safe_parse)
        return df
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        return pd.DataFrame()

def calculate_level_metrics(lob_df):
    """Calculate metrics for each level in the order book"""
    # Initialize storage for level metrics
    level_metrics = []
    
    # Process top 5 levels
    for level in range(1, 6):
        # Extract bid/ask prices and volumes
        lob_df[f'bid_price_{level}'] = lob_df['bid'].apply(
            lambda x: x[level-1][0] if len(x) >= level else np.nan)
        lob_df[f'bid_vol_{level}'] = lob_df['bid'].apply(
            lambda x: x[level-1][1] if len(x) >= level else 0)
        lob_df[f'ask_price_{level}'] = lob_df['ask'].apply(
            lambda x: x[level-1][0] if len(x) >= level else np.nan)
        lob_df[f'ask_vol_{level}'] = lob_df['ask'].apply(
            lambda x: x[level-1][1] if len(x) >= level else 0)
        
        # Calculate level-specific metrics
        lob_df[f'total_vol_{level}'] = lob_df[f'bid_vol_{level}'] + lob_df[f'ask_vol_{level}']
        lob_df[f'spread_{level}'] = lob_df[f'ask_price_{level}'] - lob_df[f'bid_price_{level}']
        lob_df[f'liquidity_{level}'] = lob_df[f'total_vol_{level}'] / (lob_df[f'spread_{level}'] + 1e-10)
        
        level_metrics.extend([
            f'bid_price_{level}', f'bid_vol_{level}', 
            f'ask_price_{level}', f'ask_vol_{level}',
            f'total_vol_{level}', f'spread_{level}',
            f'liquidity_{level}'
        ])
    
    return lob_df, level_metrics

def calculate_daily_metrics(lob_df, trade_df):
    """Calculate comprehensive daily metrics with enhanced statistics"""
    # Process level metrics
    lob_df, level_metrics = calculate_level_metrics(lob_df)
    
    # Basic metrics
    lob_df['best_bid'] = lob_df['bid'].apply(lambda x: x[0][0] if x and len(x) > 0 else np.nan)
    lob_df['best_ask'] = lob_df['ask'].apply(lambda x: x[0][0] if x and len(x) > 0 else np.nan)
    lob_df['mid_price'] = (lob_df['best_bid'] + lob_df['best_ask']) / 2
    lob_df['bid_levels'] = lob_df['bid'].apply(len)
    lob_df['ask_levels'] = lob_df['ask'].apply(len)
    
    # Sort trade data by timestamp to ensure first/last are correct
    trade_df = trade_df.sort_values('timestamp')
    
    # Merge with trade data
    merged_df = pd.merge_asof(
        trade_df,
        lob_df.sort_values('timestamp'),
        on='timestamp',
        by='date',
        direction='nearest',
        tolerance=1
    ).dropna(subset=['best_bid', 'best_ask'])
    
    # Define aggregation functions
    agg_funcs = {
        # Price metrics
        'best_bid': ['mean', 'median', 'min', 'max', 'std'],
        'best_ask': ['mean', 'median', 'min', 'max', 'std'],
        'mid_price': ['mean', 'median', 'min', 'max', 'std'],
        'price': ['mean', 'median', 'min', 'max', 'std', 'first', 'last'],
        
        # Depth metrics
        'bid_levels': ['mean', 'median'],
        'ask_levels': ['mean', 'median'],
        
        # Volume metrics
        'vol': ['sum', 'mean', 'median'],
    }
    
    # Add level-specific aggregations
    for level in range(1, 6):
        agg_funcs.update({
            f'bid_vol_{level}': ['sum', 'mean', 'median'],
            f'ask_vol_{level}': ['sum', 'mean', 'median'],
            f'total_vol_{level}': ['sum', 'mean', 'median'],
            f'spread_{level}': ['mean', 'median'],
            f'liquidity_{level}': ['mean', 'median']
        })
    
    # Group by date
    daily_metrics = merged_df.groupby('date').agg(agg_funcs)
    
    # Flatten multi-index columns
    daily_metrics.columns = ['_'.join(col).strip() for col in daily_metrics.columns.values]
    daily_metrics = daily_metrics.reset_index()
    
    # Add derived metrics
    daily_metrics['bid_ask_spread'] = daily_metrics['best_ask_mean'] - daily_metrics['best_bid_mean']
    
    # Rename first/last price columns for clarity
    daily_metrics = daily_metrics.rename(columns={
        'price_first': 'price_open',
        'price_last': 'price_close'
    })
    
    # Calculate price change metrics
    daily_metrics['price_change'] = daily_metrics['price_close'] - daily_metrics['price_open']
    daily_metrics['price_pct_change'] = daily_metrics['price_change'] / (daily_metrics['price_open'] + 1e-10)
    
    # Calculate total depth metrics
    for side in ['bid', 'ask']:
        daily_metrics[f'total_{side}_depth'] = daily_metrics[
            [f'{side}_vol_{level}_sum' for level in range(1,6)]
        ].sum(axis=1)
    
    daily_metrics['total_depth'] = daily_metrics['total_bid_depth'] + daily_metrics['total_ask_depth']
    daily_metrics['depth_imbalance'] = (
        daily_metrics['total_bid_depth'] - daily_metrics['total_ask_depth']
    ) / (daily_metrics['total_depth'] + 1e-10)
    
    # Add time features
    daily_metrics['day_of_week'] = pd.to_datetime(daily_metrics['date']).dt.day_name()
    
    return daily_metrics

def main():
    """Process all files and save results"""
    all_metrics = pd.DataFrame()
    
    for trade_file in glob.glob(os.path.join(trade_path, 'Trade_*.csv')):
        trade_date = os.path.basename(trade_file).split('_')[1][:10]
        lob_file = os.path.join(lob_path, f'LoB_{trade_date}.csv')
        
        if not os.path.exists(lob_file):
            print(f'No matching LoB file for {trade_date}')
            continue
            
        print(f'Processing {trade_date}')
        
        lob_data = read_lob_file(lob_file)
        trade_data = read_trade_file(trade_file)
        
        if not lob_data.empty and not trade_data.empty:
            metrics = calculate_daily_metrics(lob_data, trade_data)
            all_metrics = pd.concat([all_metrics, metrics])
    
    # Save results
    all_metrics.to_csv(out_file, index=False)
    print(f"Saved metrics to {out_file}")
    print("Generated metrics:", list(all_metrics.columns))

if __name__ == '__main__':
    main()