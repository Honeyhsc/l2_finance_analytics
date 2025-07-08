import pandas as pd
import ast
import numpy as np 
import warnings
warnings.filterwarnings('ignore')
import ast
import os, glob

'''This script merges trade and LoB data for the 10th or 11th day of each month in 2026, given in the input range.
It reads the data from the specified directories, processes it, and saves the merged output to a CSV file.'''
  

# Define paths
lob_path = r"C:\Users\rashm\Documents\UoB\TB-2\DSMP\project\dsmp-2024-groupt33\data\processed\all_lobs"
trade_path = r"C:\Users\rashm\Documents\UoB\TB-2\DSMP\project\dsmp-2024-groupt33\data\processed\all_trades"
out_file = r"C:\Users\rashm\Documents\UoB\TB-2\DSMP\project\dsmp-2024-groupt33\data\processed\trade_lob_merged_day10_eachmonth.csv"

# Function to read LoB files
def read_lob_file(file):
    date_str = os.path.basename(file).split('_')[1][:10]
    df = pd.read_csv(file, usecols=['timestamp', 'bid', 'ask'], dtype={'timestamp': 'float64', 'bid': 'str', 'ask': 'str'}, low_memory=False)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['date'] = date_str
    print(f"Working on {date_str} - LOB rows: {len(df):,}")
    return df, len(df)

# Function to read trade files
def read_trade_file(file):
    date_str = os.path.basename(file).split('_')[1][:10]
    df = pd.read_csv(file, names=['timestamp', 'price', 'vol'], dtype={'timestamp': 'float64'}, low_memory=False)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['date'] = date_str
    print(f"Working on {date_str} - Trade rows: {len(df):,}")
    return df, len(df)  # Return both dataframe and row count

def print_summary_metrics(lob_df, trade_df, date_str):
    # Calculate time duration
    min_time = min(lob_df['timestamp'].min(), trade_df['timestamp'].min())
    max_time = max(lob_df['timestamp'].max(), trade_df['timestamp'].max())
    duration_seconds = max_time - min_time
    
    # Calculate metrics
    trades_per_second = len(trade_df) / duration_seconds if duration_seconds > 0 else 0
    lob_updates_per_second = len(lob_df) / duration_seconds if duration_seconds > 0 else 0
    
    print(f"\nSummary for {date_str}:")
    print(f"Time duration: {duration_seconds:.2f} seconds")
    print(f"Total trades: {len(trade_df):,}")
    print(f"Total LOB updates: {len(lob_df):,}")
    print(f"Average trades per second: {trades_per_second:.2f}")
    print(f"Average LOB updates per second: {lob_updates_per_second:.2f}\n")

# Function to merge on timestamp and calculate metrics
def merge_trade_lob(lob_df, trade_df):
    # Merge dataframes on timestamp
    # Get ALL unique timestamps from both datasets
    all_timestamps = pd.concat([lob_df['timestamp'], trade_df['timestamp']]).drop_duplicates().sort_values()

    # Create a base dataframe with all timestamps
    base_df = all_timestamps.to_frame(name='timestamp')
    
    # Forward-fill LOB data for missing timestamps
    # Merge LOB data with forward-fill
    lob_df_filled = (
        base_df
        .merge(lob_df, on='timestamp', how='left')
        .sort_values('timestamp')
        .ffill()  # Forward-fill missing LOB data including the date column
    )
    
    # print(lob_df_filled.head(2))

    # Merge with Trade data (left join to keep all timestamps)
    merged_df = pd.merge(
        lob_df_filled,
        trade_df,
        on='timestamp',
        how='left',
        suffixes=('_lob', '_trade')
    )
    
    # Fill NaN trade values with 0 (optional: adjust as needed)
    trade_cols = ['price', 'vol']  # Columns to fill
    merged_df[trade_cols] = merged_df[trade_cols].fillna(0)

    # print('after merger')
    # print(merged_df.head(2))

    # Ensure date column exists after merge
    if 'date' not in lob_df_filled.columns:
        raise ValueError("Date column missing after LOB merge - check input data")
    
    # Adjust datetime (assuming timestamp is seconds since midnight)
    # 1. Convert the date string to datetime at midnight
    base_date = pd.to_datetime(merged_df['date_lob'])
    
    # 2. Add 8 hours to get market open time (08:00:00)
    market_open = base_date + pd.to_timedelta(8, unit='h')
    
    # 3. Add the timestamp seconds to get exact time
    merged_df['datetime'] = market_open + pd.to_timedelta(merged_df['timestamp'], unit='s')
    
    # 4. Format as string (keeping milliseconds)
    merged_df['datetime'] = merged_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]

    # print(merged_df.head(2))
    
    # Convert 'bid' and 'ask' columns to lists of lists
    # With safer version:
    def safe_literal_eval(x):
        try:
            return ast.literal_eval(x.strip()) if pd.notna(x) else []
        except:
            return []

    merged_df['bid'] = merged_df['bid'].apply(safe_literal_eval)
    merged_df['ask'] = merged_df['ask'].apply(safe_literal_eval)

    # Calculate bid-ask spread
    def calculate_bid_ask_spread(row):
        if not row['bid'] or not row['ask']:
            return np.nan
        best_bid = row['bid'][0][0]
        best_ask = row['ask'][0][0]
        return best_ask - best_bid

    merged_df['bid_ask_spread'] = merged_df.apply(calculate_bid_ask_spread, axis=1)

    # Calculate mid-price
    def calculate_mid_price(row):
        if not row['bid'] or not row['ask']:
            return np.nan
        best_bid = row['bid'][0][0]
        best_ask = row['ask'][0][0]
        return (best_bid + best_ask) / 2

    merged_df['mid_price'] = merged_df.apply(calculate_mid_price, axis=1)

    # # Calculate bid-ask ratio using best bid/ask volume
    # def calculate_bid_ask_ratio(row):
    #     if not row['bid'] or not row['ask']:
    #         return np.nan
    #     best_bid_vol = row['bid'][0][1]  # Volume at best bid
    #     best_ask_vol = row['ask'][0][1]  # Volume at best ask
    #     return best_bid_vol / best_ask_vol if best_ask_vol != 0 else np.nan

    # merged_df['bid_ask_ratio'] = merged_df.apply(calculate_bid_ask_ratio, axis=1)

    # # Calculate top-shelf volume (sum of best bid and best ask volumes)
    # def calculate_top_shelf_volume(row):
    #     if not row['bid'] or not row['ask']:
    #         return np.nan
    #     best_bid_vol = row['bid'][0][1]  # Volume at best bid
    #     best_ask_vol = row['ask'][0][1]  # Volume at best ask
    #     return best_bid_vol + best_ask_vol

    # merged_df['top_shelf_volume'] = merged_df.apply(calculate_top_shelf_volume, axis=1)

    return merged_df

# Get all LoB and trade files
lob_files = glob.glob(os.path.join(lob_path, 'LoB_*.csv'))
trade_files = glob.glob(os.path.join(trade_path, 'Trade_*.csv'))    ## --------> Change the file name format here

# Create an empty dataframe to store all daily metrics
all_daily_metrics = pd.DataFrame()

# trade_file_pattern = 'Trade_2026-{:02d}-{:02d}.csv'
# lob_file_pattern = 'LoB_2026-{:02d}-{:02d}.csv'

# Get the relevant day (10th or 11th day) of each month for the year 2026
def get_valid_day_for_month(year, month):
    for day in [10, 11, 9]:
        date_str = f"{year}-{month:02d}-{day:02d}"
        trade_file = os.path.join(trade_path, f'Trade_{date_str}.csv')
        lob_file = os.path.join(lob_path, f'LoB_{date_str}.csv')
        if os.path.exists(trade_file) and os.path.exists(lob_file):
            return date_str
    return None
    
# Main processing loop (modified)
total_lob_rows = 0
total_trade_rows = 0

# User input for month range
start_month = int(input("Enter start month (1–6): "))
end_month = int(input("Enter end month (1–6): "))

# Main loop over the selected months
for month in range(start_month, end_month + 1):
    date_str = get_valid_day_for_month(2026, month)
    if date_str is None:
        print(f"No valid trade/LoB file for month {month}")
        continue

    trade_file = os.path.join(trade_path, f'Trade_{date_str}.csv')
    lob_file = os.path.join(lob_path, f'LoB_{date_str}.csv')

    lob_df, lob_rows = read_lob_file(lob_file)
    trade_df, trade_rows = read_trade_file(trade_file)

    total_lob_rows += lob_rows
    total_trade_rows += trade_rows

    print_summary_metrics(lob_df, trade_df, date_str)

    merged_df = merge_trade_lob(lob_df, trade_df)
    merged_df.drop('date_trade', inplace=True, axis=1)
    merged_df.rename(columns={'date_lob': 'date'}, inplace=True)

    all_daily_metrics = pd.concat([all_daily_metrics, merged_df], ignore_index=True)

# Print total summary
print("\n=== FINAL SUMMARY ===")
print(f"Total LOB rows processed: {total_lob_rows:,}")
print(f"Total trade rows processed: {total_trade_rows:,}")

# Save to CSV
all_daily_metrics.to_csv(out_file, index=False)
print(f"\nProcessed data saved to {out_file}")