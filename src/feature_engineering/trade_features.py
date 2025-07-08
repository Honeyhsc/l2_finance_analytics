import pandas as pd
import numpy as np

def aggregate_trades(trade_df, 
                    group_col='datetime',
                    date_col='date',
                    timestamp_col='timestamp',
                    sum_cols=None,
                    exclude_cols=None):
    """
    Aggregates trade data by a given column, summing specified columns and taking mean of others.
    Automatically excludes timestamp and date columns from averaging unless specified otherwise.
    
    Parameters:
    - trade_df: Input DataFrame with trade data
    - group_col: Column to group by (default 'datetime')
    - date_col: Date column to exclude (default 'date')
    - timestamp_col: Timestamp column to exclude (default 'timestamp')
    - sum_cols: List of columns (or single column as str) to sum (default None → auto-detect)
    - exclude_cols: Additional columns to exclude from mean calculation
    
    Returns:
    - Aggregated DataFrame
    """
    # Make a copy to avoid modifying original as this function is meant to be non-destructive
    df = trade_df.copy()
    
    # Default columns to exclude from mean calculation
    default_exclude = {group_col, date_col, timestamp_col, 'bid', 'ask'}
    if exclude_cols:
        default_exclude.update(
            [exclude_cols] if isinstance(exclude_cols, str) else exclude_cols
        )
    
    # Ensure sum_cols is a list (even if a single string is passed)
    if sum_cols is not None:
        sum_cols = [sum_cols] if isinstance(sum_cols, str) else sum_cols
    else:
        # Auto-detect volume columns if sum_cols is None
        sum_cols = [
            col for col in df.columns
            if col.lower() in {'vol', 'volume', 'qty', 'quantity', 'size'}
            and col not in default_exclude
        ]
    
    # Columns for mean calculation (all others except sum_cols and excluded)
    mean_cols = [
        col for col in df.columns
        if col not in sum_cols and col not in default_exclude
    ]
    
    # Create aggregation dictionary
    agg_dict = {col: 'sum' for col in sum_cols}
    agg_dict.update({col: 'mean' for col in mean_cols})
    # print(f"Aggregating by {group_col} with sum_cols: {sum_cols} and mean_cols: {mean_cols}")
    # print(f"Excluding columns: {default_exclude}")
    # print(f"Aggregation dictionary: {agg_dict}")
    # Group and aggregate
    agg_df = df.groupby(group_col).agg(agg_dict).reset_index()
    
    return agg_df

# Function to calculate trade duration
def calculate_trade_duration(df):
    """
    Calculate trade duration in seconds between consecutive trades within each day.
    
    Parameters:
    df (pd.DataFrame): Input dataframe containing a 'datetime' column
    
    Returns:
    pd.DataFrame: Dataframe with added 'trade_duration' column and NA rows removed
    """
    df = df.copy()  # Avoid modifying the original dataframe
    df['trade_duration'] = (
        df.groupby(pd.Grouper(key='datetime', freq='D'))['datetime']
        .diff()
        .dt.total_seconds()
    )
    df = df.dropna(subset=['trade_duration'])
    return df


# # Function to calculate forward returns and remove outliers
def calculate_forward_returns(df, 
                            price_col='mid_price', 
                            duration_col='trade_duration',
                            periods=10,
                            remove_outliers=True,
                            duration_percentile=0.99,
                            return_percentile=None):
    """
    Calculates forward returns and optionally removes outliers.
    
    Args:
        df: Input DataFrame
        price_col: Column name for price data
        duration_col: Column name for trade duration
        periods: Number of periods for forward return calculation
        remove_outliers: Whether to remove outliers (bool)
        duration_percentile: Percentile cutoff for trade duration (0-1)
        return_percentile: Percentile cutoff for returns (0-1, None to skip)
    
    Returns:
        Cleaned DataFrame with forward returns column
    """
    # Create copy to avoid modifying original
    result_df = df.copy()
    
    # Calculate forward returns
    return_col = f'{price_col}_return_{periods}'
    result_df[return_col] = result_df[price_col].pct_change(periods=periods).shift(-periods)
    
    # Remove rows with NaN in key columns
    result_df = result_df.dropna(subset=[duration_col, return_col])
    
    if remove_outliers:
        # Remove duration outliers
        duration_threshold = result_df[duration_col].quantile(duration_percentile)
        result_df = result_df[result_df[duration_col] <= duration_threshold]
        
        # Remove return outliers if requested
        if return_percentile is not None:
            return_threshold = result_df[return_col].abs().quantile(return_percentile)
            result_df = result_df[result_df[return_col].abs() <= return_threshold]
    
    return result_df


def create_quantile_bins(df, 
                       value_col,
                       analysis_col=None,
                       num_bins=20,
                       use_log_scale=True,
                       precision=2,
                       label_suffix=''):
    """
    Creates quantile bins for any numerical column with optional log scaling.
    
    Args:
        df: Input DataFrame
        value_col: Column to bin (must be numeric)
        analysis_col: Optional column to analyze by bin (default None)
        num_bins: Number of quantile bins (default 20)
        use_log_scale: Whether to apply log1p transformation (default True)
        precision: Decimal places for bin labels (default 2)
        label_suffix: Optional suffix for bin labels (e.g., 'sec', 'ms')
        
    Returns:
        Tuple of (DataFrame with bin columns, bin statistics if analysis_col provided)
    """
    # Create copy to avoid modifying original
    result_df = df.copy()
    
    # Transform values if using log scale
    if use_log_scale:
        transformed = np.log1p(result_df[value_col])
        bin_col = f'log_{value_col}_bin'
    else:
        transformed = result_df[value_col]
        bin_col = f'{value_col}_bin'
    
    # Create quantile bins (handle edge cases)
    try:
        result_df[bin_col] = pd.qcut(transformed, q=num_bins, duplicates='drop')
    except ValueError as e:
        if "Bin edges must be unique" in str(e):
            # Reduce bin count if too many duplicates
            actual_bins = len(transformed.unique())
            num_bins = min(num_bins, actual_bins)
            result_df[bin_col] = pd.qcut(transformed, q=num_bins, duplicates='drop')
        else:
            raise
    
    # Generate readable labels
    interval_index = pd.IntervalIndex(result_df[bin_col].cat.categories)
    
    if use_log_scale:
        # Convert back to original scale
        left_edges = np.expm1(interval_index.left)
        right_edges = np.expm1(interval_index.right)
    else:
        left_edges = interval_index.left
        right_edges = interval_index.right
    
    # Create formatted labels
    label_template = f"{{0:.{precision}f}}{label_suffix}–{{1:.{precision}f}}{label_suffix}"
    bin_labels = [label_template.format(l, r) 
                 for l, r in zip(left_edges, right_edges)]
    
    # Apply labels as ordered categories
    label_col = f'{value_col}_bin_label'
    result_df[label_col] = pd.Categorical.from_codes(
        codes=result_df[bin_col].cat.codes,
        categories=bin_labels,
        ordered=True
    )
    
    # Calculate bin statistics if analysis column provided
    bin_stats = None
    if analysis_col is not None:
        bin_stats = result_df.groupby(label_col, observed=True)[analysis_col].agg(['mean', 'count', 'std'])
    
    return result_df, bin_stats


def count_trades_by_time_bin(
    df, 
    datetime_col='datetime', 
    bin_duration='30min', 
    output_col='trade_count',
):
    """
    Counts trades in specified time bins (e.g., 30min, 1H, 15min) from a DataFrame with a datetime column.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing trade data.
    datetime_col : str, optional (default: 'datetime')
        Name of the datetime column.
    bin_duration : str, optional (default: '30min')
        Duration of each bin (e.g., '30min', '1H', '15min').
    output_col : str, optional (default: 'trade_count')
        Name for the output count column.
    drop_zero_bins : bool, optional (default: False)
        If True, drops bins with zero trades.

    Returns:
    --------
    pd.DataFrame
        DataFrame with time bins and trade counts, indexed by bin start times.
    """
    # Ensure datetime column is in proper format
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Group by time bins and count trades
    df['time_bin'] = df[datetime_col].dt.floor(bin_duration)
    trade_counts = df.groupby('time_bin').size().rename(output_col)
    trade_counts = trade_counts.to_frame().reset_index()

    return trade_counts


def calculate_time_based_stats_old(
    df,
    datetime_col='datetime',
    value_col='price',  # Column to calculate stats on
    stat='mean',       # 'mean', 'median', 'sum', 'count', etc.
    time_bin='30min',  # Bin size (e.g., '15min', '1H', '5min')
):
    """
    Calculate time-based statistics (ignoring dates) for specified value column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with datetime and value columns
    datetime_col : str
        Name of datetime column
    value_col : str
        Column to calculate statistics on
    stat : str or function
        Aggregation function ('mean', 'median', 'sum', 'count', etc.)
    time_bin : str
        Time frequency for binning (e.g., '30min', '1H')
    market_open, market_close : str
        Trading hours in 'HH:MM' format
    
    Returns:
    --------
    pd.DataFrame with statistics indexed by time
    """
    # Ensure datetime format
    df = df.copy()
    
    # Extract time component and bin
    df['time_only'] = df[datetime_col].dt.time
    df['time_bin'] = pd.to_datetime(df['time_only'].astype(str)).dt.floor(time_bin)
    df['time_bin'] = df['time_bin'].dt.time
    
    # Group by time bin and calculate stats
    result = df.groupby('time_bin')[value_col].agg(stat).to_frame()
    result.index.name = 'time'
    result.sort_index()
    result.reset_index(inplace=True)

    return result


from typing import Union, List, Optional

def calculate_time_based_stats(
    df: pd.DataFrame,
    datetime_col: str = 'datetime',
    value_col: str = 'price',
    stat: Union[str, List[str]] = 'mean',
    time_bin: str = '30min',
    group_by_date: bool = False,
    market_hours: Optional[tuple] = None
) -> pd.DataFrame:
    """
    Calculate statistics for time bins, optionally grouping across days or keeping dates separate.
    
    Parameters:
    -----------

    
    Returns:
    --------
    pd.DataFrame with statistics indexed by time bins
    """
    # Ensure datetime
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Filter market hours if specified
    if market_hours:
        open_time = pd.to_datetime(market_hours[0]).time()
        close_time = pd.to_datetime(market_hours[1]).time()
        mask = (df[datetime_col].dt.time >= open_time) & (df[datetime_col].dt.time <= close_time)
        df = df[mask]
    
    # Create bins
    if group_by_date:
        # Date+time bins (e.g., '2023-01-01 09:30:00')
        df['bin'] = df[datetime_col].dt.floor(time_bin)
    else:
        # Time-only bins (e.g., '09:30:00')
        df['bin'] = df[datetime_col].dt.floor(time_bin).dt.time
    
    # Group and aggregate
    result = df.groupby('bin')[value_col].agg(stat)
    
    # Clean output
    if isinstance(stat, list):
        result = result.unstack()
    result.index.name = 'date_time' if group_by_date else 'time'
    return result.sort_index().reset_index()


import pandas as pd
from typing import Union, List, Optional, Dict

def calculate_time_based_stats(
    df: pd.DataFrame,
    datetime_col: str = 'datetime',
    value_col: str = 'price',
    stat: Union[str, List[str], Dict[str, str]] = 'mean',
    time_bin: str = '30min',
    group_by_date: bool = False,
    market_hours: Optional[tuple] = None,
    daily_agg: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Calculate statistics for time bins, optionally grouping across days or keeping dates separate.
    Allows statistics with daily aggregation option as well. For example,
    {'count': 'mean'} where 'daily_stat' which is count is calculated first for each date, then 'count' 
    is aggregated using 'mean' across dates to get the final statistic.
    Otherwise, group_by_date=False will aggregate across days, and calculate the given stat.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with datetime column
    datetime_col : str
        Name of datetime column (default: 'datetime')
    value_col : str
        Column to calculate statistics on (default: 'price')
    stat : str or list
        Aggregation function(s) ('mean', 'median', 'sum', 'count', etc.)
    time_bin : str
        Time frequency for binning (e.g., '15min', '1H') (default: '30min')
    group_by_date : bool
        If False, aggregates across days (time-only bins).
        If True, keeps date+time bins (default: False)
    market_hours : tuple, optional
        (open_time, close_time) as strings (e.g., ('09:30', '16:00'))
    daily_agg : Dict[str, str], optional
        If provided, calculates daily statistics first, then aggregates across days.
        Format: {'daily_stat': 'final_stat'} 
        (e.g., {'count': 'mean'} gives mean of daily counts)
    """
    # Ensure datetime
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Filter market hours
    if market_hours:
        open_time = pd.to_datetime(market_hours[0]).time()
        close_time = pd.to_datetime(market_hours[1]).time()
        mask = (df[datetime_col].dt.time >= open_time) & (df[datetime_col].dt.time <= close_time)
        df = df[mask]
    
    # Daily aggregation mode
    if daily_agg:
        # First calculate daily bins
        daily_bins = df[datetime_col].dt.floor(time_bin).dt.time
        daily_groups = df.groupby([df[datetime_col].dt.date, daily_bins])
        
        # Calculate daily stats
        # Unstack send the second level of grouping (daily_bins) as columns
        daily_stats = daily_groups[value_col].agg(list(daily_agg.keys())[0]).unstack()  # applying first aggregation
        
        # Then aggregate across days
        # We are aggregating across columns (daily_bins) as the default axis=0
        result = daily_stats.agg(list(daily_agg.values())[0])  # applying the second aggregation (the value)
        # result is a series with index as daily_bins(time) and values as the aggregated statistic
        result.name = f"{list(daily_agg.values())[0]}_of_daily_{list(daily_agg.keys())[0]}"
        # getting index to time column
        return result.reset_index().rename(columns={'index': 'time'})
    
    # Normal mode
    if group_by_date:
        bins = df[datetime_col].dt.floor(time_bin)
    else:
        bins = df[datetime_col].dt.floor(time_bin).dt.time
    
    return df.groupby(bins)[value_col].agg(stat).reset_index().rename(
        columns={datetime_col: 'date_time' if group_by_date else 'time'}
    )