import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import ast
import os, glob
from datetime import datetime, timedelta
import itertools

def plot_depth_with_price_proportions(df, depth_col='ask_depth', value_col='best_ask', 
                                     plot_type='violin', figsize=(12, 6), 
                                     title='Ask price distribution by depth with depth proportions'):
    """
    Plots distribution of a value column by depth with proportion of each depth level.
    
    Parameters:
    - df: DataFrame
    - depth_col: Column name for depth levels (used for x-axis and proportion calculation)
    - value_col: Column name for values to plot distribution of
    - plot_type: 'violin', 'box', or 'swarm' (default: 'violin')
    - figsize: Figure size
    - title: Plot title
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # --- Primary Plot (Distribution by Depth) ---
    depth_order = sorted(df[depth_col].unique(), reverse=True)

    if plot_type == 'violin':
        sns.violinplot(data=df, x=depth_col, y=value_col,
                      order=depth_order, ax=ax1, palette='Blues',
                      cut=0)
    elif plot_type == 'box':
        sns.boxplot(data=df, x=depth_col, y=value_col,
                   order=depth_order, ax=ax1, palette='Blues',
                   whis=[5,95])
    
    ax1.set_ylim(max(0, df[value_col].min()*0.9), df[value_col].max()*1.1)
    ax1.set_xlabel(f'{depth_col} (Depth Levels)', fontsize=12)
    ax1.set_ylabel(f'{value_col}', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # --- Secondary Axis (Depth Proportions) ---
    ax2 = ax1.twinx()
    depth_props = df[depth_col].value_counts(normalize=True).sort_index(ascending=False)
    proportions = depth_props.values * 100
    x_vals = depth_props.index

    sns.lineplot(x=x_vals, y=proportions, 
                 ax=ax2, color='red', marker='o', label='Proportion (%)')

    # Annotate each point with its percentage
    for x, y in zip(x_vals, proportions):
        ax2.text(x=x, y=y + 1, s=f'{y:.1f}%', color='red', fontsize=10, ha='center')

    ax2.set_ylabel(f'Proportion of {depth_col} (%)', fontsize=12, color='red')
    ax2.tick_params(axis='y', colors='red')
    ax2.set_ylim(0, max(proportions) * 1.2)
    ax2.set_yticks(range(0, int(max(proportions) * 1.2) + 1, 5))
    ax2.legend(loc='upper right', frameon=False)

    # --- Final Formatting ---
    plt.title(title, fontsize=14)
    fig.tight_layout()
    plt.show()


def plot_orderbook_snapshot(df, start_idx, window_size=100, num_xticks=10, 
                            secondary_column='spread', dual_axis_column=None):
    """
    Plots order book dynamics including best bid/ask, mid price, spread, and trade prices
    for a specified window around a given index. Additionally, adds a secondary y-axis for 
    an optional column on the primary plot.
    
    Args:
        df: DataFrame containing order book and trade data with columns:
            - best_bid: Best bid price
            - best_ask: Best ask price
            - mid_price: Mid price (optional, will calculate if not present)
            - price: Trade price (optional)
            - secondary_column: Column to plot on the secondary axis (default 'spread')
            - dual_axis_column: Optional column to plot on a dual y-axis on the primary plot
        start_idx: Starting index for the plot
        window_size: Number of observations to plot (centered around start_idx)
        num_xticks: Number of x-ticks to display on the plot
    """
    # Calculate plot boundaries
    half_window = window_size // 2
    start_idx = max(0, start_idx - half_window)
    end_idx = min(len(df), start_idx + window_size)
    plot_data = df.iloc[start_idx:end_idx].copy()
    
    # Calculate mid price if not present
    if 'mid_price' not in plot_data.columns:
        plot_data['mid_price'] = (plot_data['best_bid'] + plot_data['best_ask']) / 2
    
    # Calculate spread
    plot_data['spread'] = plot_data['best_ask'] - plot_data['best_bid']
    
    # Create figure with primary and secondary axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                  gridspec_kw={'height_ratios': [3, 1]},
                                  sharex=True)
    
    # Plot price levels on primary axis
    ax1.plot(plot_data.index, plot_data['best_bid'], 'b-', label='Best Bid', linewidth=1, alpha=0.8)
    ax1.plot(plot_data.index, plot_data['best_ask'], 'r-', label='Best Ask', linewidth=1, alpha=0.8)
    ax1.plot(plot_data.index, plot_data['mid_price'], 'g--', label='Mid Price', linewidth=1.5)
    
    # Plot trades if available
    if 'price' in plot_data.columns:
        ax1.scatter(plot_data.index, plot_data['price'], 
                   color='m', marker='o', s=30, alpha=0.7,
                   label='Trade Price', zorder=5)
    
    # Highlight spread area
    ax1.fill_between(plot_data.index, 
                    plot_data['best_bid'], 
                    plot_data['best_ask'],
                    color='gray', alpha=0.1, label='Spread')
    
    # Add dual axis if provided
    if dual_axis_column and dual_axis_column in plot_data.columns:
        ax1_dual = ax1.twinx()  # Create a twin y-axis
        ax1_dual.plot(plot_data.index, plot_data[dual_axis_column], 
                      color='orange', label=dual_axis_column.capitalize(), alpha=0.7)
        ax1_dual.set_ylabel(dual_axis_column.capitalize(), fontsize=12, color='orange')
        ax1_dual.tick_params(axis='y', labelcolor='orange')

    # Plot the secondary column (e.g., imbalance or spread) on the secondary axis
    ax2.plot(plot_data.index, plot_data[secondary_column], 
            color='purple', label=secondary_column.capitalize(), alpha=0.7)
    ax2.fill_between(plot_data.index, 0, plot_data[secondary_column],
                    color='purple', alpha=0.1)
    
    # Mark the reference point (start_idx)
    ref_time = df.index[start_idx + half_window] if start_idx + half_window < len(df) else df.index[-1]
    for ax in [ax1, ax2]:
        ax.axvline(ref_time, color='k', linestyle=':', alpha=0.5, linewidth=1)
    
    # Formatting x-ticks (dynamic number of x-ticks)
    step = max(len(plot_data) // num_xticks, 1)
    ax1.set_xticks(plot_data.index[::step])
    
    # Keep the x-ticks as numeric seconds, no conversion
    ax1.set_xticklabels(plot_data.index[::step], rotation=45, ha='right')
    
    # Axis labels and title
    ax1.set_title(f"Order Book Dynamics (Index {start_idx} to {end_idx})", fontsize=14)
    ax1.set_ylabel('Price', fontsize=12)
    ax2.set_ylabel(secondary_column.capitalize(), fontsize=12)
    ax2.set_xlabel('Time (Seconds)', fontsize=12)
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    
    if dual_axis_column and dual_axis_column in plot_data.columns:
        ax1_dual.legend(loc='upper right')
    
    # Grid and layout
    ax1.grid(True, alpha=0.2)
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()
