import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import ast, os, glob, itertools
from datetime import datetime, timedelta
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_distribution_comparison(df, col1, col2=None, 
                              col1_label=None, col2_label=None, 
                              title='Distribution Comparison with Percentiles',
                              xlabel='Value', ylabel='Density',
                              figsize=(12, 6)):
    """
    Plots the distribution comparison of one or two columns with 1% and 99% percentiles.
    
    Parameters:
    - df: DataFrame containing the data
    - col1: Name of the first column to plot (required)
    - col2: Name of the second column to plot (optional)
    - col1_label: Optional label for col1 in the legend (defaults to col1 name)
    - col2_label: Optional label for col2 in the legend (defaults to col2 name)
    - title: Plot title
    - xlabel: X-axis label
    - ylabel: Y-axis label
    - figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Set default labels if not provided
    if col1_label is None:
        col1_label = col1
    
    # Process first column
    col1_data = df[col1]
    col1_p1 = col1_data.quantile(0.01)
    col1_p99 = col1_data.quantile(0.99)
    
    # Plot first column distribution
    sns.kdeplot(data=df, x=col1, label=col1_label, fill=True, alpha=0.3, color='blue')
    plt.axvline(col1_p1, color='blue', linestyle=':', alpha=0.7, 
                label=f'{col1_label} 1% ({col1_p1:.2f})')
    plt.axvline(col1_p99, color='blue', linestyle=':', alpha=0.7, 
                label=f'{col1_label} 99% ({col1_p99:.2f})')
    
    # Process and plot second column if provided
    if col2 is not None:
        if col2_label is None:
            col2_label = col2
            
        # Filter out zeros (like in original trade price example)
        col2_data = df[df[col2] != 0][col2]
        col2_p1 = col2_data.quantile(0.01)
        col2_p99 = col2_data.quantile(0.99)
        
        # Plot second column distribution
        sns.kdeplot(data=df[df[col2] != 0], x=col2, label=col2_label, fill=True, alpha=0.3, color='orange')
        plt.axvline(col2_p1, color='orange', linestyle='--', alpha=0.7, 
                    label=f'{col2_label} 1% ({col2_p1:.2f})')
        plt.axvline(col2_p99, color='orange', linestyle='--', alpha=0.7, 
                    label=f'{col2_label} 99% ({col2_p99:.2f})')
        
        # Adjust title if using default
        if title == 'Distribution Comparison with Percentiles':
            title = f'Distribution Comparison: {col1_label} vs {col2_label}'
    
    # Formatting
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()



def plot_price_distribution_comparison(
    data,
    price_columns=['price_mean', 'best_bid_mean', 'best_ask_mean'],
    box=True,
    points="all",
    title='Price Distribution Comparison',
    xaxis_title='Metric',
    yaxis_title='Price',
    show_meanline=True,
    show_figure=True
):
    """
    Creates a violin plot comparing price distributions of different metrics.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the price metrics
    - price_columns (list): List of column names to plot (default: price_mean, best_bid_mean, best_ask_mean)
    - box (bool): Whether to show box plot inside violin (default: True)
    - points (str): How to show points ("all", "outliers", False) (default: "all")
    - title (str): Plot title (default: 'Price Distribution Comparison')
    - xaxis_title (str): X-axis title (default: 'Metric')
    - yaxis_title (str): Y-axis title (default: 'Price')
    - show_meanline (bool): Whether to show mean line (default: True)
    - show_figure (bool): Whether to immediately show the figure (default: True)
    
    Returns:
    - fig: Plotly figure object
    """
    fig = px.violin(
        data,
        y=price_columns,
        box=box,
        points=points,
        title=title,
        labels={'value': yaxis_title, 'variable': xaxis_title}
    )
    
    if show_meanline:
        fig.update_traces(meanline_visible=True)
    
    if show_figure:
        fig.show()
    
    return fig


def plot_trade_duration(
    intervals: Union[List[float], pd.Series],
    title: str = "Distribution of Trade Intervals",
    xlabel: str = "Trade Interval (seconds)",
    ylabel: str = "Frequency",
    color: str = "royalblue",
    bins: int = 30,
    figsize: tuple = (10, 6),
    show_grid: bool = True,
    show_kde: bool = True,
    style: str = "whitegrid"
) -> None:
    """
    Plot a distribution of trade intervals with customizable styling.
    
    Parameters:
    -----------
    intervals : array-like
        List or Series of trade intervals
    title : str, optional
        Plot title (default: "Distribution of Trade Intervals")
    xlabel : str, optional
        X-axis label (default: "Trade Interval (seconds)")
    ylabel : str, optional
        Y-axis label (default: "Frequency")
    color : str, optional
        Color for the histogram (default: "royalblue")
    bins : int, optional
        Number of bins for histogram (default: 30)
    figsize : tuple, optional
        Figure size (default: (10, 6))
    show_grid : bool, optional
        Whether to show grid lines (default: True)
    show_kde : bool, optional
        Whether to show KDE curve (default: True)
    style : str, optional
        Seaborn style (default: "whitegrid")
    """
    # Set style
    sns.set_style(style)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot distribution
    ax = sns.histplot(
        intervals,
        kde=show_kde,
        color=color,
        bins=bins,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.8
    )
    
    # Add labels and title
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Add grid if requested
    if show_grid:
        plt.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()



def plot_exponential_qq(
    intervals: Union[List[float], np.ndarray, pd.Series],
    title: str = "Q-Q Plot: Trade Intervals vs Exponential Distribution",
    xlabel: str = "Theoretical Quantiles (Exponential Distribution)",
    ylabel: str = "Observed Quantiles (Trade Intervals)",
    point_color: str = "royalblue",
    line_color: str = "red",
    figsize: tuple = (8, 6),
    grid: bool = True
) -> None:
    """
    Creates an exponential Q-Q plot for trade interval analysis.
    
    Parameters:
    -----------
    intervals : array-like
        Observed trade intervals (list, numpy array, or pandas Series)
    title : str, optional
        Plot title (default: "Q-Q Plot...")
    xlabel : str, optional
        X-axis label (default: "Theoretical Quantiles...")
    ylabel : str, optional
        Y-axis label (default: "Observed Quantiles...")
    point_color : str, optional
        Color for scatter points (default: "royalblue")
    line_color : str, optional
        Color for reference line (default: "red")
    figsize : tuple, optional
        Figure dimensions (default: (8, 6))
    grid : bool, optional
        Whether to show grid (default: True)
    """
    # Convert to numpy array if not already
    observed = np.asarray(intervals)
    
    # Fit exponential distribution
    lambda_ = 1 / np.mean(observed)
    
    # Generate theoretical quantiles
    n = len(observed)
    theoretical = stats.expon.ppf(
        np.linspace(0, 1, n, endpoint=False),
        scale=1/lambda_
    )
    
    # Sort observed values
    observed_sorted = np.sort(observed)
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.scatter(theoretical, observed_sorted, color=point_color, alpha=0.7)
    
    # Add reference line
    max_val = max(theoretical.max(), observed_sorted.max())
    plt.plot(
        [0, max_val], 
        [0, max_val], 
        color=line_color, 
        linestyle='--',
        label='Perfect Fit'
    )
    
    # Add labels and styling
    plt.title(title, pad=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if grid:
        plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()