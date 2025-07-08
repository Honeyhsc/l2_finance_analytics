import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates
import numpy as np
from datetime import time

def plot_burst_events(
    df,
    time_col='datetime',
    value_col='price',
    burst_col=None,
    idle_col='long_idle_cluster',
    vol_col=None
):
    """
    Plots a line chart of values (e.g. price) with burst/idle events marked as points.
    Handles non-continuous datetimes by plotting with equal spacing.

    Args:
        df: DataFrame with datetime data
        time_col: Name of datetime column
        value_col: Column to plot as line (e.g., price)
        burst_col/idle_col: Boolean columns for events
        vol_col: Optional volume column for secondary axis
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Use numeric x-values to treat time as categorical
    x = range(len(df))
    y = df[value_col]
    labels = df[time_col]

    # Optimized price line rendering
    ax.plot(x, y, 
            color='#1f77b4', 
            alpha=0.5, 
            linewidth=0.3,
            label=value_col,
            rasterized=True)  # Critical for performance
    
    # Plot event markers
    event_styles = []
    if burst_col:
        event_styles.append((df[burst_col], '#d62728', 'o', 'Burst', 15, 0.7))
    if idle_col:
        event_styles.append((df[idle_col], '#2ca02c', 'D', 'Idle Cluster', 25, 1.0))
    
    for mask, color, marker, label, size, alpha in event_styles:
        ax.scatter(
            [i for i, m in zip(x, mask) if m],
            df.loc[mask, value_col],
            color=color,
            marker=marker,
            s=size,
            alpha=alpha,
            edgecolors='black' if marker == 'D' else 'none',
            linewidth=0.5 if marker == 'D' else 0,
            label=label,
            zorder=3
        )
    
    # Original axis formatting with improvements
    ax.set_xticks(x)
    ax.set_xticklabels(labels.dt.strftime('%m-%d %H:%M'), 
                      rotation=45, 
                      ha='right',
                      fontsize=9)
    
    # Smart label reduction
    if len(x) > 20:
        visible_labels = 20
        step = max(1, len(x) // visible_labels)
        for i, label in enumerate(ax.get_xticklabels()):
            if i % step != 0:
                label.set_visible(False)
    
    # Enhanced volume plot
    if vol_col:
        ax2 = ax.twinx()
        ax2.bar(x, df[vol_col], 
                color='#7f7f7f', 
                alpha=0.08, 
                width=0.8,
                label='Volume',
                rasterized=True)
        ax2.set_ylabel('Volume', color='#7f7f7f')
        ax2.tick_params(axis='y', labelcolor='#7f7f7f')
        ax2.set_ylim(0, df[vol_col].max() * 1.5)
    
    # Professional styling
    #ax.grid(False, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.grid(False)
    ax.set_ylabel(value_col)
    
    # Unified legend
    handles, labels = ax.get_legend_handles_labels()
    if vol_col:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
    ax.legend(handles, labels, 
              loc='upper left',
              frameon=True,
              framealpha=1,
              edgecolor='black')
    
    plt.tight_layout()
    plt.show()


def plot_binned_stats(x_values, y_values, 
                     y_errors=None,
                     counts=None,
                     show_error_bar=False,
                     show_counts=False,
                     title="Average Future Return vs. Inter-Trade Interval",
                     x_label="Inter-Trade Interval (sec, log-binned)",
                     y_label="Mean Future Return",
                     figsize=(12, 6),
                     marker='o',
                     color='steelblue',
                     marker_size=8,
                     line_style='-',
                     line_width=1.5,
                     grid=True,
                     grid_style='--',
                     grid_alpha=0.4,
                     title_fontsize=16,
                     label_fontsize=14,
                     tick_fontsize=12,
                     tilt_x_ticks=False,
                     rotation=45,
                     ha='right',
                     add_trendline=False,
                     trendline_color='salmon',
                     ax=None,
                     **kwargs):
    """
    Extended binned statistics plot with optional error bars and count overlay.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot main line with optional error bars
    if show_error_bar and y_errors is not None:
        ax.errorbar(x_values, y_values, yerr=y_errors, fmt=marker+line_style, 
                    color=color, markersize=marker_size, linewidth=line_width,
                    label=y_label, **kwargs)
    else:
        ax.plot(x_values, y_values, 
                marker=marker, 
                color=color,
                markersize=marker_size,
                linestyle=line_style,
                linewidth=line_width,
                label=y_label,
                **kwargs)

    # Add trendline if requested
    if add_trendline:
        numeric_x = np.arange(len(x_values))  # For non-numeric labels
        z = np.polyfit(numeric_x, y_values, 1)
        p = np.poly1d(z)
        ax.plot(x_values, p(numeric_x), 
                color=trendline_color, 
                linestyle='--',
                linewidth=1,
                alpha=0.7,
                label='Trendline')
        ax.legend(fontsize=label_fontsize - 1)

    # Count bar plot on secondary axis
    if show_counts and counts is not None:
        ax2 = ax.twinx()
        ax2.bar(x_values, counts, width=0.5, alpha=0.2, color='gray', label='Sample Count')
        ax2.set_ylabel("Sample Count", fontsize=label_fontsize)
        ax2.tick_params(axis='y', labelsize=tick_fontsize)
        ax2.legend(loc='upper right', fontsize=label_fontsize - 2)

    # Title and labels
    ax.set_title(title, fontsize=title_fontsize, pad=15)
    ax.set_xlabel(x_label, fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel(y_label, fontsize=label_fontsize, labelpad=10)

    # Grid and ticks
    if grid:
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha)

    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    if tilt_x_ticks:
        plt.setp(ax.get_xticklabels(), 
                 rotation=rotation, 
                 ha=ha,
                 rotation_mode='anchor')

    plt.tight_layout()
    plt.show()

    return


# def plot_time_series(
#     df: pd.DataFrame,
#     time_col: str = 'time',
#     value_col: str = 'value',
#     title: str = 'Time Series Plot',
#     ylabel: str = 'Value',
#     figsize: tuple = (12, 6),
#     color: str = 'royalblue',
#     time_format: str = '%H:%M',
#     rot: int = 45,
#     grid: bool = True
# ) -> plt.Axes:
#     """
#     Simple time-value plot without aggregation or filtering.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         DataFrame with time and value columns
#     time_col : str
#         Column containing time data (datetime or time objects)
#     value_col : str
#         Column to plot on y-axis
#     title, ylabel : str
#         Plot title and y-axis label
#     figsize : tuple
#         Figure size (width, height)
#     color : str
#         Line color
#     time_format : str
#         Format for x-axis time labels (e.g., '%H:%M')
#     rot : int
#         Rotation angle for x-tick labels
#     grid : bool
#         Whether to show grid lines
#     """
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Convert time data if needed
#     if isinstance(df[time_col].iloc[0], time):
#         # Convert datetime.time to plottable format
#         x = pd.to_datetime(df[time_col].astype(str))
#     else:
#         x = pd.to_datetime(df[time_col])
    
#     ax.plot(x, df[value_col], color=color)
    
#     # Format x-axis as time
#     ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
#     plt.setp(ax.get_xticklabels(), rotation=rot, ha='right')
    
#     ax.set_title(title, pad=20)
#     ax.set_ylabel(ylabel)
#     ax.set_xlabel('Time')
#     if grid:
#         ax.grid(True, linestyle='--', alpha=0.6)
    
#     plt.tight_layout()
#     plt.show()

#     return 



from typing import Optional

def plot_time_series(
    df: pd.DataFrame,
    time_col: str = 'time',
    primary_col: str = 'value',
    secondary_col: Optional[str] = None,
    title: str = 'Time Series Plot',
    primary_ylabel: str = 'Primary Values',
    secondary_ylabel: Optional[str] = None,
    figsize: tuple = (14, 7),
    primary_color: str = 'blue',
    secondary_color: str = 'green',
    time_format: str = '%H:%M',
    rot: int = 45,
    grid: bool = True,
    title_fontsize: int = 16,
    label_fontsize: int = 14,
    tick_fontsize: int = 12
) -> plt.Axes:
    """
    Enhanced time series plot with dual-axis support and professional styling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with time and value columns
    time_col : str
        Column containing time data
    primary_col : str
        Primary y-axis column
    secondary_col : str, optional
        Secondary y-axis column
    title : str
        Plot title
    primary_ylabel, secondary_ylabel : str
        Axis labels
    figsize : tuple
        Figure size
    primary_color, secondary_color : str
        Line colors
    time_format : str
        Format for x-axis time labels
    rot : int
        Rotation angle for x-tick labels
    grid : bool
        Whether to show grid
    title_fontsize, label_fontsize, tick_fontsize : int
        Font sizes for plot elements
    """
    plt.style.use(plt.style.available[12]) # Using bright
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Convert time data
    if isinstance(df[time_col].iloc[0], time):
        x = pd.to_datetime(df[time_col].astype(str))
    else:
        x = pd.to_datetime(df[time_col])
    
    # Primary axis plot
    ax1.plot(x, df[primary_col], color=primary_color, linewidth=2, marker='o', linestyle='-', label=primary_ylabel)
    ax1.set_ylabel(primary_ylabel, fontsize=label_fontsize, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    # Secondary axis if specified
    if secondary_col:
        ax2 = ax1.twinx()
        ax2.plot(x, df[secondary_col], color=secondary_color, 
                linestyle='--', linewidth=2, marker='s', label=secondary_ylabel)
        ax2.set_ylabel(secondary_ylabel or secondary_col, 
                      fontsize=label_fontsize, fontweight='bold', color=secondary_color)
        ax2.tick_params(axis='y', labelcolor=secondary_color, labelsize=tick_fontsize)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
    plt.setp(ax1.get_xticklabels(), rotation=rot, ha='right', fontsize=tick_fontsize)
    
    # Title and labels
    ax1.set_title(title, pad=20, fontsize=title_fontsize, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=label_fontsize, fontweight='bold')
    
    # Grid and legend
    if grid:
        ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Combine legends if dual axis
    lines1, labels1 = ax1.get_legend_handles_labels()
    if secondary_col:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, 
                  fontsize=label_fontsize-2, loc='upper left')
    
    plt.tight_layout()
    plt.show()

    return