import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def plot_levels_liquidity_with_spread(daily_metrics, title="Liquidity and Spread by Levels (six months)") -> go.Figure:
    """
    Creates a combined plot of liquidity levels and bid-ask spread.
    
    Parameters:
    - daily_metrics (DataFrame): DataFrame containing the data with columns:
        * date: x-axis values
        * liquidity_{1-5}_mean: y-values for liquidity levels
        * bid_ask_spread: y-values for spread (plotted on secondary axis)
    - title (str): Plot title (default: "Liquidity and Spread by Levels (six months)")
    
    Returns:
    - fig: Plotly figure object
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add top 5 level liquidity traces
    for i in range(1, 6):
        fig.add_trace(go.Scatter(
            x=daily_metrics['date'],
            y=daily_metrics[f'liquidity_{i}_mean'],
            name=f'Level {i} Liquidity',
            mode='lines+markers'
        ))
    
    # Add spread trace on secondary axis
    fig.add_trace(go.Bar(
        x=daily_metrics['date'],
        y=daily_metrics['bid_ask_spread'],
        name='Spread',
        opacity=0.3,
        marker_color='grey'
    ), secondary_y=True)
    
    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        yaxis_title='Liquidity (Volume/Spread)',
        yaxis2_title='Spread',
        hovermode='x unified',
        margin=dict(l=30, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_bid_ask_mid_with_spread(daily_metrics: pd.DataFrame) -> go.Figure:
    """
    Plots the daily mean of best bid, mid price, and best ask prices as lines,
    with the spread between best bid and ask shown as a shaded area.

    Parameters:
    - daily_metrics (pd.DataFrame): Must include columns 'date', 'best_bid_mean',
      'mid_price_mean', and 'best_ask_mean'.

    Returns:
    - plotly.graph_objects.Figure: The constructed figure.
    """
    # Reshape data for line plot
    heatmap_data = daily_metrics.melt(
        id_vars=['date'],
        value_vars=['best_bid_mean', 'mid_price_mean', 'best_ask_mean'],
        var_name='Price Type',
        value_name='price'
    )

    # Create the line plot
    fig = px.line(
        heatmap_data,
        x='date',
        y='price',
        color='Price Type',
        title='Daily Mean of Bid-Ask-Mid Prices for six months',
        labels={'price': 'Prices', 'date': 'Date'},
        template='plotly_white'
    )

    # Add bid-ask spread as shaded area
    fig.add_trace(go.Scatter(
        x=daily_metrics['date'].tolist() + daily_metrics['date'].tolist()[::-1],
        y=daily_metrics['best_ask_mean'].tolist() + daily_metrics['best_bid_mean'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(100,100,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Bid-Ask Spread'
    ))

    # Update layout
    fig.update_layout(
        hovermode='x unified',
        width=800,
        legend=dict(
            x=0.82,
            y=0.98,
            bgcolor='rgba(255,255,255,0.5)',
        ),
        xaxis=dict(
            tickfont=dict(size=12),
            title=dict(text='Date', font=dict(size=14, family='Arial', color='black'))
        ),
        yaxis=dict(
            tickfont=dict(size=12),
            title=dict(text='Prices', font=dict(size=14, family='Arial', color='black'))
        ),
        title=dict(
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        margin=dict(l=40, r=10, t=60, b=40),
    )

    return fig

