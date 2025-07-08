import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# -*- OHLC Plot with Vol -*-
def plot_ohlc_with_volume(df, 
                         title="Daily OHLC with Traded Volumes",
                         window_size=7,
                         volume_range_start=40000,
                         width=1100,
                         height=750):
    """
    Create an OHLC candlestick chart with volume bars and trend line
    
    Parameters:
    - df: DataFrame containing OHLC data and volume
    - title: Chart title (default: "Daily OHLC with Traded Volumes")
    - window_size: Moving average window for volume trend (default: 7)
    - volume_range_start: Starting value for volume axis (default: 40000)
    - width: Figure width (default: 1100)
    - height: Figure height (default: 750)
    """
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add candlestick trace (primary y-axis)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['price_open'],
        high=df['price_max'],
        low=df['price_min'],
        close=df['price_close'],
        name='Price',
        increasing_line_color='#2ca02c',
        decreasing_line_color='#d62728'
    ), secondary_y=False)

    # Add volume bars (secondary y-axis)
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['vol_sum'],
        name='Volume',
        opacity=0.2,
        marker_color='grey',
        marker_line_width=0
    ), secondary_y=True)

    # Add light grey trend line for volume
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['vol_sum'].rolling(window_size).mean(),
        name='Volume Trend',
        line=dict(color='rgba(100, 100, 100, 0.5)', width=2, dash='dot'),
        mode='lines',
        hoverinfo='skip',
    ), secondary_y=True)

    # Find the max and min prices with their dates
    max_price = df['price_max'].max()
    min_price = df['price_min'].min()
    max_date = df['price_max'].idxmax()
    min_date = df['price_min'].idxmin()

    # Customize layout
    fig.update_layout(
        title={
            'text': f'<b>{title}</b>',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18, family='Arial')
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=width,
        height=height,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        xaxis=dict(rangeslider=dict(visible=False))
    )

    # Primary y-axis (Price) settings
    fig.update_yaxes(
        title_text="<b>Price</b>",
        secondary_y=False,
        showgrid=True,
        gridcolor='rgba(0, 0, 0, 0.1)',
        title_font=dict(size=16, family='Arial'),
        tickfont=dict(size=14, family='Arial'),
        tickangle=0
    )

    # Secondary y-axis (Volume) settings
    fig.update_yaxes(
        title_text="<b>Volume</b>",
        secondary_y=True,
        showgrid=False,
        title_font=dict(size=16, family='Arial'),
        tickfont=dict(size=14, family='Arial'),
        range=[volume_range_start, df['vol_sum'].max() * 1.1],
        tickangle=0
    )

    # X-axis settings
    fig.update_xaxes(
        tickmode='auto',
        nticks=12,
        tickangle=0,
        tickfont=dict(size=14, family='Arial'),
        title=dict(text='<b>Date</b>', font=dict(size=16, family='Arial'))
    )

    # Add annotations for max and min prices
    fig.add_annotation(
        x=max_date, y=max_price,
        text=f"<b>Max: {max_price:,.2f}</b>",
        showarrow=True,
        arrowhead=1,
        ax=0, ay=-40,
        font=dict(size=14, family='Arial', color='black'),
        bgcolor='rgba(255, 255, 255, 0.8)'
    )

    fig.add_annotation(
        x=min_date, y=min_price,
        text=f"<b>Min: {min_price:,.2f}</b>",
        showarrow=True,
        arrowhead=1,
        ax=0, ay=40,
        font=dict(size=14, family='Arial', color='black'),
        bgcolor='rgba(255, 255, 255, 0.8)'
    )

    return fig
