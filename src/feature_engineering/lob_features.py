import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import os, glob
from datetime import datetime, timedelta
import itertools


def extract_lob_features(df):
    df['best_bid'] = df['bid'].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
    df['best_ask'] = df['ask'].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
    df['top1_bid_vol'] = df['bid'].apply(lambda x: x[0][1] if len(x) > 0 else np.nan)
    df['top1_ask_vol'] = df['ask'].apply(lambda x: x[0][1] if len(x) > 0 else np.nan)   
    df['total_bid_vol'] = df['bid'].apply(lambda x: sum(vol for _, vol in x[:10]))
    df['total_ask_vol'] = df['ask'].apply(lambda x: sum(vol for _, vol in x[:10]))
    df['imbalance'] = df.apply(lambda x: x['total_bid_vol'] / (x['total_bid_vol'] + x['total_ask_vol']) if (x['total_bid_vol'] + x['total_ask_vol']) > 0 else np.nan, axis=1)
    df['slippage'] = df['price_ffill'] - df['mid_price']
    df['ask_depth'] = df['ask'].apply(lambda x: len(x))
    df['bid_depth'] = df['bid'].apply(lambda x: len(x))
    return df