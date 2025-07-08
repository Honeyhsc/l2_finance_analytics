import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import ast
import os, glob

def convert_to_list(x):
    try:
        return ast.literal_eval(x.strip()) if pd.notna(x) else []
    except:
        return []
