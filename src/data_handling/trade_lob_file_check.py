# This script matches trade files with lob files and reports any dates where LOB files exist without corresponding trade files.

import os
import glob

# Define the directories containing LOB and trade files
lob_dir = r"C:\Users\rashm\Documents\UoB\TB-2\DSMP\project\dsmp-2024-groupt33\data\processed\all_lobs"
trade_dir = r"C:\Users\rashm\Documents\UoB\TB-2\DSMP\project\dsmp-2024-groupt33\data\processed\all_trades"

# Recursively find all LOB and trade files in their respective directories
lob_files = glob.glob(os.path.join(lob_dir, 'LoB_*.csv'))
trade_files = glob.glob(os.path.join(trade_dir, 'Trade_*.csv'))

# Extract dates from LOB and trade file names
lob_dates = {os.path.basename(file).split('_')[1][:10] for file in lob_files}
trade_dates = {os.path.basename(file).split('_')[1][:10] for file in trade_files}

# Check for missing trade files
missing_dates = lob_dates - trade_dates
#missing_dates = trade_dates - lob_dates


if missing_dates:
    print("The following dates have LOB files but no corresponding trade files:")
    for date in sorted(missing_dates):
        print(f"- {date}")
else:
    print("All LOB files have corresponding trade files.")