import pandas as pd
from pathlib import Path
import sys
import yfinance as yf # Import the yfinance library

# -----------------------------
# CONFIG
# -----------------------------
STOCK_SYMBOL = "AAPL"          # Yahoo Finance uses uppercase for ticker symbols
START_DATE = "2023-01-01"
END_DATE = "2023-06-01"

# -----------------------------
# YFINANCE DATA DOWNLOAD
# -----------------------------

print(f"Downloading data for {STOCK_SYMBOL} from Yahoo Finance...")

try:
    # Use the yf.download function for daily data
    # The 'end' date is exclusive, which is standard behavior
    data = yf.download(
        tickers=STOCK_SYMBOL, 
        start=START_DATE, 
        end=END_DATE, 
        interval="1d"
    )
except Exception as e:
    print("❌ Network or API error while downloading data")
    print(e)
    sys.exit(1)

if data.empty:
    print(f"❌ No data received from Yahoo Finance for {STOCK_SYMBOL} in this date range")
    sys.exit(1)

# Reset the index to convert the 'Date' (which is the index by default) into a column
data.reset_index(inplace=True)

# -----------------------------
# COLUMN CLEANING AND FILTERING
# -----------------------------

# Rename 'Adj Close' to 'Adj_Close' to eliminate the space in the column name.
# Note: For accurate backtesting, Adj_Close is often preferred over Close.
data.rename(
    columns={
        "Adj Close": "Adj_Close"
    },
    inplace=True
)

# Keep only the desired OHLCV columns and Date
# This line now safely selects the columns because they are single-level strings.
data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]

# Ensure output directory exists
# Creates 'data/raw' relative to the script's execution directory
output_dir = Path('./data/raw')
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"{STOCK_SYMBOL.lower()}_ohlc.csv" # Lowercase for consistency
data.to_csv(output_path, index=False)

print("✅ Data downloaded successfully!")
print(f"✅ Saved to: {output_path.resolve()}")
print("\n--- Data Head ---")
print(data.head())