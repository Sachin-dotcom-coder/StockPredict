import pandas as pd
from pathlib import Path
import sys
import yfinance as yf # Import the yfinance library

# -----------------------------
# CONFIG
# -----------------------------
STOCK_SYMBOL = "AAPL"          # Yahoo Finance uses uppercase for ticker symbols
# For 1-minute data, Yahoo Finance provides maximum ~7 days of historical data
# Using period="7d" to get maximum available minute-level data
PERIOD = "7d"  # Maximum period for 1-minute interval data

# -----------------------------
# YFINANCE DATA DOWNLOAD
# -----------------------------

print(f"Downloading 1-minute data for {STOCK_SYMBOL} from Yahoo Finance...")
print(f"Note: Yahoo Finance provides maximum ~7 days of 1-minute historical data")

try:
    # Use the yf.download function for 1-minute scalping data
    # Using period="7d" to get maximum available minute-level data
    data = yf.download(
        tickers=STOCK_SYMBOL, 
        period=PERIOD,  # Maximum period for minute data
        interval="1m"   # 1-minute interval for scalping
    )
except Exception as e:
    print("[ERROR] Network or API error while downloading data")
    print(e)
    sys.exit(1)

if data.empty:
    print(f"[ERROR] No data received from Yahoo Finance for {STOCK_SYMBOL} in this date range")
    sys.exit(1)

# Reset the index to convert the datetime index into a column
data.reset_index(inplace=True)

# Handle MultiIndex columns (yfinance sometimes returns columns with ticker name)
# Flatten the column names if they're MultiIndex
if isinstance(data.columns, pd.MultiIndex):
    # Get the actual column names from the first level (OHLCV names)
    # Level 0 has the column names, Level 1 has the ticker symbol
    data.columns = data.columns.get_level_values(0)  # Use first level (OHLCV names)

# Handle column name - minute data might use 'Datetime' instead of 'Date'
if 'Datetime' in data.columns:
    data.rename(columns={'Datetime': 'Date'}, inplace=True)

# -----------------------------
# COLUMN CLEANING AND FILTERING
# -----------------------------

# Rename 'Adj Close' to 'Adj_Close' to eliminate the space in the column name.
# Note: For accurate backtesting, Adj_Close is often preferred over Close.
if "Adj Close" in data.columns:
    data.rename(
        columns={
            "Adj Close": "Adj_Close"
        },
        inplace=True
    )

# Keep only the desired OHLCV columns and Date
# Select columns that exist (some might not be present in minute data)
available_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
data = data[[col for col in available_cols if col in data.columns]]

# Ensure output directory exists
# Creates 'data/raw' relative to the script's execution directory
output_dir = Path('./data/raw')
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"{STOCK_SYMBOL.lower()}_ohlc.csv" # Lowercase for consistency
data.to_csv(output_path, index=False)

print("[SUCCESS] Data downloaded successfully!")
print(f"[SUCCESS] Saved to: {output_path.resolve()}")
print("\n--- Data Head ---")
print(data.head())