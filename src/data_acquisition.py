import pandas as pd
from pathlib import Path
import sys
import yfinance as yf # Import the yfinance library

# -----------------------------
# CONFIG
# -----------------------------
STOCK_SYMBOLS = ["AAPL", "TSLA"]  # Yahoo Finance uses uppercase for ticker symbols
# For 5-minute data, Yahoo Finance provides maximum ~60 days of historical data
# Using period="60d" to get maximum available 5-minute interval data
PERIOD = "60d"  # Maximum period for 5-minute interval data

# -----------------------------
# YFINANCE DATA DOWNLOAD
# -----------------------------

# Ensure output directory exists
output_dir = Path('./data/raw')
output_dir.mkdir(parents=True, exist_ok=True)

# Download data for each stock
for STOCK_SYMBOL in STOCK_SYMBOLS:
    print(f"\n{'='*60}")
    print(f"Downloading 5-minute data for {STOCK_SYMBOL} from Yahoo Finance...")
    print(f"Note: Yahoo Finance provides maximum ~60 days of 5-minute historical data")
    print(f"{'='*60}\n")

    try:
        # Use the yf.download function for 5-minute interval data
        # Using period="60d" to get maximum available 5-minute interval data
        data = yf.download(
            tickers=STOCK_SYMBOL, 
            period=PERIOD,  # Maximum period for 5-minute data
            interval="5m"   # 5-minute interval
        )
    except Exception as e:
        print(f"[ERROR] Network or API error while downloading data for {STOCK_SYMBOL}")
        print(e)
        continue  # Skip to next stock instead of exiting

    if data.empty:
        print(f"[ERROR] No data received from Yahoo Finance for {STOCK_SYMBOL} in this date range")
        continue  # Skip to next stock

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

    output_path = output_dir / f"{STOCK_SYMBOL.lower()}_ohlc.csv" # Lowercase for consistency
    data.to_csv(output_path, index=False)

    print(f"[SUCCESS] Data downloaded successfully for {STOCK_SYMBOL}!")
    print(f"[SUCCESS] Saved to: {output_path.resolve()}")
    print(f"\n--- Data Head for {STOCK_SYMBOL} ---")
    print(data.head())
    print(f"\n--- Data Shape for {STOCK_SYMBOL}: {data.shape} ---")

print(f"\n{'='*60}")
print("[SUCCESS] All downloads completed!")
print(f"{'='*60}")
