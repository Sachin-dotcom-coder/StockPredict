import pandas as pd
from pathlib import Path

# -------------------------------
# CONFIG
# -------------------------------
STOCK_SYMBOLS = ["AAPL", "TSLA"]

# -------------------------------
# FEATURE ENGINEERING FOR 5-MINUTE INTERVAL DATA
# -------------------------------

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Process each stock
for STOCK_SYMBOL in STOCK_SYMBOLS:
    print(f"\n{'='*60}")
    print(f"Processing features for {STOCK_SYMBOL}...")
    print(f"{'='*60}\n")
    
    # Load Raw Data
    input_path = Path(f"data/raw/{STOCK_SYMBOL.lower()}_ohlc.csv")
    
    if not input_path.exists():
        print(f"[WARNING] File not found: {input_path}")
        print(f"[WARNING] Skipping {STOCK_SYMBOL}")
        continue
    
    df = pd.read_csv(input_path)

    # REMOVE CORRUPT ROWS (Fix for TSLA row with strings)
    df = df[pd.to_numeric(df["Open"], errors="coerce").notnull()]

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Ensure numeric types
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Adjusted windows for 5-minute interval data

    # 1. Simple Moving Averages
    df["SMA_5"] = df["Close"].rolling(window=5).mean()   # 5-period SMA (25 minutes)
    df["SMA_10"] = df["Close"].rolling(window=10).mean()  # 10-period SMA (50 minutes)
    df["SMA_20"] = df["Close"].rolling(window=20).mean()  # 20-period SMA (100 minutes)

    # 2. Exponential Moving Averages
    df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()   # 5-period EMA (25 minutes)
    df["EMA_8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()  # 10-period EMA (50 minutes)
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_15"] = df["Close"].ewm(span=15, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()  # 20-period EMA (100 minutes)
    df["EMA_25"] = df["Close"].ewm(span=25, adjust=False).mean()
    df["EMA_30"] = df["Close"].ewm(span=30, adjust=False).mean()
    # 3. RSI (Relative Strength Index) - 14 periods
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # 4. MACD
    df["MACD"] = df["EMA_5"] - df["EMA_10"]
    df["Signal_Line"] = df["MACD"].ewm(span=3, adjust=False).mean()

    # 5. Bollinger Bands (10-period window)
    df["BB_Mid"] = df["Close"].rolling(window=10).mean()
    df["BB_Std"] = df["Close"].rolling(window=10).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]

    # 6. Period Returns
    df["Minute_Return"] = df["Close"].pct_change()

    # 7. Create Target Column (BUY/SELL)
    # If next period's close > current period's close â†’ BUY (1), else SELL (0)
    df["Future_Close"] = df["Close"].shift(-1)
    df["Target"] = (df["Future_Close"] > df["Close"]).astype(int)

    # Remove rows with NaN values caused by indicators
    df = df.dropna()

    # Save processed data
    output_path = output_dir / f"{STOCK_SYMBOL.lower()}_features.csv"
    df.to_csv(output_path, index=False)

    print(f"[SUCCESS] Feature engineering completed for {STOCK_SYMBOL}!")
    print(f"Saved to: {output_path}")
    print(f"Data shape: {df.shape}")
    print(f"\n--- Data Head for {STOCK_SYMBOL} ---")
    print(df.head())

print(f"\n{'='*60}")
print("[SUCCESS] All feature engineering completed!")
print(f"{'='*60}")
