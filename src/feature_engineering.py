import pandas as pd
from pathlib import Path

# -------------------------------
# CONFIG
# -------------------------------
STOCK_SYMBOLS = ["AAPL", "TSLA"]
TRANSACTION_COST_PCT = 0.0004  # 0.04% per trade (Indian market average)
MIN_PROFIT_THRESHOLD = 0.0015  # 0.15% minimum move to be profitable (0.08% costs + 0.07% buffer)

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
    
    # 7. Volume-based Indicators
    df["Volume_EMA_10"] = df["Volume"].ewm(span=10, adjust=False).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_EMA_10"]  # Current volume vs average
    df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Trend"] = (df["Volume"] > df["Volume_SMA_20"]).astype(int)  # 1 if above average
    
    # 8. ATR (Average True Range) - Volatility indicator
    high_low = df["High"] - df["Low"]
    high_close = abs(df["High"] - df["Close"].shift(1))
    low_close = abs(df["Low"] - df["Close"].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_14"] = true_range.rolling(window=14).mean()
    df["ATR_Pct"] = df["ATR_14"] / df["Close"]  # ATR as percentage of price
    
    # 9. Momentum Indicators
    df["ROC_5"] = df["Close"].pct_change(periods=5)  # Rate of Change over 5 periods
    df["ROC_10"] = df["Close"].pct_change(periods=10)  # Rate of Change over 10 periods
    df["Momentum"] = df["Close"] - df["Close"].shift(5)  # Price momentum
    
    # 10. Price Action Patterns
    df["Higher_High"] = (df["High"] > df["High"].shift(1)).astype(int)
    df["Lower_Low"] = (df["Low"] < df["Low"].shift(1)).astype(int)
    df["Higher_Close"] = (df["Close"] > df["Close"].shift(1)).astype(int)
    
    # 11. Bollinger Band Position
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])  # 0-1 scale
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]  # Band width as % of price
    
    # 12. RSI-based signals
    df["RSI_Overbought"] = (df["RSI_14"] > 70).astype(int)
    df["RSI_Oversold"] = (df["RSI_14"] < 30).astype(int)
    
    # 13. MACD Histogram (difference between MACD and Signal)
    df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]
    
    # 14. EMA Crossovers
    df["EMA_8_20_Cross"] = (df["EMA_8"] > df["EMA_20"]).astype(int)  # Golden cross indicator
    df["EMA_10_20_Cross"] = (df["EMA_10"] > df["EMA_20"]).astype(int)
    
    # 15. Price relative to EMAs
    df["Price_vs_EMA_8"] = (df["Close"] - df["EMA_8"]) / df["EMA_8"]
    df["Price_vs_EMA_20"] = (df["Close"] - df["EMA_20"]) / df["EMA_20"]

    # 16. Create Target Column (BUY/SELL) - Profit-Based
    # Only mark as BUY (1) if the move is profitable after transaction costs
    # Round trip cost = 2 * TRANSACTION_COST_PCT (0.08%), use MIN_PROFIT_THRESHOLD for safety
    df["Future_Close"] = df["Close"].shift(-1)
    
    # Calculate price change percentage
    price_change_pct = (df["Future_Close"] - df["Close"]) / df["Close"]
    
    # Target = 1 only if price change exceeds minimum profit threshold (profitable after costs)
    # Target = 0 otherwise (including small moves that would lose money due to transaction costs)
    df["Target"] = (price_change_pct > MIN_PROFIT_THRESHOLD).astype(int)
    
    # Also store the actual profit percentage for weighting during training
    df["Profit_Pct"] = price_change_pct

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
