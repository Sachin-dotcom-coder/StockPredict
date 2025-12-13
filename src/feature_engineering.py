import pandas as pd
from pathlib import Path

# -------------------------------
# Load Raw AAPL Data
# -------------------------------
input_path = Path("data/raw/aapl_ohlc.csv")
df = pd.read_csv(input_path)

# REMOVE CORRUPT ROWS (Fix for TSLA row with strings)
df = df[pd.to_numeric(df["Open"], errors="coerce").notnull()]

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Ensure numeric types
numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# -------------------------------
# FEATURE ENGINEERING FOR SCALPING (1-MINUTE DATA)
# -------------------------------

# Adjusted windows for minute-level scalping (faster signals for quick trades)

# 1. Simple Moving Averages (shorter windows for scalping)
df["SMA_5"] = df["Close"].rolling(window=5).mean()   # 5-minute SMA
df["SMA_10"] = df["Close"].rolling(window=10).mean()  # 10-minute SMA
df["SMA_20"] = df["Close"].rolling(window=20).mean()  # 20-minute SMA

# 2. Exponential Moving Averages (faster for scalping)
df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()   # 5-minute EMA
df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()  # 10-minute EMA
df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()  # 20-minute EMA

# 3. RSI (Relative Strength Index) - 14 minutes for scalping
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df["RSI_14"] = 100 - (100 / (1 + rs))

# 4. MACD (adjusted for faster scalping signals)
df["MACD"] = df["EMA_5"] - df["EMA_10"]  # Faster MACD for scalping
df["Signal_Line"] = df["MACD"].ewm(span=3, adjust=False).mean()  # Faster signal line

# 5. Bollinger Bands (10-minute window for scalping)
df["BB_Mid"] = df["Close"].rolling(window=10).mean()
df["BB_Std"] = df["Close"].rolling(window=10).std()
df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]

# 6. Minute Returns (instead of daily returns)
df["Minute_Return"] = df["Close"].pct_change()

# 7. Create Target Column (BUY/SELL)
# If next minute's close > current minute's close → BUY (1), else SELL (0)
df["Future_Close"] = df["Close"].shift(-1)
df["Target"] = (df["Future_Close"] > df["Close"]).astype(int)

# Remove rows with NaN values caused by indicators
df = df.dropna()

# -------------------------------
# SAVE PROCESSED DATA
# -------------------------------
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "aapl_features.csv"
df.to_csv(output_path, index=False)

print("[SUCCESS] Feature engineering completed!")
print("Saved to:", output_path)
print(df.head())
