import pandas as pd
from pathlib import Path

# -------------------------------
# Load Raw TSLA Data
# -------------------------------
input_path = Path("data/raw/tsla_ohlc.csv")
df = pd.read_csv(input_path)

import pandas as pd
from pathlib import Path

input_path = Path("data/raw/tsla_ohlc.csv")
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
# FEATURE ENGINEERING
# -------------------------------

# 1. Simple Moving Averages
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()

# 2. Exponential Moving Averages
df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

# 3. RSI (Relative Strength Index)
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df["RSI_14"] = 100 - (100 / (1 + rs))

# 4. MACD
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

# 5. Bollinger Bands
df["BB_Mid"] = df["Close"].rolling(window=20).mean()
df["BB_Std"] = df["Close"].rolling(window=20).std()
df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]

# 6. Daily Returns
df["Daily_Return"] = df["Close"].pct_change()

# 7. Create Target Column (BUY/SELL)
# If tomorrow's close > today's close → BUY (1), else SELL (0)
df["Future_Close"] = df["Close"].shift(-1)
df["Target"] = (df["Future_Close"] > df["Close"]).astype(int)

# Remove rows with NaN values caused by indicators
df = df.dropna()

# -------------------------------
# SAVE PROCESSED DATA
# -------------------------------
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "tsla_features.csv"
df.to_csv(output_path, index=False)

print("✅ Feature engineering completed!")
print("Saved to:", output_path)
print(df.head())
