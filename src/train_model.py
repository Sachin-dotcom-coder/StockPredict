import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import joblib
import json
from datetime import datetime
try:
    from .backtest import BacktestEngine, INITIAL_CAPITAL, TRANSACTION_COST_PCT
except ImportError:
    from backtest import BacktestEngine, INITIAL_CAPITAL, TRANSACTION_COST_PCT

# -------------------------------
# CONFIG
# -------------------------------
STOCK_SYMBOLS = ["AAPL", "TSLA"]  # Train on both stocks for better generalization
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Enhanced feature set - includes more profitable indicators
FEATURES = [
    # Trend indicators
    "EMA_8",  # Fast EMA
    "EMA_10",  # Confirm EMA
    "EMA_20",  # Anchor EMA
    "EMA_8_20_Cross",  # Trend confirmation
    "Price_vs_EMA_8",  # Price position relative to fast EMA
    "Price_vs_EMA_20",  # Price position relative to anchor EMA
    
    # Momentum indicators
    "MACD",  # MACD line
    "Signal_Line",  # MACD signal
    "MACD_Histogram",  # Trend strength
    "RSI_14",  # Momentum oscillator
    "RSI_Oversold",  # Oversold signal
    "RSI_Overbought",  # Overbought signal
    "ROC_5",  # Rate of change
    "Momentum",  # Price momentum
    
    # Volatility indicators
    "BB_Position",  # Bollinger band position
    "BB_Width",  # Bollinger band width
    "ATR_Pct",  # Volatility measure
    
    # Volume indicators
    "Volume_Ratio",  # Volume confirmation
    "Volume_Trend",  # Volume trend
    
    # Price action
    "Minute_Return",  # Period return
    "Higher_Close",  # Price action pattern
]

TARGET = "Target"

# -------------------------------
# LOAD DATA - Combine both stocks for better generalization
# -------------------------------
print("\n" + "="*60)
print("Loading and combining data from multiple stocks...")
print("="*60)

all_dfs = []
for symbol in STOCK_SYMBOLS:
    data_path = Path(f"data/processed/{symbol.lower()}_features.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        df["Stock"] = symbol
        all_dfs.append(df)
        print(f"  Loaded {len(df)} samples from {symbol}")
    else:
        print(f"  [WARNING] File not found: {data_path}")

if not all_dfs:
    raise FileNotFoundError("No feature files found. Please run feature_engineering.py first.")

# Combine all stocks
df = pd.concat(all_dfs, ignore_index=True)
print(f"\n  Total samples: {len(df)}")

# Check for required columns
missing_features = [f for f in FEATURES if f not in df.columns]
if missing_features:
    print(f"\n[WARNING] Missing features: {missing_features}")
    print("Available features:", [c for c in df.columns if c not in ["Date", "Open", "High", "Low", "Close", "Volume", "Stock", "Target", "Future_Close", "Profit_Pct"]])
    # Use only available features
    FEATURES = [f for f in FEATURES if f in df.columns]
    print(f"Using {len(FEATURES)} features: {FEATURES}")

X = df[FEATURES]
y = df[TARGET]

# Calculate profit-based sample weights
# Weight samples by actual profit potential (higher weight = more profitable trades)
if "Profit_Pct" in df.columns:
    # Use absolute profit percentage, with higher weight for more profitable moves
    profit_weights = np.abs(df["Profit_Pct"].fillna(0))
    # Normalize to 0-2 range (1 = average, 2 = double average)
    profit_weights = 1 + (profit_weights / profit_weights.mean()) if profit_weights.mean() > 0 else np.ones(len(df))
    # Boost weight for positive profit trades
    profit_weights = np.where(df["Profit_Pct"].values > 0, profit_weights * 1.5, profit_weights)
    sample_weights = profit_weights
    print(f"  Profit-based weighting: min={sample_weights.min():.2f}, max={sample_weights.max():.2f}, mean={sample_weights.mean():.2f}")
else:
    # Fallback: use price change magnitude
    if "Future_Close" in df.columns and "Close" in df.columns:
        price_change = abs(df["Future_Close"] - df["Close"]) / df["Close"]
        sample_weights = (1 + price_change / price_change.mean()).fillna(1).values
    else:
        sample_weights = np.ones(len(df))
    print("  Using price change magnitude for weighting")

# Time-based split (NO shuffle for trading - preserve temporal order)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
weights_train = sample_weights[:split]
weights_test = sample_weights[split:]

print(f"\n  Train samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Positive class ratio (train): {y_train.mean():.2%}")
print(f"  Positive class ratio (test): {y_test.mean():.2%}")

# -------------------------------
# RANDOM FOREST
# -------------------------------
print("\n" + "="*60)
print("Training Random Forest...")
print("="*60)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train, sample_weight=weights_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]  # Probability of class 1 (BUY)

rf_acc = accuracy_score(y_test, rf_preds)
rf_precision = precision_score(y_test, rf_preds, zero_division=0)
rf_recall = recall_score(y_test, rf_preds, zero_division=0)
rf_f1 = f1_score(y_test, rf_preds, zero_division=0)

print("\n Random Forest Metrics:")
print(f"  Accuracy:  {rf_acc:.4f}")
print(f"  Precision: {rf_precision:.4f}")
print(f"  Recall:    {rf_recall:.4f}")
print(f"  F1 Score:  {rf_f1:.4f}")
print("\n Classification Report:")
print(classification_report(y_test, rf_preds, zero_division=0))

# -------------------------------
# XGBOOST
# -------------------------------
print("\n" + "="*60)
print("Training XGBoost...")
print("="*60)

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train, sample_weight=weights_train)
xgb_preds = xgb.predict(X_test)
xgb_probs = xgb.predict_proba(X_test)[:, 1]  # Probability of class 1 (BUY)

xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_precision = precision_score(y_test, xgb_preds, zero_division=0)
xgb_recall = recall_score(y_test, xgb_preds, zero_division=0)
xgb_f1 = f1_score(y_test, xgb_preds, zero_division=0)

print("\n XGBoost Metrics:")
print(f"  Accuracy:  {xgb_acc:.4f}")
print(f"  Precision: {xgb_precision:.4f}")
print(f"  Recall:    {xgb_recall:.4f}")
print(f"  F1 Score:  {xgb_f1:.4f}")
print("\n Classification Report:")
print(classification_report(y_test, xgb_preds, zero_division=0))

# -------------------------------
# MODEL SELECTION (Based on Profitability - Backtest Performance)
# -------------------------------
print("\n" + "="*60)
print("Model Selection (Profit-Based)...")
print("="*60)

# Create a small validation set from test data for profit-based selection
val_split = int(len(X_test) * 0.5)
X_val = X_test.iloc[:val_split]
y_val = y_test.iloc[:val_split]

# Need to create a temporary dataframe for backtesting
# Use test data with predictions
val_df = df.iloc[split:split+val_split].copy()

# Make predictions with both models
rf_val_probs = rf.predict_proba(X_val)[:, 1]
xgb_val_probs = xgb.predict_proba(X_val)[:, 1]

# Quick backtest to compare profitability
print("\n  Running quick backtests for model selection...")

def quick_backtest_profit(probabilities, df_subset, model_name):
    """Quick backtest to estimate profit."""
    try:
        engine = BacktestEngine(INITIAL_CAPITAL, TRANSACTION_COST_PCT)
        # Use a reasonable confidence threshold (0.6) and holding period (1)
        adjusted_preds = (probabilities >= 0.6).astype(int)
        result = engine.run_backtest(
            df_subset, adjusted_preds, probabilities,
            confidence_threshold=0.6,
            holding_periods=1,
            position_pct=1.0
        )
        return result["total_profit"]
    except Exception as e:
        print(f"    Error in {model_name} backtest: {e}")
        import traceback
        traceback.print_exc()
        return -float('inf')

rf_profit = quick_backtest_profit(rf_val_probs, val_df, "Random Forest")
xgb_profit = quick_backtest_profit(xgb_val_probs, val_df, "XGBoost")

print(f"  Random Forest validation profit: Rs {rf_profit:,.2f}")
print(f"  XGBoost validation profit: Rs {xgb_profit:,.2f}")

# Select model based on profitability (primary) and F1 score (secondary)
if xgb_profit > rf_profit:
    best_model = xgb
    best_name = "XGBoost"
    best_acc = xgb_acc
    best_f1 = xgb_f1
    best_probs = xgb_probs
    best_profit = xgb_profit
    print(f"\n Selected: {best_name} (Higher Profit)")
    print(f"  Validation Profit: Rs {best_profit:,.2f}")
    print(f"  Accuracy: {best_acc:.4f}")
    print(f"  F1 Score: {best_f1:.4f}")
    joblib.dump(xgb, MODEL_DIR / "scalping_model.pkl")
elif rf_profit > xgb_profit:
    best_model = rf
    best_name = "Random Forest"
    best_acc = rf_acc
    best_f1 = rf_f1
    best_probs = rf_probs
    best_profit = rf_profit
    print(f"\n Selected: {best_name} (Higher Profit)")
    print(f"  Validation Profit: Rs {best_profit:,.2f}")
    print(f"  Accuracy: {best_acc:.4f}")
    print(f"  F1 Score: {best_f1:.4f}")
    joblib.dump(rf, MODEL_DIR / "scalping_model.pkl")
else:
    # If profits are similar, use F1 score as tiebreaker
    if xgb_f1 > rf_f1:
        best_model = xgb
        best_name = "XGBoost"
        best_acc = xgb_acc
        best_f1 = xgb_f1
        best_probs = xgb_probs
        best_profit = xgb_profit
    else:
        best_model = rf
        best_name = "Random Forest"
        best_acc = rf_acc
        best_f1 = rf_f1
        best_probs = rf_probs
        best_profit = rf_profit
    print(f"\n Selected: {best_name} (Tiebreaker: F1 Score)")
    print(f"  Validation Profit: ₹{best_profit:,.2f}")
    print(f"  Accuracy: {best_acc:.4f}")
    print(f"  F1 Score: {best_f1:.4f}")
    joblib.dump(best_model, MODEL_DIR / "scalping_model.pkl")

# Save model metadata
metadata = {
    "model_type": best_name,
    "features": FEATURES,
    "training_date": datetime.now().isoformat(),
    "accuracy": float(best_acc),
    "f1_score": float(best_f1),
    "precision": float(xgb_precision if best_name == "XGBoost" else rf_precision),
    "recall": float(xgb_recall if best_name == "XGBoost" else rf_recall),
    "validation_profit": float(best_profit),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "stocks_used": STOCK_SYMBOLS,
    "feature_count": len(FEATURES)
}

with open(MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n" + "="*60)
print("Model training and selection complete!")
print(f"Model saved to: {MODEL_DIR / 'scalping_model.pkl'}")
print(f"Metadata saved to: {MODEL_DIR / 'model_metadata.json'}")
print("="*60)
