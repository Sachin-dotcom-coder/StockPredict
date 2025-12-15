import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "data/processed/aapl_features.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# CHANGE FEATURES HERE 
FEATURES = [
    "EMA_8", #fast
    "EMA_10", #confirm
    "EMA_20", #anchor
    "MACD",
    "Signal_Line",
    "Minute_Return"
]

TARGET = "Target"

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df[TARGET]

# Time-based split (NO shuffle for trading)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# -------------------------------
# RANDOM FOREST
# -------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

print("\n Random Forest Accuracy:", round(rf_acc, 4))
print(classification_report(y_test, rf_preds))

# -------------------------------
# XGBOOST
# -------------------------------
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)

print("\n XGBoost Accuracy:", round(xgb_acc, 4))
print(classification_report(y_test, xgb_preds))

# -------------------------------
# MODEL SELECTION
# -------------------------------
if xgb_acc > rf_acc:
    print(f"\n Saving XGBoost (Accuracy: {xgb_acc:.4f})")
    joblib.dump(xgb, MODEL_DIR / "scalping_model.pkl")
else:
    print(f"\n Saving Random Forest (Accuracy: {rf_acc:.4f})")
    joblib.dump(rf, MODEL_DIR / "scalping_model.pkl")

print("\n Model training and selection complete. Best model saved.")
