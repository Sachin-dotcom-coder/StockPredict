import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

# -------------------------------
# CONFIG
# -------------------------------
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "scalping_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# -------------------------------
# LOAD MODEL AND METADATA
# -------------------------------
def load_model():
    """Load the trained model and its metadata."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
    
    model = joblib.load(MODEL_PATH)
    
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
    
    return model, metadata

# -------------------------------
# PREDICT FUNCTION
# -------------------------------
def predict(df, model=None, features=None, return_proba=True):
    """
    Make predictions on new data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with feature columns
    model : sklearn model (optional)
        Pre-loaded model. If None, will load from disk.
    features : list (optional)
        List of feature names. If None, will load from metadata.
    return_proba : bool
        If True, return probabilities. If False, return binary predictions.
    
    Returns:
    --------
    predictions : numpy.ndarray or pandas.Series
        Binary predictions (0 or 1) if return_proba=False
        Probabilities (0 to 1) if return_proba=True
    """
    if model is None:
        model, metadata = load_model()
        if features is None:
            if "features" in metadata and metadata["features"]:
                features = metadata["features"]
            else:
                # Default features from train_model.py
                features = ["EMA_8", "EMA_10", "EMA_20", "MACD", "Signal_Line", "Minute_Return"]
    else:
        if features is None:
            # Default features from train_model.py
            features = ["EMA_8", "EMA_10", "EMA_20", "MACD", "Signal_Line", "Minute_Return"]
    
    # Ensure all required features are present
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Extract features
    X = df[features]
    
    # Make predictions
    if return_proba:
        predictions = model.predict_proba(X)[:, 1]  # Probability of class 1 (BUY)
    else:
        predictions = model.predict(X)
    
    return predictions

# -------------------------------
# BATCH PREDICT
# -------------------------------
def predict_batch(data_path, output_path=None, confidence_threshold=0.5):
    """
    Make predictions on a CSV file with features.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to CSV file with features
    output_path : str or Path (optional)
        Path to save predictions. If None, returns DataFrame.
    confidence_threshold : float
        Threshold for binary classification (0.5 default)
    
    Returns:
    --------
    results : pandas.DataFrame
        DataFrame with original data + predictions
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Load model
    model, metadata = load_model()
    features = metadata.get("features", [])
    
    # Make predictions
    probabilities = predict(df, model, features, return_proba=True)
    binary_preds = (probabilities >= confidence_threshold).astype(int)
    
    # Add predictions to dataframe
    df["Prediction_Probability"] = probabilities
    df["Prediction"] = binary_preds
    df["Prediction_Label"] = df["Prediction"].map({1: "BUY", 0: "SELL"})
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
    
    return df

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <data_path> [output_path] [confidence_threshold]")
        print("Example: python predict.py data/processed/aapl_features.csv predictions.csv 0.6")
        sys.exit(1)
    
    data_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    confidence_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    print(f"Loading data from: {data_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    results = predict_batch(data_path, output_path, confidence_threshold)
    
    print(f"\nPredictions Summary:")
    print(f"  Total samples: {len(results)}")
    print(f"  BUY signals: {results['Prediction'].sum()}")
    print(f"  SELL signals: {(results['Prediction'] == 0).sum()}")
    print(f"  Average probability: {results['Prediction_Probability'].mean():.4f}")
    
    if output_path:
        print(f"\nResults saved to: {output_path}")

