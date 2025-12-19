import pandas as pd
import numpy as np
from pathlib import Path
import json
from itertools import product
from typing import Optional, List
import time
try:
    from .backtest import BacktestEngine, run_backtest
    from .predict import load_model, predict
except ImportError:
    from backtest import BacktestEngine, run_backtest
    from predict import load_model, predict

# -------------------------------
# CONFIG
# -------------------------------
INITIAL_CAPITAL = 100000  # ₹1,00,000
TARGET_PROFIT = 1000  # ₹1,000
TRANSACTION_COST_PCT = 0.0004  # 0.04%

# -------------------------------
# OPTIMIZATION PARAMETERS - Expanded Search Space
# -------------------------------
# More granular confidence thresholds for better optimization
CONFIDENCE_THRESHOLDS = [0.5, 0.55, 0.6, 0.62, 0.65, 0.67, 0.7, 0.72, 0.75, 0.77, 0.8, 0.82, 0.85, 0.87, 0.9, 0.92, 0.95]
# Extended holding periods to capture longer trends
HOLDING_PERIODS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
# More position sizing options including partial positions
POSITION_PCTS = [0.25, 0.5, 0.75, 1.0]  # Use 25%, 50%, 75%, or 100% of capital per trade

# -------------------------------
# OPTIMIZATION FUNCTIONS
# -------------------------------
def optimize_strategy(data_path: str, output_path: Optional[str] = None,
                     test_all_combinations: bool = True,
                     max_combinations: int = 100) -> pd.DataFrame:
    """
    Optimize trading strategy by testing different parameter combinations.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file with features
    output_path : str (optional)
        Path to save optimization results
    test_all_combinations : bool
        If True, test all combinations. If False, use smart search.
    max_combinations : int
        Maximum combinations to test if test_all_combinations=False
    
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame with all tested combinations and their results
    """
    print("="*60)
    print("STRATEGY OPTIMIZATION")
    print("="*60)
    print(f"Data: {data_path}")
    print(f"Initial Capital: Rs {INITIAL_CAPITAL:,.2f}")
    print(f"Target Profit: Rs {TARGET_PROFIT:,.2f}")
    print("\nTesting parameter combinations...")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Load model and make predictions
    model, metadata = load_model()
    features = metadata.get("features", [])
    if not features:
        # Default features from train_model.py
        features = ["EMA_8", "EMA_10", "EMA_20", "MACD", "Signal_Line", "Minute_Return"]
    probabilities = predict(df, model, features, return_proba=True)
    predictions = (probabilities >= 0.5).astype(int)
    
    # Generate parameter combinations
    if test_all_combinations:
        combinations = list(product(CONFIDENCE_THRESHOLDS, HOLDING_PERIODS, POSITION_PCTS))
        print(f"Total combinations to test: {len(combinations)}")
    else:
        # Smart search: focus on promising regions with more comprehensive coverage
        combinations = []
        # Test all confidence thresholds with holding_period=1 (most common for scalping)
        for conf in CONFIDENCE_THRESHOLDS:
            combinations.append((conf, 1, 1.0))
        # Test all holding periods with confidence=0.6 (balanced threshold)
        for hold in HOLDING_PERIODS:
            combinations.append((0.6, hold, 1.0))
        # Test all confidence thresholds with holding_period=2 (slightly longer)
        for conf in CONFIDENCE_THRESHOLDS[::2]:  # Every other threshold
            combinations.append((conf, 2, 1.0))
        # Test position sizes with common params
        for pos in POSITION_PCTS:
            combinations.append((0.6, 1, pos))
            combinations.append((0.7, 1, pos))
            combinations.append((0.75, 2, pos))
        # Test longer holding periods with higher confidence
        for hold in [5, 10, 15, 20]:
            combinations.append((0.7, hold, 1.0))
            combinations.append((0.75, hold, 1.0))
        # Add strategic combinations (high confidence + short hold, medium confidence + medium hold)
        combinations.append((0.8, 1, 1.0))
        combinations.append((0.85, 1, 1.0))
        combinations.append((0.65, 3, 1.0))
        combinations.append((0.7, 4, 1.0))
        # Add some random combinations for exploration
        np.random.seed(42)
        remaining = max_combinations - len(combinations)
        if remaining > 0:
            for _ in range(remaining):
                conf = np.random.choice(CONFIDENCE_THRESHOLDS)
                hold = np.random.choice(HOLDING_PERIODS)
                pos = np.random.choice(POSITION_PCTS)
                combinations.append((conf, hold, pos))
        combinations = list(set(combinations))  # Remove duplicates
        print(f"Testing {len(combinations)} combinations (smart search)")
    
    # Test each combination
    results = []
    start_time = time.time()
    
    for i, (conf_thresh, hold_periods, position_pct) in enumerate(combinations):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {i+1}/{len(combinations)} ({elapsed:.1f}s elapsed)")
        
        try:
            engine = BacktestEngine(INITIAL_CAPITAL, TRANSACTION_COST_PCT)
            
            # Adjust predictions based on confidence threshold
            adjusted_predictions = (probabilities >= conf_thresh).astype(int)
            
            result = engine.run_backtest(
                df, adjusted_predictions, probabilities,
                conf_thresh, hold_periods, position_pct
            )
            
            results.append({
                "confidence_threshold": conf_thresh,
                "holding_periods": hold_periods,
                "position_pct": position_pct,
                "total_trades": result["total_trades"],
                "winning_trades": result["winning_trades"],
                "losing_trades": result["losing_trades"],
                "win_rate": result["win_rate"],
                "total_profit": result["total_profit"],
                "final_equity": result["final_equity"],
                "return_pct": result["return_pct"],
                "avg_profit_per_trade": result["avg_profit_per_trade"],
                "sharpe_ratio": result["sharpe_ratio"],
                "max_drawdown": result["max_drawdown"],
                "profit_factor": result["profit_factor"],
                "target_achieved": result["total_profit"] >= TARGET_PROFIT
            })
        except Exception as e:
            print(f"Error testing combination {i+1}: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by total profit
    results_df = results_df.sort_values("total_profit", ascending=False)
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    if len(results_df) > 0:
        best = results_df.iloc[0]
        print(f"\nBest Strategy:")
        print(f"  Confidence Threshold: {best['confidence_threshold']:.2f}")
        print(f"  Holding Periods: {best['holding_periods']}")
        print(f"  Position Size: {best['position_pct']*100:.0f}%")
        print(f"  Total Profit: Rs {best['total_profit']:,.2f}")
        print(f"  Return: {best['return_pct']:.2f}%")
        print(f"  Win Rate: {best['win_rate']*100:.2f}%")
        print(f"  Total Trades: {int(best['total_trades'])}")
        print(f"  Sharpe Ratio: {best['sharpe_ratio']:.2f}")
        
        if best['target_achieved']:
            print(f"\n[SUCCESS] TARGET ACHIEVED! Profit of Rs {best['total_profit']:,.2f} >= Rs {TARGET_PROFIT:,.2f}")
        else:
            print(f"\n[INFO] Target not met. Best profit: Rs {best['total_profit']:,.2f}")
        
        # Show top 10 strategies
        print(f"\nTop 10 Strategies:")
        print(results_df.head(10)[["confidence_threshold", "holding_periods", "position_pct", 
                                   "total_profit", "return_pct", "win_rate", "total_trades"]].to_string(index=False))
        
        # Strategies that achieved target
        target_achieved = results_df[results_df["target_achieved"]]
        if len(target_achieved) > 0:
            print(f"\n{len(target_achieved)} strategies achieved the target profit of Rs {TARGET_PROFIT:,.2f}")
    
    # Save results
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    return results_df

# -------------------------------
# MULTI-STOCK OPTIMIZATION
# -------------------------------
def optimize_multi_stock(data_paths: List[str], output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Optimize strategy across multiple stocks.
    
    Parameters:
    -----------
    data_paths : List[str]
        List of paths to CSV files for different stocks
    output_path : str (optional)
        Path to save results
    
    Returns:
    --------
    results_df : pandas.DataFrame
        Combined results from all stocks
    """
    print("="*60)
    print("MULTI-STOCK OPTIMIZATION")
    print("="*60)
    
    all_results = []
    
    for data_path in data_paths:
        stock_name = Path(data_path).stem.replace("_features", "")
        print(f"\nOptimizing for {stock_name}...")
        
        results_df = optimize_strategy(data_path, test_all_combinations=False, max_combinations=50)
        results_df["stock"] = stock_name
        all_results.append(results_df)
    
    # Combine results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Find best overall strategy
    best_overall = combined_results.groupby(["confidence_threshold", "holding_periods", "position_pct"]).agg({
        "total_profit": "sum",
        "total_trades": "sum",
        "win_rate": "mean",
        "sharpe_ratio": "mean"
    }).reset_index().sort_values("total_profit", ascending=False)
    
    print("\n" + "="*60)
    print("BEST OVERALL STRATEGY (Combined)")
    print("="*60)
    if len(best_overall) > 0:
        best = best_overall.iloc[0]
        print(f"  Confidence Threshold: {best['confidence_threshold']:.2f}")
        print(f"  Holding Periods: {best['holding_periods']}")
        print(f"  Position Size: {best['position_pct']*100:.0f}%")
        print(f"  Combined Profit: Rs {best['total_profit']:,.2f}")
        print(f"  Combined Trades: {int(best['total_trades'])}")
        print(f"  Avg Win Rate: {best['win_rate']*100:.2f}%")
        
        if best['total_profit'] >= TARGET_PROFIT:
            print(f"\n[SUCCESS] TARGET ACHIEVED! Combined profit of Rs {best['total_profit']:,.2f} >= Rs {TARGET_PROFIT:,.2f}")
    
    if output_path:
        combined_results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    return combined_results

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python optimize_strategy.py <data_path> [output_path] [--multi-stock]")
        print("Example: python optimize_strategy.py data/processed/aapl_features.csv results.csv")
        print("Example: python optimize_strategy.py data/processed/ --multi-stock")
        sys.exit(1)
    
    data_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    multi_stock = "--multi-stock" in sys.argv
    
    if multi_stock or Path(data_path).is_dir():
        # Multi-stock optimization
        if Path(data_path).is_dir():
            data_paths = list(Path(data_path).glob("*_features.csv"))
        else:
            data_paths = [data_path]
        
        if not data_paths:
            print(f"No feature files found in {data_path}")
            sys.exit(1)
        
        optimize_multi_stock(data_paths, output_path)
    else:
        # Single stock optimization
        optimize_strategy(data_path, output_path)

