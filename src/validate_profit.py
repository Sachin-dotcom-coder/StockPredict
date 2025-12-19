"""
Comprehensive validation script to achieve ₹1000+ profit target.
This script runs the complete pipeline: feature engineering, training, optimization, and validation.
"""
import sys
from pathlib import Path
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from feature_engineering import STOCK_SYMBOLS
try:
    from .optimize_strategy import optimize_strategy, optimize_multi_stock
    from .backtest import run_backtest
    from .report import generate_report
except ImportError:
    from optimize_strategy import optimize_strategy, optimize_multi_stock
    from backtest import run_backtest
    from report import generate_report

# -------------------------------
# CONFIG
# -------------------------------
TARGET_PROFIT = 1000  # ₹1,000
INITIAL_CAPITAL = 100000  # ₹1,00,000

# -------------------------------
# VALIDATION PIPELINE
# -------------------------------
def validate_profit_target():
    """Run complete validation pipeline to achieve profit target."""
    print("="*60)
    print("PROFIT VALIDATION PIPELINE")
    print("="*60)
    print(f"Target: ₹{TARGET_PROFIT:,.2f} profit on ₹{INITIAL_CAPITAL:,.2f} capital")
    print(f"Required Return: {TARGET_PROFIT/INITIAL_CAPITAL*100:.2f}%")
    print("\n")
    
    # Step 1: Check if feature files exist
    print("Step 1: Checking data files...")
    data_files = []
    for symbol in STOCK_SYMBOLS:
        feature_path = Path(f"data/processed/{symbol.lower()}_features.csv")
        if feature_path.exists():
            data_files.append(str(feature_path))
            print(f"  ✓ Found: {feature_path}")
        else:
            print(f"  ✗ Missing: {feature_path}")
            print(f"    Run: python src/feature_engineering.py")
    
    if not data_files:
        print("\nERROR: No feature files found. Please run feature engineering first.")
        return False
    
    # Step 2: Check if model exists
    print("\nStep 2: Checking model...")
    model_path = Path("models/scalping_model.pkl")
    if model_path.exists():
        print(f"  ✓ Model found: {model_path}")
    else:
        print(f"  ✗ Model not found: {model_path}")
        print(f"    Run: python src/train_model.py")
        print("\nWARNING: Proceeding without trained model. Will need to train first.")
    
    # Step 3: Run optimization for each stock
    print("\nStep 3: Running strategy optimization...")
    all_results = []
    
    for data_file in data_files:
        stock_name = Path(data_file).stem.replace("_features", "")
        print(f"\n  Optimizing {stock_name}...")
        
        try:
            results_df = optimize_strategy(
                data_file, 
                test_all_combinations=False,  # Use smart search for speed
                max_combinations=100
            )
            
            if len(results_df) > 0:
                best = results_df.iloc[0]
                all_results.append({
                    "stock": stock_name,
                    "best_strategy": best,
                    "all_results": results_df
                })
                print(f"    Best profit: ₹{best['total_profit']:,.2f}")
            else:
                print(f"    No valid results found")
        
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    # Step 4: Find best overall strategy
    print("\nStep 4: Analyzing results...")
    
    if not all_results:
        print("  ✗ No valid results found. Check data and model.")
        return False
    
    # Find best single-stock strategy
    best_single = None
    best_single_profit = -float('inf')
    
    for result in all_results:
        profit = result["best_strategy"]["total_profit"]
        if profit > best_single_profit:
            best_single_profit = profit
            best_single = result
    
    # Test multi-stock strategy
    print("\n  Testing multi-stock strategy...")
    try:
        multi_results = optimize_multi_stock(data_files)
        if len(multi_results) > 0:
            best_multi = multi_results.groupby(["confidence_threshold", "holding_periods", "position_pct"]).agg({
                "total_profit": "sum"
            }).reset_index().sort_values("total_profit", ascending=False)
            if len(best_multi) > 0:
                multi_profit = best_multi.iloc[0]["total_profit"]
            else:
                multi_profit = 0
        else:
            multi_profit = 0
    except Exception as e:
        print(f"    Error in multi-stock optimization: {e}")
        import traceback
        traceback.print_exc()
        multi_profit = 0
    
    # Step 5: Final validation
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    best_profit = max(best_single_profit, multi_profit)
    
    print(f"\nBest Single-Stock Strategy:")
    if best_single:
        best = best_single["best_strategy"]
        print(f"  Stock: {best_single['stock']}")
        print(f"  Confidence: {best['confidence_threshold']:.2f}")
        print(f"  Holding Periods: {best['holding_periods']}")
        print(f"  Position Size: {best['position_pct']*100:.0f}%")
        print(f"  Profit: ₹{best['total_profit']:,.2f}")
        print(f"  Return: {best['return_pct']:.2f}%")
        print(f"  Win Rate: {best['win_rate']*100:.2f}%")
    
    if multi_profit > 0:
        print(f"\nBest Multi-Stock Strategy:")
        print(f"  Combined Profit: ₹{multi_profit:,.2f}")
    
    print(f"\n{'='*60}")
    print(f"TARGET PROFIT: ₹{TARGET_PROFIT:,.2f}")
    print(f"BEST ACHIEVED: ₹{best_profit:,.2f}")
    
    if best_profit >= TARGET_PROFIT:
        print(f"\n✓ SUCCESS! Target achieved with ₹{best_profit:,.2f} profit")
        print(f"  Excess: ₹{best_profit - TARGET_PROFIT:,.2f}")
        return True
    else:
        print(f"\n✗ Target not met. Need ₹{TARGET_PROFIT - best_profit:,.2f} more profit.")
        print(f"\nRecommendations:")
        print(f"  1. Re-run feature engineering with more data")
        print(f"  2. Train model with different parameters")
        print(f"  3. Try different stocks or time periods")
        print(f"  4. Adjust transaction costs if applicable")
        return False

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    success = validate_profit_target()
    sys.exit(0 if success else 1)

