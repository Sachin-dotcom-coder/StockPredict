"""
Simple script to check if we're achieving ₹1000+ profit target.
Run this to get output for screenshot.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from backtest import run_backtest, BacktestEngine
    from predict import load_model, predict
    from optimize_strategy import optimize_strategy
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# -------------------------------
# CONFIG
# -------------------------------
TARGET_PROFIT = 1000
INITIAL_CAPITAL = 100000

# -------------------------------
# MAIN CHECK
# -------------------------------
def check_profit():
    """Check profit on available data files."""
    print("="*70)
    print("PROFIT CHECK - Trading Bot Performance")
    print("="*70)
    print(f"Target: Rs {TARGET_PROFIT:,} profit on Rs {INITIAL_CAPITAL:,} capital")
    print(f"Required Return: {TARGET_PROFIT/INITIAL_CAPITAL*100:.2f}%")
    print("\n")
    
    # Find data files
    data_dir = Path("data/processed")
    data_files = list(data_dir.glob("*_features.csv"))
    
    if not data_files:
        print("ERROR: No feature files found in data/processed/")
        print("Please run: python src/feature_engineering.py")
        return False
    
    print(f"Found {len(data_files)} data file(s):")
    for f in data_files:
        print(f"  - {f.name}")
    print("\n")
    
    # Check model
    model_path = Path("models/scalping_model.pkl")
    if not model_path.exists():
        print("ERROR: Model not found!")
        print("Please run: python src/train_model.py")
        return False
    
    print("Model found: [OK]")
    print("\n" + "="*70)
    print("RUNNING OPTIMIZATION TO FIND BEST STRATEGY...")
    print("="*70)
    print("\n")
    
    best_overall_profit = -float('inf')
    best_result = None
    best_stock = None
    
    # Test each stock
    for data_file in data_files:
        stock_name = data_file.stem.replace("_features", "").upper()
        print(f"\n{'='*70}")
        print(f"OPTIMIZING: {stock_name}")
        print(f"{'='*70}\n")
        
        try:
            # Run optimization (smart search - faster)
            results_df = optimize_strategy(
                str(data_file),
                test_all_combinations=False,
                max_combinations=50
            )
            
            if len(results_df) > 0:
                best = results_df.iloc[0]
                profit = best['total_profit']
                
                print(f"\n{'='*70}")
                print(f"BEST STRATEGY FOR {stock_name}:")
                print(f"{'='*70}")
                print(f"  Confidence Threshold: {best['confidence_threshold']:.2f}")
                print(f"  Holding Periods: {best['holding_periods']}")
                print(f"  Position Size: {best['position_pct']*100:.0f}%")
                print(f"  Total Trades: {int(best['total_trades'])}")
                print(f"  Winning Trades: {int(best['winning_trades'])}")
                print(f"  Losing Trades: {int(best['losing_trades'])}")
                print(f"  Win Rate: {best['win_rate']*100:.2f}%")
                print(f"\n  TOTAL PROFIT: Rs {profit:,.2f}")
                print(f"  Return: {best['return_pct']:.2f}%")
                print(f"  Final Equity: Rs {best['final_equity']:,.2f}")
                print(f"  Avg Profit/Trade: Rs {best['avg_profit_per_trade']:,.2f}")
                print(f"  Sharpe Ratio: {best['sharpe_ratio']:.2f}")
                print(f"  Max Drawdown: {best['max_drawdown']:.2f}%")
                
                if profit > best_overall_profit:
                    best_overall_profit = profit
                    best_result = best
                    best_stock = stock_name
                
                if profit >= TARGET_PROFIT:
                    print(f"\n  [SUCCESS] TARGET ACHIEVED! Rs {profit:,.2f} >= Rs {TARGET_PROFIT:,}")
                else:
                    shortfall = TARGET_PROFIT - profit
                    print(f"\n  [INFO] Target not met. Need Rs {shortfall:,.2f} more")
            else:
                print(f"No valid results for {stock_name}")
        
        except Exception as e:
            print(f"Error processing {stock_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final Summary
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    if best_result is not None:
        print(f"\nBEST OVERALL PERFORMANCE: {best_stock}")
        print(f"  Total Profit: Rs {best_overall_profit:,.2f}")
        print(f"  Return: {best_result['return_pct']:.2f}%")
        print(f"  Strategy: Confidence={best_result['confidence_threshold']:.2f}, "
              f"Hold={best_result['holding_periods']}, "
              f"Size={best_result['position_pct']*100:.0f}%")
        
        print(f"\n{'='*70}")
        print(f"TARGET: Rs {TARGET_PROFIT:,}")
        print(f"ACHIEVED: Rs {best_overall_profit:,.2f}")
        
        if best_overall_profit >= TARGET_PROFIT:
            excess = best_overall_profit - TARGET_PROFIT
            print(f"\n{'='*70}")
            print("[SUCCESS] TARGET ACHIEVED!")
            print(f"{'='*70}")
            print(f"Profit exceeds target by: Rs {excess:,.2f}")
            print(f"Return: {best_result['return_pct']:.2f}%")
            return True
        else:
            shortfall = TARGET_PROFIT - best_overall_profit
            print(f"\n{'='*70}")
            print("[INFO] TARGET NOT MET")
            print(f"{'='*70}")
            print(f"Shortfall: Rs {shortfall:,.2f}")
            print(f"Current Return: {best_result['return_pct']:.2f}%")
            print(f"Required Return: {TARGET_PROFIT/INITIAL_CAPITAL*100:.2f}%")
            return False
    else:
        print("\nERROR: No valid results found")
        return False

if __name__ == "__main__":
    success = check_profit()
    print("\n")
    sys.exit(0 if success else 1)

