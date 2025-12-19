import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import json

# -------------------------------
# REPORTING FUNCTIONS
# -------------------------------
def generate_report(backtest_results: dict, output_dir: str = "reports", 
                   stock_name: str = "Unknown") -> dict:
    """
    Generate comprehensive performance report.
    
    Parameters:
    -----------
    backtest_results : dict
        Results from BacktestEngine.calculate_metrics()
    output_dir : str
        Directory to save reports
    stock_name : str
        Name of stock/symbol
    
    Returns:
    --------
    report : dict
        Report summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Extract data
    trades = backtest_results.get("trades", [])
    equity_curve = backtest_results.get("equity_curve", [])
    
    # Create report summary
    report = {
        "stock": stock_name,
        "summary": {
            "total_trades": backtest_results.get("total_trades", 0),
            "winning_trades": backtest_results.get("winning_trades", 0),
            "losing_trades": backtest_results.get("losing_trades", 0),
            "win_rate": backtest_results.get("win_rate", 0),
            "total_profit": backtest_results.get("total_profit", 0),
            "final_equity": backtest_results.get("final_equity", 0),
            "return_pct": backtest_results.get("return_pct", 0),
            "avg_profit_per_trade": backtest_results.get("avg_profit_per_trade", 0),
            "sharpe_ratio": backtest_results.get("sharpe_ratio", 0),
            "max_drawdown": backtest_results.get("max_drawdown", 0),
            "profit_factor": backtest_results.get("profit_factor", 0)
        },
        "generated_at": datetime.now().isoformat()
    }
    
    # Save summary as JSON
    summary_path = output_path / f"{stock_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Create trades DataFrame and save
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_path = output_path / f"{stock_name}_trades.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"Trades saved to: {trades_path}")
    
    # Generate charts
    if equity_curve:
        generate_charts(equity_curve, trades, output_path, stock_name)
    
    # Print summary
    print_report(report["summary"], stock_name)
    
    return report

def print_report(summary: dict, stock_name: str):
    """Print formatted report to console."""
    print("\n" + "="*60)
    print(f"PERFORMANCE REPORT: {stock_name}")
    print("="*60)
    print(f"\nTRADE STATISTICS:")
    print(f"  Total Trades: {summary['total_trades']}")
    print(f"  Winning Trades: {summary['winning_trades']}")
    print(f"  Losing Trades: {summary['losing_trades']}")
    print(f"  Win Rate: {summary['win_rate']*100:.2f}%")
    
    print(f"\nPROFIT & LOSS:")
    print(f"  Total Profit: ₹{summary['total_profit']:,.2f}")
    print(f"  Final Equity: ₹{summary['final_equity']:,.2f}")
    print(f"  Return: {summary['return_pct']:.2f}%")
    print(f"  Avg Profit per Trade: ₹{summary['avg_profit_per_trade']:,.2f}")
    
    print(f"\nRISK METRICS:")
    print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {summary['max_drawdown']:.2f}%")
    print(f"  Profit Factor: {summary['profit_factor']:.2f}")
    print("="*60)

def generate_charts(equity_curve: list, trades: list, output_path: Path, stock_name: str):
    """Generate performance charts."""
    if not equity_curve:
        return
    
    # Convert equity curve to DataFrame
    equity_df = pd.DataFrame(equity_curve)
    
    # Convert time to datetime if needed
    if "time" in equity_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(equity_df["time"]):
            equity_df["time"] = pd.to_datetime(equity_df["time"])
        equity_df = equity_df.set_index("time")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Performance Report: {stock_name}", fontsize=16, fontweight="bold")
    
    # 1. Equity Curve
    ax1 = axes[0, 0]
    ax1.plot(equity_df.index, equity_df["equity"], linewidth=2, label="Equity", color="blue")
    ax1.axhline(y=100000, color="green", linestyle="--", alpha=0.7, label="Initial Capital")
    ax1.set_title("Equity Curve", fontweight="bold")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Equity (₹)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
    
    # Format x-axis dates
    if isinstance(equity_df.index, pd.DatetimeIndex):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    # 2. Drawdown Chart
    ax2 = axes[0, 1]
    equity_values = equity_df["equity"].values
    peak = equity_values[0]
    drawdowns = []
    for equity in equity_values:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak * 100
        drawdowns.append(drawdown)
    
    ax2.fill_between(equity_df.index, drawdowns, 0, alpha=0.3, color="red")
    ax2.plot(equity_df.index, drawdowns, linewidth=1, color="red")
    ax2.set_title("Drawdown", fontweight="bold")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)
    
    if isinstance(equity_df.index, pd.DatetimeIndex):
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    # 3. Trade P&L Distribution
    ax3 = axes[1, 0]
    if trades:
        trades_df = pd.DataFrame(trades)
        profits = trades_df["net_profit"].values
        
        ax3.hist(profits, bins=30, edgecolor="black", alpha=0.7, color="green")
        ax3.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Break Even")
        ax3.axvline(x=profits.mean(), color="blue", linestyle="--", linewidth=2, label=f"Mean: ₹{profits.mean():,.2f}")
        ax3.set_title("Trade P&L Distribution", fontweight="bold")
        ax3.set_xlabel("Profit/Loss (₹)")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
    else:
        ax3.text(0.5, 0.5, "No trades to display", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Trade P&L Distribution", fontweight="bold")
    
    # 4. Cumulative Profit
    ax4 = axes[1, 1]
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df["cumulative_profit"] = trades_df["net_profit"].cumsum()
        
        ax4.plot(range(len(trades_df)), trades_df["cumulative_profit"], 
                marker="o", markersize=3, linewidth=2, color="green")
        ax4.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax4.axhline(y=1000, color="blue", linestyle="--", alpha=0.7, label="Target: ₹1,000")
        ax4.set_title("Cumulative Profit", fontweight="bold")
        ax4.set_xlabel("Trade Number")
        ax4.set_ylabel("Cumulative Profit (₹)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
    else:
        ax4.text(0.5, 0.5, "No trades to display", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Cumulative Profit", fontweight="bold")
    
    plt.tight_layout()
    
    # Save chart
    chart_path = output_path / f"{stock_name}_performance_charts.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    print(f"Charts saved to: {chart_path}")
    plt.close()

def generate_comparison_report(results_list: list, output_dir: str = "reports"):
    """Generate comparison report for multiple strategies."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create comparison DataFrame
    comparison_data = []
    for result in results_list:
        summary = result.get("summary", {})
        comparison_data.append({
            "Strategy": result.get("stock", "Unknown"),
            "Total Trades": summary.get("total_trades", 0),
            "Win Rate": f"{summary.get('win_rate', 0)*100:.2f}%",
            "Total Profit": f"₹{summary.get('total_profit', 0):,.2f}",
            "Return %": f"{summary.get('return_pct', 0):.2f}%",
            "Sharpe Ratio": f"{summary.get('sharpe_ratio', 0):.2f}",
            "Max Drawdown": f"{summary.get('max_drawdown', 0):.2f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison
    comparison_path = output_path / "strategy_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    print(comparison_df.to_string(index=False))
    print(f"\nComparison saved to: {comparison_path}")
    
    return comparison_df

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    import sys
    try:
        from .backtest import run_backtest
    except ImportError:
        from backtest import run_backtest
    
    if len(sys.argv) < 2:
        print("Usage: python report.py <data_path> [confidence_threshold] [holding_periods]")
        print("Example: python report.py data/processed/aapl_features.csv 0.6 1")
        sys.exit(1)
    
    data_path = sys.argv[1]
    confidence_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    holding_periods = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    stock_name = Path(data_path).stem.replace("_features", "")
    
    print(f"Running backtest and generating report for {stock_name}...")
    
    # Run backtest
    results = run_backtest(data_path, confidence_threshold=confidence_threshold,
                          holding_periods=holding_periods)
    
    # Generate report
    report = generate_report(results, stock_name=stock_name)

