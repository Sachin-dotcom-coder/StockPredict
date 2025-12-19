import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
try:
    from .predict import load_model, predict
except ImportError:
    from predict import load_model, predict

# -------------------------------
# CONFIG
# -------------------------------
INITIAL_CAPITAL = 100000  # ₹1,00,000
TRANSACTION_COST_PCT = 0.0004  # 0.04% per trade (Indian market average)
TARGET_PROFIT = 1000  # ₹1,000 target profit

# -------------------------------
# BACKTEST ENGINE
# -------------------------------
class BacktestEngine:
    """Comprehensive backtesting engine for trading strategies."""
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL, 
                 transaction_cost_pct: float = TRANSACTION_COST_PCT):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.reset()
    
    def reset(self):
        """Reset the backtest state."""
        self.cash = self.initial_capital
        self.position = 0  # Number of shares held
        self.position_entry_price = 0.0
        self.position_entry_time = None
        self.trades = []  # List of all completed trades
        self.equity_curve = []  # Track equity over time
        self.current_time = None
    
    def calculate_transaction_cost(self, trade_value: float) -> float:
        """Calculate transaction cost for a trade."""
        return trade_value * self.transaction_cost_pct
    
    def enter_position(self, price: float, time, shares: Optional[int] = None, 
                      position_pct: float = 1.0):
        """
        Enter a position (BUY).
        
        Parameters:
        -----------
        price : float
            Entry price
        time : datetime or timestamp
            Entry time
        shares : int (optional)
            Number of shares. If None, uses position_pct of available capital.
        position_pct : float
            Percentage of capital to use (0.0 to 1.0)
        """
        if self.position > 0:
            return False  # Already in position
        
        if shares is None:
            # Calculate shares based on position_pct
            available_capital = self.cash * position_pct
            shares = int(available_capital / price)
        
        if shares == 0:
            return False  # Not enough capital
        
        trade_value = shares * price
        transaction_cost = self.calculate_transaction_cost(trade_value)
        total_cost = trade_value + transaction_cost
        
        if total_cost > self.cash:
            return False  # Not enough cash
        
        self.cash -= total_cost
        self.position = shares
        self.position_entry_price = price
        self.position_entry_time = time
        self.current_time = time
        
        return True
    
    def exit_position(self, price: float, time, reason: str = "Signal"):
        """
        Exit current position (SELL).
        
        Parameters:
        -----------
        price : float
            Exit price
        time : datetime or timestamp
            Exit time
        reason : str
            Reason for exit (e.g., "Signal", "StopLoss", "TakeProfit")
        """
        if self.position == 0:
            return False  # No position to exit
        
        trade_value = self.position * price
        transaction_cost = self.calculate_transaction_cost(trade_value)
        net_proceeds = trade_value - transaction_cost
        
        # Calculate profit/loss
        gross_profit = (price - self.position_entry_price) * self.position
        net_profit = gross_profit - transaction_cost - self.calculate_transaction_cost(
            self.position * self.position_entry_price
        )
        
        # Record trade
        trade = {
            "entry_time": self.position_entry_time,
            "exit_time": time,
            "entry_price": self.position_entry_price,
            "exit_price": price,
            "shares": self.position,
            "gross_profit": gross_profit,
            "transaction_cost": transaction_cost + self.calculate_transaction_cost(
                self.position * self.position_entry_price
            ),
            "net_profit": net_profit,
            "return_pct": (price - self.position_entry_price) / self.position_entry_price * 100,
            "holding_periods": None,  # Will be calculated
            "reason": reason
        }
        
        self.trades.append(trade)
        
        # Update cash
        self.cash += net_proceeds
        
        # Reset position
        self.position = 0
        self.position_entry_price = 0.0
        self.position_entry_time = None
        self.current_time = time
        
        return True
    
    def update_equity(self, current_price: float, time):
        """Update equity curve with current portfolio value."""
        if self.position > 0:
            current_value = self.position * current_price
            equity = self.cash + current_value
        else:
            equity = self.cash
        
        self.equity_curve.append({
            "time": time,
            "equity": equity,
            "cash": self.cash,
            "position_value": self.position * current_price if self.position > 0 else 0,
            "position": self.position
        })
    
    def get_current_equity(self, current_price: float) -> float:
        """Get current total equity."""
        if self.position > 0:
            return self.cash + (self.position * current_price)
        return self.cash
    
    def run_backtest(self, df: pd.DataFrame, predictions: np.ndarray, 
                    probabilities: Optional[np.ndarray] = None,
                    confidence_threshold: float = 0.5,
                    holding_periods: int = 1,
                    position_pct: float = 1.0,
                    use_probability: bool = True) -> Dict:
        """
        Run backtest on historical data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLC data and features
        predictions : numpy.ndarray
            Binary predictions (0 or 1)
        probabilities : numpy.ndarray (optional)
            Prediction probabilities
        confidence_threshold : float
            Minimum probability to enter trade
        holding_periods : int
            Number of periods to hold position (1 = exit next period)
        position_pct : float
            Percentage of capital to use per trade
        use_probability : bool
            If True, use probabilities for entry. If False, use binary predictions.
        
        Returns:
        --------
        results : dict
            Backtest results with metrics and trades
        """
        self.reset()
        
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")
        
        if "Date" in df.columns:
            time_col = "Date"
        elif "Datetime" in df.columns:
            time_col = "Datetime"
        else:
            time_col = None
            df["Time"] = range(len(df))
            time_col = "Time"
        
        # Convert time to datetime if needed
        if time_col and not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        position_hold_counter = 0
        position_entry_idx = None
        
        for i in range(len(df)):
            current_price = df.iloc[i]["Close"]
            current_time = df.iloc[i][time_col] if time_col else i
            
            # Update equity curve
            self.update_equity(current_price, current_time)
            
            # Check if we need to exit position (holding period reached)
            if self.position > 0:
                position_hold_counter += 1
                if position_hold_counter >= holding_periods:
                    self.exit_position(current_price, current_time, reason="HoldingPeriod")
                    position_hold_counter = 0
                    position_entry_idx = None
                    continue
            
            # If in position, skip entry logic
            if self.position > 0:
                continue
            
            # Check for entry signal
            if i < len(predictions):
                if use_probability and probabilities is not None:
                    should_enter = probabilities[i] >= confidence_threshold
                else:
                    should_enter = predictions[i] == 1
                
                if should_enter:
                    if self.enter_position(current_price, current_time, 
                                          position_pct=position_pct):
                        position_entry_idx = i
                        position_hold_counter = 0
        
        # Close any open position at the end
        if self.position > 0:
            final_price = df.iloc[-1]["Close"]
            final_time = df.iloc[-1][time_col] if time_col else len(df) - 1
            self.exit_position(final_price, final_time, reason="EndOfData")
        
        # Calculate holding periods for trades
        for trade in self.trades:
            if time_col:
                if isinstance(trade["entry_time"], str):
                    entry = pd.to_datetime(trade["entry_time"])
                    exit = pd.to_datetime(trade["exit_time"])
                else:
                    entry = trade["entry_time"]
                    exit = trade["exit_time"]
                trade["holding_periods"] = (exit - entry).total_seconds() / 300  # 5-minute periods
            else:
                trade["holding_periods"] = 1
        
        # Calculate metrics
        results = self.calculate_metrics(df.iloc[-1]["Close"] if len(df) > 0 else 0)
        
        return results
    
    def calculate_metrics(self, final_price: float = 0) -> Dict:
        """Calculate performance metrics."""
        if len(self.trades) == 0:
            return {
                "total_trades": 0,
                "total_profit": 0,
                "final_equity": self.initial_capital,
                "return_pct": 0,
                "win_rate": 0,
                "avg_profit_per_trade": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "trades": []
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        total_profit = trades_df["net_profit"].sum()
        winning_trades = trades_df[trades_df["net_profit"] > 0]
        losing_trades = trades_df[trades_df["net_profit"] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        avg_profit = trades_df["net_profit"].mean()
        avg_win = winning_trades["net_profit"].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades["net_profit"].mean() if len(losing_trades) > 0 else 0
        
        # Calculate final equity
        final_equity = self.get_current_equity(final_price)
        return_pct = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        # Sharpe ratio (simplified - using returns)
        if len(self.equity_curve) > 1:
            equity_values = [e["equity"] for e in self.equity_curve]
            returns = np.diff(equity_values) / equity_values[:-1]
            if returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 288)  # Annualized (5-min data)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        if len(self.equity_curve) > 0:
            equity_values = [e["equity"] for e in self.equity_curve]
            peak = equity_values[0]
            max_drawdown = 0
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            max_drawdown = 0
        
        return {
            "total_trades": len(trades_df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "total_profit": total_profit,
            "final_equity": final_equity,
            "return_pct": return_pct,
            "win_rate": win_rate,
            "avg_profit_per_trade": avg_profit,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(winning_trades["net_profit"].sum() / losing_trades["net_profit"].sum()) if len(losing_trades) > 0 and losing_trades["net_profit"].sum() < 0 else 0,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "trades": self.trades,
            "equity_curve": self.equity_curve
        }

# -------------------------------
# CONVENIENCE FUNCTIONS
# -------------------------------
def run_backtest(data_path: str, model_path: Optional[str] = None,
                confidence_threshold: float = 0.5,
                holding_periods: int = 1,
                position_pct: float = 1.0,
                initial_capital: float = INITIAL_CAPITAL,
                transaction_cost_pct: float = TRANSACTION_COST_PCT) -> Dict:
    """
    Convenience function to run backtest from file.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file with features
    model_path : str (optional)
        Path to model. If None, uses default.
    confidence_threshold : float
        Minimum probability to enter trade
    holding_periods : int
        Number of periods to hold
    position_pct : float
        Percentage of capital per trade
    initial_capital : float
        Starting capital
    transaction_cost_pct : float
        Transaction cost percentage
    
    Returns:
    --------
    results : dict
        Backtest results
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Load model and make predictions
    model, metadata = load_model()
    features = metadata.get("features", [])
    
    probabilities = predict(df, model, features, return_proba=True)
    predictions = (probabilities >= confidence_threshold).astype(int)
    
    # Run backtest
    engine = BacktestEngine(initial_capital, transaction_cost_pct)
    results = engine.run_backtest(
        df, predictions, probabilities,
        confidence_threshold, holding_periods, position_pct
    )
    
    return results

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python backtest.py <data_path> [confidence_threshold] [holding_periods]")
        print("Example: python backtest.py data/processed/aapl_features.csv 0.6 1")
        sys.exit(1)
    
    data_path = sys.argv[1]
    confidence_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    holding_periods = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    print(f"Running backtest on: {data_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Holding periods: {holding_periods}")
    print(f"Initial capital: ₹{INITIAL_CAPITAL:,.2f}")
    print(f"Transaction cost: {TRANSACTION_COST_PCT*100:.2f}%")
    print("\n" + "="*60)
    
    results = run_backtest(data_path, confidence_threshold=confidence_threshold,
                          holding_periods=holding_periods)
    
    print("\nBACKTEST RESULTS")
    print("="*60)
    print(f"Total Trades: {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']}")
    print(f"Losing Trades: {results['losing_trades']}")
    print(f"Win Rate: {results['win_rate']*100:.2f}%")
    print(f"\nTotal Profit: ₹{results['total_profit']:,.2f}")
    print(f"Final Equity: ₹{results['final_equity']:,.2f}")
    print(f"Return: {results['return_pct']:.2f}%")
    print(f"\nAvg Profit per Trade: ₹{results['avg_profit_per_trade']:,.2f}")
    print(f"Avg Win: ₹{results['avg_win']:,.2f}")
    print(f"Avg Loss: ₹{results['avg_loss']:,.2f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"\nSharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    
    if results['total_profit'] >= TARGET_PROFIT:
        print(f"\n✓ TARGET ACHIEVED! Profit of ₹{results['total_profit']:,.2f} >= ₹{TARGET_PROFIT:,.2f}")
    else:
        print(f"\n✗ Target not met. Need ₹{TARGET_PROFIT - results['total_profit']:,.2f} more profit.")

