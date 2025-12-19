import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
from typing import Optional, Dict
try:
    from .predict import load_model, predict
    from .backtest import BacktestEngine, INITIAL_CAPITAL, TRANSACTION_COST_PCT
except ImportError:
    from predict import load_model, predict
    from backtest import BacktestEngine, INITIAL_CAPITAL, TRANSACTION_COST_PCT

# -------------------------------
# PAPER TRADING SYSTEM
# -------------------------------
class PaperTradingSystem:
    """Simulated real-time trading system for paper trading."""
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL,
                 transaction_cost_pct: float = TRANSACTION_COST_PCT,
                 confidence_threshold: float = 0.5,
                 holding_periods: int = 1,
                 position_pct: float = 1.0):
        self.engine = BacktestEngine(initial_capital, transaction_cost_pct)
        self.confidence_threshold = confidence_threshold
        self.holding_periods = holding_periods
        self.position_pct = position_pct
        self.model = None
        self.metadata = None
        self.features = None
        self.position_hold_counter = 0
        self.trade_log = []
        self.is_running = False
        
    def initialize_model(self):
        """Load the trained model."""
        self.model, self.metadata = load_model()
        self.features = self.metadata.get("features", [])
        print(f"Model loaded: {self.metadata.get('model_type', 'Unknown')}")
        print(f"Features: {self.features}")
    
    def process_tick(self, data: pd.Series) -> Dict:
        """
        Process a single data tick (5-minute candle).
        
        Parameters:
        -----------
        data : pandas.Series
            Single row of OHLC data with features
        
        Returns:
        --------
        action : dict
            Trading action taken
        """
        if self.model is None:
            self.initialize_model()
        
        current_price = data["Close"]
        current_time = data.get("Date", datetime.now())
        
        # Update equity
        self.engine.update_equity(current_price, current_time)
        
        action = {
            "time": current_time,
            "price": current_price,
            "action": "HOLD",
            "reason": None,
            "equity": self.engine.get_current_equity(current_price)
        }
        
        # Check if we need to exit position (holding period reached)
        if self.engine.position > 0:
            self.position_hold_counter += 1
            if self.position_hold_counter >= self.holding_periods:
                self.engine.exit_position(current_price, current_time, reason="HoldingPeriod")
                action["action"] = "SELL"
                action["reason"] = "HoldingPeriod"
                self.position_hold_counter = 0
                self.trade_log.append(action.copy())
                return action
        
        # If in position, skip entry logic
        if self.engine.position > 0:
            return action
        
        # Make prediction
        try:
            # Extract features
            X = pd.DataFrame([data[self.features]])
            probabilities = self.model.predict_proba(X)[0, 1]
            prediction = 1 if probabilities >= self.confidence_threshold else 0
            
            # Check for entry signal
            if prediction == 1 and probabilities >= self.confidence_threshold:
                if self.engine.enter_position(current_price, current_time, 
                                             position_pct=self.position_pct):
                    action["action"] = "BUY"
                    action["reason"] = f"Signal (prob={probabilities:.3f})"
                    action["probability"] = probabilities
                    self.position_hold_counter = 0
                    self.trade_log.append(action.copy())
            else:
                action["probability"] = probabilities
                action["prediction"] = "SELL" if prediction == 0 else "BUY"
        
        except Exception as e:
            action["error"] = str(e)
            print(f"Error processing tick: {e}")
        
        return action
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a DataFrame of historical data (for testing).
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLC data and features
        
        Returns:
        --------
        actions_df : pandas.DataFrame
            DataFrame with all actions taken
        """
        actions = []
        
        for idx, row in df.iterrows():
            action = self.process_tick(row)
            actions.append(action)
        
        return pd.DataFrame(actions)
    
    def get_status(self) -> Dict:
        """Get current trading status."""
        current_equity = self.engine.get_current_equity(0)  # Will be updated with real price
        return {
            "cash": self.engine.cash,
            "position": self.engine.position,
            "position_entry_price": self.engine.position_entry_price,
            "position_entry_time": self.engine.position_entry_time,
            "total_trades": len(self.engine.trades),
            "current_equity": current_equity,
            "total_profit": sum(t["net_profit"] for t in self.engine.trades),
            "is_running": self.is_running
        }
    
    def get_performance(self) -> Dict:
        """Get performance metrics."""
        if len(self.engine.trades) == 0:
            return {
                "total_trades": 0,
                "total_profit": 0,
                "win_rate": 0,
                "final_equity": self.engine.initial_capital
            }
        
        results = self.engine.calculate_metrics()
        return results
    
    def save_log(self, output_path: str):
        """Save trade log to file."""
        if self.trade_log:
            log_df = pd.DataFrame(self.trade_log)
            log_df.to_csv(output_path, index=False)
            print(f"Trade log saved to: {output_path}")
    
    def reset(self):
        """Reset the paper trading system."""
        self.engine.reset()
        self.position_hold_counter = 0
        self.trade_log = []
        self.is_running = False

# -------------------------------
# SIMULATED REAL-TIME TRADING
# -------------------------------
def simulate_realtime(data_path: str, output_path: Optional[str] = None,
                     confidence_threshold: float = 0.5,
                     holding_periods: int = 1,
                     position_pct: float = 1.0,
                     delay_seconds: float = 0.1):
    """
    Simulate real-time trading on historical data.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file with features
    output_path : str (optional)
        Path to save trade log
    confidence_threshold : float
        Minimum probability to enter trade
    holding_periods : int
        Number of periods to hold
    position_pct : float
        Percentage of capital per trade
    delay_seconds : float
        Delay between ticks (for simulation)
    """
    print("="*60)
    print("PAPER TRADING - SIMULATED REAL-TIME")
    print("="*60)
    
    # Load data
    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    
    print(f"Data loaded: {len(df)} rows")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Holding periods: {holding_periods}")
    print(f"Position size: {position_pct*100:.0f}%")
    print("\nStarting simulation...\n")
    
    # Initialize system
    system = PaperTradingSystem(
        confidence_threshold=confidence_threshold,
        holding_periods=holding_periods,
        position_pct=position_pct
    )
    system.initialize_model()
    system.is_running = True
    
    # Process each tick
    for idx, row in df.iterrows():
        action = system.process_tick(row)
        
        # Print action
        if action["action"] != "HOLD":
            print(f"[{action['time']}] {action['action']} @ ₹{action['price']:.2f} - {action.get('reason', '')}")
        
        # Simulate delay
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    
    # Close any open position
    if system.engine.position > 0:
        final_price = df.iloc[-1]["Close"]
        final_time = df.iloc[-1].get("Date", datetime.now())
        system.engine.exit_position(final_price, final_time, reason="EndOfSimulation")
        print(f"\n[End] SELL @ ₹{final_price:.2f} - EndOfSimulation")
    
    # Get performance
    performance = system.get_performance()
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Total Trades: {performance['total_trades']}")
    print(f"Winning Trades: {performance['winning_trades']}")
    print(f"Losing Trades: {performance['losing_trades']}")
    print(f"Win Rate: {performance['win_rate']*100:.2f}%")
    print(f"Total Profit: ₹{performance['total_profit']:,.2f}")
    print(f"Final Equity: ₹{performance['final_equity']:,.2f}")
    print(f"Return: {performance['return_pct']:.2f}%")
    print("="*60)
    
    # Save log
    if output_path:
        system.save_log(output_path)
    
    return system, performance

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python paper_trade.py <data_path> [confidence_threshold] [holding_periods] [output_path]")
        print("Example: python paper_trade.py data/processed/aapl_features.csv 0.6 1 paper_trades.csv")
        sys.exit(1)
    
    data_path = sys.argv[1]
    confidence_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    holding_periods = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    output_path = sys.argv[4] if len(sys.argv) > 4 else None
    
    simulate_realtime(data_path, output_path, confidence_threshold, holding_periods)

