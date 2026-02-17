"""
AlphaEngine — Free Demo
Run: python demo.py

Demonstrates the Momentum strategy with basic backtesting.
Full version includes 5 strategies, dashboard, optimizer, Monte Carlo, and more.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from strategies import get_strategy, list_strategies
from core.portfolio import Portfolio
from risk import PositionSizer
from utils.metrics import PerformanceMetrics


def generate_market_data(n_bars: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 40000.0
    prices = [price]
    for i in range(1, n_bars):
        drift = 0.0003 if i % 200 < 120 else 0.0
        vol = 0.015 if i % 200 < 120 else 0.008
        price = price * (1 + drift + vol * rng.standard_normal())
        prices.append(price)
    close = np.array(prices)
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="1h")
    df = pd.DataFrame({
        "open": close * (1 + rng.uniform(-0.003, 0.003, n_bars)),
        "high": close * (1 + np.abs(rng.standard_normal(n_bars)) * 0.005),
        "low": close * (1 - np.abs(rng.standard_normal(n_bars)) * 0.005),
        "close": close,
        "volume": rng.integers(5000, 50000, n_bars).astype(float),
    }, index=dates)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def run_backtest(strategy_name, data, initial_capital=10000.0, position_pct=0.02, stop_loss_pct=0.03):
    strat = get_strategy(strategy_name)
    portfolio = Portfolio(initial_capital=initial_capital)
    pos_sizer = PositionSizer()
    signals_df = strat.generate_signals(data)
    symbol = "SIM"

    for i in range(len(signals_df)):
        row = signals_df.iloc[i]
        ts = signals_df.index[i]
        close = float(row["close"])
        signal = int(row.get("signal", 0))
        pos = portfolio.get_position(symbol)

        if pos is not None:
            entry = pos["entry_price"]
            if close <= entry * (1 - stop_loss_pct):
                portfolio.close_position(symbol, close, ts, reason="stop_loss")
                pos = None

        if signal == 1 and pos is None:
            dollar_risk = pos_sizer.fixed_percentage(portfolio.equity, position_pct)
            qty = dollar_risk / close if close > 0 else 0
            if qty > 0:
                try:
                    portfolio.open_position(symbol, "long", qty, close, ts)
                except ValueError:
                    pass
        elif signal == -1 and pos is not None:
            portfolio.close_position(symbol, close, ts, reason="signal")

        portfolio.update_equity({symbol: close})
        portfolio.record_equity(ts)

    if portfolio.get_position(symbol) is not None:
        portfolio.close_position(symbol, float(signals_df.iloc[-1]["close"]),
                                 signals_df.index[-1], reason="end_of_data")

    equity_curve = portfolio.equity_curve_to_series()
    metrics = PerformanceMetrics.calculate(equity_curve, portfolio.trade_log, initial_capital)
    return metrics


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   ⚡  A L P H A  E N G I N E  —  Free Demo              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    data = generate_market_data(1000)
    print(f"📊 Generated 1000 bars | ${data['close'].min():.0f} — ${data['close'].max():.0f}")
    print()

    # Run momentum strategy
    print("  ▸ Running Momentum strategy...", end=" ")
    m = run_backtest("momentum", data)
    print(f"✓")
    print()

    print("═" * 60)
    print(f"  {'Total Return':<20} {m.get('total_return_pct', 0):+.2f}%")
    print(f"  {'Sharpe Ratio':<20} {m.get('sharpe_ratio', 0):.3f}")
    print(f"  {'Win Rate':<20} {m.get('win_rate', 0):.1f}%")
    print(f"  {'Max Drawdown':<20} {m.get('max_drawdown_pct', 0):.2f}%")
    print(f"  {'Profit Factor':<20} {m.get('profit_factor', 0):.2f}")
    print(f"  {'Total Trades':<20} {m.get('total_trades', 0)}")
    print("═" * 60)
    print()

    print("🔒 This demo includes 1 of 5 strategies.")
    print()
    print("   The full AlphaEngine Pro includes:")
    print("   ✦ 5 strategies (Momentum, Mean Reversion, Breakout, RSI+MACD, Grid)")
    print("   ✦ Interactive Streamlit dashboard with 5 tabs")
    print("   ✦ Parameter optimization with Sharpe heatmaps")
    print("   ✦ Monte Carlo stress testing (1000+ simulations)")
    print("   ✦ Walk-forward optimization")
    print("   ✦ Risk management (drawdown halts, trailing stops)")
    print("   ✦ Multi-exchange support (Binance + Alpaca)")
    print("   ✦ MQL5 Expert Advisor for MetaTrader 5")
    print()
    print("   👉 Get the full version: https://leotaby.gumroad.com/l/alphaengine")
    print()


if __name__ == "__main__":
    main()
