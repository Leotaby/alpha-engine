"""AlphaEngine — Portfolio State Tracker"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger("AlphaEngine.Portfolio")


class Portfolio:
    """Tracks the complete state of a trading portfolio."""

    def __init__(self, initial_capital: float, commission_pct: float = 0.001, slippage_pct: float = 0.0005) -> None:
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._commission_pct = commission_pct
        self._slippage_pct = slippage_pct
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._trade_log: List[Dict[str, Any]] = []
        self._realized_pnl: float = 0.0
        self._peak_equity: float = initial_capital
        self._equity_history: List[Tuple[Any, float]] = []

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._positions)

    @property
    def trade_log(self) -> List[Dict[str, Any]]:
        return list(self._trade_log)

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def unrealized_pnl(self) -> float:
        pnl = 0.0
        for pos in self._positions.values():
            mark = pos.get("mark_price", pos["entry_price"])
            if pos["side"] == "long":
                pnl += (mark - pos["entry_price"]) * pos["quantity"]
            else:
                pnl += (pos["entry_price"] - mark) * pos["quantity"]
        return pnl

    @property
    def equity(self) -> float:
        return self._cash + self.unrealized_pnl

    @property
    def initial_capital(self) -> float:
        return self._initial_capital

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def equity_history(self) -> List[Tuple[Any, float]]:
        return list(self._equity_history)

    def open_position(self, symbol: str, side: str, quantity: float, entry_price: float, timestamp: Any) -> Dict[str, Any]:
        if symbol in self._positions:
            raise ValueError(f"Position already open for {symbol}")
        if side not in ("long", "short"):
            raise ValueError(f"Invalid side '{side}'")

        if side == "long":
            fill_price = entry_price * (1.0 + self._slippage_pct)
        else:
            fill_price = entry_price * (1.0 - self._slippage_pct)

        notional = fill_price * quantity
        commission = notional * self._commission_pct
        total_cost = notional + commission

        if side == "long" and total_cost > self._cash:
            raise ValueError(f"Insufficient cash: need {total_cost:.2f}, have {self._cash:.2f}")

        position = {
            "symbol": symbol, "side": side, "quantity": quantity, "entry_price": fill_price,
            "raw_entry_price": entry_price, "commission_entry": commission,
            "timestamp_open": timestamp, "mark_price": fill_price,
        }
        self._positions[symbol] = position

        if side == "long":
            self._cash -= total_cost
        else:
            self._cash += notional - commission

        return dict(position)

    def close_position(self, symbol: str, exit_price: float, timestamp: Any, reason: str = "signal") -> Dict[str, Any]:
        if symbol not in self._positions:
            raise KeyError(f"No open position for {symbol}")

        pos = self._positions.pop(symbol)
        side = pos["side"]
        quantity = pos["quantity"]
        entry_fill = pos["entry_price"]

        if side == "long":
            fill_price = exit_price * (1.0 - self._slippage_pct)
        else:
            fill_price = exit_price * (1.0 + self._slippage_pct)

        notional_exit = fill_price * quantity
        commission_exit = notional_exit * self._commission_pct

        if side == "long":
            gross_pnl = (fill_price - entry_fill) * quantity
            self._cash += notional_exit - commission_exit
        else:
            gross_pnl = (entry_fill - fill_price) * quantity
            self._cash -= notional_exit + commission_exit
            self._cash += gross_pnl

        total_commission = pos["commission_entry"] + commission_exit
        net_pnl = gross_pnl - total_commission
        return_pct = net_pnl / (entry_fill * quantity) if entry_fill * quantity else 0.0

        try:
            duration = timestamp - pos["timestamp_open"]
        except TypeError:
            duration = None

        self._realized_pnl += net_pnl

        trade_record = {
            "symbol": symbol, "side": side, "quantity": quantity,
            "entry_price": entry_fill, "exit_price": fill_price,
            "pnl": net_pnl, "gross_pnl": gross_pnl, "return_pct": return_pct,
            "duration": duration, "reason": reason, "commission_total": total_commission,
            "timestamp_open": pos["timestamp_open"], "timestamp_close": timestamp,
        }
        self._trade_log.append(trade_record)
        return trade_record

    def update_equity(self, current_prices: Dict[str, float]) -> float:
        for symbol, pos in self._positions.items():
            if symbol in current_prices:
                pos["mark_price"] = current_prices[symbol]
        eq = self.equity
        if eq > self._peak_equity:
            self._peak_equity = eq
        return eq

    def record_equity(self, timestamp: Any) -> float:
        eq = self.equity
        if eq > self._peak_equity:
            self._peak_equity = eq
        self._equity_history.append((timestamp, eq))
        return eq

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        pos = self._positions.get(symbol)
        return dict(pos) if pos else None

    def current_drawdown_pct(self) -> float:
        if self._peak_equity == 0:
            return 0.0
        return (self._peak_equity - self.equity) / self._peak_equity

    def get_portfolio_state(self) -> Dict[str, Any]:
        return {
            "equity": self.equity, "cash": self._cash, "initial_capital": self._initial_capital,
            "unrealized_pnl": self.unrealized_pnl, "realized_pnl": self._realized_pnl,
            "num_open_positions": len(self._positions), "open_positions": self.positions,
            "current_drawdown_pct": self.current_drawdown_pct(), "peak_equity": self._peak_equity,
            "total_trades": len(self._trade_log),
        }

    def trade_log_to_dataframe(self) -> pd.DataFrame:
        if not self._trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self._trade_log)

    def equity_curve_to_series(self) -> pd.Series:
        if not self._equity_history:
            return pd.Series(dtype=float)
        timestamps, values = zip(*self._equity_history)
        return pd.Series(data=values, index=pd.Index(timestamps), name="equity")

    def __repr__(self) -> str:
        return f"Portfolio(equity={self.equity:.2f}, cash={self._cash:.2f}, positions={len(self._positions)}, trades={len(self._trade_log)})"
