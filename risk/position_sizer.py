"""AlphaEngine — Position Sizing Calculator"""

import logging
from typing import Optional

logger = logging.getLogger("AlphaEngine.PositionSizer")


class PositionSizer:
    """Calculates position size using several well-known methods."""

    def __init__(self, max_position_pct: float = 0.10) -> None:
        self.max_position_pct = max(0.0, min(max_position_pct, 1.0))

    def fixed_percentage(self, capital: float, pct: float = 0.02) -> float:
        if capital <= 0 or pct <= 0:
            return 0.0
        raw = capital * pct
        capped = min(raw, capital * self.max_position_pct)
        return round(capped, 2)

    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float, half_kelly: bool = True) -> float:
        win_rate = max(0.0, min(win_rate, 1.0))
        avg_win = abs(avg_win) if avg_win != 0 else 0.0
        avg_loss = abs(avg_loss) if avg_loss != 0 else 0.0

        if avg_loss == 0.0 or avg_win == 0.0:
            return 0.0

        r = avg_win / avg_loss
        f_star = win_rate - (1.0 - win_rate) / r

        if half_kelly:
            f_star *= 0.5

        f_star = max(0.0, min(f_star, self.max_position_pct))
        return round(f_star, 6)

    def atr_volatility(self, capital: float, atr_value: float, risk_per_atr: float = 1.0, max_pct: float = 0.05) -> float:
        if capital <= 0 or atr_value <= 0 or risk_per_atr <= 0:
            return 0.0
        dollar_risk = capital * (risk_per_atr / 100.0)
        capped = min(dollar_risk, capital * max_pct)
        return round(capped, 2)

    def calculate(self, method: str, **kwargs) -> float:
        dispatch = {
            "fixed": self.fixed_percentage,
            "kelly": self.kelly_criterion,
            "atr": self.atr_volatility,
        }
        fn = dispatch.get(method.lower())
        if fn is None:
            raise ValueError(f"Unknown sizing method '{method}'. Choose from {list(dispatch.keys())}.")
        return fn(**kwargs)
