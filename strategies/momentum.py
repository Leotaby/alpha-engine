"""
Dual EMA Crossover + ADX Filter momentum strategy.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy
from utils.indicators import ema, adx


class MomentumStrategy(BaseStrategy):
    """
    Dual EMA Crossover with ADX trend-strength and volume confirmation.
    """

    default_params: Dict[str, Any] = {
        "fast_period": 12,
        "slow_period": 26,
        "adx_period": 14,
        "adx_threshold": 25.0,
        "volume_period": 20,
        "volume_mult": 1.2,
    }

    EQUITY_DEFAULTS: Dict[str, Any] = {
        "fast_period": 10,
        "slow_period": 50,
        "adx_period": 14,
        "adx_threshold": 20.0,
        "volume_period": 20,
        "volume_mult": 1.0,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            "fast_period": [8, 10, 12, 15],
            "slow_period": [20, 26, 30, 40],
            "adx_threshold": [20, 25, 30],
            "volume_mult": [1.0, 1.2, 1.5],
        }

    @property
    def description(self) -> str:
        return (
            "Dual EMA Crossover + ADX Filter.\n"
            "Buys when the fast EMA crosses above the slow EMA in a trending\n"
            "market (ADX > threshold) with above-average volume confirmation.\n"
            "Sells on the opposite crossover with the same trend filter.\n\n"
            "Works best: strong trending markets (crypto uptrends, equity momentum).\n"
            "Fails: choppy, range-bound, or low-volume sideways markets."
        )

    @classmethod
    def for_equities(cls) -> "MomentumStrategy":
        return cls(params=cls.EQUITY_DEFAULTS)

    @classmethod
    def for_crypto(cls) -> "MomentumStrategy":
        return cls()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        fast: int = self._params["fast_period"]
        slow: int = self._params["slow_period"]
        adx_p: int = self._params["adx_period"]
        adx_thr: float = self._params["adx_threshold"]
        vol_p: int = self._params["volume_period"]
        vol_m: float = self._params["volume_mult"]

        min_rows = max(slow, adx_p * 2, vol_p) + 5
        self.validate_data(df, min_rows=min_rows)

        result = df.copy()

        result = ema(result, period=fast)
        result = ema(result, period=slow)
        result = adx(result, period=adx_p)

        fast_col = f"ema_{fast}"
        slow_col = f"ema_{slow}"
        adx_col = f"adx_{adx_p}"

        result["vol_avg"] = (
            result["volume"].rolling(window=vol_p, min_periods=vol_p).mean()
        )

        fast_above = result[fast_col] > result[slow_col]
        fast_above_prev = fast_above.shift(1, fill_value=False)

        cross_up = fast_above & ~fast_above_prev
        cross_down = ~fast_above & fast_above_prev

        trending = result[adx_col] > adx_thr
        vol_safe = result["vol_avg"].replace(0.0, np.nan)
        volume_confirm = result["volume"] > (vol_m * vol_safe)

        result["signal"] = 0
        result.loc[cross_up & trending & volume_confirm, "signal"] = 1
        result.loc[cross_down & trending, "signal"] = -1

        result.drop(columns=["vol_avg"], inplace=True)

        return result
