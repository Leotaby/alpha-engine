"""
Abstract base class for all AlphaEngine trading strategies.

Every strategy in the framework inherits from ``BaseStrategy`` and must
implement ``generate_signals``, which receives an OHLCV DataFrame and
returns it with at least a ``signal`` column (1 = buy, -1 = sell, 0 = hold).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base strategy that all AlphaEngine strategies inherit from.

    Subclasses must define ``default_params`` at the class level and
    implement ``generate_signals``.  Optional overrides include the
    ``param_grid``, ``description``, and ``required_columns`` properties.

    Parameters
    ----------
    params : dict, optional
        Strategy-specific parameters.  Any key not supplied falls back to
        the class-level ``default_params`` value.
    """

    # Subclasses MUST override this with their own defaults
    default_params: Dict[str, Any] = {}

    # Minimum columns the input DataFrame must contain
    REQUIRED_COLUMNS: Set[str] = {"open", "high", "low", "close", "volume"}

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self._params: Dict[str, Any] = {**self.default_params}
        if params is not None:
            self._params.update(params)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def params(self) -> Dict[str, Any]:
        return dict(self._params)

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        return {}

    @property
    def description(self) -> str:
        return "Base strategy — override the `description` property in your subclass."

    # ------------------------------------------------------------------
    # Data validation
    # ------------------------------------------------------------------

    def validate_data(self, df: pd.DataFrame, min_rows: int = 1) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {sorted(missing)}. "
                f"Available columns: {sorted(df.columns.tolist())}"
            )

        if len(df) < min_rows:
            raise ValueError(
                f"DataFrame has {len(df)} rows but at least {min_rows} are required."
            )

        if df["close"].dropna().empty:
            raise ValueError("The 'close' column is entirely NaN.")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
        return f"{self.name}({params_str})"

    def __str__(self) -> str:
        return self.__repr__()
