"""
AlphaEngine — Technical Indicators Library
All indicators computed from scratch (no TA-Lib dependency).
"""

import numpy as np
import pandas as pd


def sma(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
    col_name = f"sma_{period}"
    df[col_name] = df[column].rolling(window=period, min_periods=period).mean()
    return df


def ema(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
    col_name = f"ema_{period}"
    df[col_name] = df[column].ewm(span=period, adjust=False, min_periods=period).mean()
    return df


def wma(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
    col_name = f"wma_{period}"
    weights = np.arange(1, period + 1, dtype=float)
    df[col_name] = df[column].rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
    return df


def rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.DataFrame:
    col_name = f"rsi_{period}"
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[col_name] = 100.0 - (100.0 / (1.0 + rs))
    return df


def macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26,
         signal_period: int = 9, column: str = "close") -> pd.DataFrame:
    fast_ema = df[column].ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    slow_ema = df[column].ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
    df["macd_line"] = fast_ema - slow_ema
    df["macd_signal"] = df["macd_line"].ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
    return df


def bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0,
                    column: str = "close") -> pd.DataFrame:
    mid = df[column].rolling(window=period, min_periods=period).mean()
    std = df[column].rolling(window=period, min_periods=period).std()
    df["bb_middle"] = mid
    df["bb_upper"] = mid + num_std * std
    df["bb_lower"] = mid - num_std * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, np.nan)
    return df


def atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    col_name = f"atr_{period}"
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[col_name] = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return df


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    col_name = f"adx_{period}"
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    close_prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    alpha = 1.0 / period
    smoothed_tr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smoothed_plus = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smoothed_minus = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * smoothed_plus / smoothed_tr.replace(0, np.nan)
    minus_di = 100.0 * smoothed_minus / smoothed_tr.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df[col_name] = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    df[f"plus_di_{period}"] = plus_di
    df[f"minus_di_{period}"] = minus_di
    return df


def donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df[f"dc_upper_{period}"] = df["high"].rolling(window=period, min_periods=period).max()
    df[f"dc_lower_{period}"] = df["low"].rolling(window=period, min_periods=period).min()
    df[f"dc_middle_{period}"] = (df[f"dc_upper_{period}"] + df[f"dc_lower_{period}"]) / 2.0
    return df


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    lowest_low = df["low"].rolling(window=k_period, min_periods=k_period).min()
    highest_high = df["high"].rolling(window=k_period, min_periods=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    df[f"stoch_k_{k_period}"] = 100.0 * (df["close"] - lowest_low) / denom
    df[f"stoch_d_{d_period}"] = df[f"stoch_k_{k_period}"].rolling(window=d_period).mean()
    return df


def cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mad = tp.rolling(window=period, min_periods=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df[f"cci_{period}"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    return df


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    highest = df["high"].rolling(window=period, min_periods=period).max()
    lowest = df["low"].rolling(window=period, min_periods=period).min()
    denom = (highest - lowest).replace(0, np.nan)
    df[f"williams_r_{period}"] = -100.0 * (highest - df["close"]) / denom
    return df


def obv(df: pd.DataFrame) -> pd.DataFrame:
    sign = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (sign * df["volume"]).cumsum()
    return df


def vwap(df: pd.DataFrame) -> pd.DataFrame:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cumulative_tpv = (tp * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = cumulative_tpv / cumulative_vol
    return df


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    hl2 = (df["high"] + df["low"]) / 2.0
    df_atr = atr(df.copy(), period=period)
    atr_col = f"atr_{period}"
    atr_vals = df_atr[atr_col]

    upper_band = hl2 + multiplier * atr_vals
    lower_band = hl2 - multiplier * atr_vals

    supertrend_vals = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1:
            supertrend_vals.iloc[i] = lower_band.iloc[i]
        else:
            supertrend_vals.iloc[i] = upper_band.iloc[i]

    df[f"supertrend_{period}"] = supertrend_vals
    df[f"supertrend_dir_{period}"] = direction
    return df


def add_all_indicators(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Apply all indicators with configurable periods."""
    cfg = config or {}
    result = df.copy()
    result = sma(result, period=cfg.get("sma_period", 20))
    result = ema(result, period=cfg.get("ema_fast", 12))
    result = ema(result, period=cfg.get("ema_slow", 26))
    result = rsi(result, period=cfg.get("rsi_period", 14))
    result = macd(result)
    result = bollinger_bands(result, period=cfg.get("bb_period", 20))
    result = atr(result, period=cfg.get("atr_period", 14))
    result = adx(result, period=cfg.get("adx_period", 14))
    result = donchian_channel(result, period=cfg.get("dc_period", 20))
    result = stochastic(result)
    result = obv(result)
    result = vwap(result)
    return result
# End of file
