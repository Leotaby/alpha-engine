"""AlphaEngine — Project Configuration"""

import os
from typing import Any, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float = 0.0) -> float:
    raw = os.getenv(key, "")
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_int(key: str, default: int = 0) -> int:
    raw = os.getenv(key, "")
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key, "").lower()
    if raw in ("1", "true", "yes"):
        return True
    if raw in ("0", "false", "no"):
        return False
    return default


def get_config() -> Dict[str, Any]:
    """Build and return the fully merged configuration dictionary."""
    active_exchange = _env("EXCHANGE", "binance")

    cfg: Dict[str, Any] = {
        "mode": _env("ENGINE_MODE", "paper"),
        "symbol": _env("SYMBOL", "BTC/USDT"),
        "timeframe": _env("TIMEFRAME", "1h"),
        "interval_seconds": _env_int("INTERVAL_SECONDS", 60),
        "lookback_bars": _env_int("LOOKBACK_BARS", 200),
        "initial_capital": _env_float("INITIAL_CAPITAL", 100000.0),
        "exchange": active_exchange,
        "exchange_keys": {
            "api_key": _env(f"{active_exchange.upper()}_API_KEY"),
            "api_secret": _env(f"{active_exchange.upper()}_API_SECRET"),
        },
        "strategy": _env("STRATEGY_NAME", "momentum"),
        "strategy_params": {},
        "risk": {
            "max_drawdown_pct": _env_float("RISK_MAX_DRAWDOWN_PCT", 15.0),
            "max_open_positions": _env_int("RISK_MAX_OPEN_POSITIONS", 5),
            "daily_loss_limit": _env_float("RISK_DAILY_LOSS_LIMIT", 500.0),
            "commission_pct": _env_float("COMMISSION_PCT", 0.001),
            "slippage_pct": _env_float("SLIPPAGE_PCT", 0.0005),
            "stop_loss_pct": _env_float("STOP_LOSS_PCT", 0.02),
            "take_profit_pct": _env_float("TAKE_PROFIT_PCT", 0.04),
        },
        "backtest": {
            "initial_capital": _env_float("BACKTEST_INITIAL_CAPITAL", 100000.0),
            "enable_shorting": _env_bool("ENABLE_SHORTING", False),
        },
        "notifications": {
            "telegram_bot_token": _env("TELEGRAM_BOT_TOKEN"),
            "telegram_chat_id": _env("TELEGRAM_CHAT_ID"),
            "discord_webhook_url": _env("DISCORD_WEBHOOK_URL"),
        },
    }
    return cfg
# End of file
