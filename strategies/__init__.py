"""AlphaEngine Demo — Strategy Registry (1 of 5 strategies)"""

from strategies.momentum import MomentumStrategy

STRATEGY_REGISTRY = {
    "momentum": MomentumStrategy,
}

# 🔒 Full version includes: mean_reversion, breakout, rsi_macd, grid_trading
# Get all 5 strategies → https://leotaby.gumroad.com/l/alphaengine


def get_strategy(name: str, params: dict = None):
    if name not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Strategy '{name}' not available in demo. "
            f"Available: {available}. "
            f"Get all 5 strategies at https://leotaby.gumroad.com/l/alphaengine"
        )
    cls = STRATEGY_REGISTRY[name]
    return cls(params=params) if params else cls()


def list_strategies():
    return list(STRATEGY_REGISTRY.keys())
# End of file
