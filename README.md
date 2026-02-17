# ⚡ AlphaEngine — Algorithmic Trading Framework

**Professional-grade Python framework for backtesting, optimizing, and deploying trading strategies.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Strategies](https://img.shields.io/badge/Strategies-5-orange)

---

## 🖥️ Dashboard Preview

### Equity Curve & Metrics
![Overview](screenshots/overview.png)

### Trade Visualization
![Trades](screenshots/trades.png)

### Strategy Comparison
![Compare](screenshots/compare.png)

### Parameter Optimization
![Optimize](screenshots/optimize.png)

### Monte Carlo Stress Testing
![Monte Carlo](screenshots/montecarlo.png)

---

## What is AlphaEngine?

AlphaEngine is a **complete algorithmic trading framework** for serious quant traders. It includes 5 production-ready strategies, an interactive dashboard, walk-forward optimization, Monte Carlo simulation, and multi-exchange support.

This repo contains the **free demo** with 1 strategy (Momentum) and basic backtesting.

### 🔓 Free Demo (this repo)
- ✅ Momentum strategy
- ✅ 15 technical indicators (computed from scratch)
- ✅ Portfolio tracker with commission/slippage
- ✅ Performance metrics (Sharpe, Sortino, win rate, etc.)
- ✅ Position sizing

### 🔒 Full Version ([Get it here →](https://leotaby.gumroad.com/l/alphaengine))
- ✦ **5 strategies**: Momentum, Mean Reversion, Breakout, RSI+MACD, Grid Trading
- ✦ **Interactive Streamlit dashboard** with 5 tabs
- ✦ **Parameter optimization** with Sharpe ratio heatmaps
- ✦ **Monte Carlo simulation** (1000+ scenarios, fan charts)
- ✦ **Risk management suite**: drawdown halts, trailing stops, Kelly criterion
- ✦ **Multi-exchange**: Binance (crypto) + Alpaca (US equities)
- ✦ **MQL5 Expert Advisor** (.mq5 file for MetaTrader 5)
- ✦ Telegram & Discord notifications

---

## Quick Start

```bash
git clone https://github.com/Leotaby/alpha-engine.git
cd alpha-engine
pip install numpy pandas
python demo.py
```

Output:
```
╔══════════════════════════════════════════════════════════╗
║   ⚡  A L P H A  E N G I N E  —  Free Demo              ║
╚══════════════════════════════════════════════════════════╝

📊 Generated 1000 bars | $28,521 — $44,182

  ▸ Running Momentum strategy... ✓

════════════════════════════════════════════════════════════
  Total Return         +0.23%
  Sharpe Ratio         0.040
  Win Rate             100.0%
  Max Drawdown         2.05%
  Profit Factor        0.00
  Total Trades         2
════════════════════════════════════════════════════════════
```

---

## Add Your Own Strategy

Even in the free version, you can create custom strategies:

```python
from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    name = "my_strategy"
    default_params = {"fast": 10, "slow": 30}
    
    @property
    def description(self):
        return "My custom crossover strategy."
    
    @property
    def param_grid(self):
        return {"fast": [5, 10, 15], "slow": [20, 30, 40]}
    
    def generate_signals(self, df):
        from utils.indicators import ema
        df = ema(df, self.params["fast"])
        df = ema(df, self.params["slow"])
        df["signal"] = 0
        df.loc[df[f"ema_{self.params['fast']}"] > df[f"ema_{self.params['slow']}"], "signal"] = 1
        df.loc[df[f"ema_{self.params['fast']}"] < df[f"ema_{self.params['slow']}"], "signal"] = -1
        return df
```

---

## Full Version Pricing

| | Starter | Pro ⭐ | Premium |
|---|---------|--------|---------|
| **Price** | $79 | $129 | $249 |
| Strategies | 2 | **5** | **5** |
| Dashboard | Basic | **Full (5 tabs)** | **Full** |
| Optimizer | ❌ | ✅ | ✅ |
| Monte Carlo | ❌ | ✅ | ✅ |
| Risk Suite | Basic | **Full** | **Full** |
| Exchanges | 1 | **2** | **2** |
| MQL5 EA | ❌ | ❌ | ✅ |
| Support | Email | Priority | **Video call** |

### [👉 Get AlphaEngine Pro →](https://leotaby.gumroad.com/l/alphaengine)

---

## Tech Stack

- **Python 3.10+** — core framework
- **Streamlit** — interactive dashboard
- **Plotly** — professional charts
- **NumPy/Pandas** — data processing
- **MQL5** — MetaTrader 5 Expert Advisor

---

## Disclaimer

This software is for **educational and research purposes only**. It is not financial advice. Past performance does not guarantee future results. Always do your own research before trading with real money.

---

Built with 🧠 by a quantitative finance professional.
