"""AlphaEngine — Performance Metrics"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional


class PerformanceMetrics:
    """Compute comprehensive strategy analytics from equity curve and trade log."""

    @staticmethod
    def calculate(equity_curve: pd.Series, trades: List[Dict[str, Any]],
                  initial_capital: float = 100000.0) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}

        if equity_curve is None or len(equity_curve) < 2:
            return {"total_return_pct": 0, "sharpe_ratio": 0, "win_rate": 0,
                    "max_drawdown_pct": 0, "profit_factor": 0, "total_trades": 0}

        final_equity = float(equity_curve.iloc[-1])
        total_return = (final_equity / initial_capital) - 1.0
        metrics["total_return_pct"] = round(total_return * 100, 2)
        metrics["final_equity"] = round(final_equity, 2)

        returns = equity_curve.pct_change().dropna()
        if len(returns) > 0 and returns.std() != 0:
            metrics["sharpe_ratio"] = round(float(returns.mean() / returns.std() * np.sqrt(252)), 3)
        else:
            metrics["sharpe_ratio"] = 0.0

        neg_returns = returns[returns < 0]
        if len(neg_returns) > 0 and neg_returns.std() != 0:
            metrics["sortino_ratio"] = round(float(returns.mean() / neg_returns.std() * np.sqrt(252)), 3)
        else:
            metrics["sortino_ratio"] = 0.0

        running_max = equity_curve.expanding().max()
        drawdown = (running_max - equity_curve) / running_max
        metrics["max_drawdown_pct"] = round(float(drawdown.max()) * 100, 2)

        if metrics["max_drawdown_pct"] != 0:
            annualized_return = total_return * 252 / max(len(returns), 1)
            metrics["calmar_ratio"] = round(annualized_return / (metrics["max_drawdown_pct"] / 100), 3)
        else:
            metrics["calmar_ratio"] = 0.0

        metrics["total_trades"] = len(trades)

        if trades:
            pnls = [t.get("pnl", 0) for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            metrics["win_rate"] = round(len(wins) / len(pnls) * 100, 1) if pnls else 0
            metrics["avg_win"] = round(np.mean(wins), 2) if wins else 0
            metrics["avg_loss"] = round(np.mean(losses), 2) if losses else 0
            metrics["best_trade"] = round(max(pnls), 2) if pnls else 0
            metrics["worst_trade"] = round(min(pnls), 2) if pnls else 0

            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            metrics["profit_factor"] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0

            returns_pct = [t.get("return_pct", 0) * 100 for t in trades]
            metrics["avg_trade_return"] = round(np.mean(returns_pct), 2) if returns_pct else 0
        else:
            metrics["win_rate"] = 0
            metrics["profit_factor"] = 0
            metrics["avg_trade_return"] = 0

        return metrics

    @staticmethod
    def summary_string(metrics: Dict[str, Any]) -> str:
        lines = []
        for k, v in metrics.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
# End of file
