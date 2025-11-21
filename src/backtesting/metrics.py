"""
Performance-Metriken für Backtesting.

Enthält alle wichtigen Trading-Metriken wie:
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TradeResult:
    """Ergebnis eines einzelnen Trades."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    side: str  # 'long' or 'short'
    size: float
    pnl: float
    pnl_pct: float
    fees: float


class BacktestMetrics:
    """Berechnet und aggregiert Backtesting-Metriken."""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: Risikofreier Zinssatz (annualisiert)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_all(
        self,
        equity_curve: np.ndarray,
        trades: List[TradeResult],
        periods_per_year: int = 252
    ) -> Dict[str, float]:
        """
        Berechnet alle Metriken.

        Args:
            equity_curve: Array mit Portfolio-Werten über Zeit
            trades: Liste der abgeschlossenen Trades
            periods_per_year: Anzahl der Perioden pro Jahr (252 für Tage)

        Returns:
            Dictionary mit allen Metriken
        """
        returns = self._calculate_returns(equity_curve)

        metrics = {
            # Rendite-Metriken
            'total_return': self.total_return(equity_curve),
            'annualized_return': self.annualized_return(equity_curve, periods_per_year),
            'cagr': self.cagr(equity_curve, periods_per_year),

            # Risiko-Metriken
            'volatility': self.volatility(returns, periods_per_year),
            'max_drawdown': self.max_drawdown(equity_curve),
            'avg_drawdown': self.avg_drawdown(equity_curve),

            # Risikoadjustierte Rendite
            'sharpe_ratio': self.sharpe_ratio(returns, periods_per_year),
            'sortino_ratio': self.sortino_ratio(returns, periods_per_year),
            'calmar_ratio': self.calmar_ratio(equity_curve, periods_per_year),

            # Trade-Statistiken
            'total_trades': len(trades),
            'win_rate': self.win_rate(trades),
            'profit_factor': self.profit_factor(trades),
            'avg_win': self.avg_win(trades),
            'avg_loss': self.avg_loss(trades),
            'avg_trade': self.avg_trade(trades),
            'expectancy': self.expectancy(trades),

            # Weitere Metriken
            'recovery_factor': self.recovery_factor(equity_curve),
            'ulcer_index': self.ulcer_index(equity_curve),
        }

        return metrics

    def _calculate_returns(self, equity_curve: np.ndarray) -> np.ndarray:
        """Berechnet periodische Returns."""
        returns = np.diff(equity_curve) / equity_curve[:-1]
        return returns

    # ==================== Rendite-Metriken ====================

    def total_return(self, equity_curve: np.ndarray) -> float:
        """Gesamtrendite in Prozent."""
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve[-1] / equity_curve[0] - 1) * 100

    def annualized_return(
        self,
        equity_curve: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """Annualisierte Rendite."""
        if len(equity_curve) < 2:
            return 0.0

        total_return = equity_curve[-1] / equity_curve[0]
        n_periods = len(equity_curve) - 1
        years = n_periods / periods_per_year

        if years <= 0:
            return 0.0

        return (total_return ** (1 / years) - 1) * 100

    def cagr(
        self,
        equity_curve: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """Compound Annual Growth Rate."""
        return self.annualized_return(equity_curve, periods_per_year)

    # ==================== Risiko-Metriken ====================

    def volatility(
        self,
        returns: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """Annualisierte Volatilität."""
        if len(returns) < 2:
            return 0.0
        return np.std(returns) * np.sqrt(periods_per_year) * 100

    def max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Maximum Drawdown in Prozent."""
        if len(equity_curve) < 2:
            return 0.0

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return abs(np.min(drawdown)) * 100

    def avg_drawdown(self, equity_curve: np.ndarray) -> float:
        """Durchschnittlicher Drawdown."""
        if len(equity_curve) < 2:
            return 0.0

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return abs(np.mean(drawdown[drawdown < 0])) * 100 if np.any(drawdown < 0) else 0.0

    def drawdown_duration(self, equity_curve: np.ndarray) -> int:
        """Längste Drawdown-Periode in Perioden."""
        if len(equity_curve) < 2:
            return 0

        peak = np.maximum.accumulate(equity_curve)
        in_drawdown = equity_curve < peak

        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    # ==================== Risikoadjustierte Metriken ====================

    def sharpe_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """Sharpe Ratio (risikoadjustierte Rendite)."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - self.risk_free_rate / periods_per_year
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(periods_per_year)

    def sortino_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """Sortino Ratio (nur Downside-Volatilität)."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / periods_per_year
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0 if np.mean(excess_returns) <= 0 else float('inf')

        downside_std = np.std(downside_returns)
        return np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)

    def calmar_ratio(
        self,
        equity_curve: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """Calmar Ratio (CAGR / Max Drawdown)."""
        max_dd = self.max_drawdown(equity_curve)
        if max_dd == 0:
            return 0.0

        ann_return = self.annualized_return(equity_curve, periods_per_year)
        return ann_return / max_dd

    # ==================== Trade-Statistiken ====================

    def win_rate(self, trades: List[TradeResult]) -> float:
        """Gewinnrate in Prozent."""
        if not trades:
            return 0.0

        wins = sum(1 for t in trades if t.pnl > 0)
        return (wins / len(trades)) * 100

    def profit_factor(self, trades: List[TradeResult]) -> float:
        """Profit Factor (Bruttogewinn / Bruttoverlust)."""
        if not trades:
            return 0.0

        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def avg_win(self, trades: List[TradeResult]) -> float:
        """Durchschnittlicher Gewinn pro Winning Trade."""
        wins = [t.pnl for t in trades if t.pnl > 0]
        return np.mean(wins) if wins else 0.0

    def avg_loss(self, trades: List[TradeResult]) -> float:
        """Durchschnittlicher Verlust pro Losing Trade."""
        losses = [t.pnl for t in trades if t.pnl < 0]
        return np.mean(losses) if losses else 0.0

    def avg_trade(self, trades: List[TradeResult]) -> float:
        """Durchschnittlicher PnL pro Trade."""
        if not trades:
            return 0.0
        return np.mean([t.pnl for t in trades])

    def expectancy(self, trades: List[TradeResult]) -> float:
        """
        Erwartungswert pro Trade.
        (Win Rate * Avg Win) + (Loss Rate * Avg Loss)
        """
        if not trades:
            return 0.0

        win_rate = self.win_rate(trades) / 100
        avg_win = self.avg_win(trades)
        avg_loss = self.avg_loss(trades)

        return (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    def payoff_ratio(self, trades: List[TradeResult]) -> float:
        """Payoff Ratio (Avg Win / Avg Loss)."""
        avg_win = self.avg_win(trades)
        avg_loss = abs(self.avg_loss(trades))

        if avg_loss == 0:
            return float('inf') if avg_win > 0 else 0.0

        return avg_win / avg_loss

    # ==================== Weitere Metriken ====================

    def recovery_factor(self, equity_curve: np.ndarray) -> float:
        """Recovery Factor (Total Return / Max Drawdown)."""
        max_dd = self.max_drawdown(equity_curve)
        if max_dd == 0:
            return 0.0

        total_return = self.total_return(equity_curve)
        return total_return / max_dd

    def ulcer_index(self, equity_curve: np.ndarray) -> float:
        """
        Ulcer Index (Maß für Drawdown-Stress).
        Niedrigere Werte sind besser.
        """
        if len(equity_curve) < 2:
            return 0.0

        peak = np.maximum.accumulate(equity_curve)
        drawdown_pct = ((equity_curve - peak) / peak) * 100

        return np.sqrt(np.mean(drawdown_pct ** 2))

    def var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """Value at Risk."""
        if len(returns) < 2:
            return 0.0

        return np.percentile(returns, (1 - confidence) * 100) * 100

    def cvar(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """Conditional Value at Risk (Expected Shortfall)."""
        if len(returns) < 2:
            return 0.0

        var = np.percentile(returns, (1 - confidence) * 100)
        return np.mean(returns[returns <= var]) * 100

    def format_metrics(self, metrics: Dict[str, float]) -> str:
        """Formatiert Metriken für Ausgabe."""
        output = []
        output.append("=" * 50)
        output.append("BACKTEST RESULTS")
        output.append("=" * 50)

        # Rendite
        output.append("\n📈 RETURNS")
        output.append(f"  Total Return:      {metrics['total_return']:>10.2f}%")
        output.append(f"  Annualized Return: {metrics['annualized_return']:>10.2f}%")

        # Risiko
        output.append("\n📉 RISK")
        output.append(f"  Volatility:        {metrics['volatility']:>10.2f}%")
        output.append(f"  Max Drawdown:      {metrics['max_drawdown']:>10.2f}%")

        # Risikoadjustiert
        output.append("\n⚖️  RISK-ADJUSTED")
        output.append(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>10.2f}")
        output.append(f"  Sortino Ratio:     {metrics['sortino_ratio']:>10.2f}")
        output.append(f"  Calmar Ratio:      {metrics['calmar_ratio']:>10.2f}")

        # Trades
        output.append("\n🔄 TRADES")
        output.append(f"  Total Trades:      {metrics['total_trades']:>10}")
        output.append(f"  Win Rate:          {metrics['win_rate']:>10.2f}%")
        output.append(f"  Profit Factor:     {metrics['profit_factor']:>10.2f}")
        output.append(f"  Expectancy:        {metrics['expectancy']:>10.2f}")

        output.append("=" * 50)

        return "\n".join(output)
