# Backtesting module for strategy evaluation
"""
Backtesting-Engine für realistische Strategiesimulation.

Enthält:
- BacktestEngine: Hauptklasse für Backtests
- Metrics: Performance-Metriken (Sharpe, Sortino, etc.)
- WalkForward: Walk-Forward-Optimierung
"""

from .engine import BacktestEngine
from .metrics import BacktestMetrics
from .walk_forward import WalkForwardOptimizer

__all__ = ['BacktestEngine', 'BacktestMetrics', 'WalkForwardOptimizer']
