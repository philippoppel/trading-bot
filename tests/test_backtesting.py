"""
Unit Tests für Backtesting-Engine.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtesting.engine import BacktestEngine, BacktestConfig, Order, OrderType, Side, create_simple_strategy
from src.backtesting.metrics import BacktestMetrics, TradeResult
from src.backtesting.walk_forward import WalkForwardOptimizer


class TestBacktestMetrics:
    """Tests für Backtesting-Metriken."""

    @pytest.fixture
    def metrics(self):
        return BacktestMetrics()

    @pytest.fixture
    def equity_curve(self):
        # Simulierte Equity Curve
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01
        equity = 10000 * np.cumprod(1 + returns)
        return equity

    @pytest.fixture
    def trades(self):
        trades = []
        for i in range(20):
            pnl = np.random.uniform(-100, 150)
            trades.append(TradeResult(
                entry_time=pd.Timestamp.now(),
                exit_time=pd.Timestamp.now(),
                entry_price=100,
                exit_price=100 + pnl/10,
                side='long',
                size=10,
                pnl=pnl,
                pnl_pct=pnl/1000,
                fees=1.0
            ))
        return trades

    def test_total_return(self, metrics, equity_curve):
        """Test Gesamtrendite."""
        result = metrics.total_return(equity_curve)
        expected = (equity_curve[-1] / equity_curve[0] - 1) * 100
        assert abs(result - expected) < 0.01

    def test_max_drawdown(self, metrics, equity_curve):
        """Test Max Drawdown."""
        result = metrics.max_drawdown(equity_curve)
        assert result >= 0
        assert result <= 100

    def test_sharpe_ratio(self, metrics, equity_curve):
        """Test Sharpe Ratio."""
        returns = np.diff(equity_curve) / equity_curve[:-1]
        result = metrics.sharpe_ratio(returns)
        assert isinstance(result, float)

    def test_win_rate(self, metrics, trades):
        """Test Win Rate."""
        result = metrics.win_rate(trades)
        assert 0 <= result <= 100

    def test_profit_factor(self, metrics, trades):
        """Test Profit Factor."""
        result = metrics.profit_factor(trades)
        assert result >= 0

    def test_calculate_all(self, metrics, equity_curve, trades):
        """Test alle Metriken."""
        result = metrics.calculate_all(equity_curve, trades)

        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result
        assert 'win_rate' in result
        assert 'profit_factor' in result


class TestBacktestEngine:
    """Tests für Backtesting-Engine."""

    @pytest.fixture
    def config(self):
        return BacktestConfig(
            initial_capital=10000,
            fee_rate=0.001,
            slippage_rate=0.0005
        )

    @pytest.fixture
    def sample_data(self):
        """Erstellt Sample OHLCV-Daten."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        np.random.seed(42)

        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high = close + np.abs(np.random.randn(100) * 0.3)
        low = close - np.abs(np.random.randn(100) * 0.3)
        open_ = close + np.random.randn(100) * 0.2

        df = pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)

        return df

    def test_engine_creation(self, config):
        """Test Engine-Erstellung."""
        engine = BacktestEngine(config)
        assert engine.capital == 10000

    def test_simple_strategy(self, config, sample_data):
        """Test mit einfacher Strategie."""
        # Füge Signal-Spalte hinzu
        sample_data['signal'] = 0
        sample_data.iloc[10, sample_data.columns.get_loc('signal')] = 1  # Buy
        sample_data.iloc[50, sample_data.columns.get_loc('signal')] = -1  # Sell

        engine = BacktestEngine(config)
        strategy = create_simple_strategy()

        result = engine.run(sample_data, strategy, verbose=False)

        assert 'metrics' in result
        assert 'trades' in result
        assert 'equity_curve' in result

    def test_fees_applied(self, config, sample_data):
        """Test dass Gebühren angewendet werden."""
        sample_data['signal'] = 0
        sample_data.iloc[10, sample_data.columns.get_loc('signal')] = 1
        sample_data.iloc[20, sample_data.columns.get_loc('signal')] = -1

        engine = BacktestEngine(config)
        strategy = create_simple_strategy()

        result = engine.run(sample_data, strategy, verbose=False)

        if result['trades']:
            # Trades sollten Gebühren haben
            assert result['trades'][0].fees > 0


class TestWalkForwardOptimizer:
    """Tests für Walk-Forward-Optimierung."""

    @pytest.fixture
    def optimizer(self):
        return WalkForwardOptimizer(
            train_pct=0.7,
            n_windows=3,
            optimization_metric='sharpe_ratio'
        )

    @pytest.fixture
    def sample_data(self):
        """Erstellt längere Sample-Daten."""
        dates = pd.date_range(start='2023-01-01', periods=500, freq='H')
        np.random.seed(42)

        close = 100 + np.cumsum(np.random.randn(500) * 0.5)

        df = pd.DataFrame({
            'open': close + np.random.randn(500) * 0.1,
            'high': close + np.abs(np.random.randn(500) * 0.3),
            'low': close - np.abs(np.random.randn(500) * 0.3),
            'close': close,
            'volume': np.random.uniform(1000, 5000, 500)
        }, index=dates)

        return df

    def test_create_windows(self, optimizer, sample_data):
        """Test Fenster-Erstellung."""
        windows = optimizer.create_windows(sample_data)

        assert len(windows) == 3
        for w in windows:
            assert w.train_start < w.train_end
            assert w.train_end <= w.test_start
            assert w.test_start < w.test_end


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
