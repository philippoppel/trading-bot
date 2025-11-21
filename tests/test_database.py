"""
Unit Tests für Database Module.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import os

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestDatabaseManager:
    """Tests für DatabaseManager."""

    @pytest.fixture
    def db_manager(self):
        """Erstellt temporäre Datenbank."""
        from src.utils.database import DatabaseManager

        # Temporäre SQLite-Datenbank
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        db_url = f"sqlite:///{db_path}"
        manager = DatabaseManager(db_url)

        yield manager

        # Cleanup
        os.unlink(db_path)

    def test_log_trade(self, db_manager):
        """Test Trade-Logging."""
        trade_id = db_manager.log_trade(
            symbol='BTCUSDT',
            side='buy',
            price=45000.0,
            amount=0.1,
            fee=4.5
        )

        assert trade_id is not None
        assert trade_id > 0

    def test_get_trades(self, db_manager):
        """Test Trade-Abruf."""
        # Log einige Trades
        for i in range(5):
            db_manager.log_trade(
                symbol='BTCUSDT',
                side='buy' if i % 2 == 0 else 'sell',
                price=45000.0 + i * 100,
                amount=0.1
            )

        trades = db_manager.get_trades(symbol='BTCUSDT', limit=10)
        assert len(trades) == 5

    def test_log_portfolio_snapshot(self, db_manager):
        """Test Portfolio-Snapshot."""
        snapshot_id = db_manager.log_portfolio_snapshot(
            total_value=10500.0,
            cash_balance=5000.0,
            position_value=5500.0,
            unrealized_pnl=500.0,
            positions={'BTCUSDT': 0.1}
        )

        assert snapshot_id is not None

    def test_get_portfolio_history(self, db_manager):
        """Test Portfolio-History."""
        # Log einige Snapshots
        for i in range(10):
            db_manager.log_portfolio_snapshot(
                total_value=10000.0 + i * 100,
                cash_balance=5000.0
            )

        history = db_manager.get_portfolio_history(limit=100)
        assert len(history) == 10

    def test_log_alert(self, db_manager):
        """Test Alert-Logging."""
        alert_id = db_manager.log_alert(
            alert_type='drawdown',
            level='warning',
            title='Drawdown Warning',
            message='Current drawdown: 15%',
            data={'current_dd': 15.0, 'limit': 20.0}
        )

        assert alert_id is not None

    def test_get_alerts(self, db_manager):
        """Test Alert-Abruf."""
        db_manager.log_alert('test', 'info', 'Test', 'Test message')
        db_manager.log_alert('test', 'warning', 'Warning', 'Warning message')

        alerts = db_manager.get_alerts(limit=10)
        assert len(alerts) == 2

    def test_acknowledge_alert(self, db_manager):
        """Test Alert-Bestätigung."""
        alert_id = db_manager.log_alert('test', 'info', 'Test', 'Message')

        db_manager.acknowledge_alert(alert_id)

        alerts = db_manager.get_alerts(unacknowledged_only=True)
        assert len(alerts) == 0

    def test_log_prediction(self, db_manager):
        """Test Prediction-Logging."""
        pred_id = db_manager.log_prediction(
            symbol='BTCUSDT',
            model_name='lstm',
            prediction='up',
            confidence=0.75,
            probabilities={'up': 0.75, 'down': 0.15, 'sideways': 0.10}
        )

        assert pred_id is not None

    def test_update_prediction_outcome(self, db_manager):
        """Test Prediction-Outcome Update."""
        pred_id = db_manager.log_prediction(
            symbol='BTCUSDT',
            model_name='lstm',
            prediction='up',
            confidence=0.75
        )

        db_manager.update_prediction_outcome(pred_id, 'up')

        # Accuracy sollte 100% sein
        accuracy = db_manager.get_model_accuracy('lstm')
        assert accuracy['accuracy'] == 100.0

    def test_log_performance_metrics(self, db_manager):
        """Test Performance-Metriken."""
        metrics = {
            'total_return': 15.5,
            'sharpe_ratio': 1.8,
            'max_drawdown': 8.2,
            'win_rate': 58.0
        }

        metric_id = db_manager.log_performance_metrics('daily', metrics)
        assert metric_id is not None

        latest = db_manager.get_latest_metrics('daily')
        assert latest.sharpe_ratio == 1.8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
