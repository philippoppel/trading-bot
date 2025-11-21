"""
Unit Tests für Risk Management.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.risk_management.risk_manager import RiskManager, PositionSizer, RiskLevel


class TestRiskManager:
    """Tests für Risk Manager."""

    @pytest.fixture
    def risk_manager(self):
        return RiskManager(
            initial_capital=10000,
            max_position_pct=0.5,
            max_daily_loss_pct=0.10,
            max_drawdown_pct=0.20,
            max_consecutive_losses=3
        )

    def test_creation(self, risk_manager):
        """Test Erstellung."""
        assert risk_manager.initial_capital == 10000
        assert risk_manager.can_trade()

    def test_update_capital(self, risk_manager):
        """Test Kapital-Update."""
        risk_manager.update_capital(11000)
        assert risk_manager.current_capital == 11000
        assert risk_manager.peak_capital == 11000

    def test_drawdown_detection(self, risk_manager):
        """Test Drawdown-Erkennung."""
        risk_manager.update_capital(12000)  # Peak
        risk_manager.update_capital(9000)  # Drawdown

        state = risk_manager.check_risk()
        assert state.current_drawdown > 0.20
        assert not state.is_trading_allowed

    def test_consecutive_losses(self, risk_manager):
        """Test aufeinanderfolgende Verluste."""
        risk_manager.record_trade(-100)
        risk_manager.record_trade(-100)
        risk_manager.record_trade(-100)

        assert risk_manager.consecutive_losses == 3
        assert risk_manager.cooldown_until is not None

    def test_kill_switch(self, risk_manager):
        """Test Kill Switch."""
        risk_manager.activate_kill_switch("Test")

        state = risk_manager.check_risk()
        assert state.kill_switch_active
        assert not state.is_trading_allowed

        risk_manager.deactivate_kill_switch()
        assert risk_manager.can_trade()


class TestPositionSizer:
    """Tests für Position Sizer."""

    @pytest.fixture
    def sizer(self):
        return PositionSizer(
            base_risk_pct=0.02,
            kelly_fraction=0.5,
            max_position_pct=0.5
        )

    def test_fixed_fractional(self, sizer):
        """Test Fixed Fractional."""
        size = sizer.fixed_fractional(10000)
        assert size == 200  # 2% von 10000

    def test_kelly_criterion(self, sizer):
        """Test Kelly Criterion."""
        size = sizer.kelly_criterion(
            capital=10000,
            win_rate=0.6,
            avg_win=100,
            avg_loss=50
        )

        assert size > 0
        assert size <= 5000  # max 50%

    def test_volatility_adjusted(self, sizer):
        """Test Volatilitätsanpassung."""
        # Hohe Volatilität -> kleinere Position
        high_vol = sizer.volatility_adjusted(10000, 0.03, 0.02)
        low_vol = sizer.volatility_adjusted(10000, 0.01, 0.02)

        assert low_vol > high_vol

    def test_calculate_position_size(self, sizer):
        """Test Positionsberechnung mit Stop-Loss."""
        units = sizer.calculate_position_size(
            capital=10000,
            entry_price=100,
            stop_loss_price=95
        )

        # Risiko: 2% von 10000 = 200
        # Preis-Risiko: 5
        # Units: 200 / 5 = 40
        assert abs(units - 40) < 1

    def test_get_optimal_size(self, sizer):
        """Test alle Sizing-Methoden."""
        result = sizer.get_optimal_size(
            capital=10000,
            entry_price=100,
            stop_loss_price=95,
            win_rate=0.55,
            avg_win=150,
            avg_loss=100
        )

        assert 'fixed_fractional' in result
        assert 'risk_based' in result
        assert 'kelly' in result
        assert 'recommended' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
