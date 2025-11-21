# Risk Management module
"""
Risikomanagement für Trading-Operationen.

Enthält:
- RiskManager: Hauptklasse für Risikoüberwachung
- PositionSizer: Dynamische Positionsgrößenberechnung
- KillSwitch: Notfall-Stop bei anormalem Verhalten
"""

from .risk_manager import RiskManager
from .position_sizer import PositionSizer

__all__ = ['RiskManager', 'PositionSizer']
