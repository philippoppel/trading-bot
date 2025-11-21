# Monitoring module for dashboard and alerts
"""
Monitoring und Alerting System.

Enthält:
- Dashboard: Streamlit-basiertes Live-Dashboard
- AlertManager: Telegram/Discord Benachrichtigungen
"""

from .alert_manager import AlertManager

__all__ = ['AlertManager']
