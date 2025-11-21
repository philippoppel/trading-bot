"""
Risk Manager für Trading-Operationen.

Enthält:
- Position Sizing
- Drawdown-Limits
- Daily Stop Loss
- Kill Switch
- Volatilitätsbasierte Anpassungen
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger
from enum import Enum


class RiskLevel(Enum):
    """Risiko-Level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskState:
    """Aktueller Risiko-Zustand."""
    level: RiskLevel = RiskLevel.LOW
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    position_count: int = 0
    is_trading_allowed: bool = True
    kill_switch_active: bool = False
    messages: List[str] = field(default_factory=list)


class RiskManager:
    """
    Verwaltet Risiko für Trading-Operationen.

    Features:
    - Dynamische Position Sizing
    - Drawdown-Überwachung
    - Daily Stop Loss
    - Consecutive Loss Cooldown
    - Kill Switch
    """

    def __init__(
        self,
        initial_capital: float,
        max_position_pct: float = 0.5,
        max_daily_loss_pct: float = 0.10,
        max_drawdown_pct: float = 0.20,
        max_consecutive_losses: int = 5,
        cooldown_minutes: int = 30,
        volatility_window: int = 20
    ):
        """
        Args:
            initial_capital: Startkapital
            max_position_pct: Maximale Positionsgröße (% des Kapitals)
            max_daily_loss_pct: Maximaler Tagesverlust
            max_drawdown_pct: Maximaler Drawdown
            max_consecutive_losses: Max aufeinanderfolgende Verluste vor Cooldown
            cooldown_minutes: Cooldown-Dauer nach Verlustserie
            volatility_window: Fenster für Volatilitätsberechnung
        """
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_minutes = cooldown_minutes
        self.volatility_window = volatility_window

        # State
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_start_capital = initial_capital
        self.last_daily_reset = datetime.now().date()

        self.consecutive_losses = 0
        self.cooldown_until: Optional[datetime] = None
        self.trade_history: List[float] = []  # PnL pro Trade

        self.kill_switch_active = False
        self.kill_switch_reason = ""

        logger.info(f"RiskManager initialized with capital: {initial_capital}")

    def update_capital(self, new_capital: float):
        """
        Aktualisiert das aktuelle Kapital.

        Args:
            new_capital: Neuer Kapitalstand
        """
        self.current_capital = new_capital

        # Update Peak
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital

        # Daily Reset
        if datetime.now().date() != self.last_daily_reset:
            self.daily_start_capital = new_capital
            self.last_daily_reset = datetime.now().date()
            logger.info("Daily capital reset")

    def record_trade(self, pnl: float):
        """
        Zeichnet Trade-Ergebnis auf.

        Args:
            pnl: Profit/Loss des Trades
        """
        self.trade_history.append(pnl)

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Prüfe Consecutive Losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self._activate_cooldown()

    def _activate_cooldown(self):
        """Aktiviert Cooldown nach Verlustserie."""
        self.cooldown_until = datetime.now() + timedelta(minutes=self.cooldown_minutes)
        logger.warning(
            f"Cooldown activated until {self.cooldown_until} "
            f"({self.consecutive_losses} consecutive losses)"
        )

    def check_risk(self) -> RiskState:
        """
        Prüft aktuellen Risiko-Status.

        Returns:
            RiskState mit aktuellem Status
        """
        state = RiskState()
        state.consecutive_losses = self.consecutive_losses

        # Kill Switch
        if self.kill_switch_active:
            state.kill_switch_active = True
            state.is_trading_allowed = False
            state.level = RiskLevel.CRITICAL
            state.messages.append(f"Kill switch active: {self.kill_switch_reason}")
            return state

        # Cooldown
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).seconds // 60
            state.is_trading_allowed = False
            state.level = RiskLevel.HIGH
            state.messages.append(f"Cooldown active: {remaining} minutes remaining")

        # Drawdown
        if self.peak_capital > 0:
            state.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital

        if state.current_drawdown >= self.max_drawdown_pct:
            state.is_trading_allowed = False
            state.level = RiskLevel.CRITICAL
            state.messages.append(
                f"Max drawdown reached: {state.current_drawdown*100:.1f}% "
                f"(limit: {self.max_drawdown_pct*100:.1f}%)"
            )

        # Daily Loss
        state.daily_pnl = self.current_capital - self.daily_start_capital
        daily_loss_pct = -state.daily_pnl / self.daily_start_capital if state.daily_pnl < 0 else 0

        if daily_loss_pct >= self.max_daily_loss_pct:
            state.is_trading_allowed = False
            state.level = RiskLevel.CRITICAL
            state.messages.append(
                f"Daily loss limit reached: {daily_loss_pct*100:.1f}% "
                f"(limit: {self.max_daily_loss_pct*100:.1f}%)"
            )

        # Determine Risk Level
        if state.level == RiskLevel.LOW:
            if state.current_drawdown > self.max_drawdown_pct * 0.5:
                state.level = RiskLevel.MEDIUM
            elif state.current_drawdown > self.max_drawdown_pct * 0.75:
                state.level = RiskLevel.HIGH
            elif self.consecutive_losses >= self.max_consecutive_losses - 2:
                state.level = RiskLevel.MEDIUM

        return state

    def can_trade(self) -> bool:
        """Prüft ob Trading erlaubt ist."""
        state = self.check_risk()
        return state.is_trading_allowed

    def activate_kill_switch(self, reason: str = "Manual activation"):
        """
        Aktiviert Kill Switch.

        Args:
            reason: Grund für Aktivierung
        """
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def deactivate_kill_switch(self):
        """Deaktiviert Kill Switch."""
        self.kill_switch_active = False
        self.kill_switch_reason = ""
        logger.info("Kill switch deactivated")

    def reset_daily(self):
        """Setzt tägliche Metriken zurück."""
        self.daily_start_capital = self.current_capital
        self.last_daily_reset = datetime.now().date()
        logger.info("Daily metrics reset")


class PositionSizer:
    """Berechnet optimale Positionsgrößen."""

    def __init__(
        self,
        base_risk_pct: float = 0.02,
        kelly_fraction: float = 0.5,
        max_position_pct: float = 0.5
    ):
        """
        Args:
            base_risk_pct: Basis-Risiko pro Trade
            kelly_fraction: Anteil des Kelly-Kriteriums
            max_position_pct: Maximale Positionsgröße
        """
        self.base_risk_pct = base_risk_pct
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct

    def fixed_fractional(
        self,
        capital: float,
        risk_pct: Optional[float] = None
    ) -> float:
        """
        Fixed Fractional Position Sizing.

        Args:
            capital: Verfügbares Kapital
            risk_pct: Risiko-Prozent (default: base_risk_pct)

        Returns:
            Positionsgröße in Kapitaleinheiten
        """
        risk = risk_pct or self.base_risk_pct
        return capital * min(risk, self.max_position_pct)

    def kelly_criterion(
        self,
        capital: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Kelly Criterion Position Sizing.

        Args:
            capital: Verfügbares Kapital
            win_rate: Gewinnrate (0-1)
            avg_win: Durchschnittlicher Gewinn
            avg_loss: Durchschnittlicher Verlust (positiv)

        Returns:
            Positionsgröße
        """
        if avg_loss == 0:
            return self.fixed_fractional(capital)

        # Kelly Formel: f* = (bp - q) / b
        # b = avg_win / avg_loss (odds)
        # p = win_rate
        # q = 1 - p
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        kelly_pct = (b * p - q) / b

        # Verwende nur Anteil des Kelly-Wertes
        kelly_pct *= self.kelly_fraction

        # Begrenzen
        kelly_pct = max(0, min(kelly_pct, self.max_position_pct))

        return capital * kelly_pct

    def volatility_adjusted(
        self,
        capital: float,
        current_volatility: float,
        base_volatility: float,
        risk_pct: Optional[float] = None
    ) -> float:
        """
        Volatilitätsangepasste Position Sizing.

        Args:
            capital: Verfügbares Kapital
            current_volatility: Aktuelle Volatilität
            base_volatility: Basis-Volatilität
            risk_pct: Basis-Risiko

        Returns:
            Angepasste Positionsgröße
        """
        risk = risk_pct or self.base_risk_pct

        if current_volatility == 0:
            return self.fixed_fractional(capital, risk)

        # Reduziere Position bei hoher Volatilität
        vol_ratio = base_volatility / current_volatility
        adjusted_risk = risk * vol_ratio

        return capital * min(adjusted_risk, self.max_position_pct)

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Berechnet Positionsgröße basierend auf Stop-Loss.

        Args:
            capital: Verfügbares Kapital
            entry_price: Einstiegspreis
            stop_loss_price: Stop-Loss Preis

        Returns:
            Anzahl der Einheiten
        """
        risk_amount = capital * self.base_risk_pct
        price_risk = abs(entry_price - stop_loss_price)

        if price_risk == 0:
            return 0

        units = risk_amount / price_risk

        # Prüfe max Position
        max_units = (capital * self.max_position_pct) / entry_price

        return min(units, max_units)

    def get_optimal_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: float,
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        current_volatility: Optional[float] = None,
        base_volatility: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Berechnet alle Positionsgrößen-Methoden.

        Returns:
            Dictionary mit verschiedenen Größenberechnungen
        """
        results = {}

        # Fixed Fractional
        results['fixed_fractional'] = self.fixed_fractional(capital)

        # Risk-based
        results['risk_based'] = self.calculate_position_size(
            capital, entry_price, stop_loss_price
        ) * entry_price

        # Kelly
        results['kelly'] = self.kelly_criterion(
            capital, win_rate, avg_win, avg_loss
        )

        # Volatility-adjusted
        if current_volatility and base_volatility:
            results['volatility_adjusted'] = self.volatility_adjusted(
                capital, current_volatility, base_volatility
            )

        # Empfohlene Größe (Durchschnitt)
        valid_sizes = [v for v in results.values() if v > 0]
        results['recommended'] = np.mean(valid_sizes) if valid_sizes else 0

        return results
