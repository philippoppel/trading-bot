"""
Backtesting Engine für realistische Strategiesimulation.

Features:
- OHLCV-basierte Simulation
- Gebühren, Slippage, Latenz
- Stop-Loss/Take-Profit
- Detaillierte Trade-Aufzeichnung
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from loguru import logger
from enum import Enum

from .metrics import BacktestMetrics, TradeResult


class OrderType(Enum):
    """Order-Typen."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class Side(Enum):
    """Trade-Richtung."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """Aktive Position."""
    side: Side
    entry_price: float
    entry_time: pd.Timestamp
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Order:
    """Pending Order."""
    order_type: OrderType
    side: Side
    price: Optional[float]  # None für Market Orders
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class BacktestConfig:
    """Konfiguration für Backtest."""
    initial_capital: float = 10000.0
    fee_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    latency_bars: int = 0  # Verzögerung in Bars
    max_position_size: float = 1.0  # Max 100% des Kapitals
    allow_shorting: bool = True


class BacktestEngine:
    """
    Hauptklasse für Backtesting.

    Simuliert Trading mit:
    - Realistischen Gebühren
    - Slippage
    - Latenz
    - Stop-Loss/Take-Profit
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Args:
            config: Backtest-Konfiguration
        """
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self):
        """Setzt den Backtest zurück."""
        self.capital = self.config.initial_capital
        self.position: Optional[Position] = None
        self.pending_orders: List[Order] = []
        self.trades: List[TradeResult] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[pd.Timestamp] = []

    def run(
        self,
        data: pd.DataFrame,
        strategy: Callable[[pd.DataFrame, int, Optional[Position]], Optional[Order]],
        verbose: bool = True
    ) -> Dict:
        """
        Führt Backtest aus.

        Args:
            data: OHLCV DataFrame mit Spalten: open, high, low, close, volume
            strategy: Strategie-Funktion die Orders generiert
                      Signatur: (data, current_idx, position) -> Order oder None
            verbose: Detaillierte Ausgabe

        Returns:
            Dictionary mit Ergebnissen und Metriken
        """
        self.reset()

        if verbose:
            logger.info(f"Starting backtest with {len(data)} bars")
            logger.info(f"Initial capital: ${self.config.initial_capital:,.2f}")

        # Hauptschleife
        for i in range(len(data)):
            current_bar = data.iloc[i]
            timestamp = data.index[i]

            # 1. Prüfe Stop-Loss/Take-Profit
            self._check_stop_orders(current_bar, timestamp)

            # 2. Führe pending Orders aus
            self._execute_pending_orders(current_bar, timestamp)

            # 3. Generiere neuen Order von Strategie
            order = strategy(data, i, self.position)
            if order:
                # Verzögerung durch Latenz
                if self.config.latency_bars > 0:
                    self.pending_orders.append(order)
                else:
                    self._execute_order(order, current_bar, timestamp)

            # 4. Update Equity
            self._update_equity(current_bar['close'], timestamp)

        # Schließe offene Position am Ende
        if self.position:
            self._close_position(data.iloc[-1]['close'], data.index[-1], "End of backtest")

        # Berechne Metriken
        metrics_calc = BacktestMetrics()
        equity_array = np.array(self.equity_curve)
        metrics = metrics_calc.calculate_all(
            equity_array,
            self.trades,
            periods_per_year=self._estimate_periods_per_year(data)
        )

        if verbose:
            logger.info(f"Backtest completed. Final equity: ${self.equity_curve[-1]:,.2f}")
            logger.info(f"Total trades: {len(self.trades)}")

        return {
            'metrics': metrics,
            'trades': self.trades,
            'equity_curve': equity_array,
            'timestamps': self.timestamps,
            'final_capital': self.equity_curve[-1] if self.equity_curve else self.config.initial_capital
        }

    def _execute_order(
        self,
        order: Order,
        bar: pd.Series,
        timestamp: pd.Timestamp
    ):
        """Führt einen Order aus."""
        # Berechne Execution-Preis mit Slippage
        if order.order_type == OrderType.MARKET:
            base_price = bar['close']
        else:
            base_price = order.price

        # Slippage (abhängig von Richtung)
        if order.side == Side.LONG:
            exec_price = base_price * (1 + self.config.slippage_rate)
        else:
            exec_price = base_price * (1 - self.config.slippage_rate)

        # Prüfe ob Position geschlossen werden soll
        if self.position:
            if (self.position.side == Side.LONG and order.side == Side.SHORT) or \
               (self.position.side == Side.SHORT and order.side == Side.LONG):
                self._close_position(exec_price, timestamp, "Strategy signal")
                return

        # Öffne neue Position
        if not self.position:
            # Berechne Position Size
            max_size = (self.capital * self.config.max_position_size) / exec_price
            size = min(order.size, max_size)

            if size * exec_price < 10:  # Mindestordergröße
                return

            # Gebühren
            fees = size * exec_price * self.config.fee_rate
            self.capital -= fees

            self.position = Position(
                side=order.side,
                entry_price=exec_price,
                entry_time=timestamp,
                size=size,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit
            )

            logger.debug(f"Opened {order.side.value} position at {exec_price:.2f}, size: {size:.4f}")

    def _close_position(
        self,
        price: float,
        timestamp: pd.Timestamp,
        reason: str = ""
    ):
        """Schließt die aktuelle Position."""
        if not self.position:
            return

        # Berechne PnL
        if self.position.side == Side.LONG:
            pnl = (price - self.position.entry_price) * self.position.size
        else:
            pnl = (self.position.entry_price - price) * self.position.size

        # Slippage beim Schließen
        slippage_cost = price * self.position.size * self.config.slippage_rate
        pnl -= slippage_cost

        # Gebühren
        fees = price * self.position.size * self.config.fee_rate
        pnl -= fees

        # Update Capital
        self.capital += self.position.size * self.position.entry_price + pnl

        # Trade aufzeichnen
        trade = TradeResult(
            entry_time=self.position.entry_time,
            exit_time=timestamp,
            entry_price=self.position.entry_price,
            exit_price=price,
            side=self.position.side.value,
            size=self.position.size,
            pnl=pnl,
            pnl_pct=(pnl / (self.position.entry_price * self.position.size)) * 100,
            fees=fees + slippage_cost
        )
        self.trades.append(trade)

        logger.debug(f"Closed position at {price:.2f}, PnL: {pnl:.2f} ({reason})")

        self.position = None

    def _check_stop_orders(self, bar: pd.Series, timestamp: pd.Timestamp):
        """Prüft Stop-Loss und Take-Profit."""
        if not self.position:
            return

        if self.position.side == Side.LONG:
            # Stop-Loss
            if self.position.stop_loss and bar['low'] <= self.position.stop_loss:
                self._close_position(self.position.stop_loss, timestamp, "Stop-Loss")
                return

            # Take-Profit
            if self.position.take_profit and bar['high'] >= self.position.take_profit:
                self._close_position(self.position.take_profit, timestamp, "Take-Profit")
                return

        else:  # SHORT
            # Stop-Loss
            if self.position.stop_loss and bar['high'] >= self.position.stop_loss:
                self._close_position(self.position.stop_loss, timestamp, "Stop-Loss")
                return

            # Take-Profit
            if self.position.take_profit and bar['low'] <= self.position.take_profit:
                self._close_position(self.position.take_profit, timestamp, "Take-Profit")
                return

    def _execute_pending_orders(self, bar: pd.Series, timestamp: pd.Timestamp):
        """Führt verzögerte Orders aus."""
        if not self.pending_orders:
            return

        order = self.pending_orders.pop(0)
        self._execute_order(order, bar, timestamp)

    def _update_equity(self, current_price: float, timestamp: pd.Timestamp):
        """Aktualisiert die Equity Curve."""
        equity = self.capital

        if self.position:
            # Mark-to-Market
            if self.position.side == Side.LONG:
                unrealized = (current_price - self.position.entry_price) * self.position.size
            else:
                unrealized = (self.position.entry_price - current_price) * self.position.size

            equity += self.position.size * self.position.entry_price + unrealized

        self.equity_curve.append(equity)
        self.timestamps.append(timestamp)

    def _estimate_periods_per_year(self, data: pd.DataFrame) -> int:
        """Schätzt Perioden pro Jahr basierend auf Daten."""
        if len(data) < 2:
            return 252

        # Durchschnittliche Zeitdifferenz
        time_diff = (data.index[-1] - data.index[0]) / len(data)

        if time_diff.total_seconds() < 3600:  # < 1h
            return 525600  # Minuten
        elif time_diff.total_seconds() < 86400:  # < 1d
            return 8760  # Stunden
        else:
            return 252  # Tage


def create_simple_strategy(
    signal_column: str = 'signal',
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.10
) -> Callable:
    """
    Erstellt eine einfache Signal-basierte Strategie.

    Args:
        signal_column: Spaltenname mit Signalen (1=buy, -1=sell, 0=hold)
        stop_loss_pct: Stop-Loss in Prozent
        take_profit_pct: Take-Profit in Prozent

    Returns:
        Strategie-Funktion
    """
    def strategy(
        data: pd.DataFrame,
        idx: int,
        position: Optional[Position]
    ) -> Optional[Order]:
        if signal_column not in data.columns:
            return None

        signal = data.iloc[idx][signal_column]
        price = data.iloc[idx]['close']

        if signal == 1 and not position:  # Buy
            return Order(
                order_type=OrderType.MARKET,
                side=Side.LONG,
                price=None,
                size=float('inf'),  # Max size
                stop_loss=price * (1 - stop_loss_pct),
                take_profit=price * (1 + take_profit_pct)
            )
        elif signal == -1 and position:  # Sell
            return Order(
                order_type=OrderType.MARKET,
                side=Side.SHORT,
                price=None,
                size=float('inf')
            )

        return None

    return strategy
