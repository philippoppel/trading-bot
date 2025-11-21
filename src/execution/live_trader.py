"""
Live Trading Engine.

Führt Trades in Echtzeit aus mit:
- Model-basierte Signale
- Risk Management Integration
- Position Tracking
- WebSocket Streaming
"""

import time
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from loguru import logger
from datetime import datetime
import threading

from .order_manager import OrderManager, OrderResult, OrderStatus


@dataclass
class Position:
    """Aktive Position."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    amount: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0


@dataclass
class TradeStats:
    """Trading-Statistiken."""
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_equity: float = 0.0
    peak_equity: float = 0.0


class LiveTrader:
    """
    Live Trading Engine für automatisiertes Trading.

    Features:
    - Echtzeit Signal-Ausführung
    - Position Management
    - Risk Checks
    - Graceful Shutdown
    """

    def __init__(
        self,
        order_manager: OrderManager,
        symbols: list[str],
        initial_capital: float = 10000.0,
        max_position_size: float = 0.5,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        cooldown_seconds: int = 60
    ):
        """
        Args:
            order_manager: Order Manager Instanz
            symbols: Liste der zu handelnden Symbole
            initial_capital: Startkapital
            max_position_size: Max Position als Anteil des Kapitals
            stop_loss_pct: Standard Stop-Loss Prozent
            take_profit_pct: Standard Take-Profit Prozent
            cooldown_seconds: Mindestzeit zwischen Trades
        """
        self.order_manager = order_manager
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.cooldown_seconds = cooldown_seconds

        # State
        self.positions: Dict[str, Position] = {}
        self.stats = TradeStats(current_equity=initial_capital, peak_equity=initial_capital)
        self.last_trade_time: Dict[str, datetime] = {}
        self.running = False
        self.paused = False

        # Callbacks
        self.on_trade_callback: Optional[Callable] = None
        self.on_signal_callback: Optional[Callable] = None

        logger.info(f"LiveTrader initialized for {symbols}")

    def set_callbacks(
        self,
        on_trade: Optional[Callable] = None,
        on_signal: Optional[Callable] = None
    ):
        """Setzt Callbacks für Events."""
        self.on_trade_callback = on_trade
        self.on_signal_callback = on_signal

    def start(
        self,
        signal_generator: Callable[[str], int],
        interval_seconds: int = 60
    ):
        """
        Startet den Live-Trading-Loop.

        Args:
            signal_generator: Funktion die Signale generiert (symbol) -> signal (-1, 0, 1)
            interval_seconds: Intervall zwischen Signal-Checks
        """
        self.running = True
        logger.info("Starting live trading...")

        try:
            while self.running:
                if not self.paused:
                    self._trading_cycle(signal_generator)

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()

    def _trading_cycle(self, signal_generator: Callable):
        """Ein Trading-Zyklus."""
        for symbol in self.symbols:
            try:
                # Prüfe Cooldown
                if not self._check_cooldown(symbol):
                    continue

                # Generiere Signal
                signal = signal_generator(symbol)

                if self.on_signal_callback:
                    self.on_signal_callback(symbol, signal)

                # Update Position
                self._update_position(symbol)

                # Verarbeite Signal
                if signal != 0:
                    self._process_signal(symbol, signal)

            except Exception as e:
                logger.error(f"Error in trading cycle for {symbol}: {e}")

        # Update Stats
        self._update_stats()

    def _check_cooldown(self, symbol: str) -> bool:
        """Prüft ob Cooldown abgelaufen ist."""
        if symbol not in self.last_trade_time:
            return True

        elapsed = (datetime.now() - self.last_trade_time[symbol]).total_seconds()
        return elapsed >= self.cooldown_seconds

    def _update_position(self, symbol: str):
        """Aktualisiert Position mit aktuellem Preis."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        ticker = self.order_manager.get_ticker(symbol)
        current_price = ticker['last']

        # Berechne unrealized PnL
        if pos.side == 'long':
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.amount
        else:
            pos.unrealized_pnl = (pos.entry_price - current_price) * pos.amount

        # Prüfe Stop-Loss
        if pos.stop_loss:
            if (pos.side == 'long' and current_price <= pos.stop_loss) or \
               (pos.side == 'short' and current_price >= pos.stop_loss):
                logger.warning(f"Stop-loss triggered for {symbol}")
                self._close_position(symbol, reason="Stop-Loss")
                return

        # Prüfe Take-Profit
        if pos.take_profit:
            if (pos.side == 'long' and current_price >= pos.take_profit) or \
               (pos.side == 'short' and current_price <= pos.take_profit):
                logger.info(f"Take-profit triggered for {symbol}")
                self._close_position(symbol, reason="Take-Profit")
                return

    def _process_signal(self, symbol: str, signal: int):
        """Verarbeitet Trading-Signal."""
        has_position = symbol in self.positions

        if signal == 1:  # Buy
            if has_position and self.positions[symbol].side == 'short':
                self._close_position(symbol, reason="Signal reversal")
            if not has_position:
                self._open_position(symbol, 'long')

        elif signal == -1:  # Sell
            if has_position and self.positions[symbol].side == 'long':
                self._close_position(symbol, reason="Signal reversal")
            elif not has_position:
                self._open_position(symbol, 'short')

    def _open_position(self, symbol: str, side: str):
        """Öffnet neue Position."""
        try:
            # Hole aktuellen Preis
            ticker = self.order_manager.get_ticker(symbol)
            price = ticker['last']

            # Berechne Position Size
            balance = self.order_manager.get_balance('USDT')
            available = balance.get('free', 0)
            position_value = available * self.max_position_size
            amount = position_value / price

            if amount <= 0:
                logger.warning(f"Insufficient balance for {symbol}")
                return

            # Berechne SL/TP
            if side == 'long':
                stop_loss = price * (1 - self.stop_loss_pct)
                take_profit = price * (1 + self.take_profit_pct)
                order_side = 'buy'
            else:
                stop_loss = price * (1 + self.stop_loss_pct)
                take_profit = price * (1 - self.take_profit_pct)
                order_side = 'sell'

            # Erstelle Order
            result = self.order_manager.create_market_order(symbol, order_side, amount)

            if result.status == OrderStatus.CLOSED:
                # Position erstellen
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    entry_price=result.price or price,
                    amount=result.filled,
                    entry_time=datetime.now(),
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

                self.last_trade_time[symbol] = datetime.now()
                self.stats.total_trades += 1

                logger.info(f"Opened {side} position: {result.filled} {symbol} @ {result.price}")

                if self.on_trade_callback:
                    self.on_trade_callback('open', symbol, side, result)

        except Exception as e:
            logger.error(f"Error opening position for {symbol}: {e}")

    def _close_position(self, symbol: str, reason: str = ""):
        """Schließt Position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        try:
            # Erstelle Close Order
            order_side = 'sell' if pos.side == 'long' else 'buy'
            result = self.order_manager.create_market_order(symbol, order_side, pos.amount)

            if result.status == OrderStatus.CLOSED:
                # Berechne PnL
                exit_price = result.price or self.order_manager.get_ticker(symbol)['last']

                if pos.side == 'long':
                    pnl = (exit_price - pos.entry_price) * pos.amount
                else:
                    pnl = (pos.entry_price - exit_price) * pos.amount

                pnl -= result.fee

                # Update Stats
                self.stats.total_pnl += pnl
                if pnl > 0:
                    self.stats.winning_trades += 1

                self.last_trade_time[symbol] = datetime.now()

                logger.info(f"Closed {pos.side} position: {symbol}, PnL: {pnl:.2f} ({reason})")

                if self.on_trade_callback:
                    self.on_trade_callback('close', symbol, pos.side, result, pnl)

                # Position entfernen
                del self.positions[symbol]

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")

    def _update_stats(self):
        """Aktualisiert Trading-Statistiken."""
        # Berechne aktuelle Equity
        balance = self.order_manager.get_balance('USDT')
        equity = balance.get('total', 0)

        # Addiere unrealized PnL
        for pos in self.positions.values():
            equity += pos.unrealized_pnl

        self.stats.current_equity = equity

        # Update Peak und Drawdown
        if equity > self.stats.peak_equity:
            self.stats.peak_equity = equity

        if self.stats.peak_equity > 0:
            drawdown = (self.stats.peak_equity - equity) / self.stats.peak_equity
            self.stats.max_drawdown = max(self.stats.max_drawdown, drawdown)

    def stop(self):
        """Stoppt den Trader."""
        self.running = False
        logger.info("Stopping live trader...")

        # Schließe alle Positionen
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, reason="Shutdown")

        # Storniere alle Orders
        self.order_manager.cancel_all_orders()

        logger.info("Live trader stopped")

    def pause(self):
        """Pausiert den Trader."""
        self.paused = True
        logger.info("Live trader paused")

    def resume(self):
        """Setzt den Trader fort."""
        self.paused = False
        logger.info("Live trader resumed")

    def emergency_stop(self):
        """Notfall-Stop: Schließt alles sofort."""
        logger.warning("EMERGENCY STOP triggered!")
        self.stop()

    def get_status(self) -> Dict:
        """Gibt aktuellen Status zurück."""
        return {
            'running': self.running,
            'paused': self.paused,
            'positions': {
                symbol: {
                    'side': pos.side,
                    'entry_price': pos.entry_price,
                    'amount': pos.amount,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit
                }
                for symbol, pos in self.positions.items()
            },
            'stats': {
                'total_trades': self.stats.total_trades,
                'winning_trades': self.stats.winning_trades,
                'win_rate': (self.stats.winning_trades / self.stats.total_trades * 100)
                           if self.stats.total_trades > 0 else 0,
                'total_pnl': self.stats.total_pnl,
                'current_equity': self.stats.current_equity,
                'max_drawdown': self.stats.max_drawdown * 100
            }
        }

    def get_positions_df(self) -> pd.DataFrame:
        """Gibt Positionen als DataFrame zurück."""
        if not self.positions:
            return pd.DataFrame()

        data = []
        for symbol, pos in self.positions.items():
            data.append({
                'Symbol': symbol,
                'Side': pos.side,
                'Entry Price': pos.entry_price,
                'Amount': pos.amount,
                'Unrealized PnL': pos.unrealized_pnl,
                'Stop Loss': pos.stop_loss,
                'Take Profit': pos.take_profit,
                'Entry Time': pos.entry_time
            })

        return pd.DataFrame(data)
