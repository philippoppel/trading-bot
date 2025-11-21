"""
Paper trading system for live simulation with virtual money.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.data.binance_client import BinanceDataClient
from src.data.preprocessor import DataPreprocessor
from src.utils.logger import get_logger

logger = get_logger()


class Position:
    """Represents a trading position."""

    def __init__(
        self,
        symbol: str,
        entry_price: float,
        amount: float,
        entry_time: datetime
    ):
        self.symbol = symbol
        self.entry_price = entry_price
        self.amount = amount
        self.entry_time = entry_time
        self.exit_price: float | None = None
        self.exit_time: datetime | None = None
        self.closed = False

    def close(self, exit_price: float, exit_time: datetime) -> float:
        """
        Close the position and calculate profit.

        Returns:
            Profit/loss from the trade
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.closed = True

        profit = (self.exit_price - self.entry_price) * self.amount
        return profit

    @property
    def current_value(self) -> float:
        """Get current value at entry price."""
        return self.entry_price * self.amount

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized profit/loss."""
        return (current_price - self.entry_price) * self.amount

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "amount": self.amount,
            "entry_time": self.entry_time.isoformat(),
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "closed": self.closed
        }


class PaperTrader:
    """
    Paper trading system that simulates live trading with virtual money.

    Tracks positions, balance, and provides risk management.
    """

    def __init__(self, initial_balance: float | None = None):
        """
        Initialize the paper trader.

        Args:
            initial_balance: Starting balance in USDT
        """
        self.settings = get_settings()

        if initial_balance is None:
            initial_balance = self.settings.trading.initial_balance

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: dict[str, Position] = {}  # symbol -> Position
        self.trade_history: list[dict] = []
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.utcnow().date()

        # Risk management settings
        self.risk_config = self.settings.paper_trading.risk_management

        # Data client
        self.data_client = BinanceDataClient()

        logger.info(f"Paper trader initialized with balance: {self.balance}")

    def _check_daily_reset(self) -> None:
        """Reset daily PnL counter if new day."""
        current_date = datetime.utcnow().date()
        if current_date > self.last_reset_date:
            logger.info(f"Daily reset. Previous day PnL: {self.daily_pnl:.2f}")
            self.daily_pnl = 0.0
            self.last_reset_date = current_date

    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value (balance + open positions).

        Returns:
            Total portfolio value in USDT
        """
        total = self.balance

        for symbol, position in self.positions.items():
            if not position.closed:
                current_price = self.data_client.get_current_price(symbol)
                total += position.amount * current_price

        return total

    def can_open_position(self, symbol: str, amount_usdt: float) -> tuple[bool, str]:
        """
        Check if a new position can be opened.

        Args:
            symbol: Trading pair symbol
            amount_usdt: Amount to invest in USDT

        Returns:
            Tuple of (allowed, reason)
        """
        self._check_daily_reset()

        # Check if already in position
        if symbol in self.positions and not self.positions[symbol].closed:
            return False, f"Already have open position in {symbol}"

        # Check balance
        if amount_usdt > self.balance:
            return False, f"Insufficient balance. Required: {amount_usdt}, Available: {self.balance}"

        # Check max position size
        portfolio_value = self.get_portfolio_value()
        max_position_value = portfolio_value * self.risk_config.max_position_pct

        if amount_usdt > max_position_value:
            return False, f"Position size {amount_usdt} exceeds max {max_position_value:.2f}"

        # Check daily loss limit
        if self.daily_pnl < -portfolio_value * self.risk_config.daily_loss_limit:
            return False, f"Daily loss limit reached. PnL: {self.daily_pnl:.2f}"

        return True, "OK"

    def open_position(
        self,
        symbol: str,
        amount_usdt: float,
        price: float | None = None
    ) -> Position | None:
        """
        Open a new position.

        Args:
            symbol: Trading pair symbol
            amount_usdt: Amount to invest in USDT
            price: Execution price (uses current price if None)

        Returns:
            Position object if successful, None otherwise
        """
        # Check if allowed
        allowed, reason = self.can_open_position(symbol, amount_usdt)
        if not allowed:
            logger.warning(f"Cannot open position: {reason}")
            return None

        # Get current price if not provided
        if price is None:
            price = self.data_client.get_current_price(symbol)

        # Apply slippage
        execution_price = price * (1 + self.settings.trading.slippage)

        # Calculate fees
        fee = amount_usdt * self.settings.trading.trading_fee

        # Calculate amount of crypto to buy
        amount_crypto = (amount_usdt - fee) / execution_price

        # Create position
        position = Position(
            symbol=symbol,
            entry_price=execution_price,
            amount=amount_crypto,
            entry_time=datetime.utcnow()
        )

        # Update balance
        self.balance -= amount_usdt
        self.positions[symbol] = position

        logger.info(
            f"Opened position: {symbol}, "
            f"amount={amount_crypto:.6f}, "
            f"price={execution_price:.2f}, "
            f"cost={amount_usdt:.2f}"
        )

        return position

    def close_position(
        self,
        symbol: str,
        price: float | None = None
    ) -> float | None:
        """
        Close an existing position.

        Args:
            symbol: Trading pair symbol
            price: Execution price (uses current price if None)

        Returns:
            Profit/loss if successful, None otherwise
        """
        if symbol not in self.positions or self.positions[symbol].closed:
            logger.warning(f"No open position for {symbol}")
            return None

        position = self.positions[symbol]

        # Get current price if not provided
        if price is None:
            price = self.data_client.get_current_price(symbol)

        # Apply slippage
        execution_price = price * (1 - self.settings.trading.slippage)

        # Calculate proceeds
        gross_proceeds = position.amount * execution_price
        fee = gross_proceeds * self.settings.trading.trading_fee
        net_proceeds = gross_proceeds - fee

        # Close position and calculate profit
        exit_time = datetime.utcnow()
        entry_value = position.current_value
        profit = net_proceeds - entry_value
        position.close(execution_price, exit_time)

        # Update balance and daily PnL
        self.balance += net_proceeds
        self.daily_pnl += profit

        # Record trade
        trade_record = {
            "symbol": symbol,
            "entry_price": position.entry_price,
            "exit_price": execution_price,
            "amount": position.amount,
            "entry_time": position.entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "entry_value": entry_value,
            "exit_value": net_proceeds,
            "profit": profit,
            "profit_pct": (profit / entry_value) * 100
        }
        self.trade_history.append(trade_record)

        logger.info(
            f"Closed position: {symbol}, "
            f"price={execution_price:.2f}, "
            f"profit={profit:.2f} ({trade_record['profit_pct']:.2f}%)"
        )

        return profit

    def check_stop_loss(self, symbol: str) -> bool:
        """
        Check if stop loss is triggered for a position.

        Args:
            symbol: Trading pair symbol

        Returns:
            True if stop loss triggered
        """
        if symbol not in self.positions or self.positions[symbol].closed:
            return False

        position = self.positions[symbol]
        current_price = self.data_client.get_current_price(symbol)

        loss_pct = (position.entry_price - current_price) / position.entry_price

        if loss_pct >= self.risk_config.stop_loss_pct:
            logger.warning(f"Stop loss triggered for {symbol}: {loss_pct*100:.2f}% loss")
            return True

        return False

    def check_take_profit(self, symbol: str) -> bool:
        """
        Check if take profit is triggered for a position.

        Args:
            symbol: Trading pair symbol

        Returns:
            True if take profit triggered
        """
        if symbol not in self.positions or self.positions[symbol].closed:
            return False

        position = self.positions[symbol]
        current_price = self.data_client.get_current_price(symbol)

        profit_pct = (current_price - position.entry_price) / position.entry_price

        if profit_pct >= self.risk_config.take_profit_pct:
            logger.info(f"Take profit triggered for {symbol}: {profit_pct*100:.2f}% profit")
            return True

        return False

    def execute_risk_management(self) -> list[str]:
        """
        Execute risk management checks for all open positions.

        Returns:
            List of symbols that were closed
        """
        closed_symbols = []

        for symbol in list(self.positions.keys()):
            if self.positions[symbol].closed:
                continue

            if self.check_stop_loss(symbol):
                self.close_position(symbol)
                closed_symbols.append(symbol)
            elif self.check_take_profit(symbol):
                self.close_position(symbol)
                closed_symbols.append(symbol)

        return closed_symbols

    def get_stats(self) -> dict:
        """
        Get trading statistics.

        Returns:
            Dictionary with trading statistics
        """
        if not self.trade_history:
            return {
                "total_trades": 0,
                "balance": self.balance,
                "portfolio_value": self.get_portfolio_value(),
                "total_return": 0.0,
                "win_rate": 0.0
            }

        profits = [t["profit"] for t in self.trade_history]
        winning = [p for p in profits if p > 0]

        portfolio_value = self.get_portfolio_value()

        return {
            "total_trades": len(self.trade_history),
            "winning_trades": len(winning),
            "win_rate": len(winning) / len(profits) if profits else 0,
            "total_profit": sum(profits),
            "avg_profit": np.mean(profits),
            "best_trade": max(profits),
            "worst_trade": min(profits),
            "balance": self.balance,
            "portfolio_value": portfolio_value,
            "total_return": (portfolio_value - self.initial_balance) / self.initial_balance,
            "daily_pnl": self.daily_pnl
        }

    def save_state(self, filepath: str | Path) -> None:
        """Save trader state to file."""
        state = {
            "initial_balance": self.initial_balance,
            "balance": self.balance,
            "daily_pnl": self.daily_pnl,
            "last_reset_date": self.last_reset_date.isoformat(),
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "trade_history": self.trade_history
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved trader state to {filepath}")

    def load_state(self, filepath: str | Path) -> None:
        """Load trader state from file."""
        filepath = Path(filepath)

        if not filepath.exists():
            logger.warning(f"State file not found: {filepath}")
            return

        with open(filepath, "r") as f:
            state = json.load(f)

        self.initial_balance = state["initial_balance"]
        self.balance = state["balance"]
        self.daily_pnl = state["daily_pnl"]
        self.last_reset_date = datetime.fromisoformat(state["last_reset_date"]).date()
        self.trade_history = state["trade_history"]

        # Reconstruct positions
        self.positions = {}
        for symbol, pos_dict in state["positions"].items():
            position = Position(
                symbol=pos_dict["symbol"],
                entry_price=pos_dict["entry_price"],
                amount=pos_dict["amount"],
                entry_time=datetime.fromisoformat(pos_dict["entry_time"])
            )
            if pos_dict["closed"]:
                position.exit_price = pos_dict["exit_price"]
                position.exit_time = datetime.fromisoformat(pos_dict["exit_time"])
                position.closed = True
            self.positions[symbol] = position

        logger.info(f"Loaded trader state from {filepath}")
