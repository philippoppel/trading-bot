"""
Paper Trading mit Live-Daten von Binance.
Verwendet das trainierte SAC-Modell für Echtzeit-Trading-Entscheidungen.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta
import time
import asyncio

# Projekt-Root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.binance_client import BinanceDataClient
from src.data.preprocessor import DataPreprocessor
from stable_baselines3 import SAC


class PaperTrader:
    """Paper Trading System mit Live-Daten."""

    def __init__(
        self,
        model_path: str = 'models/sac_best',
        symbol: str = 'BTCUSDT',
        interval: str = '1h',
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001,
        observation_window: int = 60
    ):
        self.symbol = symbol
        self.interval = interval
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.fee_rate = fee_rate
        self.observation_window = observation_window

        # Position tracking
        self.position = 0.0  # -1 to 1
        self.position_value = 0.0
        self.entry_price = 0.0
        self.total_fees = 0.0
        self.num_trades = 0

        # History
        self.trade_history = []
        self.portfolio_history = []

        # Load model
        logger.info(f"Loading model from {model_path}...")
        self.model = SAC.load(model_path)
        logger.info("Model loaded successfully")

        # Data client
        self.client = BinanceDataClient()
        self.preprocessor = DataPreprocessor()

        # Feature columns (will be set after first data load)
        self.feature_cols = None

    def get_live_data(self, lookback_hours: int = 200):
        """Holt aktuelle Live-Daten."""
        # Hole mehr Daten als nötig für Indikatoren
        days = max(lookback_hours // 24 + 5, 10)

        df = self.client.get_data(
            symbol=self.symbol,
            interval=self.interval,
            days=days
        )

        # Preprocessing
        df = self.preprocessor.process(df, normalize=True)

        # Feature columns
        if self.feature_cols is None:
            self.feature_cols = [c for c in df.columns if c.endswith('_norm')]

        return df

    def get_observation(self, df):
        """Erstellt Observation für das Modell."""
        # Letzte observation_window Zeilen
        features = df[self.feature_cols].values[-self.observation_window:]

        # Flatten
        obs = features.flatten().astype(np.float32)

        # Account info anhängen
        current_price = df['close'].iloc[-1]
        portfolio_value = self.balance + self.position_value

        account_info = np.array([
            self.position,
            self.balance / self.initial_balance,
            portfolio_value / self.initial_balance,
            0.0  # Unrealized PnL placeholder
        ], dtype=np.float32)

        obs = np.concatenate([obs, account_info])

        return obs, current_price

    def execute_action(self, action: float, current_price: float):
        """Führt Trading-Aktion aus."""
        target_position = np.clip(action, -1.0, 1.0)

        # Position difference
        position_diff = target_position - self.position

        if abs(position_diff) < 0.01:
            return  # Keine signifikante Änderung

        # Calculate trade value
        portfolio_value = self.balance + self.position_value
        trade_value = abs(position_diff) * portfolio_value

        # Fees
        fee = trade_value * self.fee_rate
        self.total_fees += fee

        # Update position
        old_position = self.position
        self.position = target_position

        # Update balance and position value
        if position_diff > 0:  # Buying
            self.balance -= trade_value + fee
            self.position_value = self.position * portfolio_value
        else:  # Selling
            self.balance += trade_value - fee
            self.position_value = self.position * portfolio_value

        self.entry_price = current_price
        self.num_trades += 1

        # Log trade
        trade = {
            'timestamp': datetime.now(),
            'price': current_price,
            'old_position': old_position,
            'new_position': self.position,
            'trade_value': trade_value,
            'fee': fee,
            'balance': self.balance,
            'portfolio_value': self.balance + self.position_value
        }
        self.trade_history.append(trade)

        action_str = "BUY" if position_diff > 0 else "SELL"
        logger.info(f"{action_str}: {old_position:.2f} -> {self.position:.2f} @ ${current_price:,.2f}")

    def update_position_value(self, current_price: float):
        """Aktualisiert Position Value basierend auf aktuellem Preis."""
        if self.position != 0 and self.entry_price > 0:
            # PnL calculation
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            self.position_value = self.position * self.initial_balance * (1 + pnl_pct)

    def get_portfolio_value(self):
        """Berechnet aktuellen Portfolio-Wert."""
        return self.balance + self.position_value

    def run_once(self):
        """Führt einen Trading-Zyklus aus."""
        try:
            # Get live data
            df = self.get_live_data()

            # Get observation
            obs, current_price = self.get_observation(df)

            # Update position value
            self.update_position_value(current_price)

            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            action = float(action[0])

            # Execute
            self.execute_action(action, current_price)

            # Log status
            portfolio_value = self.get_portfolio_value()
            ret = (portfolio_value / self.initial_balance - 1) * 100

            # Save snapshot
            self.portfolio_history.append({
                'timestamp': datetime.now(),
                'price': current_price,
                'position': self.position,
                'balance': self.balance,
                'portfolio_value': portfolio_value,
                'return_pct': ret
            })

            return {
                'price': current_price,
                'position': self.position,
                'portfolio_value': portfolio_value,
                'return_pct': ret,
                'num_trades': self.num_trades
            }

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return None

    def run_loop(self, interval_seconds: int = 3600):
        """Läuft kontinuierlich und tradet bei jedem Intervall."""
        logger.info("=" * 60)
        logger.info("PAPER TRADING STARTED")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Interval: {self.interval}")
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info("=" * 60)

        try:
            while True:
                result = self.run_once()

                if result:
                    logger.info("-" * 40)
                    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info(f"BTC Price: ${result['price']:,.2f}")
                    logger.info(f"Position: {result['position']:.2f}")
                    logger.info(f"Portfolio: ${result['portfolio_value']:,.2f}")
                    logger.info(f"Return: {result['return_pct']:+.2f}%")
                    logger.info(f"Trades: {result['num_trades']}")
                    logger.info("-" * 40)

                # Wait for next interval
                logger.info(f"Waiting {interval_seconds}s for next update...")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("\nPaper trading stopped by user")
            self.print_summary()

    def print_summary(self):
        """Zeigt Trading-Zusammenfassung."""
        portfolio_value = self.get_portfolio_value()
        ret = (portfolio_value / self.initial_balance - 1) * 100

        logger.info("\n" + "=" * 60)
        logger.info("PAPER TRADING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"Final Portfolio: ${portfolio_value:,.2f}")
        logger.info(f"Total Return: {ret:+.2f}%")
        logger.info(f"Total Trades: {self.num_trades}")
        logger.info(f"Total Fees: ${self.total_fees:,.2f}")
        logger.info("=" * 60)

        if self.trade_history:
            logger.info("\nRecent Trades:")
            for trade in self.trade_history[-5:]:
                logger.info(f"  {trade['timestamp'].strftime('%H:%M')} - "
                          f"{trade['old_position']:.2f} -> {trade['new_position']:.2f} "
                          f"@ ${trade['price']:,.2f}")


def main():
    """Hauptfunktion für Paper Trading."""
    import argparse

    parser = argparse.ArgumentParser(description='Paper Trading mit Live-Daten')
    parser.add_argument('--model', type=str, default='models/sac_best',
                       help='Pfad zum trainierten Modell')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Trading-Symbol')
    parser.add_argument('--balance', type=float, default=10000.0,
                       help='Startkapital')
    parser.add_argument('--interval', type=int, default=60,
                       help='Update-Intervall in Sekunden')
    parser.add_argument('--once', action='store_true',
                       help='Nur einmal ausführen (kein Loop)')

    args = parser.parse_args()

    trader = PaperTrader(
        model_path=args.model,
        symbol=args.symbol,
        initial_balance=args.balance
    )

    if args.once:
        result = trader.run_once()
        if result:
            logger.info(f"Price: ${result['price']:,.2f}")
            logger.info(f"Action: Position {result['position']:.2f}")
            logger.info(f"Portfolio: ${result['portfolio_value']:,.2f}")
    else:
        trader.run_loop(interval_seconds=args.interval)


if __name__ == '__main__':
    main()
