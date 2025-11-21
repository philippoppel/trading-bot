"""
Multi-Symbol Paper Trading mit Live-Daten.
Zeigt übersichtliche Performance-Tabelle für alle Cryptos.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime
import time

# Projekt-Root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.binance_client import BinanceDataClient
from src.data.preprocessor import DataPreprocessor
from stable_baselines3 import SAC


class MultiSymbolTrader:
    """Paper Trading für mehrere Symbole gleichzeitig."""

    def __init__(
        self,
        config_path: str = 'models/multi_symbol_config.json',
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001,
        observation_window: int = 60
    ):
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.observation_window = observation_window

        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.symbols = config['symbols']
        self.model_paths = config['models']

        # Initialize traders for each symbol
        self.traders = {}
        self.models = {}
        self.feature_cols = {}

        # Data client
        self.client = BinanceDataClient()
        self.preprocessor = DataPreprocessor()

        # Load models
        for symbol in self.symbols:
            logger.info(f"Loading model for {symbol}...")
            model_path = self.model_paths[symbol]
            self.models[symbol] = SAC.load(model_path)

            # Initialize trader state
            self.traders[symbol] = {
                'balance': initial_balance,
                'position': 0.0,
                'position_value': 0.0,
                'entry_price': 0.0,
                'total_fees': 0.0,
                'num_trades': 0,
                'trade_history': [],
                'current_price': 0.0
            }

        logger.info(f"Loaded {len(self.symbols)} models")

    def get_live_data(self, symbol: str, lookback_hours: int = 200):
        """Holt aktuelle Live-Daten für ein Symbol."""
        days = max(lookback_hours // 24 + 5, 10)

        df = self.client.get_data(
            symbol=symbol,
            interval='1h',
            days=days,
            force_refresh=True  # Immer neue Daten von der API holen
        )

        # Preprocessing
        df = self.preprocessor.process(df, normalize=True)

        # Feature columns
        if symbol not in self.feature_cols:
            self.feature_cols[symbol] = [c for c in df.columns if c.endswith('_norm')]

        return df

    def get_observation(self, df, symbol: str):
        """Erstellt Observation für das Modell."""
        feature_cols = self.feature_cols[symbol]
        features = df[feature_cols].values[-self.observation_window:]

        # Flatten
        obs = features.flatten().astype(np.float32)

        # Account info
        trader = self.traders[symbol]
        current_price = df['close'].iloc[-1]
        portfolio_value = trader['balance'] + trader['position_value']

        account_info = np.array([
            trader['position'],
            trader['balance'] / self.initial_balance,
            portfolio_value / self.initial_balance,
            0.0
        ], dtype=np.float32)

        obs = np.concatenate([obs, account_info])

        return obs, current_price

    def execute_action(self, symbol: str, action: float, current_price: float):
        """Führt Trading-Aktion aus."""
        trader = self.traders[symbol]
        target_position = np.clip(action, -1.0, 1.0)

        position_diff = target_position - trader['position']

        if abs(position_diff) < 0.01:
            return

        # Calculate trade
        portfolio_value = trader['balance'] + trader['position_value']
        trade_value = abs(position_diff) * portfolio_value

        # Fees
        fee = trade_value * self.fee_rate
        trader['total_fees'] += fee

        # Update position
        old_position = trader['position']
        trader['position'] = target_position

        if position_diff > 0:  # Buying
            trader['balance'] -= trade_value + fee
            trader['position_value'] = trader['position'] * portfolio_value
        else:  # Selling
            trader['balance'] += trade_value - fee
            trader['position_value'] = trader['position'] * portfolio_value

        trader['entry_price'] = current_price
        trader['num_trades'] += 1

    def update_position_value(self, symbol: str, current_price: float):
        """Aktualisiert Position Value."""
        trader = self.traders[symbol]
        if trader['position'] != 0 and trader['entry_price'] > 0:
            pnl_pct = (current_price - trader['entry_price']) / trader['entry_price']
            trader['position_value'] = trader['position'] * self.initial_balance * (1 + pnl_pct)
        trader['current_price'] = current_price

    def get_portfolio_value(self, symbol: str):
        """Berechnet Portfolio-Wert für ein Symbol."""
        trader = self.traders[symbol]
        return trader['balance'] + trader['position_value']

    def trade_symbol(self, symbol: str):
        """Führt einen Trading-Zyklus für ein Symbol aus."""
        try:
            # Get data
            df = self.get_live_data(symbol)

            # Get observation
            obs, current_price = self.get_observation(df, symbol)

            # Update position value
            self.update_position_value(symbol, current_price)

            # Get action
            action, _ = self.models[symbol].predict(obs, deterministic=True)
            action = float(action[0])

            # Execute
            self.execute_action(symbol, action, current_price)

            return True
        except Exception as e:
            logger.error(f"Error trading {symbol}: {e}")
            return False

    def print_overview(self, show_detailed: bool = False):
        """Zeigt übersichtliche Performance-Tabelle."""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')

        print("\n" + "=" * 80)
        print("🤖 MULTI-SYMBOL PAPER TRADING - LIVE PERFORMANCE")
        print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

        # Header
        print(f"{'Symbol':<10} {'Price':>12} {'Position':>10} {'Portfolio':>14} {'Return':>10} {'Trades':>8}")
        print("-" * 80)

        total_value = 0
        total_initial = 0
        total_fees = 0

        for symbol in self.symbols:
            trader = self.traders[symbol]
            portfolio_value = self.get_portfolio_value(symbol)
            ret = (portfolio_value / self.initial_balance - 1) * 100

            total_value += portfolio_value
            total_initial += self.initial_balance
            total_fees += trader['total_fees']

            # Position indicator
            pos = trader['position']
            if pos > 0.1:
                pos_str = f"LONG {pos:.2f}"
            elif pos < -0.1:
                pos_str = f"SHORT {abs(pos):.2f}"
            else:
                pos_str = "FLAT"

            # Color for return
            ret_str = f"{ret:+.2f}%"

            print(f"{symbol:<10} ${trader['current_price']:>10,.2f} {pos_str:>10} ${portfolio_value:>12,.2f} {ret_str:>10} {trader['num_trades']:>8}")

        # Total
        print("-" * 80)
        total_return = (total_value / total_initial - 1) * 100
        avg_return = total_return / len(self.symbols) if self.symbols else 0

        print(f"{'TOTAL':<10} {'':<12} {'':<10} ${total_value:>12,.2f} {total_return:+.2f}%")
        print(f"{'AVERAGE':<10} {'':<12} {'':<10} {'':<14} {avg_return:+.2f}%")
        print("=" * 80)

        # Best/Worst
        results = [(s, (self.get_portfolio_value(s) / self.initial_balance - 1) * 100) for s in self.symbols]
        best = max(results, key=lambda x: x[1])
        worst = min(results, key=lambda x: x[1])

        print(f"\n📈 Best:  {best[0]} ({best[1]:+.2f}%)")
        print(f"📉 Worst: {worst[0]} ({worst[1]:+.2f}%)")

        # Detailed statistics
        if show_detailed:
            print("\n" + "=" * 80)
            print("📊 DETAILED STATISTICS")
            print("=" * 80)

            total_trades = sum(t['num_trades'] for t in self.traders.values())

            print(f"\n💰 Cost Analysis:")
            print(f"   Total Fees Paid:        ${total_fees:>10,.2f}")
            print(f"   Total Trades:           {total_trades:>10}")
            print(f"   Avg Fee per Trade:      ${total_fees/max(total_trades, 1):>10,.2f}")
            print(f"   Fee Rate:               {self.fee_rate*100:>10.2f}%")

            print(f"\n📈 Performance Metrics:")
            print(f"   Gross Return:           {total_return:>+10.2f}%")
            print(f"   Return if no Fees:      {((total_value + total_fees) / total_initial - 1) * 100:>+10.2f}%")
            print(f"   Fee Impact:             {(total_fees / total_initial) * 100:>10.2f}%")

            print(f"\n🔄 Trading Activity by Symbol:")
            for symbol in self.symbols:
                trader = self.traders[symbol]
                print(f"   {symbol:<10} {trader['num_trades']:>4} trades | ${trader['total_fees']:>8,.2f} fees")

            print("=" * 80)

        print("\nPress Ctrl+C for final summary | Press 'd' + Enter for detailed stats\n")

    def run(self, interval_seconds: int = 60, detailed_every: int = 10):
        """Startet Multi-Symbol Trading Loop."""
        logger.info("Starting Multi-Symbol Paper Trading...")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Initial Balance per Symbol: ${self.initial_balance:,.2f}")
        logger.info(f"Update Interval: {interval_seconds}s")

        iteration = 0
        try:
            while True:
                # Trade all symbols
                for symbol in self.symbols:
                    self.trade_symbol(symbol)

                # Show overview (detailed every N iterations)
                iteration += 1
                show_detailed = (iteration % detailed_every == 0)
                self.print_overview(show_detailed=show_detailed)

                # Wait
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\nPaper Trading stopped by user")
            self.print_final_summary()

    def print_final_summary(self):
        """Zeigt finale Zusammenfassung."""
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)

        total_value = 0
        total_fees = 0
        total_trades = 0

        for symbol in self.symbols:
            trader = self.traders[symbol]
            portfolio_value = self.get_portfolio_value(symbol)
            ret = (portfolio_value / self.initial_balance - 1) * 100

            total_value += portfolio_value
            total_fees += trader['total_fees']
            total_trades += trader['num_trades']

            print(f"{symbol}: {ret:+.2f}% | ${portfolio_value:,.2f} | {trader['num_trades']} trades | ${trader['total_fees']:.2f} fees")

        print("-" * 60)
        total_initial = self.initial_balance * len(self.symbols)
        total_return = (total_value / total_initial - 1) * 100

        print(f"Total Portfolio: ${total_value:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Total Fees: ${total_fees:.2f}")
        print("=" * 60)


def main():
    """Hauptfunktion für Multi-Symbol Paper Trading."""
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Symbol Paper Trading')
    parser.add_argument('--config', type=str, default='models/multi_symbol_config.json',
                       help='Pfad zur Config-Datei')
    parser.add_argument('--balance', type=float, default=10000.0,
                       help='Startkapital pro Symbol')
    parser.add_argument('--interval', type=int, default=60,
                       help='Update-Intervall in Sekunden')
    parser.add_argument('--detailed-every', type=int, default=10,
                       help='Zeige detaillierte Stats alle N Iterationen')

    args = parser.parse_args()

    trader = MultiSymbolTrader(
        config_path=args.config,
        initial_balance=args.balance
    )

    trader.run(interval_seconds=args.interval, detailed_every=args.detailed_every)


if __name__ == '__main__':
    main()
