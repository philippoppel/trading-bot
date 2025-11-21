"""
Realistisches Paper Trading mit Live-Daten.
Berücksichtigt: Slippage, Spread, Funding Rates, Multiple Symbols.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta
import time
import random

# Projekt-Root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.binance_client import BinanceDataClient
from src.data.preprocessor import DataPreprocessor
from stable_baselines3 import SAC


class RealisticPaperTrader:
    """Realistisches Paper Trading System mit allen Kosten."""

    # Realistische Binance Gebühren und Kosten
    TRADING_FEE = 0.001       # 0.1% Maker/Taker Fee
    SLIPPAGE_BASE = 0.0005    # 0.05% Base Slippage
    SLIPPAGE_VOLATILITY = 0.001  # Zusätzlicher Slippage bei hoher Volatilität
    SPREAD_PCT = 0.0001       # 0.01% Spread (Bid/Ask)
    FUNDING_RATE = 0.0001     # 0.01% Funding Rate (alle 8h für Perpetuals)

    def __init__(
        self,
        model_path: str = 'models/sac_best',
        symbols: list = None,
        interval: str = '1h',
        initial_balance: float = 10000.0,
        observation_window: int = 60,
        use_perpetuals: bool = True  # Perpetuals haben Funding Rates
    ):
        self.symbols = symbols or ['BTCUSDT']
        self.interval = interval
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.observation_window = observation_window
        self.use_perpetuals = use_perpetuals

        # Position tracking per symbol
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.entry_prices = {symbol: 0.0 for symbol in self.symbols}
        self.position_values = {symbol: 0.0 for symbol in self.symbols}

        # Cost tracking
        self.total_fees = 0.0
        self.total_slippage_cost = 0.0
        self.total_spread_cost = 0.0
        self.total_funding_paid = 0.0
        self.num_trades = 0

        # History
        self.trade_history = []
        self.portfolio_history = []
        self.funding_history = []

        # Load model
        logger.info(f"Loading model from {model_path}...")
        self.model = SAC.load(model_path)
        logger.info("Model loaded successfully")

        # Data client
        self.client = BinanceDataClient()
        self.preprocessor = DataPreprocessor()

        # Feature columns (will be set after first data load)
        self.feature_cols = None

        # Last funding time
        self.last_funding_time = datetime.now()

    def calculate_slippage(self, trade_value: float, volatility: float = 0.02) -> float:
        """Berechnet realistischen Slippage basierend auf Volumen und Volatilität."""
        # Größere Orders haben mehr Slippage
        size_factor = min(trade_value / 10000, 2.0)  # Max 2x für große Orders

        # Höhere Volatilität = mehr Slippage
        vol_factor = 1 + (volatility * 10)  # z.B. 2% vol -> 1.2x

        # Random component für Realismus
        random_factor = random.uniform(0.8, 1.2)

        slippage = self.SLIPPAGE_BASE * size_factor * vol_factor * random_factor
        return slippage

    def calculate_spread_cost(self, trade_value: float) -> float:
        """Berechnet Spread-Kosten (Bid/Ask Differenz)."""
        return trade_value * self.SPREAD_PCT

    def apply_funding_rate(self, current_time: datetime):
        """Wendet Funding Rate an (alle 8 Stunden für Perpetuals)."""
        if not self.use_perpetuals:
            return

        hours_since_funding = (current_time - self.last_funding_time).total_seconds() / 3600

        if hours_since_funding >= 8:
            # Funding wird auf offene Positionen angewendet
            for symbol in self.symbols:
                if self.positions[symbol] != 0:
                    position_value = abs(self.position_values[symbol])

                    # Long zahlt, Short erhält (vereinfacht)
                    if self.positions[symbol] > 0:
                        funding_cost = position_value * self.FUNDING_RATE
                        self.balance -= funding_cost
                        self.total_funding_paid += funding_cost
                    else:
                        funding_received = position_value * self.FUNDING_RATE
                        self.balance += funding_received
                        self.total_funding_paid -= funding_received

                    self.funding_history.append({
                        'timestamp': current_time,
                        'symbol': symbol,
                        'position': self.positions[symbol],
                        'funding': funding_cost if self.positions[symbol] > 0 else -funding_received
                    })

            self.last_funding_time = current_time
            logger.debug(f"Funding applied. Total paid: ${self.total_funding_paid:.2f}")

    def get_live_data(self, symbol: str, lookback_hours: int = 200):
        """Holt aktuelle Live-Daten für ein Symbol."""
        days = max(lookback_hours // 24 + 5, 10)

        df = self.client.get_data(
            symbol=symbol,
            interval=self.interval,
            days=days
        )

        # Preprocessing
        df = self.preprocessor.process(df, normalize=True)

        # Feature columns
        if self.feature_cols is None:
            self.feature_cols = [c for c in df.columns if c.endswith('_norm')]

        return df

    def get_observation(self, df, symbol: str):
        """Erstellt Observation für das Modell."""
        features = df[self.feature_cols].values[-self.observation_window:]
        obs = features.flatten().astype(np.float32)

        current_price = df['close'].iloc[-1]
        portfolio_value = self.get_portfolio_value()

        account_info = np.array([
            self.positions[symbol],
            self.balance / self.initial_balance,
            portfolio_value / self.initial_balance,
            0.0
        ], dtype=np.float32)

        obs = np.concatenate([obs, account_info])
        return obs, current_price

    def execute_action(self, symbol: str, action: float, current_price: float, volatility: float = 0.02):
        """Führt Trading-Aktion mit realistischen Kosten aus."""
        target_position = np.clip(action, -1.0, 1.0)
        position_diff = target_position - self.positions[symbol]

        if abs(position_diff) < 0.01:
            return  # Keine signifikante Änderung

        # Calculate trade value
        portfolio_value = self.get_portfolio_value()
        trade_value = abs(position_diff) * portfolio_value

        # 1. Trading Fee
        fee = trade_value * self.TRADING_FEE
        self.total_fees += fee

        # 2. Slippage
        slippage_pct = self.calculate_slippage(trade_value, volatility)
        slippage_cost = trade_value * slippage_pct
        self.total_slippage_cost += slippage_cost

        # 3. Spread Cost
        spread_cost = self.calculate_spread_cost(trade_value)
        self.total_spread_cost += spread_cost

        # Total costs
        total_cost = fee + slippage_cost + spread_cost

        # Adjusted execution price (worse due to slippage/spread)
        if position_diff > 0:  # Buying
            execution_price = current_price * (1 + slippage_pct + self.SPREAD_PCT/2)
        else:  # Selling
            execution_price = current_price * (1 - slippage_pct - self.SPREAD_PCT/2)

        # Update position
        old_position = self.positions[symbol]
        self.positions[symbol] = target_position

        # Update balance
        if position_diff > 0:  # Buying
            self.balance -= trade_value + total_cost
        else:  # Selling
            self.balance += trade_value - total_cost

        self.position_values[symbol] = self.positions[symbol] * portfolio_value
        self.entry_prices[symbol] = execution_price
        self.num_trades += 1

        # Log trade
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'price': current_price,
            'execution_price': execution_price,
            'old_position': old_position,
            'new_position': self.positions[symbol],
            'trade_value': trade_value,
            'fee': fee,
            'slippage': slippage_cost,
            'spread': spread_cost,
            'total_cost': total_cost,
            'balance': self.balance
        }
        self.trade_history.append(trade)

        action_str = "BUY" if position_diff > 0 else "SELL"
        logger.info(f"{symbol} {action_str}: {old_position:.2f} -> {self.positions[symbol]:.2f}")
        logger.info(f"  Price: ${current_price:,.2f} -> Exec: ${execution_price:,.2f}")
        logger.info(f"  Costs: Fee=${fee:.2f}, Slip=${slippage_cost:.2f}, Spread=${spread_cost:.2f}")

    def update_position_values(self, prices: dict):
        """Aktualisiert alle Position Values."""
        for symbol in self.symbols:
            if self.positions[symbol] != 0 and self.entry_prices[symbol] > 0:
                current_price = prices.get(symbol, self.entry_prices[symbol])
                pnl_pct = (current_price - self.entry_prices[symbol]) / self.entry_prices[symbol]
                self.position_values[symbol] = self.positions[symbol] * self.initial_balance * (1 + pnl_pct)

    def get_portfolio_value(self):
        """Berechnet aktuellen Portfolio-Wert."""
        return self.balance + sum(self.position_values.values())

    def get_total_costs(self):
        """Gibt alle Kosten zurück."""
        return {
            'fees': self.total_fees,
            'slippage': self.total_slippage_cost,
            'spread': self.total_spread_cost,
            'funding': self.total_funding_paid,
            'total': self.total_fees + self.total_slippage_cost + self.total_spread_cost + self.total_funding_paid
        }

    def run_once(self):
        """Führt einen Trading-Zyklus für alle Symbole aus."""
        try:
            current_time = datetime.now()
            prices = {}

            # Apply funding rate if needed
            self.apply_funding_rate(current_time)

            for symbol in self.symbols:
                # Get live data
                df = self.get_live_data(symbol)

                # Get observation and price
                obs, current_price = self.get_observation(df, symbol)
                prices[symbol] = current_price

                # Calculate volatility from recent data
                returns = df['close'].pct_change().dropna()
                volatility = returns.tail(24).std()  # 24h volatility

                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                action = float(action[0])

                # Execute with realistic costs
                self.execute_action(symbol, action, current_price, volatility)

            # Update all position values
            self.update_position_values(prices)

            # Calculate results
            portfolio_value = self.get_portfolio_value()
            ret = (portfolio_value / self.initial_balance - 1) * 100
            costs = self.get_total_costs()

            # Save snapshot
            self.portfolio_history.append({
                'timestamp': current_time,
                'prices': prices.copy(),
                'positions': self.positions.copy(),
                'balance': self.balance,
                'portfolio_value': portfolio_value,
                'return_pct': ret,
                'total_costs': costs['total']
            })

            return {
                'prices': prices,
                'positions': self.positions.copy(),
                'portfolio_value': portfolio_value,
                'return_pct': ret,
                'num_trades': self.num_trades,
                'costs': costs
            }

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_loop(self, interval_seconds: int = 3600):
        """Läuft kontinuierlich und tradet bei jedem Intervall."""
        logger.info("=" * 60)
        logger.info("REALISTIC PAPER TRADING STARTED")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Interval: {self.interval}")
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"Perpetuals (Funding): {'Yes' if self.use_perpetuals else 'No'}")
        logger.info("=" * 60)
        logger.info("Realistic costs enabled:")
        logger.info(f"  - Trading Fee: {self.TRADING_FEE*100:.2f}%")
        logger.info(f"  - Slippage: ~{self.SLIPPAGE_BASE*100:.3f}%")
        logger.info(f"  - Spread: {self.SPREAD_PCT*100:.3f}%")
        if self.use_perpetuals:
            logger.info(f"  - Funding Rate: {self.FUNDING_RATE*100:.3f}% per 8h")
        logger.info("=" * 60)

        try:
            while True:
                result = self.run_once()

                if result:
                    logger.info("-" * 50)
                    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                    for symbol in self.symbols:
                        price = result['prices'].get(symbol, 0)
                        pos = result['positions'].get(symbol, 0)
                        logger.info(f"{symbol}: ${price:,.2f} | Pos: {pos:.2f}")

                    logger.info(f"Portfolio: ${result['portfolio_value']:,.2f}")
                    logger.info(f"Return: {result['return_pct']:+.2f}%")
                    logger.info(f"Trades: {result['num_trades']}")

                    costs = result['costs']
                    logger.info(f"Costs: Fee=${costs['fees']:.2f}, Slip=${costs['slippage']:.2f}, "
                              f"Spread=${costs['spread']:.2f}, Fund=${costs['funding']:.2f}")
                    logger.info(f"Total Costs: ${costs['total']:.2f}")
                    logger.info("-" * 50)

                logger.info(f"Waiting {interval_seconds}s for next update...")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("\nPaper trading stopped by user")
            self.print_summary()

    def print_summary(self):
        """Zeigt detaillierte Trading-Zusammenfassung."""
        portfolio_value = self.get_portfolio_value()
        ret = (portfolio_value / self.initial_balance - 1) * 100
        costs = self.get_total_costs()

        logger.info("\n" + "=" * 60)
        logger.info("REALISTIC PAPER TRADING SUMMARY")
        logger.info("=" * 60)

        logger.info("\nPERFORMANCE:")
        logger.info(f"  Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"  Final Portfolio: ${portfolio_value:,.2f}")
        logger.info(f"  Gross Return: {ret:+.2f}%")

        logger.info("\nCOSTS BREAKDOWN:")
        logger.info(f"  Trading Fees: ${costs['fees']:.2f}")
        logger.info(f"  Slippage: ${costs['slippage']:.2f}")
        logger.info(f"  Spread: ${costs['spread']:.2f}")
        logger.info(f"  Funding: ${costs['funding']:.2f}")
        logger.info(f"  TOTAL COSTS: ${costs['total']:.2f}")

        net_return = ret - (costs['total'] / self.initial_balance * 100)
        logger.info(f"\nNET RETURN (after costs): {net_return:+.2f}%")

        logger.info(f"\nTRADING ACTIVITY:")
        logger.info(f"  Total Trades: {self.num_trades}")
        if self.num_trades > 0:
            logger.info(f"  Avg Cost per Trade: ${costs['total']/self.num_trades:.2f}")

        logger.info("\nPOSITIONS:")
        for symbol in self.symbols:
            logger.info(f"  {symbol}: {self.positions[symbol]:.2f}")

        logger.info("=" * 60)


def main():
    """Hauptfunktion für realistisches Paper Trading."""
    import argparse

    parser = argparse.ArgumentParser(description='Realistisches Paper Trading')
    parser.add_argument('--model', type=str, default='models/sac_best',
                       help='Pfad zum trainierten Modell')
    parser.add_argument('--symbols', type=str, nargs='+',
                       default=['BTCUSDT'],
                       help='Trading-Symbole (z.B. BTCUSDT ETHUSDT)')
    parser.add_argument('--balance', type=float, default=10000.0,
                       help='Startkapital')
    parser.add_argument('--interval', type=int, default=60,
                       help='Update-Intervall in Sekunden')
    parser.add_argument('--no-perpetuals', action='store_true',
                       help='Keine Funding Rates (Spot Trading)')

    args = parser.parse_args()

    # Warnung für Multi-Symbol
    if len(args.symbols) > 1:
        logger.warning("=" * 60)
        logger.warning("WARNUNG: Multi-Symbol Trading")
        logger.warning("Das Modell wurde nur auf BTCUSDT trainiert!")
        logger.warning("Für beste Ergebnisse sollte jedes Symbol")
        logger.warning("ein eigenes trainiertes Modell haben.")
        logger.warning("=" * 60)

    trader = RealisticPaperTrader(
        model_path=args.model,
        symbols=args.symbols,
        initial_balance=args.balance,
        use_perpetuals=not args.no_perpetuals
    )

    trader.run_loop(interval_seconds=args.interval)


if __name__ == '__main__':
    main()
