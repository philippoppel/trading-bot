"""
LIVE Multi-Symbol Trading auf Binance TESTNET mit umfassenden Risk Management Features.

⚠️  WICHTIG: Dieser Bot führt ECHTE Trades aus!
   - Standardmäßig auf TESTNET (sicher)
   - Für Production: EXTREM vorsichtig sein!
   - Niemals ohne ausgiebige Testnet-Tests zu Production wechseln!
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta
import time
import shutil
from typing import Optional
import requests

# Projekt-Root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.binance_client import BinanceDataClient
from src.data.preprocessor import DataPreprocessor
from src.trading.live_trader import LiveBinanceTrader
from stable_baselines3 import SAC


class LiveMultiSymbolTrader:
    """Live Trading mit echten Binance Orders und umfassenden Sicherheitsmechanismen."""

    def __init__(
        self,
        config_path: str = 'models/multi_symbol_config.json',
        initial_balance: float = 10000.0,
        observation_window: int = 60,
        # RISK MANAGEMENT PARAMETERS
        max_position_size: float = 0.3,        # Max 30% pro Position
        max_loss_per_symbol: float = -0.15,    # -15% Stop-Loss pro Symbol
        max_total_drawdown: float = -0.20,     # -20% Total Portfolio Stop
        max_trades_per_hour: int = 5,          # Max Trades pro Stunde
        min_trade_interval: int = 300,         # Min 5 Min zwischen Trades
        volatility_threshold: float = 0.05,    # Pause bei >5% Volatilität
        use_stop_loss: bool = True,
        use_take_profit: bool = True,
        take_profit_pct: float = 0.10,         # 10% Take Profit
        # LIVE TRADING PARAMETERS
        testnet: bool = True,                  # IMMER True für Tests!
        min_order_size: float = 15.0,          # Minimum $15 per order (Binance minimum ~$10)
        # PERSISTENCE PARAMETERS
        state_file: Optional[str] = None,
        auto_save: bool = True,
        save_interval: int = 300,
        keep_backups: int = 5,
        # DASHBOARD UPLOAD
        upload_to_dashboard: bool = True,
        dashboard_url: Optional[str] = None
    ):
        self.initial_balance = initial_balance
        self.observation_window = observation_window

        # RISK PARAMETERS
        self.max_position_size = max_position_size
        self.max_loss_per_symbol = max_loss_per_symbol
        self.max_total_drawdown = max_total_drawdown
        self.max_trades_per_hour = max_trades_per_hour
        self.min_trade_interval = min_trade_interval
        self.volatility_threshold = volatility_threshold
        self.use_stop_loss = use_stop_loss
        self.use_take_profit = use_take_profit
        self.take_profit_pct = take_profit_pct

        # LIVE TRADING
        self.testnet = testnet
        self.min_order_size = min_order_size

        # PERSISTENCE
        self.auto_save = auto_save
        self.save_interval = save_interval
        self.keep_backups = keep_backups
        self.last_save_time = None

        # DASHBOARD UPLOAD
        self.upload_to_dashboard = upload_to_dashboard
        self.dashboard_url = dashboard_url or os.getenv('DASHBOARD_UPLOAD_URL')

        # Setup state file
        if state_file is None:
            state_dir = Path('data/trading_state')
            state_dir.mkdir(parents=True, exist_ok=True)
            mode_str = 'testnet' if testnet else 'live'
            self.state_file = state_dir / f'live_multi_symbol_{mode_str}_state.json'
        else:
            self.state_file = Path(state_file)
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Emergency brake
        self.emergency_stopped = False
        self.emergency_reason = None

        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.symbols = config['symbols']
        self.model_paths = config['models']

        # Initialize Live Trader
        api_key = os.getenv('BINANCE_TESTNET_API_KEY' if testnet else 'BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_TESTNET_API_SECRET' if testnet else 'BINANCE_API_SECRET')

        if not api_key or not api_secret:
            raise ValueError(
                f"API Keys not found! Set {'BINANCE_TESTNET_API_KEY/SECRET' if testnet else 'BINANCE_API_KEY/SECRET'}"
            )

        self.live_trader = LiveBinanceTrader(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            min_notional=min_order_size
        )

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
                'balance': 0.0,  # Will be updated from real balance
                'position': 0.0,  # Will be updated from real positions
                'position_value': 0.0,
                'entry_price': 0.0,
                'total_fees': 0.0,
                'num_trades': 0,
                'trade_history': [],
                'current_price': 0.0,
                'last_trade_time': None,
                'trades_last_hour': [],
                'max_loss_reached': False,
                'highest_value': initial_balance,
                'initial_balance': initial_balance,
            }

        # Try to load existing state
        self.session_start_time = datetime.now()
        if self.load_state():
            logger.info("✅ Restored previous trading state")
        else:
            logger.info("🆕 Starting fresh trading session")
            # Sync with real account
            self.sync_with_exchange()

        logger.info(f"Loaded {len(self.symbols)} models")
        logger.info("=" * 60)
        logger.info("⚠️  LIVE TRADING SETTINGS:")
        logger.info(f"   Mode:                   {'🧪 TESTNET (Safe)' if testnet else '⚠️  PRODUCTION (REAL MONEY!)'}")
        logger.info(f"   Max Position Size:      {max_position_size*100:.0f}%")
        logger.info(f"   Stop-Loss per Symbol:   {max_loss_per_symbol*100:.0f}%")
        logger.info(f"   Total Drawdown Limit:   {max_total_drawdown*100:.0f}%")
        logger.info(f"   Max Trades/Hour:        {max_trades_per_hour}")
        logger.info(f"   Min Trade Interval:     {min_trade_interval}s")
        logger.info(f"   Min Order Size:         ${min_order_size}")
        logger.info("=" * 60)

    def sync_with_exchange(self):
        """Synchronisiere State mit echten Binance Balances."""
        logger.info("🔄 Syncing with exchange...")

        try:
            # Get USDT balance
            usdt_balance = self.live_trader.get_account_balance('USDT')
            logger.info(f"   USDT Balance: ${usdt_balance:,.2f}")

            # Distribute equally among symbols (or use saved allocation)
            balance_per_symbol = usdt_balance / len(self.symbols)

            for symbol in self.symbols:
                trader = self.traders[symbol]

                # Get current position
                position_balance = self.live_trader.get_asset_balance(symbol)

                if position_balance > 0:
                    # We have a position
                    current_price = self.live_trader.get_current_price(symbol)
                    trader['position'] = position_balance
                    trader['position_value'] = position_balance * current_price
                    trader['current_price'] = current_price
                    logger.info(f"   {symbol}: {position_balance:.8f} ({trader['position_value']:.2f} USDT)")
                else:
                    trader['position'] = 0.0
                    trader['position_value'] = 0.0

                # Set balance
                if trader['balance'] == 0:
                    trader['balance'] = balance_per_symbol

            logger.info("✅ Sync complete")

        except Exception as e:
            logger.error(f"❌ Sync failed: {e}")
            raise

    def get_live_data(self, symbol: str, lookback_hours: int = 200):
        """Holt aktuelle Live-Daten für ein Symbol."""
        days = max(lookback_hours // 24 + 5, 10)

        df = self.client.get_data(
            symbol=symbol,
            interval='1h',
            days=days,
            force_refresh=True
        )

        # Preprocessing
        df = self.preprocessor.process(df, normalize=True)

        # Feature columns
        if symbol not in self.feature_cols:
            self.feature_cols[symbol] = [c for c in df.columns if c.endswith('_norm')]

        return df

    def _serialize_trader_state(self, trader: dict) -> dict:
        """Konvertiert Trader State zu JSON-serialisierbarem Format."""
        serialized = trader.copy()

        if trader['last_trade_time']:
            serialized['last_trade_time'] = trader['last_trade_time'].isoformat()

        serialized['trades_last_hour'] = [
            t.isoformat() for t in trader['trades_last_hour']
        ]

        return serialized

    def _deserialize_trader_state(self, trader: dict) -> dict:
        """Konvertiert JSON State zurück zu Trader State."""
        deserialized = trader.copy()

        if trader['last_trade_time']:
            deserialized['last_trade_time'] = datetime.fromisoformat(trader['last_trade_time'])
        else:
            deserialized['last_trade_time'] = None

        deserialized['trades_last_hour'] = [
            datetime.fromisoformat(t) for t in trader['trades_last_hour']
        ]

        return deserialized

    def save_state(self, force: bool = False) -> bool:
        """Speichert aktuellen Trading State."""
        if not force and self.last_save_time:
            elapsed = (datetime.now() - self.last_save_time).total_seconds()
            if elapsed < self.save_interval:
                return False

        try:
            state = {
                'version': '1.0',
                'timestamp': datetime.now().isoformat(),
                'session_start_time': self.session_start_time.isoformat(),
                'initial_balance': self.initial_balance,
                'emergency_stopped': self.emergency_stopped,
                'emergency_reason': self.emergency_reason,
                'testnet': self.testnet,
                'traders': {}
            }

            for symbol, trader in self.traders.items():
                state['traders'][symbol] = self._serialize_trader_state(trader)

            # Create backup
            if self.state_file.exists() and self.keep_backups > 0:
                self._create_backup()

            # Atomic write
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)

            shutil.move(str(temp_file), str(self.state_file))

            self.last_save_time = datetime.now()
            logger.debug(f"💾 State saved to {self.state_file}")

            # Upload to dashboard
            if self.upload_to_dashboard and self.dashboard_url:
                self.upload_state_to_dashboard(state)

            return True

        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False

    def upload_state_to_dashboard(self, state: dict):
        """Upload state to dashboard via API."""
        try:
            response = requests.post(
                self.dashboard_url,
                json=state,
                timeout=5
            )

            if response.status_code == 200:
                logger.debug("📤 State uploaded to dashboard")
            else:
                logger.warning(f"Dashboard upload failed: {response.status_code}")

        except Exception as e:
            logger.debug(f"Dashboard upload error (non-critical): {e}")

    def _create_backup(self):
        """Erstellt ein Backup des State Files."""
        try:
            backup_dir = self.state_file.parent / 'backups'
            backup_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_dir / f"{self.state_file.stem}_{timestamp}.json"

            shutil.copy2(self.state_file, backup_file)
            logger.debug(f"📦 Backup created: {backup_file}")

            # Clean old backups
            backups = sorted(backup_dir.glob(f"{self.state_file.stem}_*.json"))
            if len(backups) > self.keep_backups:
                for old_backup in backups[:-self.keep_backups]:
                    old_backup.unlink()
                    logger.debug(f"🗑️  Deleted old backup: {old_backup}")

        except Exception as e:
            logger.warning(f"Error creating backup: {e}")

    def load_state(self) -> bool:
        """Lädt gespeicherten Trading State."""
        if not self.state_file.exists():
            logger.info("No saved state found")
            return False

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            if state.get('version') != '1.0':
                logger.warning(f"Incompatible state version: {state.get('version')}")
                return False

            # Restore emergency state
            self.emergency_stopped = state.get('emergency_stopped', False)
            self.emergency_reason = state.get('emergency_reason')

            # Restore session start time
            if 'session_start_time' in state:
                self.session_start_time = datetime.fromisoformat(state['session_start_time'])

            # Restore trader states
            for symbol, trader_state in state.get('traders', {}).items():
                if symbol in self.traders:
                    self.traders[symbol] = self._deserialize_trader_state(trader_state)
                else:
                    logger.warning(f"Symbol {symbol} in saved state but not in config")

            logger.info(f"📂 Loaded state from {state['timestamp']}")
            return True

        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False

    def calculate_volatility(self, df: pd.DataFrame, window: int = 24) -> float:
        """Berechnet aktuelle Volatilität."""
        if len(df) < window:
            return 0.0
        returns = df['close'].pct_change().tail(window)
        return returns.std()

    def check_market_conditions(self, symbol: str, df: pd.DataFrame) -> tuple[bool, str]:
        """Prüft ob Market Conditions safe für Trading sind."""
        volatility = self.calculate_volatility(df)
        if volatility > self.volatility_threshold:
            return False, f"High volatility: {volatility*100:.2f}%"

        if len(df) < self.observation_window:
            return False, "Insufficient data"

        return True, "OK"

    def check_stop_loss(self, symbol: str) -> bool:
        """Prüft ob Stop-Loss erreicht wurde."""
        if not self.use_stop_loss:
            return False

        trader = self.traders[symbol]
        portfolio_value = self.get_portfolio_value(symbol)
        loss = (portfolio_value / trader['initial_balance'] - 1)

        if loss <= self.max_loss_per_symbol:
            logger.warning(f"🛑 STOP-LOSS triggered for {symbol}: {loss*100:.2f}%")
            trader['max_loss_reached'] = True

            # Close position on exchange
            if trader['position'] > 0:
                try:
                    self.live_trader.close_position(symbol)
                    logger.info(f"✅ Closed position for {symbol}")

                    # Record trade
                    self._record_trade(
                        symbol=symbol,
                        action_type='STOP_LOSS',
                        old_position=trader['position'],
                        new_position=0.0,
                        price=trader['current_price'],
                        reasoning=f"Stop-loss triggered at {loss*100:.2f}% loss"
                    )

                    trader['position'] = 0.0
                    trader['position_value'] = 0.0

                except Exception as e:
                    logger.error(f"❌ Failed to close position: {e}")

            return True

        return False

    def check_total_drawdown(self) -> bool:
        """Prüft ob maximaler Total Drawdown erreicht wurde."""
        total_value = sum(self.get_portfolio_value(s) for s in self.symbols)
        total_highest = sum(self.traders[s]['highest_value'] for s in self.symbols)

        for symbol in self.symbols:
            trader = self.traders[symbol]
            current_val = self.get_portfolio_value(symbol)
            trader['highest_value'] = max(trader['highest_value'], current_val)

        drawdown = (total_value / total_highest - 1)

        if drawdown <= self.max_total_drawdown:
            self.emergency_stopped = True
            self.emergency_reason = f"Max Total Drawdown reached: {drawdown*100:.2f}%"
            logger.critical(f"🚨 EMERGENCY STOP: {self.emergency_reason}")

            # Close all positions
            for symbol in self.symbols:
                try:
                    self.live_trader.close_position(symbol)
                except:
                    pass

            return True

        return False

    def check_trade_limits(self, symbol: str) -> tuple[bool, str]:
        """Prüft ob Trade-Limits eingehalten werden."""
        trader = self.traders[symbol]

        if trader['max_loss_reached']:
            return False, "Max loss reached - trading paused"

        if trader['last_trade_time']:
            time_since_last = (datetime.now() - trader['last_trade_time']).total_seconds()
            if time_since_last < self.min_trade_interval:
                return False, f"Min interval not met ({time_since_last:.0f}s < {self.min_trade_interval}s)"

        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        trader['trades_last_hour'] = [t for t in trader['trades_last_hour'] if t > hour_ago]

        if len(trader['trades_last_hour']) >= self.max_trades_per_hour:
            return False, f"Max trades/hour reached ({self.max_trades_per_hour})"

        return True, "OK"

    def get_observation(self, df, symbol: str):
        """Erstellt Observation für das Modell."""
        feature_cols = self.feature_cols[symbol]
        features = df[feature_cols].values[-self.observation_window:]

        obs = features.flatten().astype(np.float32)

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

    def _record_trade(
        self,
        symbol: str,
        action_type: str,
        old_position: float,
        new_position: float,
        price: float,
        reasoning: str,
        order_id: Optional[int] = None,
        fee: float = 0.0
    ):
        """Records a trade in history."""
        trader = self.traders[symbol]
        trade_time = datetime.now()

        trade_record = {
            'timestamp': trade_time.isoformat(),
            'symbol': symbol,
            'action_type': action_type,
            'old_position': old_position,
            'new_position': new_position,
            'position_change': new_position - old_position,
            'price': price,
            'trade_value': abs(new_position - old_position) * price,
            'fee': fee,
            'slippage': 0.0,  # Real slippage from order
            'total_cost': fee,
            'balance_before': trader['balance'],
            'balance_after': trader['balance'],
            'portfolio_value_before': self.get_portfolio_value(symbol),
            'portfolio_value_after': self.get_portfolio_value(symbol),
            'reasoning': reasoning,
            'model_action': 0.0,
            'order_id': order_id
        }

        trader['trade_history'].append(trade_record)
        trader['num_trades'] += 1
        trader['last_trade_time'] = trade_time
        trader['trades_last_hour'].append(trade_time)

    def execute_action(self, symbol: str, action: float, current_price: float, reasoning: str = "Model decision"):
        """Führt Trading-Aktion aus mit echten Binance Orders."""
        trader = self.traders[symbol]

        # Apply max position size limit
        target_position = np.clip(action, -self.max_position_size, self.max_position_size)
        position_diff = target_position - trader['position']

        if abs(position_diff) < 0.01:
            return

        old_position = trader['position']

        # Calculate trade value
        portfolio_value = trader['balance'] + trader['position_value']
        trade_value_usdt = abs(position_diff) * portfolio_value

        # Check minimum order size
        if trade_value_usdt < self.min_order_size:
            logger.debug(f"{symbol}: Trade value ${trade_value_usdt:.2f} below minimum ${self.min_order_size}")
            return

        try:
            # Execute real order on Binance
            if position_diff > 0:
                # BUY
                action_type = "BUY" if old_position >= 0 else "COVER"
                logger.info(f"📈 {action_type} {symbol}: ${trade_value_usdt:.2f} worth")

                order = self.live_trader.execute_market_buy(
                    symbol=symbol,
                    quote_order_qty=trade_value_usdt
                )

                executed_qty = float(order['executedQty'])
                avg_price = float(order['fills'][0]['price']) if order['fills'] else current_price
                total_fee = sum(float(fill['commission']) for fill in order['fills'])

                trader['position'] = executed_qty
                trader['position_value'] = executed_qty * avg_price
                trader['entry_price'] = avg_price

                self._record_trade(
                    symbol=symbol,
                    action_type=action_type,
                    old_position=old_position,
                    new_position=executed_qty,
                    price=avg_price,
                    reasoning=reasoning,
                    order_id=order['orderId'],
                    fee=total_fee
                )

                logger.info(f"✅ {action_type} executed: {executed_qty:.8f} @ ${avg_price:.2f}")

            else:
                # SELL
                action_type = "SELL" if old_position > 0 else "SHORT"

                # Can't short on Spot (need Futures for that)
                if old_position <= 0:
                    logger.debug(f"{symbol}: Can't SHORT on Spot exchange")
                    return

                sell_qty = min(old_position, abs(position_diff))

                logger.info(f"📉 {action_type} {symbol}: {sell_qty:.8f}")

                order = self.live_trader.execute_market_sell(
                    symbol=symbol,
                    quantity=sell_qty
                )

                executed_qty = float(order['executedQty'])
                avg_price = float(order['fills'][0]['price']) if order['fills'] else current_price
                total_fee = sum(float(fill['commission']) for fill in order['fills'])

                trader['position'] -= executed_qty
                trader['position_value'] = trader['position'] * avg_price if trader['position'] > 0 else 0

                # Update balance (approximate - real balance will sync)
                trader['balance'] += executed_qty * avg_price - total_fee

                self._record_trade(
                    symbol=symbol,
                    action_type=action_type,
                    old_position=old_position,
                    new_position=trader['position'],
                    price=avg_price,
                    reasoning=reasoning,
                    order_id=order['orderId'],
                    fee=total_fee
                )

                logger.info(f"✅ {action_type} executed: {executed_qty:.8f} @ ${avg_price:.2f}")

            # Auto-save after trade
            if self.auto_save:
                self.save_state()

        except Exception as e:
            logger.error(f"❌ Order failed for {symbol}: {e}")

    def update_position_value(self, symbol: str, current_price: float):
        """Aktualisiert Position Value."""
        trader = self.traders[symbol]
        if trader['position'] > 0:
            trader['position_value'] = trader['position'] * current_price
        trader['current_price'] = current_price

    def get_portfolio_value(self, symbol: str):
        """Berechnet Portfolio-Wert für ein Symbol."""
        trader = self.traders[symbol]
        return trader['balance'] + trader['position_value']

    def trade_symbol(self, symbol: str):
        """Führt einen Trading-Zyklus für ein Symbol aus."""
        try:
            if self.emergency_stopped:
                return False

            # Get data
            df = self.get_live_data(symbol)

            # Get observation
            obs, current_price = self.get_observation(df, symbol)

            # Update position value
            self.update_position_value(symbol, current_price)

            # Check Stop-Loss
            if self.check_stop_loss(symbol):
                return False

            # Check market conditions
            safe, reason = self.check_market_conditions(symbol, df)
            if not safe:
                logger.debug(f"{symbol}: Unsafe market conditions - {reason}")
                return False

            # Check trade limits
            can_trade, reason = self.check_trade_limits(symbol)
            if not can_trade:
                logger.debug(f"{symbol}: Trade limit - {reason}")
                return False

            # Get action from model
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
        os.system('clear' if os.name == 'posix' else 'cls')

        print("\n" + "=" * 80)
        print(f"{'🧪 TESTNET' if self.testnet else '⚠️  LIVE'} MULTI-SYMBOL TRADING - LIVE PERFORMANCE")
        print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

        if self.emergency_stopped:
            print(f"🚨 EMERGENCY STOP ACTIVE: {self.emergency_reason}")
            print("=" * 80 + "\n")

        print(f"{'Symbol':<10} {'Price':>12} {'Position':>10} {'Portfolio':>14} {'Return':>10} {'Status':>10}")
        print("-" * 80)

        total_value = 0
        total_initial = 0
        total_fees = 0

        for symbol in self.symbols:
            trader = self.traders[symbol]
            portfolio_value = self.get_portfolio_value(symbol)
            ret = (portfolio_value / trader['initial_balance'] - 1) * 100

            total_value += portfolio_value
            total_initial += trader['initial_balance']
            total_fees += trader['total_fees']

            pos = trader['position']
            pos_str = f"{pos:.4f}" if pos != 0 else "FLAT"

            if trader['max_loss_reached']:
                status = "🛑 STOP"
            elif ret > 5:
                status = "✅ GOOD"
            elif ret < -5:
                status = "⚠️  WARN"
            else:
                status = "➖ OK"

            ret_str = f"{ret:+.2f}%"

            print(f"{symbol:<10} ${trader['current_price']:>10,.2f} {pos_str:>10} ${portfolio_value:>12,.2f} {ret_str:>10} {status:>10}")

        print("-" * 80)
        total_return = (total_value / total_initial - 1) * 100

        total_highest = sum(self.traders[s]['highest_value'] for s in self.symbols)
        drawdown = (total_value / total_highest - 1) * 100

        print(f"{'TOTAL':<10} {'':<12} {'':<10} ${total_value:>12,.2f} {total_return:+.2f}%")
        print(f"{'DRAWDOWN':<10} {'':<12} {'':<10} {'':<14} {drawdown:+.2f}%")
        print("=" * 80)

        print(f"\n⚠️  RISK METRICS:")
        print(f"   Total Drawdown:         {drawdown:>+8.2f}% (Limit: {self.max_total_drawdown*100:.0f}%)")
        print(f"   Total Fees Paid:        ${total_fees:>8,.2f}")
        print(f"   Emergency Stop:         {'YES 🚨' if self.emergency_stopped else 'NO ✅'}")

        print("\nPress Ctrl+C for final summary\n")

    def run(self, interval_seconds: int = 60, detailed_every: int = 10):
        """Startet Live Trading Loop."""
        logger.info("Starting LIVE Multi-Symbol Trading...")
        logger.info(f"Mode: {'🧪 TESTNET' if self.testnet else '⚠️  PRODUCTION'}")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Update Interval: {interval_seconds}s")

        iteration = 0
        try:
            while True:
                if self.check_total_drawdown():
                    self.print_overview(show_detailed=True)
                    logger.critical("EMERGENCY STOP - Trading halted!")
                    self.save_state(force=True)
                    break

                for symbol in self.symbols:
                    if not self.emergency_stopped:
                        self.trade_symbol(symbol)

                if self.auto_save:
                    self.save_state()

                iteration += 1
                show_detailed = (iteration % detailed_every == 0)
                self.print_overview(show_detailed=show_detailed)

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\nLive Trading stopped by user")
            logger.info("Saving final state...")
            self.save_state(force=True)
            self.print_final_summary()

    def print_final_summary(self):
        """Zeigt finale Zusammenfassung."""
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)

        total_value = 0
        total_fees = 0
        total_trades = 0

        for symbol in self.symbols:
            trader = self.traders[symbol]
            portfolio_value = self.get_portfolio_value(symbol)
            ret = (portfolio_value / trader['initial_balance'] - 1) * 100

            total_value += portfolio_value
            total_fees += trader['total_fees']
            total_trades += trader['num_trades']

            status = "🛑" if trader['max_loss_reached'] else ""
            print(f"{symbol}: {ret:+.2f}% | ${portfolio_value:,.2f} | {trader['num_trades']} trades | ${trader['total_fees']:.2f} fees {status}")

        print("-" * 80)
        total_initial = sum(t['initial_balance'] for t in self.traders.values())
        total_return = (total_value / total_initial - 1) * 100

        print(f"Total Portfolio: ${total_value:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Total Fees: ${total_fees:.2f}")

        if self.emergency_stopped:
            print(f"\n🚨 EMERGENCY STOPPED: {self.emergency_reason}")

        print("=" * 80)


def main():
    """Hauptfunktion für LIVE Multi-Symbol Trading."""
    import argparse

    parser = argparse.ArgumentParser(description='LIVE Multi-Symbol Trading')
    parser.add_argument('--config', type=str, default='models/multi_symbol_config.json')
    parser.add_argument('--balance', type=float, default=10000.0)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--testnet', action='store_true', default=True,
                       help='Use Testnet (default: True)')
    parser.add_argument('--production', action='store_true',
                       help='⚠️  Use PRODUCTION (REAL MONEY!)')

    args = parser.parse_args()

    # Safety check for production
    if args.production:
        print("\n" + "=" * 80)
        print("⚠️  WARNING: You are about to trade with REAL MONEY!")
        print("=" * 80)
        response = input("Type 'I UNDERSTAND THE RISKS' to continue: ")
        if response != 'I UNDERSTAND THE RISKS':
            print("Aborted.")
            return
        testnet = False
    else:
        testnet = True

    trader = LiveMultiSymbolTrader(
        config_path=args.config,
        initial_balance=args.balance,
        testnet=testnet
    )

    trader.run(interval_seconds=args.interval)


if __name__ == '__main__':
    main()
