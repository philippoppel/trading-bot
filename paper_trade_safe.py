"""
SAFE Multi-Symbol Paper Trading mit umfassenden Risk Management Features.

WICHTIG: Selbst mit allen Sicherheitsmechanismen gibt es KEINE Garantie für Profit!
Trading mit echtem Geld hat immer Risiken.
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
import requests
from typing import Optional

# Projekt-Root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.binance_client import BinanceDataClient
from src.data.preprocessor import DataPreprocessor
from stable_baselines3 import SAC


class SafeMultiSymbolTrader:
    """Paper Trading mit umfassenden Sicherheitsmechanismen."""

    def __init__(
        self,
        config_path: str = 'models/multi_symbol_config.json',
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001,
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
        # PERSISTENCE PARAMETERS
        state_file: Optional[str] = None,      # State save file
        auto_save: bool = True,                # Auto-save after trades
        save_interval: int = 300,              # Save every N seconds
        keep_backups: int = 5,                 # Number of backups to keep
        # DASHBOARD UPLOAD
        upload_to_dashboard: bool = True,
        dashboard_url: Optional[str] = None
    ):
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
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

        # PERSISTENCE PARAMETERS
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
            self.state_file = state_dir / 'safe_multi_symbol_state.json'
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

            # Initialize trader state (will be overwritten if we load saved state)
            self.traders[symbol] = {
                'balance': initial_balance,
                'position': 0.0,
                'position_value': 0.0,
                'entry_price': 0.0,
                'total_fees': 0.0,
                'num_trades': 0,
                'trade_history': [],
                'current_price': 0.0,
                'last_trade_time': None,
                'trades_last_hour': [],
                'max_loss_reached': False,
                'highest_value': initial_balance,  # Track peak for drawdown
            }

        # Try to load existing state
        self.session_start_time = datetime.now()
        if self.load_state():
            logger.info("✅ Restored previous trading state")
        else:
            logger.info("🆕 Starting fresh trading session")

        logger.info(f"Loaded {len(self.symbols)} models")
        logger.info("=" * 60)
        logger.info("⚠️  RISK MANAGEMENT SETTINGS:")
        logger.info(f"   Max Position Size:      {max_position_size*100:.0f}%")
        logger.info(f"   Stop-Loss per Symbol:   {max_loss_per_symbol*100:.0f}%")
        logger.info(f"   Total Drawdown Limit:   {max_total_drawdown*100:.0f}%")
        logger.info(f"   Max Trades/Hour:        {max_trades_per_hour}")
        logger.info(f"   Min Trade Interval:     {min_trade_interval}s")
        logger.info(f"   Volatility Threshold:   {volatility_threshold*100:.0f}%")
        logger.info("=" * 60)
        logger.info("💾 PERSISTENCE SETTINGS:")
        logger.info(f"   State File:             {self.state_file}")
        logger.info(f"   Auto-Save:              {'Enabled' if auto_save else 'Disabled'}")
        logger.info(f"   Save Interval:          {save_interval}s")
        logger.info(f"   Backups to Keep:        {keep_backups}")
        logger.info("=" * 60)

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

        # Convert datetime objects to ISO strings
        if trader['last_trade_time']:
            serialized['last_trade_time'] = trader['last_trade_time'].isoformat()

        serialized['trades_last_hour'] = [
            t.isoformat() for t in trader['trades_last_hour']
        ]

        return serialized

    def _deserialize_trader_state(self, trader: dict) -> dict:
        """Konvertiert JSON State zurück zu Trader State."""
        deserialized = trader.copy()

        # Convert ISO strings back to datetime
        if trader['last_trade_time']:
            deserialized['last_trade_time'] = datetime.fromisoformat(trader['last_trade_time'])
        else:
            deserialized['last_trade_time'] = None

        deserialized['trades_last_hour'] = [
            datetime.fromisoformat(t) for t in trader['trades_last_hour']
        ]

        return deserialized

    def save_state(self, force: bool = False) -> bool:
        """
        Speichert aktuellen Trading State.

        Args:
            force: Wenn True, ignoriert save_interval

        Returns:
            True wenn erfolgreich gespeichert
        """
        # Check if we should save (based on interval)
        if not force and self.last_save_time:
            elapsed = (datetime.now() - self.last_save_time).total_seconds()
            if elapsed < self.save_interval:
                return False

        try:
            # Prepare state data
            state = {
                'version': '1.0',
                'timestamp': datetime.now().isoformat(),
                'session_start_time': self.session_start_time.isoformat(),
                'initial_balance': self.initial_balance,
                'emergency_stopped': self.emergency_stopped,
                'emergency_reason': self.emergency_reason,
                'traders': {}
            }

            # Serialize each trader
            for symbol, trader in self.traders.items():
                state['traders'][symbol] = self._serialize_trader_state(trader)

            # Create backup of existing state file
            if self.state_file.exists() and self.keep_backups > 0:
                self._create_backup()

            # Atomic write: write to temp file then rename
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)

            # Atomic rename
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

            # Create timestamped backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_dir / f"{self.state_file.stem}_{timestamp}.json"

            shutil.copy2(self.state_file, backup_file)
            logger.debug(f"📦 Backup created: {backup_file}")

            # Clean old backups (keep only N most recent)
            backups = sorted(backup_dir.glob(f"{self.state_file.stem}_*.json"))
            if len(backups) > self.keep_backups:
                for old_backup in backups[:-self.keep_backups]:
                    old_backup.unlink()
                    logger.debug(f"🗑️  Deleted old backup: {old_backup}")

        except Exception as e:
            logger.warning(f"Error creating backup: {e}")

    def load_state(self) -> bool:
        """
        Lädt gespeicherten Trading State.

        Returns:
            True wenn State erfolgreich geladen wurde
        """
        if not self.state_file.exists():
            logger.info("No saved state found")
            return False

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Validate version
            if state.get('version') != '1.0':
                logger.warning(f"Incompatible state version: {state.get('version')}")
                return False

            # Check if initial balance matches
            if abs(state['initial_balance'] - self.initial_balance) > 0.01:
                logger.warning(
                    f"State initial balance ({state['initial_balance']}) "
                    f"doesn't match config ({self.initial_balance})"
                )
                logger.warning("Using saved state anyway - be aware of this discrepancy")

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
            logger.info(f"   Session started: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Print restored positions
            for symbol in self.symbols:
                trader = self.traders[symbol]
                if trader['position'] != 0:
                    logger.info(f"   {symbol}: Position {trader['position']:.2f} | "
                              f"{trader['num_trades']} trades | "
                              f"${trader['total_fees']:.2f} fees")

            return True

        except Exception as e:
            logger.error(f"Error loading state: {e}")
            logger.error("Starting with fresh state")
            return False

    def calculate_volatility(self, df: pd.DataFrame, window: int = 24) -> float:
        """Berechnet aktuelle Volatilität."""
        if len(df) < window:
            return 0.0
        returns = df['close'].pct_change().tail(window)
        return returns.std()

    def check_market_conditions(self, symbol: str, df: pd.DataFrame) -> tuple[bool, str]:
        """Prüft ob Market Conditions safe für Trading sind."""
        # Check volatility
        volatility = self.calculate_volatility(df)
        if volatility > self.volatility_threshold:
            return False, f"High volatility: {volatility*100:.2f}%"

        # Check if we have enough data
        if len(df) < self.observation_window:
            return False, "Insufficient data"

        return True, "OK"

    def check_stop_loss(self, symbol: str) -> bool:
        """Prüft ob Stop-Loss erreicht wurde."""
        if not self.use_stop_loss:
            return False

        trader = self.traders[symbol]
        portfolio_value = self.get_portfolio_value(symbol)
        loss = (portfolio_value / self.initial_balance - 1)

        if loss <= self.max_loss_per_symbol:
            logger.warning(f"🛑 STOP-LOSS triggered for {symbol}: {loss*100:.2f}%")
            trader['max_loss_reached'] = True

            # Record trade history for stop-loss closure
            if trader['position'] != 0:
                current_price = trader['current_price']
                trade_time = datetime.now()

                trade_record = {
                    'timestamp': trade_time.isoformat(),
                    'symbol': symbol,
                    'action_type': 'STOP_LOSS',
                    'old_position': trader['position'],
                    'new_position': 0.0,
                    'position_change': -trader['position'],
                    'price': current_price,
                    'trade_value': abs(trader['position_value']),
                    'fee': 0.0,
                    'slippage': 0.0,
                    'total_cost': 0.0,
                    'balance_before': trader['balance'],
                    'balance_after': portfolio_value,
                    'portfolio_value_before': portfolio_value,
                    'portfolio_value_after': portfolio_value,
                    'reasoning': f"Stop-loss triggered at {loss*100:.2f}% loss",
                    'model_action': 0.0
                }

                trader['trade_history'].append(trade_record)

            # Close all positions
            trader['position'] = 0.0
            trader['balance'] = portfolio_value
            trader['position_value'] = 0.0
            return True

        return False

    def check_take_profit(self, symbol: str) -> bool:
        """Prüft ob Take-Profit erreicht wurde."""
        if not self.use_take_profit:
            return False

        trader = self.traders[symbol]
        if trader['position'] == 0 or trader['entry_price'] == 0:
            return False

        current_price = trader['current_price']
        pnl_pct = (current_price - trader['entry_price']) / trader['entry_price']

        # Bei Long: Profit wenn Preis steigt
        # Bei Short: Profit wenn Preis fällt (pnl_pct negativ ist gut)
        should_take_profit = False
        profit_text = ""

        if trader['position'] > 0:  # Long
            if pnl_pct >= self.take_profit_pct:
                should_take_profit = True
                profit_text = f"+{pnl_pct*100:.2f}%"
        else:  # Short
            if pnl_pct <= -self.take_profit_pct:
                should_take_profit = True
                profit_text = f"+{abs(pnl_pct)*100:.2f}%"

        if should_take_profit:
            logger.info(f"✅ TAKE-PROFIT triggered for {symbol}: {profit_text}")

            portfolio_value_before = self.get_portfolio_value(symbol)
            trade_time = datetime.now()

            trade_record = {
                'timestamp': trade_time.isoformat(),
                'symbol': symbol,
                'action_type': 'TAKE_PROFIT',
                'old_position': trader['position'],
                'new_position': 0.0,
                'position_change': -trader['position'],
                'price': current_price,
                'trade_value': abs(trader['position_value']),
                'fee': 0.0,
                'slippage': 0.0,
                'total_cost': 0.0,
                'balance_before': trader['balance'],
                'balance_after': portfolio_value_before,
                'portfolio_value_before': portfolio_value_before,
                'portfolio_value_after': portfolio_value_before,
                'reasoning': f"Take-profit triggered at {profit_text} gain",
                'model_action': 0.0
            }

            trader['trade_history'].append(trade_record)

            # Close position
            trader['balance'] = portfolio_value_before
            trader['position'] = 0.0
            trader['position_value'] = 0.0
            return True

        return False

    def check_total_drawdown(self) -> bool:
        """Prüft ob maximaler Total Drawdown erreicht wurde."""
        total_value = sum(self.get_portfolio_value(s) for s in self.symbols)
        total_initial = self.initial_balance * len(self.symbols)

        # Track highest value for drawdown calculation
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
            return True

        return False

    def check_trade_limits(self, symbol: str) -> tuple[bool, str]:
        """Prüft ob Trade-Limits eingehalten werden."""
        trader = self.traders[symbol]

        # Check if symbol has reached max loss
        if trader['max_loss_reached']:
            return False, "Max loss reached - trading paused"

        # Check min time between trades
        if trader['last_trade_time']:
            time_since_last = (datetime.now() - trader['last_trade_time']).total_seconds()
            if time_since_last < self.min_trade_interval:
                return False, f"Min interval not met ({time_since_last:.0f}s < {self.min_trade_interval}s)"

        # Check max trades per hour
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

    def execute_action(self, symbol: str, action: float, current_price: float, reasoning: str = "Model decision"):
        """Führt Trading-Aktion aus mit Risk Management."""
        trader = self.traders[symbol]

        # Apply max position size limit
        target_position = np.clip(action, -self.max_position_size, self.max_position_size)

        position_diff = target_position - trader['position']

        if abs(position_diff) < 0.01:
            return

        # Calculate trade
        portfolio_value_before = trader['balance'] + trader['position_value']
        trade_value = abs(position_diff) * portfolio_value_before

        # Fees (inkl. Slippage estimate)
        fee = trade_value * self.fee_rate
        slippage = trade_value * 0.0005  # 0.05% Slippage estimate
        total_cost = fee + slippage

        trader['total_fees'] += total_cost

        # Update position
        old_position = trader['position']
        old_balance = trader['balance']
        trader['position'] = target_position

        # Determine action type
        if position_diff > 0:  # Buying/Going Long
            action_type = "BUY" if old_position >= 0 else "COVER"
            trader['balance'] -= trade_value + total_cost
            trader['position_value'] = trader['position'] * portfolio_value_before
        else:  # Selling/Going Short
            action_type = "SELL" if old_position > 0 else "SHORT"
            trader['balance'] += trade_value - total_cost
            trader['position_value'] = trader['position'] * portfolio_value_before

        trader['entry_price'] = current_price
        trader['num_trades'] += 1
        trade_time = datetime.now()
        trader['last_trade_time'] = trade_time
        trader['trades_last_hour'].append(trade_time)

        portfolio_value_after = trader['balance'] + trader['position_value']

        # Record detailed trade history
        trade_record = {
            'timestamp': trade_time.isoformat(),
            'symbol': symbol,
            'action_type': action_type,
            'old_position': old_position,
            'new_position': target_position,
            'position_change': position_diff,
            'price': current_price,
            'trade_value': trade_value,
            'fee': fee,
            'slippage': slippage,
            'total_cost': total_cost,
            'balance_before': old_balance,
            'balance_after': trader['balance'],
            'portfolio_value_before': portfolio_value_before,
            'portfolio_value_after': portfolio_value_after,
            'reasoning': reasoning,
            'model_action': action
        }

        trader['trade_history'].append(trade_record)

        # Log trade
        logger.info(f"Trade {symbol}: {action_type} {old_position:.2f} → {target_position:.2f} @ ${current_price:,.2f} | Fee: ${total_cost:.2f} | {reasoning}")

        # Auto-save after trade
        if self.auto_save:
            self.save_state()

    def update_position_value(self, symbol: str, current_price: float):
        """Aktualisiert Position Value."""
        trader = self.traders[symbol]
        if trader['position'] != 0 and trader['entry_price'] > 0:
            # P&L Berechnung
            pnl_pct = (current_price - trader['entry_price']) / trader['entry_price']
            # Position value: negativ bei SHORT, positiv bei LONG
            # Bei SHORT: wenn Preis fällt (pnl_pct negativ), wird position_value weniger negativ = Gewinn
            trader['position_value'] = trader['position'] * self.initial_balance * (1 + pnl_pct)
        trader['current_price'] = current_price

    def get_portfolio_value(self, symbol: str):
        """Berechnet Portfolio-Wert für ein Symbol."""
        trader = self.traders[symbol]
        return trader['balance'] + trader['position_value']

    def trade_symbol(self, symbol: str):
        """Führt einen Trading-Zyklus für ein Symbol aus."""
        try:
            # Check if emergency stopped
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

            # Check Take-Profit
            if self.check_take_profit(symbol):
                return True

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
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')

        print("\n" + "=" * 80)
        print("🛡️  SAFE MULTI-SYMBOL PAPER TRADING - LIVE PERFORMANCE")
        print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

        if self.emergency_stopped:
            print(f"🚨 EMERGENCY STOP ACTIVE: {self.emergency_reason}")
            print("=" * 80 + "\n")

        # Header
        print(f"{'Symbol':<10} {'Price':>12} {'Position':>10} {'Portfolio':>14} {'Return':>10} {'Status':>10}")
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

            # Status
            if trader['max_loss_reached']:
                status = "🛑 STOP"
            elif ret > 5:
                status = "✅ GOOD"
            elif ret < -5:
                status = "⚠️  WARN"
            else:
                status = "➖ OK"

            # Color for return
            ret_str = f"{ret:+.2f}%"

            print(f"{symbol:<10} ${trader['current_price']:>10,.2f} {pos_str:>10} ${portfolio_value:>12,.2f} {ret_str:>10} {status:>10}")

        # Total
        print("-" * 80)
        total_return = (total_value / total_initial - 1) * 100
        avg_return = total_return / len(self.symbols) if self.symbols else 0

        # Drawdown
        total_highest = sum(self.traders[s]['highest_value'] for s in self.symbols)
        drawdown = (total_value / total_highest - 1) * 100

        print(f"{'TOTAL':<10} {'':<12} {'':<10} ${total_value:>12,.2f} {total_return:+.2f}%")
        print(f"{'DRAWDOWN':<10} {'':<12} {'':<10} {'':<14} {drawdown:+.2f}%")
        print("=" * 80)

        # Risk Metrics
        print(f"\n⚠️  RISK METRICS:")
        print(f"   Total Drawdown:         {drawdown:>+8.2f}% (Limit: {self.max_total_drawdown*100:.0f}%)")
        print(f"   Total Fees Paid:        ${total_fees:>8,.2f}")
        print(f"   Emergency Stop:         {'YES 🚨' if self.emergency_stopped else 'NO ✅'}")

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

            print(f"\n🔄 Trading Activity by Symbol:")
            for symbol in self.symbols:
                trader = self.traders[symbol]
                trades_last_hour = len(trader['trades_last_hour'])
                print(f"   {symbol:<10} {trader['num_trades']:>4} trades | ${trader['total_fees']:>8,.2f} fees | {trades_last_hour} last hour")

            print("=" * 80)

        print("\nPress Ctrl+C for final summary\n")

    def run(self, interval_seconds: int = 60, detailed_every: int = 10):
        """Startet Multi-Symbol Trading Loop."""
        logger.info("Starting SAFE Multi-Symbol Paper Trading...")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Initial Balance per Symbol: ${self.initial_balance:,.2f}")
        logger.info(f"Update Interval: {interval_seconds}s")

        iteration = 0
        try:
            while True:
                # Check total drawdown first
                if self.check_total_drawdown():
                    self.print_overview(show_detailed=True)
                    logger.critical("EMERGENCY STOP - Trading halted!")
                    # Save state before exiting
                    self.save_state(force=True)
                    break

                # Trade all symbols
                for symbol in self.symbols:
                    if not self.emergency_stopped:
                        self.trade_symbol(symbol)

                # Periodic save (even if no trades)
                if self.auto_save:
                    self.save_state()

                # Show overview (detailed every N iterations)
                iteration += 1
                show_detailed = (iteration % detailed_every == 0)
                self.print_overview(show_detailed=show_detailed)

                # Wait
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\nPaper Trading stopped by user")
            # Save final state before exiting
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
            ret = (portfolio_value / self.initial_balance - 1) * 100

            total_value += portfolio_value
            total_fees += trader['total_fees']
            total_trades += trader['num_trades']

            status = "🛑" if trader['max_loss_reached'] else ""
            print(f"{symbol}: {ret:+.2f}% | ${portfolio_value:,.2f} | {trader['num_trades']} trades | ${trader['total_fees']:.2f} fees {status}")

        print("-" * 80)
        total_initial = self.initial_balance * len(self.symbols)
        total_return = (total_value / total_initial - 1) * 100

        # Sharpe Ratio (simplified)
        returns = [(self.get_portfolio_value(s) / self.initial_balance - 1) for s in self.symbols]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) if returns else 0

        print(f"Total Portfolio: ${total_value:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Total Fees: ${total_fees:.2f}")
        print(f"Sharpe Ratio: {sharpe:.2f}")

        if self.emergency_stopped:
            print(f"\n🚨 EMERGENCY STOPPED: {self.emergency_reason}")

        print("=" * 80)


def main():
    """Hauptfunktion für SAFE Multi-Symbol Paper Trading."""
    import argparse

    parser = argparse.ArgumentParser(description='SAFE Multi-Symbol Paper Trading')
    parser.add_argument('--config', type=str, default='models/multi_symbol_config.json',
                       help='Pfad zur Config-Datei')
    parser.add_argument('--balance', type=float, default=10000.0,
                       help='Startkapital pro Symbol')
    parser.add_argument('--interval', type=int, default=60,
                       help='Update-Intervall in Sekunden')
    parser.add_argument('--detailed-every', type=int, default=10,
                       help='Zeige detaillierte Stats alle N Iterationen')

    # Risk Management Parameters
    parser.add_argument('--max-position', type=float, default=0.3,
                       help='Max Position Size (0.0-1.0)')
    parser.add_argument('--max-loss', type=float, default=-0.15,
                       help='Stop-Loss per Symbol (z.B. -0.15 = -15%%)')
    parser.add_argument('--max-drawdown', type=float, default=-0.20,
                       help='Max Total Drawdown (z.B. -0.20 = -20%%)')
    parser.add_argument('--max-trades-hour', type=int, default=5,
                       help='Max Trades pro Stunde')

    args = parser.parse_args()

    trader = SafeMultiSymbolTrader(
        config_path=args.config,
        initial_balance=args.balance,
        max_position_size=args.max_position,
        max_loss_per_symbol=args.max_loss,
        max_total_drawdown=args.max_drawdown,
        max_trades_per_hour=args.max_trades_hour,
        dashboard_url='https://trading-dashboard-83z7kxm6d-philipps-projects-0f51423d.vercel.app/api/upload'
    )

    trader.run(interval_seconds=args.interval, detailed_every=args.detailed_every)


if __name__ == '__main__':
    main()
