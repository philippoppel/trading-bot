"""
Trainiert RL-Modelle für mehrere Kryptowährungen.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime

# Projekt-Root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.binance_client import BinanceDataClient
from src.data.preprocessor import DataPreprocessor
from src.environment.advanced_trading_env import AdvancedTradingEnv, create_sac_agent
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3 import SAC


class ProfitCallback(BaseCallback):
    """Callback das Training stoppt wenn Profit erreicht wird."""

    def __init__(self, test_env, symbol, check_freq=10000, target_return=0.0, verbose=1):
        super().__init__(verbose)
        self.test_env = test_env
        self.symbol = symbol
        self.check_freq = check_freq
        self.target_return = target_return
        self.best_return = -float('inf')

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            obs, info = self.test_env.reset()
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.test_env.step(action)
                done = terminated or truncated

            final_value = info['portfolio_value']
            ret = (final_value / 10000 - 1) * 100

            if ret > self.best_return:
                self.best_return = ret
                self.model.save(f'models/sac_{self.symbol.lower()}_best')
                logger.info(f"[{self.symbol}] New best! Return: {ret:+.2f}%")

            logger.info(f"[{self.symbol}] Step {self.n_calls}: Return={ret:+.2f}%, Value=${final_value:,.2f}")

            if ret >= self.target_return:
                logger.info(f"[{self.symbol}] TARGET REACHED! Return: {ret:+.2f}%")
                return False

        return True


def train_symbol(symbol: str, days: int = 365, max_timesteps: int = 100000, target_return: float = 5.0):
    """Trainiert ein Modell für ein einzelnes Symbol."""
    logger.info("=" * 60)
    logger.info(f"TRAINING {symbol}")
    logger.info("=" * 60)

    # Daten laden
    client = BinanceDataClient()

    try:
        df = client.get_data(symbol, '1h', days)
    except Exception as e:
        logger.error(f"Failed to load data for {symbol}: {e}")
        return None

    if len(df) < 1000:
        logger.warning(f"Not enough data for {symbol}: {len(df)} rows")
        return None

    preprocessor = DataPreprocessor()
    df = preprocessor.process(df, normalize=True)

    logger.info(f"[{symbol}] Data: {len(df)} rows")

    # Features
    feature_cols = [c for c in df.columns if c.endswith('_norm')]
    logger.info(f"[{symbol}] Using {len(feature_cols)} features")

    # Train/Test Split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)

    # Environments
    train_env = AdvancedTradingEnv(
        df=train_df,
        feature_columns=feature_cols,
        initial_balance=10000.0,
        fee_rate=0.001,
        observation_window=60
    )

    test_env = AdvancedTradingEnv(
        df=test_df,
        feature_columns=feature_cols,
        initial_balance=10000.0,
        fee_rate=0.001,
        observation_window=60
    )

    # SAC Agent
    agent = create_sac_agent(
        train_env,
        learning_rate=1e-4,
        buffer_size=100000,
        batch_size=512,
        learning_starts=2000,
        verbose=1
    )

    # Callbacks
    os.makedirs(f'models/checkpoints_{symbol.lower()}', exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=f'models/checkpoints_{symbol.lower()}',
        name_prefix='sac'
    )

    profit_callback = ProfitCallback(
        test_env=test_env,
        symbol=symbol,
        check_freq=10000,
        target_return=target_return,
        verbose=1
    )

    # Training
    logger.info(f"[{symbol}] Training for up to {max_timesteps} timesteps (target: {target_return}% return)...")

    agent.learn(
        total_timesteps=max_timesteps,
        callback=[checkpoint_callback, profit_callback],
        progress_bar=False
    )

    # Final evaluation
    best_model = SAC.load(f'models/sac_{symbol.lower()}_best', env=test_env)

    obs, info = test_env.reset()
    done = False

    while not done:
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

    final_value = info['portfolio_value']
    ret = (final_value / 10000 - 1) * 100

    logger.info(f"[{symbol}] Final Return: {ret:+.2f}%, Value: ${final_value:,.2f}, Trades: {info['num_trades']}")

    return {
        'symbol': symbol,
        'return': ret,
        'final_value': final_value,
        'trades': info['num_trades'],
        'model_path': f'models/sac_{symbol.lower()}_best.zip'
    }


def main():
    """Trainiert Modelle für mehrere Kryptowährungen."""
    logger.info("=" * 60)
    logger.info("MULTI-SYMBOL TRAINING")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Symbols to train
    symbols = [
        'BTCUSDT',   # Bitcoin
        'ETHUSDT',   # Ethereum
        'BNBUSDT',   # Binance Coin
        'SOLUSDT',   # Solana
        'XRPUSDT',   # Ripple
    ]

    # Training parameters
    days = 365           # 1 Jahr Daten
    max_timesteps = 100000  # Max Steps pro Symbol
    target_return = 5.0  # 5% Ziel-Return

    # Results
    results = []

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n[{i}/{len(symbols)}] Training {symbol}...")

        result = train_symbol(
            symbol=symbol,
            days=days,
            max_timesteps=max_timesteps,
            target_return=target_return
        )

        if result:
            results.append(result)
        else:
            logger.warning(f"Training failed for {symbol}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)

    if results:
        logger.info(f"\nSuccessfully trained {len(results)}/{len(symbols)} models:\n")

        for r in results:
            status = "✓" if r['return'] > 0 else "✗"
            logger.info(f"{status} {r['symbol']}: {r['return']:+.2f}% | ${r['final_value']:,.2f} | {r['trades']} trades")
            logger.info(f"  Model: {r['model_path']}")

        # Best performer
        best = max(results, key=lambda x: x['return'])
        logger.info(f"\nBest performer: {best['symbol']} with {best['return']:+.2f}%")

        # Calculate average
        avg_return = np.mean([r['return'] for r in results])
        logger.info(f"Average return: {avg_return:+.2f}%")

    logger.info("\n" + "=" * 60)
    logger.info("Training completed!")
    logger.info(f"Models saved to: ./models/")
    logger.info("=" * 60)

    # Create config for paper trading
    config = {
        'symbols': [r['symbol'] for r in results if r['return'] > 0],
        'models': {r['symbol']: r['model_path'] for r in results if r['return'] > 0}
    }

    import json
    with open('models/multi_symbol_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"\nConfig saved to models/multi_symbol_config.json")
    logger.info(f"Profitable symbols: {config['symbols']}")


if __name__ == '__main__':
    main()
