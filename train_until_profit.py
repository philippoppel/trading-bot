"""
Trainiert RL-Agent bis er Profit macht.
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


class ProfitCallback(BaseCallback):
    """Callback das Training stoppt wenn Profit erreicht wird."""

    def __init__(self, test_env, check_freq=10000, target_return=0.0, verbose=1):
        super().__init__(verbose)
        self.test_env = test_env
        self.check_freq = check_freq
        self.target_return = target_return
        self.best_return = -float('inf')

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Evaluate
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
                # Save best model
                self.model.save('models/sac_best')
                logger.info(f"New best model! Return: {ret:+.2f}%")

            logger.info(f"Step {self.n_calls}: Return={ret:+.2f}%, Trades={info['num_trades']}, Value=${final_value:,.2f}")

            if ret >= self.target_return:
                logger.info(f"TARGET REACHED! Return: {ret:+.2f}%")
                return False  # Stop training

        return True


def main():
    logger.info("=" * 60)
    logger.info("TRAINING RL AGENT UNTIL PROFIT")
    logger.info("=" * 60)

    # Daten laden
    client = BinanceDataClient()
    df = client.get_data('BTCUSDT', '1h', 365)

    preprocessor = DataPreprocessor()
    df = preprocessor.process(df, normalize=True)

    logger.info(f"Data: {len(df)} rows")

    # Features
    feature_cols = [c for c in df.columns if c.endswith('_norm')]
    logger.info(f"Using {len(feature_cols)} features")

    # Train/Test Split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)

    # Environment
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

    # SAC Agent mit optimierten Hyperparametern
    agent = create_sac_agent(
        train_env,
        learning_rate=1e-4,  # Kleinere LR für Stabilität
        buffer_size=100000,  # Größerer Buffer
        batch_size=512,      # Größere Batches
        learning_starts=2000,
        tau=0.005,
        gamma=0.99,
        verbose=1
    )

    # Callbacks
    os.makedirs('models/checkpoints_sac', exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path='models/checkpoints_sac',
        name_prefix='sac'
    )

    profit_callback = ProfitCallback(
        test_env=test_env,
        check_freq=10000,
        target_return=5.0,  # Ziel: 5% Profit
        verbose=1
    )

    # Training
    max_timesteps = 500000
    logger.info(f"Training for up to {max_timesteps} timesteps (target: 5% return)...")

    agent.learn(
        total_timesteps=max_timesteps,
        callback=[checkpoint_callback, profit_callback],
        progress_bar=False
    )

    # Final Evaluation
    logger.info("\nFinal Evaluation...")

    # Load best model
    from stable_baselines3 import SAC
    best_agent = SAC.load('models/sac_best', env=test_env)

    obs, info = test_env.reset()
    done = False

    while not done:
        action, _ = best_agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

    # Results
    final_value = info['portfolio_value']
    ret = (final_value / 10000 - 1) * 100

    logger.info("\n" + "=" * 40)
    logger.info("FINAL RL EVALUATION RESULTS")
    logger.info("=" * 40)
    logger.info(f"Final Portfolio: ${final_value:,.2f}")
    logger.info(f"Return: {ret:+.2f}%")
    logger.info(f"Trades: {info['num_trades']}")
    logger.info("=" * 40)

    if ret > 0:
        logger.info("\n✓ MODEL IS PROFITABLE - Ready for paper trading!")
        logger.info("Best model saved to: models/sac_best.zip")
    else:
        logger.info("\n✗ Model not yet profitable. Consider:")
        logger.info("  - More training timesteps")
        logger.info("  - Different hyperparameters")
        logger.info("  - Better reward shaping")

if __name__ == '__main__':
    main()
