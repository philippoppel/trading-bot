"""
Trainiert nur den RL-Agent (SAC).
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
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    logger.info("=" * 60)
    logger.info("TRAINING RL AGENT (SAC)")
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

    # SAC Agent
    agent = create_sac_agent(
        train_env,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=256,
        verbose=1
    )

    # Callbacks
    os.makedirs('models/checkpoints_sac', exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='models/checkpoints_sac',
        name_prefix='sac'
    )

    # Training
    logger.info("Training for 50000 timesteps...")

    agent.learn(
        total_timesteps=50000,
        callback=checkpoint_callback,
        progress_bar=False
    )

    # Save
    agent.save('models/sac_final')
    logger.info("Model saved to models/sac_final")

    # Evaluation
    logger.info("\nEvaluating...")

    test_env = AdvancedTradingEnv(
        df=test_df,
        feature_columns=feature_cols,
        initial_balance=10000.0,
        observation_window=60
    )

    obs, info = test_env.reset()
    done = False

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

    # Results
    final_value = info['portfolio_value']
    ret = (final_value / 10000 - 1) * 100

    logger.info("\n" + "=" * 40)
    logger.info("RL EVALUATION RESULTS")
    logger.info("=" * 40)
    logger.info(f"Final Portfolio: ${final_value:,.2f}")
    logger.info(f"Return: {ret:+.2f}%")
    logger.info(f"Trades: {info['num_trades']}")
    logger.info("=" * 40)

if __name__ == '__main__':
    main()
