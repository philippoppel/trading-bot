"""
Trainings-Skript für alle Modelle.

Trainiert:
1. LSTM-Modell für Trendvorhersage
2. Transformer-Modell für Trendvorhersage
3. RL-Agent (SAC) mit erweiterter Umgebung
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.binance_client import BinanceDataClient
from src.data.preprocessor import DataPreprocessor
from src.models.model_manager import ModelManager
from src.models.lstm_model import create_sequences, create_trend_labels
from src.environment.advanced_trading_env import AdvancedTradingEnv, create_sac_agent


def load_and_prepare_data(symbol: str = 'BTCUSDT', days: int = 365):
    """Lädt und bereitet Daten vor."""
    logger.info(f"Loading data for {symbol}...")

    client = BinanceDataClient()

    # Verwende get_data Methode die days unterstützt
    df = client.get_data(symbol=symbol, interval='1h', days=days)

    logger.info(f"Loaded {len(df)} rows")

    # Preprocessing
    preprocessor = DataPreprocessor()
    df = preprocessor.process(df, normalize=True, normalize_method='zscore')

    logger.info(f"Processed data: {df.shape}")
    logger.info(f"Features: {len(df.columns)}")

    return df, preprocessor


def train_lstm_model(df: pd.DataFrame, save_dir: str = 'models'):
    """Trainiert LSTM-Modell."""
    logger.info("=" * 60)
    logger.info("TRAINING LSTM MODEL")
    logger.info("=" * 60)

    # Feature-Spalten (normalisierte)
    feature_cols = [c for c in df.columns if c.endswith('_norm')]
    features = df[feature_cols].values

    logger.info(f"Using {len(feature_cols)} features")

    # Labels erstellen
    labels = create_trend_labels(df['close'].values, threshold=0.005)

    # Sequenzen erstellen
    seq_length = 60
    X, y = create_sequences(features[:-1], labels, seq_length)

    logger.info(f"Created {len(X)} sequences")
    logger.info(f"Sequence shape: {X.shape}")

    # Class Distribution
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        logger.info(f"  Class {cls}: {count} ({count/len(y)*100:.1f}%)")

    # Train/Val/Test Split
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Model Manager
    manager = ModelManager(device='cpu')

    # LSTM erstellen
    model = manager.create_lstm(
        'lstm_trader',
        input_size=X.shape[2],
        hidden_size=128,
        num_layers=2,
        num_classes=3,
        dropout=0.3,
        learning_rate=0.001
    )

    # Training
    save_path = os.path.join(save_dir, 'lstm_best.pt')

    history = manager.train_model(
        'lstm_trader',
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=64,
        early_stopping_patience=10,
        save_path=save_path
    )

    # Evaluation
    eval_result = manager.evaluate_model('lstm_trader', X_test, y_test)

    logger.info("\n" + "=" * 40)
    logger.info("LSTM EVALUATION RESULTS")
    logger.info("=" * 40)
    logger.info(f"Test Loss: {eval_result['loss']:.4f}")
    logger.info(f"Test Accuracy: {eval_result['accuracy']*100:.2f}%")
    logger.info("Class Accuracy:")
    for cls, acc in eval_result['class_accuracy'].items():
        logger.info(f"  Class {cls}: {acc*100:.2f}%")

    return manager, history, eval_result


def train_transformer_model(df: pd.DataFrame, save_dir: str = 'models'):
    """Trainiert Transformer-Modell."""
    logger.info("=" * 60)
    logger.info("TRAINING TRANSFORMER MODEL")
    logger.info("=" * 60)

    # Feature-Spalten
    feature_cols = [c for c in df.columns if c.endswith('_norm')]
    features = df[feature_cols].values

    # Labels
    labels = create_trend_labels(df['close'].values, threshold=0.005)

    # Sequenzen (längere für Transformer)
    seq_length = 100
    X, y = create_sequences(features[:-1], labels, seq_length)

    logger.info(f"Created {len(X)} sequences with length {seq_length}")

    # Split
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    # Model Manager
    manager = ModelManager(device='cpu')

    # Transformer erstellen
    model = manager.create_transformer(
        'transformer_trader',
        input_size=X.shape[2],
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        num_classes=3,
        dropout=0.1,
        learning_rate=0.0001
    )

    # Training
    save_path = os.path.join(save_dir, 'transformer_best.pt')

    history = manager.train_model(
        'transformer_trader',
        X_train, y_train,
        X_val, y_val,
        epochs=30,
        batch_size=32,
        early_stopping_patience=8,
        save_path=save_path
    )

    # Evaluation
    eval_result = manager.evaluate_model('transformer_trader', X_test, y_test)

    logger.info("\n" + "=" * 40)
    logger.info("TRANSFORMER EVALUATION RESULTS")
    logger.info("=" * 40)
    logger.info(f"Test Loss: {eval_result['loss']:.4f}")
    logger.info(f"Test Accuracy: {eval_result['accuracy']*100:.2f}%")

    return manager, history, eval_result


def train_rl_agent(df: pd.DataFrame, save_dir: str = 'models', timesteps: int = 100000):
    """Trainiert RL-Agent mit SAC."""
    logger.info("=" * 60)
    logger.info("TRAINING RL AGENT (SAC)")
    logger.info("=" * 60)

    # Feature-Spalten
    feature_cols = [c for c in df.columns if c.endswith('_norm')]

    logger.info(f"Using {len(feature_cols)} features for RL")

    # Train/Test Split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)

    logger.info(f"Train data: {len(train_df)} rows")
    logger.info(f"Test data: {len(test_df)} rows")

    # Training Environment
    train_env = AdvancedTradingEnv(
        df=train_df,
        feature_columns=feature_cols,
        initial_balance=10000.0,
        fee_rate=0.001,
        slippage_rate=0.0005,
        observation_window=60,
        reward_scaling=1.0,
        sharpe_window=20,
        max_position=1.0
    )

    # SAC Agent erstellen
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

    agent = create_sac_agent(
        train_env,
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=256,
        verbose=1
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(save_dir, 'checkpoints_sac'),
        name_prefix='sac'
    )

    # Training
    logger.info(f"Training SAC for {timesteps} timesteps...")

    agent.learn(
        total_timesteps=timesteps,
        callback=checkpoint_callback,
        progress_bar=False
    )

    # Save final model
    save_path = os.path.join(save_dir, 'sac_final')
    agent.save(save_path)
    logger.info(f"Model saved to {save_path}")

    # Evaluation
    logger.info("\nEvaluating on test data...")

    test_env = AdvancedTradingEnv(
        df=test_df,
        feature_columns=feature_cols,
        initial_balance=10000.0,
        fee_rate=0.001,
        slippage_rate=0.0005,
        observation_window=60
    )

    # Run evaluation episodes
    n_eval_episodes = 1  # One full pass through test data
    total_rewards = []
    final_values = []

    for ep in range(n_eval_episodes):
        obs, info = test_env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
        final_values.append(info['portfolio_value'])

    # Results
    avg_reward = np.mean(total_rewards)
    avg_final_value = np.mean(final_values)
    avg_return = (avg_final_value / 10000 - 1) * 100

    logger.info("\n" + "=" * 40)
    logger.info("RL AGENT EVALUATION RESULTS")
    logger.info("=" * 40)
    logger.info(f"Average Reward: {avg_reward:.2f}")
    logger.info(f"Final Portfolio Value: ${avg_final_value:,.2f}")
    logger.info(f"Return: {avg_return:+.2f}%")
    logger.info(f"Number of Trades: {info['num_trades']}")

    return agent, {'avg_reward': avg_reward, 'avg_return': avg_return, 'final_value': avg_final_value}


def main():
    """Hauptfunktion für Training."""
    logger.info("=" * 60)
    logger.info("CRYPTO TRADING BOT - MODEL TRAINING")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Verzeichnisse erstellen
    os.makedirs('models', exist_ok=True)

    # Daten laden
    df, preprocessor = load_and_prepare_data('BTCUSDT', days=365)

    logger.info(f"\nTotal samples: {len(df)}")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

    # 1. LSTM Training
    lstm_manager, lstm_history, lstm_eval = train_lstm_model(df, 'models')

    # 2. Transformer Training
    transformer_manager, transformer_history, transformer_eval = train_transformer_model(df, 'models')

    # 3. RL Agent Training
    rl_agent, rl_eval = train_rl_agent(df, 'models', timesteps=50000)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)

    logger.info("\nLSTM Model:")
    logger.info(f"  Test Accuracy: {lstm_eval['accuracy']*100:.2f}%")

    logger.info("\nTransformer Model:")
    logger.info(f"  Test Accuracy: {transformer_eval['accuracy']*100:.2f}%")

    logger.info("\nRL Agent (SAC):")
    logger.info(f"  Test Return: {rl_eval['avg_return']:+.2f}%")
    logger.info(f"  Final Value: ${rl_eval['final_value']:,.2f}")

    logger.info("\n" + "=" * 60)
    logger.info("Training completed!")
    logger.info(f"Models saved to: ./models/")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
