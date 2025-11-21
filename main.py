#!/usr/bin/env python3
"""
Main entry point for the Crypto Trading Bot.

Usage:
    python main.py fetch      # Fetch and cache market data
    python main.py train      # Train the RL agent
    python main.py evaluate   # Evaluate trained model
    python main.py paper      # Run paper trading
"""

import argparse
import sys
from pathlib import Path

from src.config.settings import get_settings
from src.utils.logger import setup_logger, get_logger


def fetch_data():
    """Fetch and cache historical market data."""
    from src.data.binance_client import BinanceDataClient

    logger = get_logger()
    settings = get_settings()

    logger.info("Fetching market data...")

    client = BinanceDataClient()
    data = client.fetch_and_save_data(
        symbols=settings.trading.symbols,
        interval=settings.trading.timeframe,
        days=settings.data.history_days
    )

    for symbol, df in data.items():
        logger.info(f"{symbol}: {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    logger.info("Data fetch complete!")


def train_model():
    """Train the RL trading agent."""
    from src.data.binance_client import BinanceDataClient
    from src.data.preprocessor import DataPreprocessor
    from src.environment.trading_env import create_env
    from src.agent.rl_agent import TradingAgent

    logger = get_logger()
    settings = get_settings()

    logger.info("Starting training pipeline...")

    # Load and preprocess data
    client = BinanceDataClient()
    preprocessor = DataPreprocessor()

    # For simplicity, train on first symbol
    symbol = settings.trading.symbols[0]
    logger.info(f"Training on {symbol}")

    # Get data
    df = client.get_data(
        symbol=symbol,
        interval=settings.trading.timeframe,
        days=settings.data.history_days
    )

    # Preprocess
    df = preprocessor.process(df)

    # Split data
    train_df, val_df, test_df = preprocessor.split_data(df)

    # Get feature columns
    feature_columns = preprocessor.get_feature_columns(df)
    logger.info(f"Using {len(feature_columns)} features")

    # Create environments
    train_env = create_env(train_df, feature_columns)
    val_env = create_env(val_df, feature_columns)

    # Create and train agent
    agent = TradingAgent(
        algorithm=settings.agent.algorithm,
        policy=settings.agent.policy
    )

    agent.create_model(train_env)
    agent.train(eval_env=val_env)

    # Save final model
    model_path = settings.get_model_dir() / f"{symbol}_final_model"
    agent.save(model_path)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_env = create_env(test_df, feature_columns)
    results = agent.evaluate(test_env)

    logger.info(f"Test Results:")
    logger.info(f"  Return: {results['mean_return']*100:.2f}%")
    logger.info(f"  Sharpe Ratio: {results['mean_sharpe']:.2f}")
    logger.info(f"  Max Drawdown: {results['mean_max_drawdown']*100:.2f}%")
    logger.info(f"  Win Rate: {results['mean_win_rate']*100:.1f}%")

    logger.info(f"Model saved to {model_path}")


def evaluate_model():
    """Evaluate a trained model."""
    from src.data.binance_client import BinanceDataClient
    from src.data.preprocessor import DataPreprocessor
    from src.environment.trading_env import create_env
    from src.agent.rl_agent import TradingAgent

    logger = get_logger()
    settings = get_settings()

    symbol = settings.trading.symbols[0]
    model_path = settings.get_model_dir() / f"{symbol}_final_model.zip"

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Run 'python main.py train' first")
        return

    # Load data
    client = BinanceDataClient()
    preprocessor = DataPreprocessor()

    df = client.get_data(
        symbol=symbol,
        interval=settings.trading.timeframe,
        days=settings.data.history_days
    )

    df = preprocessor.process(df)
    _, _, test_df = preprocessor.split_data(df)
    feature_columns = preprocessor.get_feature_columns(df)

    # Create environment
    test_env = create_env(test_df, feature_columns)

    # Load and evaluate agent
    agent = TradingAgent(algorithm=settings.agent.algorithm)
    agent.load(model_path)

    results = agent.evaluate(test_env, n_episodes=20)

    logger.info(f"\nEvaluation Results (20 episodes):")
    logger.info(f"  Mean Return: {results['mean_return']*100:.2f}%")
    logger.info(f"  Sharpe Ratio: {results['mean_sharpe']:.2f}")
    logger.info(f"  Max Drawdown: {results['mean_max_drawdown']*100:.2f}%")
    logger.info(f"  Win Rate: {results['mean_win_rate']*100:.1f}%")
    logger.info(f"  Avg Trades: {results['mean_trades']:.1f}")


def run_paper_trading():
    """Run live paper trading simulation."""
    import time
    from src.data.binance_client import BinanceDataClient
    from src.data.preprocessor import DataPreprocessor
    from src.environment.trading_env import create_env
    from src.agent.rl_agent import TradingAgent
    from src.trading.paper_trader import PaperTrader

    logger = get_logger()
    settings = get_settings()

    symbol = settings.trading.symbols[0]
    model_path = settings.get_model_dir() / f"{symbol}_final_model.zip"

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Run 'python main.py train' first")
        return

    # Initialize components
    client = BinanceDataClient()
    preprocessor = DataPreprocessor()
    trader = PaperTrader()

    # Load agent
    agent = TradingAgent(algorithm=settings.agent.algorithm)
    agent.load(model_path)

    logger.info(f"Starting paper trading for {symbol}")
    logger.info(f"Initial balance: {trader.balance}")
    logger.info(f"Update interval: {settings.paper_trading.update_interval}s")

    try:
        while True:
            # Fetch recent data
            df = client.get_data(
                symbol=symbol,
                interval=settings.trading.timeframe,
                days=7,  # Last week for observation window
                force_refresh=True
            )

            # Preprocess
            df = preprocessor.process(df)
            feature_columns = preprocessor.get_feature_columns(df)

            # Create environment with latest data
            env = create_env(df.tail(100), feature_columns)
            obs, _ = env.reset()

            # Get agent decision
            action, _ = agent.predict(obs)
            current_price = client.get_current_price(symbol)

            logger.info(f"Price: {current_price:.2f}, Action: {['HOLD', 'BUY', 'SELL'][action]}")

            # Execute trade based on action
            if action == 1:  # BUY
                position_size = trader.balance * settings.paper_trading.risk_management.max_position_pct
                trader.open_position(symbol, position_size, current_price)

            elif action == 2:  # SELL
                if symbol in trader.positions and not trader.positions[symbol].closed:
                    trader.close_position(symbol, current_price)

            # Check risk management
            closed = trader.execute_risk_management()
            if closed:
                logger.info(f"Risk management closed: {closed}")

            # Log status
            stats = trader.get_stats()
            logger.info(
                f"Portfolio: {stats['portfolio_value']:.2f}, "
                f"Return: {stats['total_return']*100:.2f}%, "
                f"Trades: {stats['total_trades']}"
            )

            # Save state
            state_path = settings.get_data_dir() / "paper_trader_state.json"
            trader.save_state(state_path)

            # Wait for next update
            logger.info(f"Waiting {settings.paper_trading.update_interval}s for next update...")
            time.sleep(settings.paper_trading.update_interval)

    except KeyboardInterrupt:
        logger.info("\nPaper trading stopped by user")
        stats = trader.get_stats()
        logger.info(f"\nFinal Stats:")
        logger.info(f"  Portfolio Value: {stats['portfolio_value']:.2f}")
        logger.info(f"  Total Return: {stats['total_return']*100:.2f}%")
        logger.info(f"  Total Trades: {stats['total_trades']}")
        logger.info(f"  Win Rate: {stats['win_rate']*100:.1f}%")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Crypto Trading Bot with RL")
    parser.add_argument(
        "command",
        choices=["fetch", "train", "evaluate", "paper"],
        help="Command to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger()
    logger = get_logger()

    logger.info(f"Running command: {args.command}")

    # Execute command
    if args.command == "fetch":
        fetch_data()
    elif args.command == "train":
        train_model()
    elif args.command == "evaluate":
        evaluate_model()
    elif args.command == "paper":
        run_paper_trading()


if __name__ == "__main__":
    main()
