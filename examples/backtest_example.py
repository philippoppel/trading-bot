"""
Beispiel-Backtest mit verschiedenen Strategien.

Demonstriert:
- LSTM-basierte Strategie
- Einfache Moving Average Strategie
- Walk-Forward-Optimierung
"""

import sys
from pathlib import Path

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from src.data.binance_client import BinanceDataClient
from src.data.preprocessor import DataPreprocessor
from src.backtesting.engine import BacktestEngine, BacktestConfig, Order, OrderType, Side
from src.backtesting.metrics import BacktestMetrics
from src.backtesting.walk_forward import WalkForwardOptimizer
from src.models.model_manager import ModelManager
from src.models.lstm_model import create_sequences, create_trend_labels


def load_data(symbol: str = 'BTCUSDT', days: int = 365) -> pd.DataFrame:
    """Lädt historische Daten."""
    client = BinanceDataClient()
    df = client.get_historical_klines(
        symbol=symbol,
        interval='1h',
        days=days
    )
    logger.info(f"Loaded {len(df)} rows for {symbol}")
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bereitet Features vor."""
    preprocessor = DataPreprocessor()
    df_features = preprocessor.add_technical_indicators(df)
    df_features = preprocessor.add_price_features(df_features)
    df_features = df_features.dropna()
    logger.info(f"Prepared {len(df_features.columns)} features")
    return df_features


# ====== STRATEGIEN ======

def ma_crossover_strategy(params: dict):
    """
    Moving Average Crossover Strategie.

    Args:
        params: {'fast_period': int, 'slow_period': int}
    """
    fast = params['fast_period']
    slow = params['slow_period']

    def strategy(data, idx, position):
        if idx < slow:
            return None

        # Berechne MAs
        fast_ma = data['close'].iloc[idx-fast:idx].mean()
        slow_ma = data['close'].iloc[idx-slow:idx].mean()

        price = data['close'].iloc[idx]

        # Crossover
        prev_fast = data['close'].iloc[idx-fast-1:idx-1].mean()
        prev_slow = data['close'].iloc[idx-slow-1:idx-1].mean()

        # Buy Signal: Fast kreuzt über Slow
        if prev_fast <= prev_slow and fast_ma > slow_ma and not position:
            return Order(
                order_type=OrderType.MARKET,
                side=Side.LONG,
                price=None,
                size=float('inf'),
                stop_loss=price * 0.95,
                take_profit=price * 1.10
            )

        # Sell Signal: Fast kreuzt unter Slow
        elif prev_fast >= prev_slow and fast_ma < slow_ma and position:
            return Order(
                order_type=OrderType.MARKET,
                side=Side.SHORT,
                price=None,
                size=float('inf')
            )

        return None

    return strategy


def rsi_strategy(params: dict):
    """
    RSI Overbought/Oversold Strategie.

    Args:
        params: {'oversold': int, 'overbought': int}
    """
    oversold = params['oversold']
    overbought = params['overbought']

    def strategy(data, idx, position):
        if 'rsi' not in data.columns or idx < 14:
            return None

        rsi = data['rsi'].iloc[idx]
        price = data['close'].iloc[idx]

        # Buy wenn überverkauft
        if rsi < oversold and not position:
            return Order(
                order_type=OrderType.MARKET,
                side=Side.LONG,
                price=None,
                size=float('inf'),
                stop_loss=price * 0.95,
                take_profit=price * 1.10
            )

        # Sell wenn überkauft
        elif rsi > overbought and position:
            return Order(
                order_type=OrderType.MARKET,
                side=Side.SHORT,
                price=None,
                size=float('inf')
            )

        return None

    return strategy


def run_simple_backtest():
    """Führt einfachen Backtest durch."""
    logger.info("=" * 50)
    logger.info("SIMPLE BACKTEST")
    logger.info("=" * 50)

    # Lade Daten
    df = load_data('BTCUSDT', days=180)
    df = prepare_features(df)

    # Backtest Config
    config = BacktestConfig(
        initial_capital=10000,
        fee_rate=0.001,
        slippage_rate=0.0005,
        max_position_size=0.5
    )

    # Teste MA Crossover
    logger.info("\n--- MA Crossover Strategy ---")

    engine = BacktestEngine(config)
    strategy = ma_crossover_strategy({'fast_period': 10, 'slow_period': 30})
    result = engine.run(df, strategy, verbose=False)

    metrics = BacktestMetrics()
    print(metrics.format_metrics(result['metrics']))

    # Teste RSI Strategy
    logger.info("\n--- RSI Strategy ---")

    engine = BacktestEngine(config)
    strategy = rsi_strategy({'oversold': 30, 'overbought': 70})
    result = engine.run(df, strategy, verbose=False)

    print(metrics.format_metrics(result['metrics']))

    return result


def run_walk_forward_optimization():
    """Führt Walk-Forward-Optimierung durch."""
    logger.info("=" * 50)
    logger.info("WALK-FORWARD OPTIMIZATION")
    logger.info("=" * 50)

    # Lade Daten
    df = load_data('BTCUSDT', days=365)
    df = prepare_features(df)

    # Parameter Grid
    param_grid = {
        'fast_period': [5, 10, 15],
        'slow_period': [20, 30, 50]
    }

    # Walk-Forward Optimizer
    optimizer = WalkForwardOptimizer(
        train_pct=0.7,
        n_windows=3,
        optimization_metric='sharpe_ratio'
    )

    config = BacktestConfig(
        initial_capital=10000,
        fee_rate=0.001,
        slippage_rate=0.0005
    )

    # Optimiere
    results = optimizer.optimize(
        df,
        ma_crossover_strategy,
        param_grid,
        config
    )

    # Analysiere
    analysis = optimizer.analyze_results(results)
    print(optimizer.format_analysis(analysis))

    return results, analysis


def run_lstm_backtest():
    """Führt Backtest mit LSTM-Modell durch."""
    logger.info("=" * 50)
    logger.info("LSTM MODEL BACKTEST")
    logger.info("=" * 50)

    # Lade und bereite Daten vor
    df = load_data('BTCUSDT', days=365)
    df = prepare_features(df)

    # Feature-Spalten (ohne OHLCV)
    feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
    features = df[feature_cols].values

    # Erstelle Labels
    labels = create_trend_labels(df['close'].values, threshold=0.005)

    # Erstelle Sequenzen
    seq_length = 60
    X, y = create_sequences(features[:-1], labels, seq_length)

    logger.info(f"Created {len(X)} sequences with {X.shape[2]} features")

    # Train/Test Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Erstelle und trainiere Modell
    manager = ModelManager(device='cpu')
    model = manager.create_lstm(
        'lstm_trader',
        input_size=X.shape[2],
        hidden_size=64,
        num_layers=2
    )

    logger.info("Training LSTM model...")
    history = manager.train_model(
        'lstm_trader',
        X_train, y_train,
        X_test, y_test,
        epochs=20,
        batch_size=32,
        early_stopping_patience=5
    )

    # Evaluiere
    eval_result = manager.evaluate_model('lstm_trader', X_test, y_test)
    logger.info(f"Test Accuracy: {eval_result['accuracy']:.4f}")

    # Backtest mit Modell-Signalen
    test_df = df.iloc[seq_length + split_idx:]

    # Generiere Signale
    predictions = manager.predict('lstm_trader', X_test, return_probs=False)

    # Mappe Predictions zu Signalen (0=down->sell, 1=sideways->hold, 2=up->buy)
    test_df = test_df.copy()
    signal_map = {0: -1, 1: 0, 2: 1}
    test_df['signal'] = [signal_map[p] for p in predictions[:len(test_df)]]

    # Backtest
    from src.backtesting.engine import create_simple_strategy

    config = BacktestConfig(
        initial_capital=10000,
        fee_rate=0.001,
        slippage_rate=0.0005
    )

    engine = BacktestEngine(config)
    strategy = create_simple_strategy('signal', stop_loss_pct=0.03, take_profit_pct=0.06)
    result = engine.run(test_df, strategy, verbose=False)

    metrics = BacktestMetrics()
    print(metrics.format_metrics(result['metrics']))

    return result


def main():
    """Hauptfunktion."""
    logger.info("Starting Backtest Examples")
    logger.info("=" * 60)

    # 1. Einfacher Backtest
    run_simple_backtest()

    # 2. Walk-Forward Optimierung
    # run_walk_forward_optimization()

    # 3. LSTM Backtest
    # run_lstm_backtest()

    logger.info("\nBacktest Examples Completed!")


if __name__ == '__main__':
    main()
