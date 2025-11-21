"""
Unit Tests für Data Module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestDataPreprocessor:
    """Tests für DataPreprocessor."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Erstellt Sample OHLCV-Daten."""
        np.random.seed(42)
        n = 500

        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_ = close + np.random.randn(n) * 0.2

        df = pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 5000, n)
        })

        return df

    def test_add_technical_indicators(self, sample_ohlcv):
        """Test technische Indikatoren."""
        from src.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        df = preprocessor.add_technical_indicators(sample_ohlcv)

        # Prüfe wichtige Indikatoren
        assert 'rsi' in df.columns
        assert 'macd' in df.columns
        assert 'bb_upper' in df.columns
        assert 'adx' in df.columns
        assert 'mfi' in df.columns
        assert 'obv' in df.columns
        assert 'cci' in df.columns
        assert 'williams_r' in df.columns

    def test_add_price_features(self, sample_ohlcv):
        """Test Preis-Features."""
        from src.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        df = preprocessor.add_price_features(sample_ohlcv)

        assert 'returns' in df.columns
        assert 'log_returns' in df.columns
        assert 'volatility_7' in df.columns
        assert 'momentum_7' in df.columns
        assert 'price_position' in df.columns

    def test_add_market_regime_features(self, sample_ohlcv):
        """Test Marktregime-Features."""
        from src.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        df = preprocessor.add_technical_indicators(sample_ohlcv)
        df = preprocessor.add_price_features(df)
        df = preprocessor.add_market_regime_features(df)

        assert 'trend_up' in df.columns
        assert 'trend_down' in df.columns
        assert 'high_volatility' in df.columns
        assert 'overbought' in df.columns
        assert 'oversold' in df.columns
        assert 'regime_score' in df.columns

    def test_add_zscore_features(self, sample_ohlcv):
        """Test Z-Score Features."""
        from src.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        df = preprocessor.add_technical_indicators(sample_ohlcv)
        df = preprocessor.add_price_features(df)
        df = preprocessor.add_zscore_features(df)

        assert 'price_zscore' in df.columns
        assert 'volume_zscore' in df.columns
        assert 'rsi_zscore' in df.columns

    def test_normalize_features(self, sample_ohlcv):
        """Test Normalisierung."""
        from src.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        df = preprocessor.add_technical_indicators(sample_ohlcv)
        df = preprocessor.normalize_features(df, method='zscore')

        # Prüfe normalisierte Spalten
        norm_cols = [c for c in df.columns if c.endswith('_norm')]
        assert len(norm_cols) > 0

    def test_full_process_pipeline(self, sample_ohlcv):
        """Test vollständige Pipeline."""
        from src.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        df = preprocessor.process(sample_ohlcv)

        # Sollte keine NaN-Werte haben
        assert df.isna().sum().sum() == 0

        # Sollte viele Features haben
        assert len(df.columns) > 50


class TestWebSocketClient:
    """Tests für WebSocket Client."""

    def test_websocket_creation(self):
        """Test WebSocket-Erstellung."""
        from src.data.websocket_client import BinanceWebSocket

        ws = BinanceWebSocket(testnet=True)
        assert ws is not None
        assert ws.running == False

    def test_callback_registration(self):
        """Test Callback-Registrierung."""
        from src.data.websocket_client import BinanceWebSocket

        ws = BinanceWebSocket()

        callback_called = [False]

        def on_kline(symbol, data):
            callback_called[0] = True

        ws.on_kline(on_kline)

        assert len(ws.callbacks['kline']) == 1

    def test_realtime_data_manager(self):
        """Test RealtimeDataManager."""
        from src.data.websocket_client import RealtimeDataManager

        manager = RealtimeDataManager(['BTCUSDT', 'ETHUSDT'], ['1m', '5m'])

        assert len(manager.symbols) == 2
        assert len(manager.intervals) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
