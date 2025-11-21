"""
Data preprocessing and feature engineering for trading data.
"""

import numpy as np
import pandas as pd
from ta import momentum, trend, volatility, volume

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger()


class DataPreprocessor:
    """
    Preprocessor for OHLCV data with technical indicator feature engineering.

    Transforms raw price data into features suitable for RL training.
    """

    def __init__(self):
        """Initialize the preprocessor with settings."""
        self.settings = get_settings()
        self.feature_config = self.settings.features

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        indicators = self.feature_config.indicators

        # RSI - Relative Strength Index
        rsi_period = indicators.rsi.get("period", 14)
        df["rsi"] = momentum.rsi(df["close"], window=rsi_period)

        # MACD - Moving Average Convergence Divergence
        macd_fast = indicators.macd.get("fast", 12)
        macd_slow = indicators.macd.get("slow", 26)
        macd_signal = indicators.macd.get("signal", 9)
        macd_indicator = trend.MACD(
            df["close"],
            window_fast=macd_fast,
            window_slow=macd_slow,
            window_sign=macd_signal
        )
        df["macd"] = macd_indicator.macd()
        df["macd_signal"] = macd_indicator.macd_signal()
        df["macd_hist"] = macd_indicator.macd_diff()

        # Bollinger Bands
        bb_period = indicators.bollinger.get("period", 20)
        bb_std = indicators.bollinger.get("std", 2)
        bb_indicator = volatility.BollingerBands(
            df["close"],
            window=bb_period,
            window_dev=bb_std
        )
        df["bb_upper"] = bb_indicator.bollinger_hband()
        df["bb_middle"] = bb_indicator.bollinger_mavg()
        df["bb_lower"] = bb_indicator.bollinger_lband()
        df["bb_width"] = bb_indicator.bollinger_wband()

        # ATR - Average True Range
        atr_period = indicators.atr.get("period", 14)
        df["atr"] = volatility.average_true_range(
            df["high"], df["low"], df["close"], window=atr_period
        )

        # Volume SMA
        vol_period = indicators.volume_sma.get("period", 20)
        df["volume_sma"] = df["volume"].rolling(window=vol_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Additional useful indicators
        # EMA - Exponential Moving Averages
        df["ema_9"] = trend.ema_indicator(df["close"], window=9)
        df["ema_21"] = trend.ema_indicator(df["close"], window=21)
        df["ema_50"] = trend.ema_indicator(df["close"], window=50)

        # Stochastic
        stoch_indicator = momentum.StochasticOscillator(
            df["high"], df["low"], df["close"]
        )
        df["stoch_k"] = stoch_indicator.stoch()
        df["stoch_d"] = stoch_indicator.stoch_signal()

        # ADX - Average Directional Index
        adx_indicator = trend.ADXIndicator(df["high"], df["low"], df["close"])
        df["adx"] = adx_indicator.adx()

        # MFI - Money Flow Index
        df["mfi"] = volume.money_flow_index(
            df["high"], df["low"], df["close"], df["volume"], window=14
        )

        # OBV - On Balance Volume
        df["obv"] = volume.on_balance_volume(df["close"], df["volume"])
        df["obv_sma"] = df["obv"].rolling(window=20).mean()
        df["obv_ratio"] = df["obv"] / (df["obv_sma"] + 1e-8)

        # VWAP - Volume Weighted Average Price (approximation)
        df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()

        # CCI - Commodity Channel Index
        df["cci"] = trend.cci(df["high"], df["low"], df["close"], window=20)

        # Williams %R
        df["williams_r"] = momentum.williams_r(df["high"], df["low"], df["close"], lbp=14)

        logger.debug(f"Added technical indicators. Columns: {list(df.columns)}")
        return df

    def add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime classification features.

        Args:
            df: DataFrame with OHLCV and indicators

        Returns:
            DataFrame with market regime features
        """
        df = df.copy()

        # Trend Classification based on EMAs
        if "ema_21" in df.columns and "ema_50" in df.columns:
            # Uptrend: EMA21 > EMA50 and price > EMA21
            df["trend_up"] = ((df["ema_21"] > df["ema_50"]) & (df["close"] > df["ema_21"])).astype(int)
            # Downtrend: EMA21 < EMA50 and price < EMA21
            df["trend_down"] = ((df["ema_21"] < df["ema_50"]) & (df["close"] < df["ema_21"])).astype(int)
            # Sideways: neither
            df["trend_sideways"] = ((df["trend_up"] == 0) & (df["trend_down"] == 0)).astype(int)

        # Trend Strength based on ADX
        if "adx" in df.columns:
            df["strong_trend"] = (df["adx"] > 25).astype(int)
            df["weak_trend"] = (df["adx"] < 20).astype(int)

        # Volatility Regime
        if "volatility_14" in df.columns:
            vol_median = df["volatility_14"].rolling(window=100).median()
            vol_std = df["volatility_14"].rolling(window=100).std()

            df["high_volatility"] = (df["volatility_14"] > vol_median + vol_std).astype(int)
            df["low_volatility"] = (df["volatility_14"] < vol_median - vol_std).astype(int)
            df["normal_volatility"] = ((df["high_volatility"] == 0) & (df["low_volatility"] == 0)).astype(int)

        # Volume Regime
        if "volume_ratio" in df.columns:
            df["high_volume"] = (df["volume_ratio"] > 1.5).astype(int)
            df["low_volume"] = (df["volume_ratio"] < 0.5).astype(int)

        # Momentum Regime based on RSI
        if "rsi" in df.columns:
            df["overbought"] = (df["rsi"] > 70).astype(int)
            df["oversold"] = (df["rsi"] < 30).astype(int)
            df["neutral_rsi"] = ((df["rsi"] >= 30) & (df["rsi"] <= 70)).astype(int)

        # Bollinger Band Position
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            bb_range = df["bb_upper"] - df["bb_lower"]
            df["bb_position"] = (df["close"] - df["bb_lower"]) / (bb_range + 1e-8)
            df["above_bb"] = (df["close"] > df["bb_upper"]).astype(int)
            df["below_bb"] = (df["close"] < df["bb_lower"]).astype(int)

        # Combined Regime Score
        # Higher score = more bullish conditions
        regime_score = 0
        if "trend_up" in df.columns:
            regime_score += df["trend_up"]
        if "strong_trend" in df.columns:
            regime_score += df["strong_trend"] * df.get("trend_up", 0)
        if "oversold" in df.columns:
            regime_score += df["oversold"]  # Oversold can be bullish reversal
        if "high_volume" in df.columns:
            regime_score += df["high_volume"] * df.get("trend_up", 0)

        df["regime_score"] = regime_score

        logger.debug("Added market regime features")
        return df

    def add_zscore_features(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Add Z-score features for mean reversion signals.

        Args:
            df: DataFrame with price data
            window: Rolling window for Z-score calculation

        Returns:
            DataFrame with Z-score features
        """
        df = df.copy()

        # Price Z-score
        price_mean = df["close"].rolling(window=window).mean()
        price_std = df["close"].rolling(window=window).std()
        df["price_zscore"] = (df["close"] - price_mean) / (price_std + 1e-8)

        # Volume Z-score
        vol_mean = df["volume"].rolling(window=window).mean()
        vol_std = df["volume"].rolling(window=window).std()
        df["volume_zscore"] = (df["volume"] - vol_mean) / (vol_std + 1e-8)

        # RSI Z-score
        if "rsi" in df.columns:
            rsi_mean = df["rsi"].rolling(window=window).mean()
            rsi_std = df["rsi"].rolling(window=window).std()
            df["rsi_zscore"] = (df["rsi"] - rsi_mean) / (rsi_std + 1e-8)

        # Returns Z-score
        if "returns" in df.columns:
            ret_mean = df["returns"].rolling(window=window).mean()
            ret_std = df["returns"].rolling(window=window).std()
            df["returns_zscore"] = (df["returns"] - ret_mean) / (ret_std + 1e-8)

        logger.debug("Added Z-score features")
        return df

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features like returns and volatility.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added price features
        """
        df = df.copy()

        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Rolling returns for different periods
        for period in self.feature_config.lookback_periods:
            df[f"returns_{period}"] = df["close"].pct_change(periods=period)

        # Volatility (rolling standard deviation of returns)
        df["volatility_7"] = df["returns"].rolling(window=7).std()
        df["volatility_14"] = df["returns"].rolling(window=14).std()
        df["volatility_30"] = df["returns"].rolling(window=30).std()

        # Price momentum
        df["momentum_7"] = df["close"] - df["close"].shift(7)
        df["momentum_14"] = df["close"] - df["close"].shift(14)

        # Price relative to recent high/low
        df["high_14"] = df["high"].rolling(window=14).max()
        df["low_14"] = df["low"].rolling(window=14).min()
        df["price_position"] = (df["close"] - df["low_14"]) / (df["high_14"] - df["low_14"] + 1e-8)

        # Candle features
        df["body"] = df["close"] - df["open"]
        df["body_pct"] = df["body"] / df["open"]
        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

        logger.debug("Added price features")
        return df

    def normalize_features(self, df: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
        """
        Normalize features for RL training.

        Args:
            df: DataFrame with features
            method: Normalization method ("zscore", "minmax", "robust")

        Returns:
            DataFrame with normalized features
        """
        df = df.copy()

        # Columns to normalize (exclude non-numeric and target columns)
        exclude_cols = ["symbol", "close_timestamp"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [c for c in numeric_cols if c not in exclude_cols]

        if method == "zscore":
            # Z-score normalization
            for col in cols_to_normalize:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[f"{col}_norm"] = (df[col] - mean) / std
                else:
                    df[f"{col}_norm"] = 0.0

        elif method == "minmax":
            # Min-max normalization to [0, 1]
            for col in cols_to_normalize:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[f"{col}_norm"] = 0.5

        elif method == "robust":
            # Robust scaling using median and IQR
            for col in cols_to_normalize:
                median = df[col].median()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    df[f"{col}_norm"] = (df[col] - median) / iqr
                else:
                    df[f"{col}_norm"] = 0.0

        logger.debug(f"Normalized features using {method} method")
        return df

    def process(
        self,
        df: pd.DataFrame,
        normalize: bool = True,
        normalize_method: str = "zscore"
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline.

        Args:
            df: Raw OHLCV DataFrame
            normalize: Whether to normalize features
            normalize_method: Normalization method to use

        Returns:
            Processed DataFrame with all features
        """
        logger.info(f"Processing {len(df)} rows of data")

        # Add all features
        df = self.add_technical_indicators(df)
        df = self.add_price_features(df)
        df = self.add_market_regime_features(df)
        df = self.add_zscore_features(df)

        if normalize:
            df = self.normalize_features(df, method=normalize_method)

        # Drop rows with NaN values (from rolling calculations)
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)

        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with NaN values")

        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        return df

    def get_feature_columns(self, df: pd.DataFrame, normalized_only: bool = True) -> list[str]:
        """
        Get list of feature columns for RL training.

        Args:
            df: Processed DataFrame
            normalized_only: If True, only return normalized columns

        Returns:
            List of feature column names
        """
        if normalized_only:
            return [c for c in df.columns if c.endswith("_norm")]
        else:
            # Return all numeric columns except raw price data
            exclude = ["open", "high", "low", "close", "volume", "quote_volume",
                      "trades", "close_timestamp", "symbol"]
            return [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in exclude]

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Processed DataFrame
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            (test_ratio = 1 - train_ratio - val_ratio)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df


def prepare_training_data(
    symbols: list[str] | None = None,
    interval: str = "1h",
    days: int | None = None
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Convenience function to prepare training data for multiple symbols.

    Args:
        symbols: List of symbols to process
        interval: Kline interval
        days: Days of historical data

    Returns:
        Dictionary mapping symbol to (train, val, test) DataFrames
    """
    from src.data.binance_client import BinanceDataClient

    settings = get_settings()
    if symbols is None:
        symbols = settings.trading.symbols
    if days is None:
        days = settings.data.history_days

    client = BinanceDataClient()
    preprocessor = DataPreprocessor()

    results = {}
    for symbol in symbols:
        logger.info(f"Preparing data for {symbol}")

        # Get raw data
        df = client.get_data(symbol, interval, days)

        # Process data
        df = preprocessor.process(df)

        # Split data
        train, val, test = preprocessor.split_data(df)

        results[symbol] = (train, val, test)

    return results
