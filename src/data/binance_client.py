"""
Binance API client for fetching historical and live market data.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger()


class BinanceDataClient:
    """
    Client for fetching cryptocurrency data from Binance.

    Handles both historical data downloads and live data streaming.
    """

    # Binance kline interval constants
    INTERVALS = {
        "1m": Client.KLINE_INTERVAL_1MINUTE,
        "5m": Client.KLINE_INTERVAL_5MINUTE,
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
    }

    def __init__(self):
        """Initialize the Binance client with API credentials."""
        self.settings = get_settings()

        # Initialize client - use public API for data fetching (no keys needed)
        # Only use testnet/keys for actual trading
        self.client = Client(
            api_key=self.settings.binance_api_key or "",
            api_secret=self.settings.binance_api_secret or ""
        )
        logger.info("Initialized Binance client")

    def get_historical_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int | None = None
    ) -> pd.DataFrame:
        """
        Fetch historical kline (candlestick) data for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Kline interval (e.g., "1h", "4h", "1d")
            start_date: Start date for historical data
            end_date: End date for historical data
            limit: Maximum number of klines to fetch

        Returns:
            DataFrame with OHLCV data
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=self.settings.data.history_days)

        # Convert interval string to Binance constant
        kline_interval = self.INTERVALS.get(interval)
        if kline_interval is None:
            raise ValueError(f"Invalid interval: {interval}. Valid intervals: {list(self.INTERVALS.keys())}")

        logger.info(f"Fetching {symbol} klines from {start_date} to {end_date}")

        try:
            # Fetch klines from Binance
            kline_args = {
                "symbol": symbol,
                "interval": kline_interval,
                "start_str": start_date.strftime("%d %b %Y %H:%M:%S"),
                "end_str": end_date.strftime("%d %b %Y %H:%M:%S")
            }
            if limit is not None:
                kline_args["limit"] = limit

            klines = self.client.get_historical_klines(**kline_args)

            # Convert to DataFrame
            df = self._klines_to_dataframe(klines, symbol)

            logger.info(f"Fetched {len(df)} klines for {symbol}")
            return df

        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise

    def _klines_to_dataframe(self, klines: list, symbol: str) -> pd.DataFrame:
        """
        Convert Binance klines to a pandas DataFrame.

        Args:
            klines: Raw klines data from Binance API
            symbol: Trading pair symbol

        Returns:
            Formatted DataFrame with OHLCV data
        """
        # Binance kline format:
        # [open_time, open, high, low, close, volume, close_time,
        #  quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]

        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]

        df = pd.DataFrame(klines, columns=columns)

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_timestamp"] = pd.to_datetime(df["close_time"], unit="ms")

        # Convert price/volume columns to float
        numeric_columns = ["open", "high", "low", "close", "volume", "quote_volume"]
        for col in numeric_columns:
            df[col] = df[col].astype(float)

        # Convert trades to int
        df["trades"] = df["trades"].astype(int)

        # Add symbol column
        df["symbol"] = symbol

        # Select and reorder columns
        df = df[[
            "timestamp", "symbol", "open", "high", "low", "close",
            "volume", "quote_volume", "trades", "close_timestamp"
        ]]

        # Set timestamp as index
        df.set_index("timestamp", inplace=True)

        return df

    def get_current_price(self, symbol: str) -> float:
        """
        Get the current price for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current price as float
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except BinanceAPIException as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            raise

    def get_all_prices(self, symbols: list[str] | None = None) -> dict[str, float]:
        """
        Get current prices for multiple symbols.

        Args:
            symbols: List of symbols. If None, uses symbols from config.

        Returns:
            Dictionary mapping symbol to price
        """
        if symbols is None:
            symbols = self.settings.trading.symbols

        prices = {}
        for symbol in symbols:
            prices[symbol] = self.get_current_price(symbol)

        return prices

    def fetch_and_save_data(
        self,
        symbols: list[str] | None = None,
        interval: str = "1h",
        days: int | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols and save to disk.

        Args:
            symbols: List of symbols to fetch
            interval: Kline interval
            days: Number of days of history (uses config default if None)

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        if symbols is None:
            symbols = self.settings.trading.symbols
        if days is None:
            days = self.settings.data.history_days

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        data_dir = self.settings.get_data_dir()
        results = {}

        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}...")

            # Fetch data
            df = self.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )

            # Save to parquet file
            filename = f"{symbol}_{interval}_{days}d.parquet"
            filepath = data_dir / filename
            df.to_parquet(filepath)
            logger.info(f"Saved {symbol} data to {filepath}")

            results[symbol] = df

        return results

    def load_cached_data(
        self,
        symbol: str,
        interval: str = "1h",
        days: int | None = None
    ) -> pd.DataFrame | None:
        """
        Load cached data from disk if available.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            days: Number of days of history

        Returns:
            DataFrame if cache exists, None otherwise
        """
        if days is None:
            days = self.settings.data.history_days

        data_dir = self.settings.get_data_dir()
        filename = f"{symbol}_{interval}_{days}d.parquet"
        filepath = data_dir / filename

        if filepath.exists():
            logger.info(f"Loading cached data from {filepath}")
            return pd.read_parquet(filepath)

        return None

    def get_data(
        self,
        symbol: str,
        interval: str = "1h",
        days: int | None = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get data for a symbol, using cache if available.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            days: Number of days of history
            force_refresh: If True, ignore cache and fetch fresh data

        Returns:
            DataFrame with OHLCV data
        """
        if not force_refresh:
            cached = self.load_cached_data(symbol, interval, days)
            if cached is not None:
                return cached

        # Fetch and cache
        if days is None:
            days = self.settings.data.history_days

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        df = self.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )

        # Save to cache
        data_dir = self.settings.get_data_dir()
        filename = f"{symbol}_{interval}_{days}d.parquet"
        filepath = data_dir / filename
        df.to_parquet(filepath)

        return df

    def get_exchange_info(self, symbol: str) -> dict:
        """
        Get exchange information for a symbol (trading rules, precision, etc.).

        Args:
            symbol: Trading pair symbol

        Returns:
            Exchange info dictionary
        """
        info = self.client.get_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                return s
        raise ValueError(f"Symbol {symbol} not found on exchange")

    def get_account_balance(self) -> dict[str, float]:
        """
        Get account balances (requires API key with read permissions).

        Returns:
            Dictionary mapping asset to free balance
        """
        try:
            account = self.client.get_account()
            balances = {}
            for balance in account["balances"]:
                free = float(balance["free"])
                if free > 0:
                    balances[balance["asset"]] = free
            return balances
        except BinanceAPIException as e:
            logger.error(f"Error fetching account balance: {e}")
            raise
