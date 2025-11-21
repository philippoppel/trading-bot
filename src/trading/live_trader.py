"""
Live Trading Modul für Binance (Testnet & Production).

ACHTUNG: Dieses Modul führt ECHTE Trades aus!
Immer zuerst mit Testnet testen!
"""

import time
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, Any
from datetime import datetime

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from loguru import logger


class LiveBinanceTrader:
    """
    Führt echte Trades auf Binance aus (Testnet oder Production).

    WICHTIG:
    - Starte IMMER mit Testnet!
    - Teste alle Funktionen ausgiebig
    - Verstehe die Risiken
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        min_notional: float = 10.0  # Minimum trade size in USDT
    ):
        """
        Initialize Live Trader.

        Args:
            api_key: Binance API Key
            api_secret: Binance API Secret
            testnet: If True, use Testnet. If False, use REAL MONEY!
            min_notional: Minimum trade value in USDT
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.min_notional = min_notional

        # Initialize client
        if testnet:
            self.client = Client(
                api_key=api_key,
                api_secret=api_secret,
                testnet=True
            )
            # Testnet base URL
            self.client.API_URL = 'https://testnet.binance.vision/api'
            logger.info("🧪 Initialized TESTNET trader (safe mode)")
        else:
            self.client = Client(
                api_key=api_key,
                api_secret=api_secret
            )
            logger.warning("⚠️ Initialized PRODUCTION trader - REAL MONEY AT RISK!")

        # Cache for exchange info
        self.exchange_info_cache: Dict[str, Dict] = {}

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get trading rules for a symbol.

        Returns lot size, price filters, min notional, etc.
        """
        if symbol in self.exchange_info_cache:
            return self.exchange_info_cache[symbol]

        try:
            exchange_info = self.client.get_exchange_info()

            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    self.exchange_info_cache[symbol] = s
                    return s

            raise ValueError(f"Symbol {symbol} not found on exchange")

        except BinanceAPIException as e:
            logger.error(f"Error getting symbol info: {e}")
            raise

    def round_quantity(self, symbol: str, quantity: float) -> str:
        """
        Round quantity to match symbol's LOT_SIZE filter.

        Binance requires specific precision for each symbol.
        """
        symbol_info = self.get_symbol_info(symbol)

        # Find LOT_SIZE filter
        lot_size_filter = None
        for f in symbol_info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                lot_size_filter = f
                break

        if not lot_size_filter:
            raise ValueError(f"No LOT_SIZE filter found for {symbol}")

        # Round to step size
        step_size = Decimal(lot_size_filter['stepSize'])
        min_qty = Decimal(lot_size_filter['minQty'])

        # Convert to Decimal for precision
        qty_decimal = Decimal(str(quantity))

        # Round down to nearest step
        precision = abs(step_size.as_tuple().exponent)
        rounded = qty_decimal.quantize(step_size, rounding=ROUND_DOWN)

        # Check minimum
        if rounded < min_qty:
            logger.warning(f"Quantity {rounded} below minimum {min_qty}")
            return "0"

        return str(rounded)

    def get_account_balance(self, asset: str = 'USDT') -> float:
        """Get current balance for an asset."""
        try:
            account = self.client.get_account()

            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])

            return 0.0

        except BinanceAPIException as e:
            logger.error(f"Error getting balance: {e}")
            raise

    def get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            raise

    def execute_market_buy(
        self,
        symbol: str,
        quantity: Optional[float] = None,
        quote_order_qty: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a market BUY order.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            quantity: Amount of base asset to buy (e.g., BTC amount)
            quote_order_qty: Amount of quote asset to spend (e.g., USDT amount)

        Returns:
            Order response from Binance

        Note: Specify either quantity OR quote_order_qty, not both
        """
        try:
            # Check minimum notional
            if quote_order_qty and quote_order_qty < self.min_notional:
                raise ValueError(
                    f"Order value ${quote_order_qty} below minimum ${self.min_notional}"
                )

            # Build order parameters
            order_params = {
                'symbol': symbol,
                'side': 'BUY',
                'type': 'MARKET'
            }

            if quantity is not None:
                # Round quantity
                rounded_qty = self.round_quantity(symbol, quantity)
                if rounded_qty == "0":
                    raise ValueError(f"Quantity too small after rounding")
                order_params['quantity'] = rounded_qty
            elif quote_order_qty is not None:
                order_params['quoteOrderQty'] = quote_order_qty
            else:
                raise ValueError("Must specify either quantity or quote_order_qty")

            # Execute order
            logger.info(f"📈 Executing BUY: {symbol} {order_params}")
            order = self.client.create_order(**order_params)

            logger.info(f"✅ BUY order executed: {order['orderId']}")
            logger.info(f"   Filled: {order.get('executedQty', 'N/A')} @ avg ${order.get('fills', [{}])[0].get('price', 'N/A') if order.get('fills') else 'N/A'}")

            return order

        except BinanceAPIException as e:
            logger.error(f"❌ BUY order failed: {e}")
            raise
        except BinanceOrderException as e:
            logger.error(f"❌ BUY order rejected: {e}")
            raise

    def execute_market_sell(
        self,
        symbol: str,
        quantity: float
    ) -> Dict[str, Any]:
        """
        Execute a market SELL order.

        Args:
            symbol: Trading pair
            quantity: Amount of base asset to sell

        Returns:
            Order response from Binance
        """
        try:
            # Round quantity
            rounded_qty = self.round_quantity(symbol, quantity)
            if rounded_qty == "0":
                raise ValueError(f"Quantity too small after rounding")

            # Execute order
            order_params = {
                'symbol': symbol,
                'side': 'SELL',
                'type': 'MARKET',
                'quantity': rounded_qty
            }

            logger.info(f"📉 Executing SELL: {symbol} {order_params}")
            order = self.client.create_order(**order_params)

            logger.info(f"✅ SELL order executed: {order['orderId']}")
            logger.info(f"   Filled: {order.get('executedQty', 'N/A')} @ avg ${order.get('fills', [{}])[0].get('price', 'N/A') if order.get('fills') else 'N/A'}")

            return order

        except BinanceAPIException as e:
            logger.error(f"❌ SELL order failed: {e}")
            raise
        except BinanceOrderException as e:
            logger.error(f"❌ SELL order rejected: {e}")
            raise

    def get_asset_balance(self, symbol: str) -> float:
        """
        Get balance of the base asset for a symbol.

        E.g., for BTCUSDT returns BTC balance
        """
        # Extract base asset (BTC from BTCUSDT)
        symbol_info = self.get_symbol_info(symbol)
        base_asset = symbol_info['baseAsset']

        return self.get_account_balance(base_asset)

    def close_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Close all holdings for a symbol by selling everything.

        Returns:
            Order response if position existed, None if no position
        """
        balance = self.get_asset_balance(symbol)

        if balance > 0:
            logger.info(f"Closing position: selling {balance} of {symbol}")
            return self.execute_market_sell(symbol, balance)
        else:
            logger.info(f"No position to close for {symbol}")
            return None

    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """Get all open orders (or for specific symbol)."""
        try:
            if symbol:
                return self.client.get_open_orders(symbol=symbol)
            else:
                return self.client.get_open_orders()
        except BinanceAPIException as e:
            logger.error(f"Error getting open orders: {e}")
            raise

    def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all open orders."""
        try:
            open_orders = self.get_open_orders(symbol)

            for order in open_orders:
                self.client.cancel_order(
                    symbol=order['symbol'],
                    orderId=order['orderId']
                )
                logger.info(f"Cancelled order {order['orderId']} for {order['symbol']}")

        except BinanceAPIException as e:
            logger.error(f"Error cancelling orders: {e}")
            raise
