"""
Order Manager für Live-Trading.

Verwaltet Orders mit:
- Retry-Logic
- Cancel-Protection
- Rate-Limiting
"""

import ccxt
import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import asyncio


class OrderStatus(Enum):
    """Order-Status."""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"


@dataclass
class OrderResult:
    """Ergebnis einer Order-Ausführung."""
    order_id: str
    symbol: str
    side: str
    order_type: str
    amount: float
    price: Optional[float]
    filled: float
    remaining: float
    status: OrderStatus
    fee: float
    timestamp: int
    raw: Dict


class OrderManager:
    """
    Verwaltet Order-Ausführung über CCXT.

    Features:
    - Automatische Retries
    - Rate-Limit-Handling
    - Order-Tracking
    - Cancel-Protection
    """

    def __init__(
        self,
        exchange_id: str = 'binance',
        api_key: str = '',
        api_secret: str = '',
        testnet: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Args:
            exchange_id: CCXT Exchange ID
            api_key: API Key
            api_secret: API Secret
            testnet: Testnet verwenden
            max_retries: Maximale Retry-Versuche
            retry_delay: Verzögerung zwischen Retries
        """
        self.exchange_id = exchange_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Exchange initialisieren
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })

        # Testnet konfigurieren
        if testnet and 'test' in self.exchange.urls:
            self.exchange.set_sandbox_mode(True)
            logger.info(f"Using {exchange_id} testnet")
        elif testnet:
            logger.warning(f"{exchange_id} does not support testnet")

        # Order-Tracking
        self.open_orders: Dict[str, OrderResult] = {}
        self.order_history: List[OrderResult] = []

        logger.info(f"OrderManager initialized for {exchange_id}")

    def _execute_with_retry(self, func, *args, **kwargs) -> Any:
        """Führt Funktion mit Retry-Logic aus."""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)

            except ccxt.NetworkError as e:
                last_exception = e
                logger.warning(f"Network error (attempt {attempt + 1}): {e}")
                time.sleep(self.retry_delay * (attempt + 1))

            except ccxt.ExchangeNotAvailable as e:
                last_exception = e
                logger.warning(f"Exchange not available (attempt {attempt + 1}): {e}")
                time.sleep(self.retry_delay * (attempt + 1))

            except ccxt.RateLimitExceeded as e:
                last_exception = e
                logger.warning(f"Rate limit exceeded, waiting...")
                time.sleep(self.retry_delay * 5)

            except ccxt.InvalidOrder as e:
                logger.error(f"Invalid order: {e}")
                raise

            except ccxt.InsufficientFunds as e:
                logger.error(f"Insufficient funds: {e}")
                raise

        raise last_exception

    def _parse_order_result(self, order: Dict) -> OrderResult:
        """Parst CCXT Order zu OrderResult."""
        return OrderResult(
            order_id=order['id'],
            symbol=order['symbol'],
            side=order['side'],
            order_type=order['type'],
            amount=order['amount'],
            price=order.get('price'),
            filled=order.get('filled', 0),
            remaining=order.get('remaining', order['amount']),
            status=OrderStatus(order['status']),
            fee=order.get('fee', {}).get('cost', 0),
            timestamp=order.get('timestamp', int(time.time() * 1000)),
            raw=order
        )

    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float
    ) -> OrderResult:
        """
        Erstellt Market Order.

        Args:
            symbol: Trading-Paar (z.B. 'BTC/USDT')
            side: 'buy' oder 'sell'
            amount: Menge

        Returns:
            OrderResult
        """
        logger.info(f"Creating market {side} order: {amount} {symbol}")

        order = self._execute_with_retry(
            self.exchange.create_market_order,
            symbol, side, amount
        )

        result = self._parse_order_result(order)
        self.order_history.append(result)

        logger.info(f"Market order executed: {result.order_id}, filled: {result.filled}")

        return result

    def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float
    ) -> OrderResult:
        """
        Erstellt Limit Order.

        Args:
            symbol: Trading-Paar
            side: 'buy' oder 'sell'
            amount: Menge
            price: Limit-Preis

        Returns:
            OrderResult
        """
        logger.info(f"Creating limit {side} order: {amount} {symbol} @ {price}")

        order = self._execute_with_retry(
            self.exchange.create_limit_order,
            symbol, side, amount, price
        )

        result = self._parse_order_result(order)
        self.open_orders[result.order_id] = result
        self.order_history.append(result)

        logger.info(f"Limit order created: {result.order_id}")

        return result

    def create_stop_loss_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
        limit_price: Optional[float] = None
    ) -> OrderResult:
        """
        Erstellt Stop-Loss Order.

        Args:
            symbol: Trading-Paar
            side: 'buy' oder 'sell'
            amount: Menge
            stop_price: Trigger-Preis
            limit_price: Optional Limit-Preis (sonst Market)

        Returns:
            OrderResult
        """
        logger.info(f"Creating stop-loss order: {amount} {symbol} @ stop {stop_price}")

        params = {'stopPrice': stop_price}

        if limit_price:
            order = self._execute_with_retry(
                self.exchange.create_order,
                symbol, 'stop_limit', side, amount, limit_price, params
            )
        else:
            order = self._execute_with_retry(
                self.exchange.create_order,
                symbol, 'stop_market', side, amount, None, params
            )

        result = self._parse_order_result(order)
        self.open_orders[result.order_id] = result
        self.order_history.append(result)

        return result

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Storniert eine Order.

        Args:
            order_id: Order ID
            symbol: Trading-Paar

        Returns:
            True wenn erfolgreich
        """
        try:
            self._execute_with_retry(
                self.exchange.cancel_order,
                order_id, symbol
            )

            if order_id in self.open_orders:
                del self.open_orders[order_id]

            logger.info(f"Order {order_id} canceled")
            return True

        except ccxt.OrderNotFound:
            logger.warning(f"Order {order_id} not found")
            return False

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Storniert alle offenen Orders.

        Args:
            symbol: Optional Symbol-Filter

        Returns:
            Anzahl stornierter Orders
        """
        canceled = 0

        try:
            if symbol:
                orders = self._execute_with_retry(
                    self.exchange.cancel_all_orders,
                    symbol
                )
            else:
                # Storniere alle Symbole einzeln
                for oid, order in list(self.open_orders.items()):
                    if self.cancel_order(oid, order.symbol):
                        canceled += 1
                return canceled

            canceled = len(orders) if orders else 0
            self.open_orders.clear()

        except Exception as e:
            logger.error(f"Error canceling orders: {e}")

        logger.info(f"Canceled {canceled} orders")
        return canceled

    def get_order_status(self, order_id: str, symbol: str) -> OrderResult:
        """
        Holt aktuellen Order-Status.

        Args:
            order_id: Order ID
            symbol: Trading-Paar

        Returns:
            OrderResult
        """
        order = self._execute_with_retry(
            self.exchange.fetch_order,
            order_id, symbol
        )

        result = self._parse_order_result(order)

        # Update lokales Tracking
        if result.status in [OrderStatus.CLOSED, OrderStatus.CANCELED]:
            if order_id in self.open_orders:
                del self.open_orders[order_id]
        else:
            self.open_orders[order_id] = result

        return result

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """
        Holt alle offenen Orders.

        Args:
            symbol: Optional Symbol-Filter

        Returns:
            Liste von OrderResult
        """
        orders = self._execute_with_retry(
            self.exchange.fetch_open_orders,
            symbol
        )

        results = [self._parse_order_result(o) for o in orders]

        # Update lokales Tracking
        self.open_orders = {r.order_id: r for r in results}

        return results

    def get_balance(self, currency: Optional[str] = None) -> Dict:
        """
        Holt Kontostand.

        Args:
            currency: Optional Währungsfilter

        Returns:
            Balance Dictionary
        """
        balance = self._execute_with_retry(
            self.exchange.fetch_balance
        )

        if currency:
            return {
                'free': balance.get(currency, {}).get('free', 0),
                'used': balance.get(currency, {}).get('used', 0),
                'total': balance.get(currency, {}).get('total', 0)
            }

        return balance

    def get_ticker(self, symbol: str) -> Dict:
        """
        Holt aktuellen Ticker.

        Args:
            symbol: Trading-Paar

        Returns:
            Ticker Dictionary
        """
        return self._execute_with_retry(
            self.exchange.fetch_ticker,
            symbol
        )

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100
    ) -> List:
        """
        Holt OHLCV Daten.

        Args:
            symbol: Trading-Paar
            timeframe: Zeitrahmen
            limit: Anzahl der Kerzen

        Returns:
            Liste von OHLCV
        """
        return self._execute_with_retry(
            self.exchange.fetch_ohlcv,
            symbol, timeframe, limit=limit
        )

    def close(self):
        """Schließt Exchange-Verbindung."""
        if hasattr(self.exchange, 'close'):
            self.exchange.close()
        logger.info("OrderManager closed")
