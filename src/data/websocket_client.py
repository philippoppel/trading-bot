"""
WebSocket Client für Echtzeit-Datenstreaming.

Features:
- Kline/Candlestick Streams
- Trade Streams
- Order Book Streams
- Automatisches Reconnect
- Multi-Symbol Support
"""

import asyncio
import json
import websockets
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
from loguru import logger
import pandas as pd
from collections import defaultdict


class BinanceWebSocket:
    """
    WebSocket Client für Binance Echtzeit-Daten.

    Unterstützt:
    - Kline/Candlestick Streams
    - Aggregated Trade Streams
    - Order Book Depth Streams
    - Ticker Streams
    """

    BASE_URL = "wss://stream.binance.com:9443/ws"
    BASE_URL_TESTNET = "wss://testnet.binance.vision/ws"

    def __init__(self, testnet: bool = False):
        """
        Args:
            testnet: Testnet verwenden
        """
        self.base_url = self.BASE_URL_TESTNET if testnet else self.BASE_URL
        self.ws = None
        self.running = False
        self.reconnect_delay = 5
        self.max_reconnect_attempts = 10

        # Callbacks für verschiedene Stream-Typen
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Daten-Buffer
        self.kline_buffer: Dict[str, pd.DataFrame] = {}
        self.trade_buffer: Dict[str, List[Dict]] = defaultdict(list)
        self.orderbook: Dict[str, Dict] = {}

        logger.info(f"WebSocket client initialized ({'testnet' if testnet else 'mainnet'})")

    def on_kline(self, callback: Callable[[str, Dict], None]):
        """Registriert Callback für Kline-Updates."""
        self.callbacks['kline'].append(callback)

    def on_trade(self, callback: Callable[[str, Dict], None]):
        """Registriert Callback für Trade-Updates."""
        self.callbacks['trade'].append(callback)

    def on_orderbook(self, callback: Callable[[str, Dict], None]):
        """Registriert Callback für Orderbook-Updates."""
        self.callbacks['depth'].append(callback)

    def on_ticker(self, callback: Callable[[str, Dict], None]):
        """Registriert Callback für Ticker-Updates."""
        self.callbacks['ticker'].append(callback)

    async def connect(self, streams: List[str]):
        """
        Verbindet zu WebSocket mit mehreren Streams.

        Args:
            streams: Liste von Stream-Namen (z.B. ['btcusdt@kline_1m', 'ethusdt@trade'])
        """
        if not streams:
            logger.warning("No streams specified")
            return

        # Combined Stream URL
        stream_path = "/".join(streams)
        url = f"{self.base_url}/{stream_path}"

        self.running = True
        reconnect_count = 0

        while self.running and reconnect_count < self.max_reconnect_attempts:
            try:
                logger.info(f"Connecting to WebSocket: {len(streams)} streams")

                async with websockets.connect(url, ping_interval=20) as ws:
                    self.ws = ws
                    reconnect_count = 0
                    logger.info("WebSocket connected")

                    async for message in ws:
                        await self._handle_message(message)

            except websockets.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                reconnect_count += 1

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                reconnect_count += 1

            if self.running:
                delay = self.reconnect_delay * reconnect_count
                logger.info(f"Reconnecting in {delay}s (attempt {reconnect_count})")
                await asyncio.sleep(delay)

        logger.info("WebSocket client stopped")

    async def _handle_message(self, message: str):
        """Verarbeitet eingehende WebSocket-Nachricht."""
        try:
            data = json.loads(message)

            # Combined Stream Format
            if 'stream' in data:
                stream = data['stream']
                payload = data['data']
            else:
                # Single Stream Format
                stream = data.get('e', 'unknown')
                payload = data

            # Dispatch basierend auf Event-Typ
            event_type = payload.get('e', '')

            if event_type == 'kline':
                await self._process_kline(payload)

            elif event_type == 'aggTrade':
                await self._process_trade(payload)

            elif event_type == 'depthUpdate':
                await self._process_depth(payload)

            elif event_type == '24hrTicker':
                await self._process_ticker(payload)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _process_kline(self, data: Dict):
        """Verarbeitet Kline-Daten."""
        symbol = data['s']
        kline = data['k']

        kline_data = {
            'timestamp': pd.to_datetime(kline['t'], unit='ms'),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'closed': kline['x']
        }

        # Callbacks aufrufen
        for callback in self.callbacks['kline']:
            try:
                callback(symbol, kline_data)
            except Exception as e:
                logger.error(f"Kline callback error: {e}")

        # In Buffer speichern wenn Kerze geschlossen
        if kline_data['closed']:
            if symbol not in self.kline_buffer:
                self.kline_buffer[symbol] = pd.DataFrame()

            new_row = pd.DataFrame([kline_data])
            self.kline_buffer[symbol] = pd.concat(
                [self.kline_buffer[symbol], new_row],
                ignore_index=True
            ).tail(1000)  # Behalte nur letzte 1000

    async def _process_trade(self, data: Dict):
        """Verarbeitet Trade-Daten."""
        symbol = data['s']

        trade_data = {
            'timestamp': pd.to_datetime(data['T'], unit='ms'),
            'price': float(data['p']),
            'quantity': float(data['q']),
            'is_buyer_maker': data['m']
        }

        # Callbacks aufrufen
        for callback in self.callbacks['trade']:
            try:
                callback(symbol, trade_data)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")

        # In Buffer speichern
        self.trade_buffer[symbol].append(trade_data)
        if len(self.trade_buffer[symbol]) > 1000:
            self.trade_buffer[symbol] = self.trade_buffer[symbol][-1000:]

    async def _process_depth(self, data: Dict):
        """Verarbeitet Orderbook-Updates."""
        symbol = data['s']

        depth_data = {
            'timestamp': pd.to_datetime(data['E'], unit='ms'),
            'bids': [[float(p), float(q)] for p, q in data['b']],
            'asks': [[float(p), float(q)] for p, q in data['a']]
        }

        # Callbacks aufrufen
        for callback in self.callbacks['depth']:
            try:
                callback(symbol, depth_data)
            except Exception as e:
                logger.error(f"Depth callback error: {e}")

        # In Orderbook speichern
        self.orderbook[symbol] = depth_data

    async def _process_ticker(self, data: Dict):
        """Verarbeitet Ticker-Daten."""
        symbol = data['s']

        ticker_data = {
            'timestamp': pd.to_datetime(data['E'], unit='ms'),
            'price': float(data['c']),
            'price_change': float(data['p']),
            'price_change_pct': float(data['P']),
            'high': float(data['h']),
            'low': float(data['l']),
            'volume': float(data['v']),
            'quote_volume': float(data['q'])
        }

        # Callbacks aufrufen
        for callback in self.callbacks['ticker']:
            try:
                callback(symbol, ticker_data)
            except Exception as e:
                logger.error(f"Ticker callback error: {e}")

    def stop(self):
        """Stoppt den WebSocket Client."""
        self.running = False
        if self.ws:
            asyncio.create_task(self.ws.close())
        logger.info("WebSocket client stopping")

    def get_klines(self, symbol: str) -> pd.DataFrame:
        """Gibt gepufferte Klines zurück."""
        return self.kline_buffer.get(symbol, pd.DataFrame())

    def get_trades(self, symbol: str) -> List[Dict]:
        """Gibt gepufferte Trades zurück."""
        return self.trade_buffer.get(symbol, [])

    def get_orderbook(self, symbol: str) -> Dict:
        """Gibt aktuelles Orderbook zurück."""
        return self.orderbook.get(symbol, {})


class RealtimeDataManager:
    """
    Manager für Echtzeit-Datenstreaming.

    Verwaltet WebSocket-Verbindungen und stellt
    Daten für Live-Trading bereit.
    """

    def __init__(self, symbols: List[str], intervals: List[str] = ['1m']):
        """
        Args:
            symbols: Liste der Symbole (z.B. ['BTCUSDT', 'ETHUSDT'])
            intervals: Kline-Intervalle (z.B. ['1m', '5m'])
        """
        self.symbols = [s.lower() for s in symbols]
        self.intervals = intervals
        self.ws_client = BinanceWebSocket()

        # Aktuelle Preise
        self.current_prices: Dict[str, float] = {}

        # Registriere Callbacks
        self.ws_client.on_kline(self._on_kline_update)
        self.ws_client.on_ticker(self._on_ticker_update)

    def _on_kline_update(self, symbol: str, data: Dict):
        """Callback für Kline-Updates."""
        self.current_prices[symbol] = data['close']

    def _on_ticker_update(self, symbol: str, data: Dict):
        """Callback für Ticker-Updates."""
        self.current_prices[symbol] = data['price']

    async def start(self):
        """Startet Datenstreaming."""
        streams = []

        for symbol in self.symbols:
            # Kline Streams
            for interval in self.intervals:
                streams.append(f"{symbol}@kline_{interval}")

            # Ticker Stream
            streams.append(f"{symbol}@ticker")

            # Trade Stream (optional)
            # streams.append(f"{symbol}@aggTrade")

        await self.ws_client.connect(streams)

    def stop(self):
        """Stoppt Datenstreaming."""
        self.ws_client.stop()

    def get_price(self, symbol: str) -> float:
        """Gibt aktuellen Preis zurück."""
        return self.current_prices.get(symbol.upper(), 0.0)

    def get_klines(self, symbol: str) -> pd.DataFrame:
        """Gibt gepufferte Klines zurück."""
        return self.ws_client.get_klines(symbol.upper())


async def example_usage():
    """Beispiel für WebSocket-Verwendung."""
    # Erstelle Client
    ws = BinanceWebSocket()

    # Registriere Callbacks
    def on_kline(symbol: str, data: Dict):
        if data['closed']:
            logger.info(f"{symbol} Kline closed: {data['close']}")

    def on_trade(symbol: str, data: Dict):
        logger.info(f"{symbol} Trade: {data['price']} x {data['quantity']}")

    ws.on_kline(on_kline)
    ws.on_trade(on_trade)

    # Starte Streams
    streams = [
        'btcusdt@kline_1m',
        'ethusdt@kline_1m',
        'btcusdt@aggTrade'
    ]

    try:
        await ws.connect(streams)
    except KeyboardInterrupt:
        ws.stop()


if __name__ == '__main__':
    asyncio.run(example_usage())
