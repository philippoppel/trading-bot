"""
Test Script für Live Trading auf Binance TESTNET.

Dieses Script testet die Live-Trading-Funktionalität sicher auf Testnet.
KEIN echtes Geld wird verwendet!
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.trading.live_trader import LiveBinanceTrader


def test_testnet_connection():
    """Test Testnet Connection und Basic Functions."""

    # API Keys aus Environment Variables
    api_key = os.getenv('BINANCE_TESTNET_API_KEY')
    api_secret = os.getenv('BINANCE_TESTNET_API_SECRET')

    if not api_key or not api_secret:
        logger.error("❌ Testnet API Keys nicht gefunden!")
        logger.info("Setze diese Environment Variables:")
        logger.info("  export BINANCE_TESTNET_API_KEY='dein_key'")
        logger.info("  export BINANCE_TESTNET_API_SECRET='dein_secret'")
        logger.info("")
        logger.info("Oder füge sie in .env.testnet ein")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("🧪 TESTNET LIVE TRADING TEST")
    logger.info("=" * 80)

    # Initialize trader
    trader = LiveBinanceTrader(
        api_key=api_key,
        api_secret=api_secret,
        testnet=True  # WICHTIG: Testnet mode!
    )

    # Test 1: Get Account Balance
    logger.info("\n📊 Test 1: Account Balance")
    logger.info("-" * 80)
    try:
        usdt_balance = trader.get_account_balance('USDT')
        btc_balance = trader.get_account_balance('BTC')

        logger.info(f"✅ USDT Balance: ${usdt_balance:,.2f}")
        logger.info(f"✅ BTC Balance:  {btc_balance:.8f} BTC")

        if usdt_balance < 10:
            logger.warning("⚠️  USDT Balance zu niedrig für Tests!")
            logger.info("Gehe zu https://testnet.binance.vision/ und hole dir Test-Guthaben")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Balance Check failed: {e}")
        sys.exit(1)

    # Test 2: Get Current Price
    logger.info("\n💰 Test 2: Current Prices")
    logger.info("-" * 80)
    try:
        btc_price = trader.get_current_price('BTCUSDT')
        eth_price = trader.get_current_price('ETHUSDT')

        logger.info(f"✅ BTC Price: ${btc_price:,.2f}")
        logger.info(f"✅ ETH Price: ${eth_price:,.2f}")

    except Exception as e:
        logger.error(f"❌ Price Check failed: {e}")
        sys.exit(1)

    # Test 3: Symbol Info & Rounding
    logger.info("\n🔍 Test 3: Symbol Info & Quantity Rounding")
    logger.info("-" * 80)
    try:
        symbol_info = trader.get_symbol_info('BTCUSDT')
        logger.info(f"✅ Symbol: {symbol_info['symbol']}")
        logger.info(f"   Status: {symbol_info['status']}")
        logger.info(f"   Base Asset: {symbol_info['baseAsset']}")
        logger.info(f"   Quote Asset: {symbol_info['quoteAsset']}")

        # Test quantity rounding
        test_qty = 0.123456789
        rounded = trader.round_quantity('BTCUSDT', test_qty)
        logger.info(f"✅ Quantity rounding: {test_qty} → {rounded}")

    except Exception as e:
        logger.error(f"❌ Symbol Info failed: {e}")
        sys.exit(1)

    # Test 4: Small Buy Order (OPTIONAL - uncomment to test)
    logger.info("\n🛒 Test 4: Execute Small BUY Order (OPTIONAL)")
    logger.info("-" * 80)
    logger.info("⚠️  Dieser Test führt einen ECHTEN Trade auf Testnet aus!")
    logger.info("   (Kein echtes Geld, aber echter Trade-Prozess)")

    response = input("\nMöchtest du einen Test-Trade ausführen? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        try:
            # Small buy: $15 worth of BTC
            test_amount = 15.0  # USDT

            logger.info(f"🔨 Buying ${test_amount} worth of BTC...")

            order = trader.execute_market_buy(
                symbol='BTCUSDT',
                quote_order_qty=test_amount
            )

            logger.info("✅ ORDER SUCCESSFUL!")
            logger.info(f"   Order ID: {order['orderId']}")
            logger.info(f"   Status: {order['status']}")
            logger.info(f"   Executed Qty: {order.get('executedQty', 'N/A')}")

            # Check new balance
            new_btc_balance = trader.get_account_balance('BTC')
            new_usdt_balance = trader.get_account_balance('USDT')

            logger.info(f"\n📊 New Balances:")
            logger.info(f"   BTC:  {new_btc_balance:.8f} BTC")
            logger.info(f"   USDT: ${new_usdt_balance:,.2f}")

        except Exception as e:
            logger.error(f"❌ Buy Order failed: {e}")
            logger.error("Check your Testnet balance and try again")
    else:
        logger.info("⏭️  Skipped trade test")

    # Test 5: Get Open Orders
    logger.info("\n📋 Test 5: Open Orders")
    logger.info("-" * 80)
    try:
        open_orders = trader.get_open_orders()
        if open_orders:
            logger.info(f"✅ Found {len(open_orders)} open orders")
            for order in open_orders:
                logger.info(f"   - {order['symbol']}: {order['side']} {order['type']}")
        else:
            logger.info("✅ No open orders")

    except Exception as e:
        logger.error(f"❌ Open Orders check failed: {e}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ TESTNET TESTS COMPLETED!")
    logger.info("=" * 80)
    logger.info("\nNächste Schritte:")
    logger.info("1. Alle Tests erfolgreich → Integriere Live Trading in deinen Bot")
    logger.info("2. Teste ausgiebig auf Testnet (Tage/Wochen)")
    logger.info("3. Erst dann: Erwäge Production (mit SEHR kleinen Beträgen!)")
    logger.info("")
    logger.info("⚠️  NIEMALS direkt zu Production ohne ausgiebige Testnet-Tests!")


if __name__ == '__main__':
    test_testnet_connection()
