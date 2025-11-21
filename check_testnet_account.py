"""
Zeige deine Testnet Account Details - Balance, Trades, Orders.
"""

import os
import sys
from pathlib import Path
from loguru import logger
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.trading.live_trader import LiveBinanceTrader


def show_account_details():
    """Zeige alle Account Details."""

    # API Keys
    api_key = os.getenv('BINANCE_TESTNET_API_KEY')
    api_secret = os.getenv('BINANCE_TESTNET_API_SECRET')

    if not api_key or not api_secret:
        logger.error("Testnet API Keys nicht gefunden!")
        sys.exit(1)

    # Initialize trader
    trader = LiveBinanceTrader(
        api_key=api_key,
        api_secret=api_secret,
        testnet=True
    )

    print("\n" + "=" * 80)
    print("📊 TESTNET ACCOUNT OVERVIEW")
    print("=" * 80)

    # Account Balance
    print("\n💰 ACCOUNT BALANCE")
    print("-" * 80)

    try:
        account = trader.client.get_account()

        print(f"Account Type: {account.get('accountType', 'N/A')}")
        print(f"Can Trade: {account.get('canTrade', False)}")
        print(f"Can Withdraw: {account.get('canWithdraw', False)}")
        print(f"Can Deposit: {account.get('canDeposit', False)}")
        print()

        # Show all non-zero balances
        print("Assets with Balance:")
        for balance in account['balances']:
            free = float(balance['free'])
            locked = float(balance['locked'])
            total = free + locked

            if total > 0:
                print(f"  {balance['asset']:<8} Free: {free:>15.8f}  Locked: {locked:>15.8f}  Total: {total:>15.8f}")

    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        return

    # Recent Trades
    print("\n\n📈 RECENT TRADES (Last 10)")
    print("-" * 80)

    try:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

        all_trades = []
        for symbol in symbols:
            try:
                trades = trader.client.get_my_trades(symbol=symbol, limit=10)
                for trade in trades:
                    trade['symbol'] = symbol
                    all_trades.append(trade)
            except:
                pass

        # Sort by time
        all_trades.sort(key=lambda x: x['time'], reverse=True)

        if all_trades:
            print(f"{'Time':<20} {'Symbol':<10} {'Side':<6} {'Price':<15} {'Qty':<15} {'Commission':<12}")
            print("-" * 80)

            for trade in all_trades[:10]:
                time_str = datetime.fromtimestamp(trade['time']/1000).strftime('%Y-%m-%d %H:%M:%S')
                side = "BUY" if trade['isBuyer'] else "SELL"
                price = float(trade['price'])
                qty = float(trade['qty'])
                commission = float(trade['commission'])

                print(f"{time_str:<20} {trade['symbol']:<10} {side:<6} ${price:>13,.2f} {qty:>14.8f} {commission:>11.8f}")
        else:
            print("No trades yet")

    except Exception as e:
        logger.error(f"Error getting trades: {e}")

    # Order History
    print("\n\n📋 ORDER HISTORY (Last 10)")
    print("-" * 80)

    try:
        all_orders = []
        for symbol in symbols:
            try:
                orders = trader.client.get_all_orders(symbol=symbol, limit=10)
                all_orders.extend(orders)
            except:
                pass

        # Sort by time
        all_orders.sort(key=lambda x: x['time'], reverse=True)

        if all_orders:
            print(f"{'Time':<20} {'Symbol':<10} {'Side':<6} {'Type':<8} {'Status':<10} {'Executed':<15}")
            print("-" * 80)

            for order in all_orders[:10]:
                time_str = datetime.fromtimestamp(order['time']/1000).strftime('%Y-%m-%d %H:%M:%S')

                print(f"{time_str:<20} {order['symbol']:<10} {order['side']:<6} {order['type']:<8} {order['status']:<10} {order['executedQty']:<15}")
        else:
            print("No orders yet")

    except Exception as e:
        logger.error(f"Error getting orders: {e}")

    # Open Orders
    print("\n\n🔓 OPEN ORDERS")
    print("-" * 80)

    try:
        open_orders = trader.get_open_orders()

        if open_orders:
            for order in open_orders:
                print(f"  {order['symbol']}: {order['side']} {order['type']} - {order['origQty']} @ {order['price']}")
        else:
            print("No open orders")

    except Exception as e:
        logger.error(f"Error getting open orders: {e}")

    print("\n" + "=" * 80)
    print("\n✅ Das ist deine Testnet-Aktivität!")
    print("   Die normale Testnet-Website zeigt diese Infos NICHT an.")
    print("   Du musst die API verwenden oder ein externes Tool nutzen.\n")


if __name__ == '__main__':
    show_account_details()
