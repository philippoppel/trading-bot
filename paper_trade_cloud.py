#!/usr/bin/env python3
"""
Cloud Trading Bot - Runs continuously and uploads state to Vercel

This version is optimized for cloud deployment (Railway.app).
It automatically uploads trading state to Vercel Blob storage
so the dashboard can display it in real-time.
"""

import os
import sys
import time
import requests
import json
from pathlib import Path
from loguru import logger
from paper_trade_safe import SafeMultiSymbolTrader

# Configuration from environment
VERCEL_DASHBOARD_URL = os.getenv('VERCEL_DASHBOARD_URL')
UPLOAD_API_KEY = os.getenv('UPLOAD_API_KEY')
TRADING_INTERVAL = int(os.getenv('TRADING_INTERVAL', '3600'))  # 1 hour default
UPLOAD_INTERVAL = int(os.getenv('UPLOAD_INTERVAL', '30'))  # 30 seconds

# Validate environment
if not VERCEL_DASHBOARD_URL:
    logger.error("VERCEL_DASHBOARD_URL environment variable not set!")
    sys.exit(1)

if not UPLOAD_API_KEY or UPLOAD_API_KEY == 'your-secret-key-here':
    logger.error("UPLOAD_API_KEY environment variable not set properly!")
    sys.exit(1)


def upload_state_to_vercel(state_file: Path) -> bool:
    """Upload trading state to Vercel Blob storage."""
    try:
        if not state_file.exists():
            logger.warning(f"State file not found: {state_file}")
            return False

        with open(state_file, 'r') as f:
            state_data = json.load(f)

        response = requests.post(
            f"{VERCEL_DASHBOARD_URL}/api/upload",
            json=state_data,
            headers={
                'X-API-Key': UPLOAD_API_KEY,
                'Content-Type': 'application/json'
            },
            timeout=30
        )

        if response.status_code == 200:
            logger.success("✅ State uploaded to Vercel")
            return True
        else:
            logger.error(f"❌ Upload failed: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        return False


def main():
    """Main trading loop."""
    logger.info("=" * 80)
    logger.info("🚀 CLOUD TRADING BOT STARTING")
    logger.info("=" * 80)
    logger.info(f"Dashboard URL: {VERCEL_DASHBOARD_URL}")
    logger.info(f"Trading Interval: {TRADING_INTERVAL}s")
    logger.info(f"Upload Interval: {UPLOAD_INTERVAL}s")
    logger.info("=" * 80)

    # Initialize trader
    try:
        trader = SafeMultiSymbolTrader(
            config_path='models/multi_symbol_config.json',
            initial_balance=10000.0,
            max_position_size=0.3,
            auto_save=True,
            save_interval=30
        )
        logger.success("✅ Trading bot initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize trader: {e}")
        sys.exit(1)

    # Trading loop
    last_trade_time = 0
    last_upload_time = 0
    iteration = 0

    logger.info("\n🔄 Starting trading loop...\n")

    while True:
        try:
            current_time = time.time()
            iteration += 1

            logger.info(f"--- Iteration {iteration} ---")

            # Execute trades
            if current_time - last_trade_time >= TRADING_INTERVAL:
                for symbol in trader.symbols:
                    try:
                        result = trader.trade_symbol(symbol)
                        trader_state = trader.traders[symbol]
                        portfolio_value = trader.get_portfolio_value(symbol)

                        logger.info(
                            f"{symbol}: "
                            f"Pos={trader_state['position']:.2f} | "
                            f"Price=${trader_state['current_price']:,.2f} | "
                            f"Portfolio=${portfolio_value:,.2f}"
                        )
                    except Exception as e:
                        logger.error(f"Error trading {symbol}: {e}")

                last_trade_time = current_time
                trader.save_state(force=True)

            # Upload state to Vercel
            if current_time - last_upload_time >= UPLOAD_INTERVAL:
                upload_state_to_vercel(trader.state_file)
                last_upload_time = current_time

            # Check for emergency stop
            if trader.emergency_stopped:
                logger.error(f"🚨 EMERGENCY STOP: {trader.emergency_reason}")
                # Upload final state
                upload_state_to_vercel(trader.state_file)
                break

            # Sleep until next action
            sleep_time = min(
                TRADING_INTERVAL - (time.time() - last_trade_time),
                UPLOAD_INTERVAL - (time.time() - last_upload_time)
            )

            if sleep_time > 0:
                logger.debug(f"Sleeping for {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("\n⚠️  Shutting down gracefully...")
            trader.save_state(force=True)
            upload_state_to_vercel(trader.state_file)
            break

        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}")
            time.sleep(60)  # Wait 1 minute before retry

    logger.info("\n✅ Trading bot stopped")


if __name__ == '__main__':
    main()
