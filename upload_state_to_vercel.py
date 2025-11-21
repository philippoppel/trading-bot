#!/usr/bin/env python3
"""
Upload Trading State to Vercel Dashboard

This script uploads the trading state to Vercel Blob storage
so the dashboard can display it in real-time.
"""

import json
import os
import sys
import requests
from pathlib import Path
from loguru import logger

# Configuration
VERCEL_DASHBOARD_URL = os.getenv('VERCEL_DASHBOARD_URL', 'https://trading-dashboard-5oqf34l8u-philipps-projects-0f51423d.vercel.app')
UPLOAD_API_KEY = os.getenv('UPLOAD_API_KEY', 'your-secret-key-here')  # Set this in your environment
STATE_FILE = Path('data/trading_state/safe_multi_symbol_state.json')


def upload_state(state_file: Path = STATE_FILE):
    """Upload trading state to Vercel dashboard."""

    if not state_file.exists():
        logger.error(f"State file not found: {state_file}")
        return False

    try:
        # Read state file
        with open(state_file, 'r') as f:
            state_data = json.load(f)

        logger.info(f"Uploading state to {VERCEL_DASHBOARD_URL}/api/upload")

        # Upload to Vercel
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
            result = response.json()
            logger.success(f"✅ State uploaded successfully!")
            logger.info(f"   Blob URL: {result.get('url', 'N/A')}")
            logger.info(f"   Timestamp: {result.get('timestamp', 'N/A')}")
            return True
        else:
            logger.error(f"❌ Upload failed: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Network error: {e}")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in state file: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("📤 UPLOADING TRADING STATE TO VERCEL")
    logger.info("=" * 60)

    # Check if API key is set
    if UPLOAD_API_KEY == 'your-secret-key-here':
        logger.warning("⚠️  Using default API key. Set UPLOAD_API_KEY environment variable!")

    success = upload_state()

    if success:
        logger.info("\n✅ Upload complete! Check your dashboard:")
        logger.info(f"   {VERCEL_DASHBOARD_URL}")
    else:
        logger.error("\n❌ Upload failed. Check the logs above.")
        sys.exit(1)
