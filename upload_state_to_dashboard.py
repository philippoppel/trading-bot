"""
Upload Trading State zum Dashboard.
"""

import json
import requests
from pathlib import Path
import sys

def upload_state():
    """Upload state to dashboard."""

    state_file = Path('data/trading_state/safe_multi_symbol_state.json')

    if not state_file.exists():
        print(f"❌ State file not found: {state_file}")
        return False

    # Read state
    with open(state_file, 'r') as f:
        state = json.load(f)

    print("📤 Uploading state to dashboard...")
    print(f"   Timestamp: {state.get('timestamp', 'N/A')}")
    print(f"   Symbols: {len(state.get('traders', {}))}")

    # Count total trades
    total_trades = sum(len(t.get('trade_history', [])) for t in state.get('traders', {}).values())
    print(f"   Total Trades in History: {total_trades}")

    # Upload to dashboard
    dashboard_url = "https://trading-dashboard-three-virid.vercel.app/api/upload"

    # Use a simple API key (you can change this in Vercel env vars)
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': 'trading-bot-2024'  # Simple key for testing
    }

    try:
        response = requests.post(
            dashboard_url,
            json=state,
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            print("✅ State uploaded successfully!")
            print(f"   Response: {response.json()}")
            print("")
            print("🔗 View on Dashboard:")
            print("   https://trading-dashboard-three-virid.vercel.app")
            return True
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

if __name__ == '__main__':
    success = upload_state()
    sys.exit(0 if success else 1)
