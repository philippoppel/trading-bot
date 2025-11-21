#!/bin/bash
#
# Sync Trading State to Vercel Dashboard
#
# This script runs in the background and uploads the trading state
# to Vercel every 30 seconds so the dashboard is always up-to-date.
#
# Usage: ./sync_to_vercel.sh

# Configuration
UPLOAD_INTERVAL=30  # Upload every 30 seconds
STATE_FILE="data/trading_state/safe_multi_symbol_state.json"

echo "=================================="
echo "🔄 VERCEL STATE SYNC"
echo "=================================="
echo "Upload interval: ${UPLOAD_INTERVAL}s"
echo "State file: ${STATE_FILE}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Activate virtual environment
source venv/bin/activate

# Run upload loop
while true; do
    if [ -f "${STATE_FILE}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Uploading state..."
        python upload_state_to_vercel.py
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  State file not found: ${STATE_FILE}"
    fi

    sleep ${UPLOAD_INTERVAL}
done
