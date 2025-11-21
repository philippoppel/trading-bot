#!/bin/bash

# Script to test the trading dashboard locally
# This script will start both the trading bot (in demo mode) and the dashboard frontend

set -e

echo "======================================"
echo "🧪 Local Trading Dashboard Test"
echo "======================================"
echo ""

# Check if we're in the right directory
if [ ! -f "paper_trade_safe.py" ]; then
    echo "Error: Must run from project root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q flask flask-cors

# Check if state file exists
STATE_FILE="data/trading_state/safe_multi_symbol_state.json"
if [ ! -f "$STATE_FILE" ]; then
    echo ""
    echo "⚠️  Warning: No state file found at $STATE_FILE"
    echo "Please run the trading bot first to generate state data:"
    echo "  python3 paper_trade_safe.py --balance 10000 --interval 60"
    echo ""
    read -p "Press Enter to continue anyway (will show error in dashboard)..."
fi

# Start the dashboard frontend in the background
echo ""
echo "Starting Next.js frontend..."
cd trading-dashboard
npm install > /dev/null 2>&1
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait a bit for frontend to start
echo "Waiting for frontend to start..."
sleep 5

echo ""
echo "======================================"
echo "✅ Dashboard is running!"
echo "======================================"
echo ""
echo "Frontend: http://localhost:3000"
echo ""
echo "The dashboard is now reading from:"
echo "  $STATE_FILE"
echo ""
echo "To generate test data, run in another terminal:"
echo "  python3 paper_trade_safe.py --balance 10000 --interval 60"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "======================================"

# Wait for user to stop
trap "echo ''; echo 'Stopping services...'; kill $FRONTEND_PID 2>/dev/null; exit 0" INT

# Keep script running
wait $FRONTEND_PID
