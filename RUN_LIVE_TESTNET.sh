#!/bin/bash

# Quick-Start Script für Live Trading auf TESTNET

echo "========================================================================"
echo "🧪 STARTING LIVE TRADING BOT ON TESTNET"
echo "========================================================================"
echo ""
echo "⚠️  This bot will execute REAL trades on Binance Testnet!"
echo "   (Testnet = Virtual money, completely safe)"
echo ""

# Load environment variables
if [ -f .env.testnet ]; then
    echo "📋 Loading Testnet API Keys..."
    export $(cat .env.testnet | xargs)
else
    echo "❌ ERROR: .env.testnet file not found!"
    echo "   Please create it with your Testnet API keys"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if API keys are set
if [ -z "$BINANCE_TESTNET_API_KEY" ] || [ -z "$BINANCE_TESTNET_API_SECRET" ]; then
    echo "❌ ERROR: Testnet API Keys not set!"
    echo "   Check your .env.testnet file"
    exit 1
fi

echo "✅ Environment configured"
echo ""
echo "🚀 Starting Live Trading Bot..."
echo "   - Mode: TESTNET (safe)"
echo "   - Symbols: BTC, ETH, BNB, SOL, XRP"
echo "   - Update Interval: 60s"
echo "   - Dashboard Upload: Enabled"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run the bot
python live_trade_safe.py --testnet

echo ""
echo "========================================================================"
echo "Bot stopped. Final state saved."
echo "========================================================================"
