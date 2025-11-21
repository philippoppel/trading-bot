"""
Lokaler Test-Server für das Trading Dashboard.
Ersetzt Vercel Blob für lokale Tests.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# Path to the state file (same as in paper_trade_safe.py)
STATE_FILE = Path('data/trading_state/safe_multi_symbol_state.json')


def load_state():
    """Load the current trading state from file."""
    if not STATE_FILE.exists():
        return None

    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading state: {e}")
        return None


def calculate_metrics(state):
    """Calculate trading metrics from state."""
    if not state or 'traders' not in state:
        return None

    traders = state['traders']
    initial_balance = state.get('initial_balance', 10000)
    symbols = list(traders.keys())

    total_value = 0
    total_fees = 0
    total_trades = 0
    returns = []

    for symbol in symbols:
        trader = traders[symbol]
        portfolio_value = trader['balance'] + trader['position_value']
        total_value += portfolio_value
        total_fees += trader['total_fees']

        # Use trade_history length if available, otherwise num_trades
        trade_count = len(trader.get('trade_history', []))
        total_trades += trade_count

        ret = (portfolio_value / initial_balance - 1) * 100
        returns.append(ret)

    total_initial = initial_balance * len(symbols)
    total_return = (total_value / total_initial - 1) * 100

    avg_return = sum(returns) / len(returns) if returns else 0

    # Calculate std deviation
    if len(returns) > 1:
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5
    else:
        std_dev = 0

    sharpe = avg_return / std_dev if std_dev > 0 else 0

    # Calculate drawdown
    max_values = [traders[s].get('highest_value', traders[s]['balance'] + traders[s]['position_value'])
                  for s in symbols]
    current_values = [traders[s]['balance'] + traders[s]['position_value'] for s in symbols]
    total_max = max(sum(max_values), 1e-6)
    total_current = sum(current_values)
    drawdown = (total_current / total_max - 1) * 100

    return {
        'totalValue': total_value,
        'totalReturn': total_return,
        'totalFees': total_fees,
        'totalTrades': total_trades,
        'avgReturn': avg_return,
        'sharpe': sharpe,
        'drawdown': drawdown,
        'symbolCount': len(symbols)
    }


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get the current trading state with metrics."""
    state = load_state()

    if not state:
        return jsonify({'error': 'State not found. Start the trading bot first.'}), 404

    metrics = calculate_metrics(state)

    return jsonify({
        **state,
        'metrics': metrics
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get trade history from all traders."""
    state = load_state()

    if not state or 'traders' not in state:
        return jsonify({'error': 'State not found'}), 404

    traders = state['traders']
    all_trades = []

    # Extract trade history from all traders
    for symbol, trader in traders.items():
        trade_history = trader.get('trade_history', [])
        for trade in trade_history:
            all_trades.append({
                **trade,
                'symbol': symbol  # Ensure symbol is set
            })

    # Sort by timestamp (newest first)
    all_trades.sort(key=lambda t: t['timestamp'], reverse=True)

    # Optional: filter by symbol
    symbol_filter = request.args.get('symbol')
    if symbol_filter:
        all_trades = [t for t in all_trades if t['symbol'] == symbol_filter]

    return jsonify({
        'trades': all_trades,
        'total_count': len(all_trades),
        'last_updated': state.get('timestamp', datetime.now().isoformat())
    })


@app.route('/api/upload', methods=['POST'])
def upload_state():
    """Receive state updates from the trading bot (for compatibility)."""
    try:
        state_data = request.get_json()

        if not state_data or not isinstance(state_data, dict):
            return jsonify({'error': 'Invalid state data'}), 400

        # Save to file (this is done by the trading bot, but we accept it anyway)
        print(f"[{datetime.now().isoformat()}] Received state upload from bot")

        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in upload: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    state = load_state()
    return jsonify({
        'status': 'ok',
        'state_file_exists': STATE_FILE.exists(),
        'has_state': state is not None,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Local Trading Dashboard Server")
    print("=" * 60)
    print(f"State file: {STATE_FILE.absolute()}")
    print(f"State exists: {STATE_FILE.exists()}")
    print()
    print("Endpoints:")
    print("  - GET  http://localhost:5001/api/state")
    print("  - GET  http://localhost:5001/api/history")
    print("  - POST http://localhost:5001/api/upload")
    print("  - GET  http://localhost:5001/health")
    print()
    print("Frontend should proxy to: http://localhost:5001")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5001, debug=True)
