"""
Fügt Mock-Trade-History zum State hinzu zum Testen des Dashboards.
"""

import json
from datetime import datetime, timedelta
import random
from pathlib import Path

def create_mock_trades():
    """Erstellt realistische Mock-Trades."""

    state_file = Path('data/trading_state/safe_multi_symbol_state.json')

    if not state_file.exists():
        print("❌ State file not found!")
        return

    # Load current state
    with open(state_file, 'r') as f:
        state = json.load(f)

    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']

    # Create mock trades for each symbol
    for symbol in symbols:
        if symbol not in state['traders']:
            continue

        trader = state['traders'][symbol]

        # Create 5-10 mock trades
        num_trades = random.randint(5, 10)

        base_time = datetime.now() - timedelta(hours=2)
        current_price = trader.get('current_price', 100)

        trades = []

        for i in range(num_trades):
            trade_time = base_time + timedelta(minutes=i * 15)

            # Alternate between BUY and SELL
            if i % 2 == 0:
                action_type = random.choice(['BUY', 'BUY', 'BUY', 'STOP_LOSS'])
                old_position = 0.0
                new_position = random.uniform(0.1, 0.3)
            else:
                action_type = random.choice(['SELL', 'SELL', 'SELL', 'TAKE_PROFIT'])
                old_position = random.uniform(0.1, 0.3)
                new_position = 0.0

            # Vary price slightly
            price_variation = random.uniform(-0.02, 0.02)
            price = current_price * (1 + price_variation)

            trade_value = abs(new_position - old_position) * 10000
            fee = trade_value * 0.001

            portfolio_before = 10000 + random.uniform(-500, 500)
            portfolio_after = portfolio_before + (random.uniform(-50, 100) if action_type in ['BUY', 'SELL'] else 0)

            reasoning = {
                'BUY': 'Model decision: Strong bullish signal',
                'SELL': 'Model decision: Bearish signal detected',
                'STOP_LOSS': f'Stop-loss triggered at -12% loss',
                'TAKE_PROFIT': f'Take-profit triggered at +8% gain'
            }[action_type]

            trade = {
                'timestamp': trade_time.isoformat(),
                'symbol': symbol,
                'action_type': action_type,
                'old_position': old_position,
                'new_position': new_position,
                'position_change': new_position - old_position,
                'price': price,
                'trade_value': trade_value,
                'fee': fee,
                'slippage': fee * 0.5,
                'total_cost': fee * 1.5,
                'balance_before': portfolio_before - (new_position - old_position) * 10000,
                'balance_after': portfolio_after - new_position * 10000,
                'portfolio_value_before': portfolio_before,
                'portfolio_value_after': portfolio_after,
                'reasoning': reasoning,
                'model_action': new_position - old_position
            }

            trades.append(trade)

        # Add to trader
        trader['trade_history'] = trades
        trader['num_trades'] = len(trades)

        print(f"✅ Added {len(trades)} mock trades for {symbol}")

    # Save updated state
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"\n✅ Mock trades added to {state_file}")
    print(f"📊 Dashboard should now show trade history!")
    print(f"🔄 Refresh dashboard: https://trading-dashboard-three-virid.vercel.app")

if __name__ == '__main__':
    create_mock_trades()
