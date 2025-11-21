"""
Entfernt Mock-Trades aus dem State und behält echte Bot-Daten.
"""

import json
from pathlib import Path

def clear_mock_trades():
    """Clear mock trades from state."""

    state_file = Path('data/trading_state/safe_multi_symbol_state.json')

    if not state_file.exists():
        print("❌ State file not found!")
        return

    # Load current state
    with open(state_file, 'r') as f:
        state = json.load(f)

    print("🧹 Clearing mock trades from state...")

    # Clear trade_history for each symbol but keep real trading data
    for symbol in state.get('traders', {}).keys():
        trader = state['traders'][symbol]

        # Keep all the real trading metrics
        # Just clear the trade_history array
        old_count = len(trader.get('trade_history', []))
        trader['trade_history'] = []

        print(f"   {symbol}: Removed {old_count} mock trades")
        print(f"            Real data preserved:")
        print(f"              - Balance: ${trader.get('balance', 0):.2f}")
        print(f"              - Position: {trader.get('position', 0):.2f}")
        print(f"              - Fees: ${trader.get('total_fees', 0):.2f}")

    # Save cleaned state
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    print("\n✅ Mock trades cleared!")
    print("   Real trading metrics preserved")
    print("   Ready for bot restart with new code")

if __name__ == '__main__':
    clear_mock_trades()
