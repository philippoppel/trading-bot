"""
Quick test script to verify state persistence functionality.
"""

from pathlib import Path
import json

# Check if state file exists
state_file = Path('data/trading_state/safe_multi_symbol_state.json')

if state_file.exists():
    print("=" * 60)
    print("✅ STATE FILE FOUND")
    print("=" * 60)

    with open(state_file, 'r') as f:
        state = json.load(f)

    print(f"\n📂 Saved State Information:")
    print(f"   Version:           {state.get('version')}")
    print(f"   Last Saved:        {state.get('timestamp')}")
    print(f"   Session Started:   {state.get('session_start_time')}")
    print(f"   Initial Balance:   ${state.get('initial_balance'):,.2f}")
    print(f"   Emergency Stop:    {state.get('emergency_stopped')}")

    print(f"\n💼 Trader States:")
    for symbol, trader in state.get('traders', {}).items():
        print(f"\n   {symbol}:")
        print(f"      Balance:        ${trader['balance']:,.2f}")
        print(f"      Position:       {trader['position']:.2f}")
        print(f"      Entry Price:    ${trader['entry_price']:,.2f}")
        print(f"      Total Trades:   {trader['num_trades']}")
        print(f"      Total Fees:     ${trader['total_fees']:.2f}")
        print(f"      Current Price:  ${trader['current_price']:,.2f}")

        portfolio = trader['balance'] + trader['position_value']
        initial = state.get('initial_balance')
        ret = (portfolio / initial - 1) * 100
        print(f"      Portfolio:      ${portfolio:,.2f} ({ret:+.2f}%)")

    # Check backups
    backup_dir = state_file.parent / 'backups'
    if backup_dir.exists():
        backups = list(backup_dir.glob('*.json'))
        print(f"\n📦 Backups: {len(backups)} files found")
        for backup in sorted(backups)[-5:]:  # Show last 5
            print(f"      {backup.name}")

    print("\n" + "=" * 60)
    print("✨ State can be restored on next run!")
    print("=" * 60)

else:
    print("=" * 60)
    print("❌ NO STATE FILE FOUND")
    print("=" * 60)
    print(f"\nExpected location: {state_file}")
    print("\nRun the trading bot first to create a state file.")
    print("=" * 60)
