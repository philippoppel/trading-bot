"""
Detaillierte Analyse einer Trading Session
Zeigt umfassende Statistiken und Performance Metrics
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

def analyze_trading_session(state_file='data/trading_state/safe_multi_symbol_state.json'):
    """Analysiert eine Trading Session im Detail."""

    state_path = Path(state_file)
    if not state_path.exists():
        print(f"❌ State file not found: {state_file}")
        print("Run the bot first to create a state file.")
        return

    with open(state_path, 'r') as f:
        state = json.load(f)

    # Header
    print("\n" + "=" * 80)
    print("📊 TRADING SESSION ANALYSIS")
    print("=" * 80)

    # Session Info
    print(f"\n🕐 SESSION INFORMATION:")
    print(f"   Started:           {state['session_start_time']}")
    print(f"   Last Update:       {state['timestamp']}")

    # Calculate session duration
    start = datetime.fromisoformat(state['session_start_time'])
    end = datetime.fromisoformat(state['timestamp'])
    duration = end - start
    hours = duration.total_seconds() / 3600

    print(f"   Duration:          {duration} ({hours:.1f} hours)")
    print(f"   Initial Balance:   ${state['initial_balance']:,.2f} per symbol")
    print(f"   Emergency Stop:    {state['emergency_stopped']}")
    if state['emergency_reason']:
        print(f"   Stop Reason:       {state['emergency_reason']}")

    # Portfolio Analysis
    traders = state['traders']
    initial_balance = state['initial_balance']

    print(f"\n💼 PORTFOLIO OVERVIEW:")
    print(f"   Symbols Traded:    {len(traders)}")

    total_value = 0
    total_initial = initial_balance * len(traders)
    total_fees = 0
    total_trades = 0

    returns = []

    for symbol, trader in traders.items():
        portfolio_value = trader['balance'] + trader['position_value']
        total_value += portfolio_value
        total_fees += trader['total_fees']
        total_trades += trader['num_trades']

        ret = (portfolio_value / initial_balance - 1) * 100
        returns.append(ret)

    total_return = (total_value / total_initial - 1) * 100

    print(f"   Total Portfolio:   ${total_value:,.2f}")
    print(f"   Total Return:      {total_return:+.2f}%")
    print(f"   Total Trades:      {total_trades}")
    print(f"   Total Fees:        ${total_fees:,.2f}")

    # Performance Metrics
    print(f"\n📈 PERFORMANCE METRICS:")

    avg_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = (avg_return / std_return) if std_return > 0 else 0

    print(f"   Average Return:    {avg_return:+.2f}%")
    print(f"   Best Symbol:       {max(zip(traders.keys(), returns), key=lambda x: x[1])[0]} ({max(returns):+.2f}%)")
    print(f"   Worst Symbol:      {min(zip(traders.keys(), returns), key=lambda x: x[1])[0]} ({min(returns):+.2f}%)")
    print(f"   Return StdDev:     {std_return:.2f}%")
    print(f"   Sharpe Ratio:      {sharpe:.2f}")

    # Cost Analysis
    print(f"\n💰 COST ANALYSIS:")
    print(f"   Total Fees:        ${total_fees:,.2f}")
    print(f"   Fee as % Capital:  {(total_fees / total_initial) * 100:.3f}%")
    print(f"   Avg Fee/Trade:     ${total_fees / max(total_trades, 1):,.2f}")
    print(f"   Trades/Hour:       {total_trades / max(hours, 0.01):.1f}")

    # Return without fees
    return_no_fees = ((total_value + total_fees) / total_initial - 1) * 100
    fee_impact = return_no_fees - total_return

    print(f"   Return w/o Fees:   {return_no_fees:+.2f}%")
    print(f"   Fee Impact:        {fee_impact:.2f}%")

    # Per-Symbol Breakdown
    print(f"\n📊 PER-SYMBOL BREAKDOWN:")
    print(f"{'Symbol':<10} {'Pos':<8} {'Entry':<12} {'Current':<12} {'Portfolio':<14} {'Return':<10} {'Trades':<8} {'Fees':<10}")
    print("-" * 90)

    for symbol, trader in sorted(traders.items()):
        portfolio_value = trader['balance'] + trader['position_value']
        ret = (portfolio_value / initial_balance - 1) * 100

        pos_str = f"{trader['position']:+.2f}"
        entry_str = f"${trader['entry_price']:,.2f}" if trader['entry_price'] > 0 else "N/A"
        current_str = f"${trader['current_price']:,.2f}"
        portfolio_str = f"${portfolio_value:,.2f}"
        ret_str = f"{ret:+.2f}%"
        trades_str = str(trader['num_trades'])
        fees_str = f"${trader['total_fees']:,.2f}"

        print(f"{symbol:<10} {pos_str:<8} {entry_str:<12} {current_str:<12} {portfolio_str:<14} {ret_str:<10} {trades_str:<8} {fees_str:<10}")

    # Trading Activity
    print(f"\n🔄 TRADING ACTIVITY:")

    most_active = max(traders.items(), key=lambda x: x[1]['num_trades'])
    least_active = min(traders.items(), key=lambda x: x[1]['num_trades'])

    print(f"   Most Active:       {most_active[0]} ({most_active[1]['num_trades']} trades)")
    print(f"   Least Active:      {least_active[0]} ({least_active[1]['num_trades']} trades)")

    # Current positions
    positions = [(s, t['position']) for s, t in traders.items() if abs(t['position']) > 0.01]
    if positions:
        print(f"\n📍 CURRENT POSITIONS:")
        for symbol, pos in positions:
            pos_type = "LONG" if pos > 0 else "SHORT"
            print(f"   {symbol:<10} {pos_type:<6} {abs(pos):.2f}")
    else:
        print(f"\n📍 CURRENT POSITIONS: All flat")

    # Risk Analysis
    print(f"\n⚠️  RISK ANALYSIS:")

    # Max drawdown calculation
    max_values = [t['highest_value'] for t in traders.values()]
    current_values = [t['balance'] + t['position_value'] for t in traders.values()]

    total_max = sum(max_values)
    total_current = sum(current_values)
    drawdown = (total_current / total_max - 1) * 100

    print(f"   Current Drawdown:  {drawdown:+.2f}%")
    print(f"   Peak Value:        ${total_max:,.2f}")

    # Symbols at loss
    symbols_at_loss = [s for s, r in zip(traders.keys(), returns) if r < 0]
    if symbols_at_loss:
        print(f"   Symbols at Loss:   {len(symbols_at_loss)}/{len(traders)} ({', '.join(symbols_at_loss)})")
    else:
        print(f"   Symbols at Loss:   0/{len(traders)} ✅")

    # ROI Analysis
    if hours > 0:
        print(f"\n📅 TIME-BASED METRICS:")
        hourly_return = total_return / hours
        daily_return = hourly_return * 24
        weekly_return = daily_return * 7
        monthly_return = daily_return * 30
        yearly_return = daily_return * 365

        print(f"   Return/Hour:       {hourly_return:+.3f}%")
        print(f"   Projected Daily:   {daily_return:+.2f}%")
        print(f"   Projected Weekly:  {weekly_return:+.2f}%")
        print(f"   Projected Monthly: {monthly_return:+.2f}%")
        print(f"   Projected Yearly:  {yearly_return:+.2f}%")
        print(f"   ⚠️  Note: Projections assume constant performance (unrealistic)")

    print("\n" + "=" * 80)

    # Recommendations
    print(f"\n💡 OBSERVATIONS:")

    if total_return < 0:
        print(f"   ⚠️  Portfolio is in loss ({total_return:.2f}%)")
    elif total_return < 1:
        print(f"   ℹ️  Small gains ({total_return:.2f}%) - consider longer timeframe")
    else:
        print(f"   ✅ Portfolio is profitable ({total_return:.2f}%)")

    if total_fees / total_initial > 0.01:
        print(f"   ⚠️  High fee ratio ({(total_fees/total_initial)*100:.2f}%) - reduce trading frequency")

    if total_trades / max(hours, 1) > 5:
        print(f"   ⚠️  High trading frequency ({total_trades/max(hours,1):.1f} trades/hour)")

    if abs(drawdown) > 10:
        print(f"   ⚠️  Significant drawdown ({drawdown:.1f}%) - monitor risk limits")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        state_file = sys.argv[1]
        analyze_trading_session(state_file)
    else:
        analyze_trading_session()
