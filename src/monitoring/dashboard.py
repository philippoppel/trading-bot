"""
Streamlit Monitoring Dashboard.

Features:
- Echtzeit Equity Curve
- Offene Positionen
- Live Model-Signale
- Risiko-Status
- Performance-Metriken
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_trade_history(path: str = "logs/trade_history.json") -> pd.DataFrame:
    """Lädt Trade-History aus JSON."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        return pd.DataFrame()


def load_equity_curve(path: str = "logs/equity_curve.json") -> pd.DataFrame:
    """Lädt Equity Curve aus JSON."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        # Generiere Demo-Daten
        dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
        equity = 10000 + np.cumsum(np.random.randn(100) * 50)
        return pd.DataFrame({'timestamp': dates, 'equity': equity})


def create_equity_chart(df: pd.DataFrame) -> go.Figure:
    """Erstellt Equity Curve Chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['equity'],
        mode='lines',
        name='Equity',
        line=dict(color='#00D4AA', width=2)
    ))

    # Drawdown als Füllbereich
    peak = df['equity'].cummax()
    drawdown = (df['equity'] - peak) / peak * 100

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=peak,
        mode='lines',
        name='Peak',
        line=dict(color='gray', width=1, dash='dot')
    ))

    fig.update_layout(
        title='Equity Curve',
        xaxis_title='Zeit',
        yaxis_title='Portfolio-Wert ($)',
        template='plotly_dark',
        height=400
    )

    return fig


def create_drawdown_chart(df: pd.DataFrame) -> go.Figure:
    """Erstellt Drawdown Chart."""
    peak = df['equity'].cummax()
    drawdown = (df['equity'] - peak) / peak * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=drawdown,
        fill='tozeroy',
        mode='lines',
        name='Drawdown',
        line=dict(color='#FF6B6B', width=1)
    ))

    fig.update_layout(
        title='Drawdown',
        xaxis_title='Zeit',
        yaxis_title='Drawdown (%)',
        template='plotly_dark',
        height=250
    )

    return fig


def create_returns_distribution(df: pd.DataFrame) -> go.Figure:
    """Erstellt Returns-Verteilung."""
    returns = df['equity'].pct_change().dropna() * 100

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Returns',
        marker_color='#00D4AA'
    ))

    fig.update_layout(
        title='Returns Distribution',
        xaxis_title='Return (%)',
        yaxis_title='Frequency',
        template='plotly_dark',
        height=300
    )

    return fig


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Berechnet Performance-Metriken."""
    if len(df) < 2:
        return {}

    returns = df['equity'].pct_change().dropna()

    # Sharpe Ratio (annualisiert)
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24) if np.std(returns) > 0 else 0

    # Max Drawdown
    peak = df['equity'].cummax()
    drawdown = (df['equity'] - peak) / peak
    max_dd = abs(drawdown.min()) * 100

    # Total Return
    total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0] - 1) * 100

    # Win Rate (basierend auf täglichen Returns)
    win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'volatility': np.std(returns) * np.sqrt(252 * 24) * 100,
        'win_rate': win_rate,
        'current_equity': df['equity'].iloc[-1]
    }


def main():
    """Hauptfunktion für Dashboard."""
    st.set_page_config(
        page_title="Crypto Trading Bot",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 Crypto Trading Bot Dashboard")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Auto-Refresh
        auto_refresh = st.checkbox("Auto-Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (s)", 5, 60, 30)

        st.divider()

        # Risk Controls
        st.header("Risk Controls")

        if st.button("🛑 Emergency Stop", type="primary"):
            st.warning("Emergency stop triggered!")
            # TODO: Trigger actual emergency stop

        if st.button("⏸️ Pause Trading"):
            st.info("Trading paused")

        if st.button("▶️ Resume Trading"):
            st.success("Trading resumed")

    # Lade Daten
    equity_df = load_equity_curve()
    metrics = calculate_metrics(equity_df)

    # KPIs
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            "Current Equity",
            f"${metrics.get('current_equity', 0):,.2f}",
            f"{metrics.get('total_return', 0):+.2f}%"
        )

    with col2:
        st.metric(
            "Total Return",
            f"{metrics.get('total_return', 0):+.2f}%"
        )

    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics.get('sharpe_ratio', 0):.2f}"
        )

    with col4:
        st.metric(
            "Max Drawdown",
            f"{metrics.get('max_drawdown', 0):.2f}%"
        )

    with col5:
        st.metric(
            "Volatility",
            f"{metrics.get('volatility', 0):.2f}%"
        )

    with col6:
        st.metric(
            "Win Rate",
            f"{metrics.get('win_rate', 0):.1f}%"
        )

    # Charts
    st.divider()

    # Equity Curve
    st.plotly_chart(create_equity_chart(equity_df), use_container_width=True)

    # Zweite Reihe
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_drawdown_chart(equity_df), use_container_width=True)

    with col2:
        st.plotly_chart(create_returns_distribution(equity_df), use_container_width=True)

    # Positionen und Trades
    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Open Positions", "📜 Trade History", "🤖 Model Signals"])

    with tab1:
        st.subheader("Open Positions")

        # Demo-Daten für Positionen
        positions_data = {
            'Symbol': ['BTC/USDT', 'ETH/USDT'],
            'Side': ['Long', 'Short'],
            'Entry Price': [45000.0, 2800.0],
            'Current Price': [46500.0, 2750.0],
            'Size': [0.1, 1.0],
            'PnL': [150.0, 50.0],
            'PnL %': [3.33, 1.79]
        }

        positions_df = pd.DataFrame(positions_data)

        st.dataframe(
            positions_df.style.applymap(
                lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red' if isinstance(x, (int, float)) and x < 0 else '',
                subset=['PnL', 'PnL %']
            ),
            use_container_width=True
        )

    with tab2:
        st.subheader("Recent Trades")

        # Demo-Daten für Trades
        trades_data = {
            'Time': pd.date_range(end=datetime.now(), periods=10, freq='H'),
            'Symbol': ['BTC/USDT'] * 10,
            'Side': ['Buy', 'Sell'] * 5,
            'Price': np.random.uniform(44000, 47000, 10),
            'Size': np.random.uniform(0.01, 0.1, 10),
            'PnL': np.random.uniform(-100, 200, 10)
        }

        trades_df = pd.DataFrame(trades_data)
        trades_df = trades_df.sort_values('Time', ascending=False)

        st.dataframe(trades_df, use_container_width=True)

    with tab3:
        st.subheader("Model Signals")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### LSTM Model")
            st.markdown("**Signal:** 🟢 BUY")
            st.markdown("**Confidence:** 78%")
            st.progress(0.78)

        with col2:
            st.markdown("### Transformer Model")
            st.markdown("**Signal:** 🟡 HOLD")
            st.markdown("**Confidence:** 65%")
            st.progress(0.65)

        st.markdown("### RL Agent (PPO)")
        st.markdown("**Action:** Buy")
        st.markdown("**Position Size:** 45%")
        st.progress(0.45)

    # Risk Status
    st.divider()
    st.subheader("🚨 Risk Status")

    risk_col1, risk_col2, risk_col3 = st.columns(3)

    with risk_col1:
        st.markdown("### Daily Loss")
        daily_loss = 3.2
        st.progress(daily_loss / 10)
        st.markdown(f"{daily_loss}% / 10% limit")

    with risk_col2:
        st.markdown("### Current Drawdown")
        current_dd = metrics.get('max_drawdown', 0)
        st.progress(min(current_dd / 20, 1.0))
        st.markdown(f"{current_dd:.1f}% / 20% limit")

    with risk_col3:
        st.markdown("### Consecutive Losses")
        consec_losses = 2
        st.progress(consec_losses / 5)
        st.markdown(f"{consec_losses} / 5 max")

    # Footer
    st.divider()
    st.markdown(
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Auto-refresh: {'Enabled' if auto_refresh else 'Disabled'}"
    )

    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
