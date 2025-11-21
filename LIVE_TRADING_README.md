# 🚀 Live Trading Bot - Quick Start Guide

## Was ist das?

Ein **Live Trading Bot** der:
- ✅ Deine ML-Modelle für Trading-Entscheidungen nutzt
- ✅ **ECHTE Orders** auf Binance ausführt
- ✅ Standardmäßig auf **TESTNET** läuft (virtuelles Geld, kein Risiko!)
- ✅ Alle Risk Management Features hat (Stop-Loss, Take-Profit, Drawdown-Limits)
- ✅ State automatisch speichert und zum Dashboard hochlädt

## ⚠️ WICHTIG

**TESTNET = Sicher!** 🧪
- Virtuelles Geld
- Echte Trading-Erfahrung
- Kein Risiko
- Perfekt zum Testen

**PRODUCTION = GEFÄHRLICH!** 💰
- Echtes Geld
- Echte Verluste möglich
- NUR nach Wochen von Testnet-Tests!

## 🎯 Quick Start (3 Schritte)

### 1. API Keys sind bereits konfiguriert ✅
```bash
# Deine .env.testnet ist bereits fertig!
BINANCE_TESTNET_API_KEY=9j4VVwmv8P5ySSvb7aSn6W3oCw2k1PhrMtYpDfjRBvbYbfI8EbHzVzbD9Fa8nEgk
BINANCE_TESTNET_API_SECRET=E78tJvVdtXmhpXcWYoAcFr8CxuyXjsqiEPoe8nBAoKy5HxpqHWG5HQYE8uBFYyqp
```

### 2. Starte den Bot
```bash
./RUN_LIVE_TESTNET.sh
```

### 3. Beobachte die Trades!
- **Terminal**: Zeigt Live-Performance
- **Dashboard**: https://trading-dashboard-three-virid.vercel.app
- **Trade History**: Wird automatisch hochgeladen

## 📊 Was der Bot macht

```
┌─────────────────────────────────────────┐
│  1. Lädt ML-Modelle                      │
│  2. Synct mit Binance Testnet           │
│  3. Holt Live-Marktdaten                 │
│  4. Modell gibt Trading-Signal           │
│  5. ⚡ ECHTE Order auf Binance Testnet   │
│  6. Tracked Position & Portfolio         │
│  7. Speichert State & Upload Dashboard  │
│  └─> Repeat alle 60 Sekunden            │
└─────────────────────────────────────────┘
```

## 🛡️ Safety Features

### Risk Management
- ✅ Stop-Loss: -15% per Symbol
- ✅ Total Drawdown Limit: -20%
- ✅ Max Position Size: 30% per Symbol
- ✅ Max 5 Trades pro Stunde
- ✅ Min 5 Minuten zwischen Trades
- ✅ Volatility Check (pausiert bei >5% Volatilität)

### Trade Execution
- ✅ Minimum Order Size: $15 (Binance Minimum ~$10)
- ✅ Automatic Quantity Rounding (Binance Precision)
- ✅ Error Handling für Failed Orders
- ✅ Commission Tracking

### State Management
- ✅ Auto-Save alle 5 Minuten
- ✅ 5 Backups aufbewahrt
- ✅ Atomic Writes (keine Corruption)
- ✅ Dashboard Upload

## 📈 Live-Performance überwachen

### Terminal Output
```
================================================================================
🧪 TESTNET MULTI-SYMBOL TRADING - LIVE PERFORMANCE
   2025-11-21 14:00:00
================================================================================

Symbol            Price   Position      Portfolio     Return     Status
--------------------------------------------------------------------------------
BTCUSDT    $ 83,500.00   0.0002 $   10,050.00     +0.50%       ✅ GOOD
ETHUSDT    $  2,730.00   0.0000 $    9,950.00     -0.50%       ➖ OK
...
--------------------------------------------------------------------------------
TOTAL                              $   49,950.00 -0.10%
DRAWDOWN                                          -0.12%
```

### Dashboard
- **Real-Time Metrics**: Balance, Positions, Returns
- **Trade History Table**: Alle Trades mit Details
  - Zeitpunkt
  - Aktion (BUY, SELL, STOP_LOSS, etc.)
  - Preis, Menge, Gebühren
  - P&L, Reasoning
  - Binance Order ID
- **Auto-Refresh**: Alle 10 Sekunden

## 🔄 State Management

### Wo wird gespeichert?
```
data/trading_state/
├── live_multi_symbol_testnet_state.json  # Current State
└── backups/
    ├── live_multi_symbol_testnet_state_20251121_140000.json
    ├── live_multi_symbol_testnet_state_20251121_135500.json
    └── ... (5 most recent backups)
```

### Was wird gespeichert?
- Alle Positionen
- Trade History
- Balances
- Risk Metrics
- Emergency Stop Status

### State wiederherstellen
Der Bot lädt automatisch den letzten gespeicherten State beim Start!

## 🎮 Commands

### Start Testnet Bot
```bash
./RUN_LIVE_TESTNET.sh
```

### Start mit Custom Settings
```bash
python live_trade_safe.py --testnet --balance 5000 --interval 120
```

### Stoppen
```bash
# Im Terminal: Ctrl+C
# Bot speichert automatisch den finalen State
```

### Account Status checken
```bash
export $(cat .env.testnet | xargs)
source venv/bin/activate
python check_testnet_account.py
```

## 📋 Command Line Options

```bash
python live_trade_safe.py [OPTIONS]

Options:
  --config PATH          Model config file (default: models/multi_symbol_config.json)
  --balance FLOAT        Initial balance per symbol (default: 10000.0)
  --interval SECONDS     Update interval (default: 60)
  --testnet              Use Testnet (default: True)
  --production          ⚠️  Use PRODUCTION - requires confirmation!
```

## 🔍 Monitoring & Debugging

### Log Files
```bash
# Logs werden in Terminal ausgegeben
# Für File Logging:
python live_trade_safe.py --testnet 2>&1 | tee trading.log
```

### Check Positions on Binance
```python
python check_testnet_account.py
```

### Check Last Trade
```bash
cat data/trading_state/live_multi_symbol_testnet_state.json | jq '.traders.BTCUSDT.trade_history[-1]'
```

## ⚠️ Troubleshooting

### "API Keys not found"
```bash
# Check .env.testnet exists
cat .env.testnet

# Load environment
export $(cat .env.testnet | xargs)
```

### "Order Failed"
- Check Testnet Balance: `python check_testnet_account.py`
- Check Minimum Order Size (needs $15+)
- Check Symbol is trading: Maybe market hours?

### "Position mismatch"
```python
# Re-sync with exchange
# Bot does this automatically on start
```

## 🚨 Emergency Stop

### Automatic Triggers
- Total Drawdown reaches -20%
- Emergency wird aktiviert
- Alle Positionen werden geschlossen

### Manual Stop
```bash
# Ctrl+C im Terminal
# Bot speichert State und zeigt Final Summary
```

## 📊 Performance Analysis

### After Trading Session
```bash
# Check final state
cat data/trading_state/live_multi_symbol_testnet_state.json | jq '.traders'

# View all trades
cat data/trading_state/live_multi_symbol_testnet_state.json | jq '.traders[].trade_history[]'
```

## 🎓 Next Steps

### Phase 1: Testnet Testing (1-2 Wochen)
- ✅ Teste alle Features
- ✅ Beobachte Performance
- ✅ Finde Bugs
- ✅ Verstehe Trade-Entscheidungen

### Phase 2: Optimize
- Tune Risk Parameters
- Adjust Position Sizes
- Test verschiedene Modelle
- Analyze Trade History

### Phase 3: Production (NUR wenn profitabel!)
- Start mit Minimum ($50-100)
- Sehr konservative Settings
- 24/7 Monitoring
- Langsam erhöhen

## 🎯 Key Differences: Paper vs Live

| Feature | Paper Trading | Live Trading |
|---------|--------------|--------------|
| Orders | Simuliert | ✅ **ECHTE Binance Orders** |
| Balance | Lokale Variable | ✅ Echte Binance Balance |
| Slippage | Geschätzt (0.05%) | ✅ Echter Slippage |
| Fees | Geschätzt (0.1%) | ✅ Echte Commission |
| Positions | Simuliert | ✅ Echte Binance Positions |
| Errors | Keine | ✅ Order Failures möglich |
| Risk | 0% | 🧪 0% auf Testnet, ⚠️  100% auf Production |

## ❓ FAQ

**Q: Ist Testnet wirklich sicher?**
A: Ja! Virtuelles Geld, keine realen Verluste möglich.

**Q: Wie lange sollte ich auf Testnet testen?**
A: Mindestens 1-2 Wochen, besser 1 Monat.

**Q: Kann ich auf Production wechseln?**
A: Technisch ja, aber NICHT empfohlen ohne Wochen von Testnet-Tests!

**Q: Was passiert bei Internet-Ausfall?**
A: Bot stoppt. State ist gespeichert. Positionen bleiben auf Binance.

**Q: Kann ich mehrere Bots parallel laufen lassen?**
A: Ja, aber verschiedene API Keys verwenden und State Files trennen.

**Q: Wie oft tradet der Bot?**
A: Abhängig von Modell-Signalen, aber max 5 Trades/Stunde per Symbol.

## 📞 Support

Bei Fragen oder Problemen:
1. Check Logs im Terminal
2. Check `check_testnet_account.py` für Balance/Positions
3. Check Dashboard für Trade History
4. Review State File

## 🎉 Happy Trading!

**Remember:**
- 🧪 Testnet = Safe & Fun
- ⚠️  Production = Risky
- 📊 Always Monitor Performance
- 🛡️ Risk Management is Key
