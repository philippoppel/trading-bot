# 💾 State Persistence Guide

## Übersicht

Der Safe Multi-Symbol Trader speichert jetzt automatisch seinen Status und kann nach einem Neustart, Netzwerkunterbrechung oder Absturz nahtlos weitermachen.

## Features

### ✅ Was wird gespeichert:

- **Portfolio State**: Balance, Positionen, Entry Prices
- **Trading History**: Anzahl Trades, Total Fees
- **Risk State**: Stop-Loss Status, Emergency Stop
- **Session Info**: Start Zeit, Timestamps
- **Trade Limits**: Trades per Hour Counter

### ✅ Wann wird gespeichert:

1. **Nach jedem Trade** (Auto-Save)
2. **Alle 5 Minuten** (Periodic Save, auch ohne Trades)
3. **Bei Ctrl+C** (Clean Exit)
4. **Bei Emergency Stop** (Max Drawdown)

### ✅ Sicherheitsfeatures:

- **Atomic Writes**: Temp file + rename (keine korrupten Dateien)
- **Backups**: Letzte 5 States werden behalten
- **Crash Recovery**: State bleibt erhalten bei Absturz
- **Validation**: Version Check beim Laden

---

## Verwendung

### Standard Start (mit Auto-Recovery):

```bash
python paper_trade_safe.py --balance 10000
```

**Beim ersten Start:**
- Erstellt neuen State
- Startet mit fresh Portfolio

**Bei nachfolgenden Starts:**
- Lädt automatisch letzten State
- Setzt alle Positionen fort
- Behält Trade History

### State File Location:

```
data/trading_state/
├── safe_multi_symbol_state.json    # Aktueller State
└── backups/
    ├── safe_multi_symbol_state_20250120_143022.json
    ├── safe_multi_symbol_state_20250120_143523.json
    └── ...
```

### State anzeigen:

```bash
python test_persistence.py
```

Output:
```
============================================================
✅ STATE FILE FOUND
============================================================

📂 Saved State Information:
   Version:           1.0
   Last Saved:        2025-01-20T14:35:45.123456
   Session Started:   2025-01-20T10:00:00.000000
   Initial Balance:   $10,000.00
   Emergency Stop:    False

💼 Trader States:

   BTCUSDT:
      Balance:        $5,234.56
      Position:       0.30
      Entry Price:    $91,464.00
      Total Trades:   14
      Total Fees:     $23.45
      Current Price:  $92,100.00
      Portfolio:      $10,123.45 (+1.23%)
...
```

---

## Konfiguration

### Persistence Parameter:

```python
trader = SafeMultiSymbolTrader(
    # State file path (optional)
    state_file='data/my_custom_state.json',

    # Auto-save nach jedem Trade
    auto_save=True,

    # Speichern alle N Sekunden
    save_interval=300,  # 5 Minuten

    # Anzahl Backups behalten
    keep_backups=5
)
```

### Command Line:

```bash
# Mit Custom Settings
python paper_trade_safe.py \
  --balance 10000 \
  --interval 60
  # State-Settings sind hardcoded im Script
```

---

## Recovery Szenarien

### 1. **Normaler Restart (Ctrl+C)**

```bash
# Session 1
$ python paper_trade_safe.py --balance 10000
> Trading läuft...
> Ctrl+C
> Saving final state...
> ✅ State saved

# Session 2
$ python paper_trade_safe.py --balance 10000
> ✅ Restored previous trading state
> Session started: 2025-01-20 10:00:00
> BTCUSDT: Position 0.30 | 14 trades | $23.45 fees
> Trading continues...
```

**Result**: Nahtlose Fortsetzung, alle Positionen erhalten

---

### 2. **Crash / Kill Process**

```bash
# Session läuft, Netzwerk bricht ab oder Kill -9
$ python paper_trade_safe.py --balance 10000
> Trading läuft...
> [CRASH - Process killed]

# Restart
$ python paper_trade_safe.py --balance 10000
> ✅ Restored previous trading state
> State from last save (max 5 min alt)
> Positionen werden restored
```

**Result**: State vom letzten Auto-Save (max 5 Min Verlust)

---

### 3. **Emergency Stop**

```bash
$ python paper_trade_safe.py --balance 10000
> Trading läuft...
> 🚨 EMERGENCY STOP: Max Total Drawdown reached: -20.5%
> Saving final state...
> ✅ State saved

# Restart
$ python paper_trade_safe.py --balance 10000
> ✅ Restored previous trading state
> Emergency Stop: ACTIVE
> ⚠️  Trading wird NICHT fortgesetzt (Emergency Flag gesetzt)
```

**Result**: State erhalten, aber Trading gestoppt

---

### 4. **Mehrere Tage Pause**

```bash
# Tag 1
$ python paper_trade_safe.py --balance 10000
> Trading for 12 hours
> Ctrl+C
> ✅ State saved

# Tag 5 (Pause)
$ python paper_trade_safe.py --balance 10000
> ✅ Restored previous trading state
> Session started: 5 days ago
> ⚠️  ACHTUNG: Preise haben sich geändert!
> Positions werden mit aktuellen Preisen bewertet
```

**Result**: Fortsetzung möglich, aber P&L kann unterschiedlich sein

---

## State File Format

### JSON Structure:

```json
{
  "version": "1.0",
  "timestamp": "2025-01-20T14:35:45.123456",
  "session_start_time": "2025-01-20T10:00:00.000000",
  "initial_balance": 10000.0,
  "emergency_stopped": false,
  "emergency_reason": null,
  "traders": {
    "BTCUSDT": {
      "balance": 5234.56,
      "position": 0.3,
      "position_value": 4900.0,
      "entry_price": 91464.0,
      "total_fees": 23.45,
      "num_trades": 14,
      "trade_history": [],
      "current_price": 92100.0,
      "last_trade_time": "2025-01-20T14:30:00.000000",
      "trades_last_hour": [
        "2025-01-20T14:00:00.000000",
        "2025-01-20T14:15:00.000000"
      ],
      "max_loss_reached": false,
      "highest_value": 10500.0
    }
  }
}
```

---

## Backups

### Automatic Backups:

- Erstellt vor jedem Überschreiben
- Timestamped Filename
- Behält letzte 5 (konfigurierbar)

### Backup Location:

```
data/trading_state/backups/
```

### Manuelles Restore von Backup:

```bash
# 1. Finde gewünschtes Backup
ls -la data/trading_state/backups/

# 2. Kopiere zu Main State
cp data/trading_state/backups/safe_multi_symbol_state_20250120_120000.json \
   data/trading_state/safe_multi_symbol_state.json

# 3. Starte Bot
python paper_trade_safe.py
```

---

## Troubleshooting

### Problem: "State initial balance doesn't match"

**Ursache**: Du startest mit anderer Balance als im State

**Lösung**:
```bash
# Option 1: Benutze gleiche Balance
python paper_trade_safe.py --balance 10000  # Wie im State

# Option 2: Delete State für Fresh Start
rm data/trading_state/safe_multi_symbol_state.json
python paper_trade_safe.py --balance 5000
```

---

### Problem: "Incompatible state version"

**Ursache**: State Format hat sich geändert (Update)

**Lösung**:
```bash
# Backup alter State
mv data/trading_state/safe_multi_symbol_state.json \
   data/trading_state/old_state_backup.json

# Fresh Start
python paper_trade_safe.py
```

---

### Problem: Corrupted State File

**Ursache**: Crash während Save (sehr selten dank Atomic Writes)

**Lösung**:
```bash
# Check backups
ls -la data/trading_state/backups/

# Restore latest backup
cp data/trading_state/backups/safe_multi_symbol_state_LATEST.json \
   data/trading_state/safe_multi_symbol_state.json
```

---

## Best Practices

### ✅ DO:

1. **Let it auto-save** - Nicht deaktivieren
2. **Keep backups** - Mindestens 5
3. **Check logs** - Bei Problemen zuerst Logs checken
4. **Test recovery** - Mal absichtlich Ctrl+C und neu starten

### ❌ DON'T:

1. **Nicht manuell editieren** - State File ist für Bot, nicht für Menschen
2. **Nicht während Save stoppen** - Warte auf Completion
3. **Nicht State zwischen Configs teilen** - Jede Config = eigener State
4. **Nicht alte States mit neuer Code Version** - Migration kann brechen

---

## Performance

### Speicher Impact:

- **State File Size**: ~5-10 KB pro Symbol
- **Backup Size**: ~50 KB für 5 Backups
- **Save Duration**: ~10-50ms (negligible)

### CPU Impact:

- **Auto-Save**: < 0.1% CPU
- **Periodic Save**: Jede 5 Min, < 1s
- **Load on Start**: < 100ms

→ **Negligible Performance Impact!**

---

## Advanced: Disaster Recovery

### Kompletter Datenverlust Szenario:

```bash
# Alle State Files verloren
# Trading läuft seit Tagen
# Positionen sind offen

# Was passiert:
$ python paper_trade_safe.py --balance 10000
> 🆕 Starting fresh trading session
> Alle Positionen VERLOREN
> Start mit 10000 USD Balance
> ❌ Trade History verloren
```

**Prevention**:
- Backup `data/trading_state/` zu externem Storage
- Cron job für tägliche Backups
- Cloud sync (Dropbox, Google Drive)

```bash
# Beispiel Backup Script
#!/bin/bash
cp -r data/trading_state/ ~/Dropbox/trading_backups/$(date +%Y%m%d)/
```

---

## Migration zwischen Versionen

### Wenn Code Update State Format ändert:

```python
# Future: Migration Script
def migrate_state_v1_to_v2(old_state):
    new_state = {
        'version': '2.0',
        # ... migration logic
    }
    return new_state
```

**Aktuell**: Keine Migration nötig (Version 1.0)

---

## FAQ

### Q: Was passiert wenn ich Balance ändere?

**A**: Warning, aber State wird geladen. Neue Trades nutzen neue Balance.

### Q: Kann ich State zwischen Machines teilen?

**A**: Ja, kopiere `data/trading_state/` Ordner. Aber: Keine parallelen Runs!

### Q: Wie oft wird gespeichert?

**A**: Nach jedem Trade + alle 5 Min + bei Exit

### Q: Was wenn State korrupt ist?

**A**: Bot startet fresh, aber Backups sind verfügbar

### Q: Performance Impact?

**A**: Minimal (<0.1% CPU, <50ms per save)

---

## Summary

**Persistence ist KRITISCH für Production Trading:**

✅ **Keine Angst vor Restarts**
✅ **Keine Angst vor Crashes**
✅ **Keine Angst vor Netzwerk Issues**
✅ **Trade History bleibt erhalten**
✅ **Positionen werden restored**

**Läuft automatisch, zero config needed!** 🎉
