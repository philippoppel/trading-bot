# Crypto Trading Bot

Ein vollständiges, modular aufgebautes KI-gestütztes Krypto-Trading-System für autonomen Handel.

## Features

- **Multi-Model Support**: LSTM, Transformer, Reinforcement Learning (PPO, DQN, A2C)
- **Umfassendes Feature Engineering**: 42+ technische Indikatoren
- **Realistische Backtesting-Engine**: Mit Gebühren, Slippage, Latenz
- **Live Trading**: CCXT-Integration für echte Order-Ausführung
- **Risikomanagement**: Stop-Loss, Take-Profit, Drawdown-Limits, Kill-Switch
- **Monitoring Dashboard**: Echtzeit-Überwachung mit Streamlit
- **Alerts**: Telegram/Discord Benachrichtigungen

## Architektur

```
src/
├── data/               # Datenbeschaffung & Feature Engineering
├── models/             # KI-Modelle (LSTM, Transformer)
├── agent/              # Reinforcement Learning Agent
├── environment/        # Trading Environment (Gym)
├── backtesting/        # Backtesting Engine
├── execution/          # Live Trading Execution
├── risk_management/    # Risikomanagement
├── monitoring/         # Dashboard & Alerts
└── utils/              # Logging & Utilities
```

## Installation

### Voraussetzungen

- Python 3.10+
- pip oder conda

### Setup

1. **Repository klonen**
```bash
git clone <repository-url>
cd traidingbot
```

2. **Virtual Environment erstellen**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate  # Windows
```

3. **Dependencies installieren**
```bash
pip install -r requirements.txt
```

4. **Konfiguration erstellen**
```bash
cp .env.example .env
# Bearbeite .env mit deinen API-Keys
```

## Konfiguration

### Environment Variables (.env)

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true
INITIAL_BALANCE=10000
TRADING_FEE=0.001
LOG_LEVEL=INFO
```

### Config File (config/config.yaml)

Die Hauptkonfiguration erfolgt über `config/config.yaml`:

- **Trading**: Symbole, Zeitrahmen, Balance
- **Features**: Indikatoren, Lookback-Perioden
- **Models**: Hyperparameter für LSTM, Transformer, RL
- **Risk**: Stop-Loss, Take-Profit, Max Drawdown
- **Monitoring**: Dashboard-Port, Alert-Einstellungen

## Usage

### Daten sammeln

```bash
python main.py --mode collect --symbol BTCUSDT --days 365
```

### Modelle trainieren

**Reinforcement Learning (PPO)**
```bash
python main.py --mode train --model ppo --timesteps 1000000
```

**LSTM Modell**
```bash
python main.py --mode train --model lstm --epochs 100
```

**Transformer Modell**
```bash
python main.py --mode train --model transformer --epochs 100
```

### Backtesting

```bash
python main.py --mode backtest --strategy lstm --start 2023-01-01 --end 2023-12-31
```

### Paper Trading

```bash
python main.py --mode paper --model ppo
```

### Live Trading

```bash
python main.py --mode live --model ppo
```

### Monitoring Dashboard

```bash
python main.py --mode dashboard
# oder
streamlit run src/monitoring/dashboard.py
```

## Modelle

### LSTM (Long Short-Term Memory)

- Input: Multivariate Zeitreihen (OHLCV + Features)
- Output: Trendklassifikation (up/down/sideways)
- Architektur: 2-Layer LSTM mit Attention

### Transformer

- Multi-Head-Attention Architektur
- Unterstützt lange Sequenzen (500-1000 Steps)
- Positional Encoding für Zeitreihen

### Reinforcement Learning

- **Algorithmen**: PPO, DQN, A2C, DDPG, SAC
- **State**: Preise, Features, Position, Portfolio
- **Actions**: Buy, Sell, Hold mit Position Sizing
- **Reward**: Risikoadjustierte Rendite (Sharpe Ratio)

## Backtesting

Die Backtesting-Engine simuliert realistisches Trading:

- **Gebühren**: Konfigurierbare Trading-Fees
- **Slippage**: Realistische Order-Ausführung
- **Latenz**: Simulierte Netzwerk-Verzögerungen

### Metriken

- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Calmar Ratio

### Walk-Forward-Optimierung

Automatische Optimierung über rollierende Zeitfenster zur Vermeidung von Overfitting.

## Risikomanagement

- **Position Sizing**: Max. 50% des Portfolios pro Trade
- **Stop-Loss**: Automatischer Ausstieg bei Verlust (default: 5%)
- **Take-Profit**: Gewinnmitnahme (default: 15%)
- **Daily Stop**: Max. Tagesverlust (default: 10%)
- **Max Drawdown**: Globales Drawdown-Limit
- **Kill-Switch**: Notfall-Stop bei anormalem Verhalten
- **Cooldown**: Pause nach Verlustserie

## Monitoring

### Dashboard Features

- Echtzeit Equity Curve
- Offene Positionen
- Live Model-Signale
- Risiko-Status
- Performance-Metriken
- Trade-History

### Alerts

- Telegram/Discord Benachrichtigungen
- Trade-Öffnung/Schließung
- Drawdown-Warnung
- Technische Fehler

## Testing

```bash
# Alle Tests ausführen
pytest tests/

# Mit Coverage
pytest tests/ --cov=src --cov-report=html

# Spezifisches Modul
pytest tests/test_backtesting.py
```

## Docker

### Build

```bash
docker build -t crypto-trading-bot .
```

### Run

```bash
docker-compose up -d
```

### Services

- **bot**: Trading Bot
- **dashboard**: Streamlit Dashboard
- **tensorboard**: Training Monitoring

## Projektstruktur

```
traidingbot/
├── config/
│   └── config.yaml           # Hauptkonfiguration
├── data/                     # Gecachte Daten (Parquet)
├── logs/                     # Log-Dateien
├── models/                   # Trainierte Modelle
│   ├── best/                 # Beste Modelle
│   └── checkpoints/          # Training Checkpoints
├── src/
│   ├── agent/                # RL Agent
│   ├── backtesting/          # Backtesting Engine
│   ├── config/               # Settings
│   ├── data/                 # Data Collection & Preprocessing
│   ├── environment/          # Trading Environment
│   ├── execution/            # Live Trading
│   ├── models/               # LSTM, Transformer
│   ├── monitoring/           # Dashboard & Alerts
│   ├── risk_management/      # Risk Management
│   └── utils/                # Utilities
├── tests/                    # Unit & Integration Tests
├── .env.example              # Environment Template
├── docker-compose.yaml       # Docker Compose
├── Dockerfile                # Docker Build
├── main.py                   # Entry Point
├── README.md                 # Diese Dokumentation
└── requirements.txt          # Dependencies
```

## Performance-Tipps

1. **GPU-Training**: Nutze CUDA für schnelleres Training
2. **Daten-Caching**: Parquet-Format für schnelles Laden
3. **Batch-Größe**: Anpassen an verfügbaren VRAM
4. **Checkpoints**: Regelmäßiges Speichern bei langem Training

## Troubleshooting

### API Rate Limits

```python
# In config.yaml
data:
  rate_limit_delay: 0.5  # Sekunden zwischen Requests
```

### Speicherprobleme

```python
# Kleinere Sequenzen verwenden
environment:
  observation_window: 60  # statt 100
```

### WebSocket Disconnects

Das System reconnected automatisch. Bei häufigen Disconnects:
- Netzwerkverbindung prüfen
- API-Limits der Börse beachten

## Contributing

1. Fork das Repository
2. Feature Branch erstellen (`git checkout -b feature/AmazingFeature`)
3. Änderungen committen (`git commit -m 'Add AmazingFeature'`)
4. Branch pushen (`git push origin feature/AmazingFeature`)
5. Pull Request öffnen

## Lizenz

MIT License - siehe [LICENSE](LICENSE) für Details.

## Disclaimer

**WICHTIG**: Dieses System ist für Bildungszwecke gedacht. Trading mit Kryptowährungen ist hochriskant. Verwende nur Kapital, dessen Verlust du dir leisten kannst. Die Entwickler übernehmen keine Haftung für finanzielle Verluste.

## Support

- GitHub Issues: Bug Reports und Feature Requests
- Dokumentation: Diese README und Code-Kommentare

---

Made with Python, PyTorch, and Stable-Baselines3
