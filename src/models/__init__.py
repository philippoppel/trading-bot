# Models module for LSTM, Transformer and ensemble models
"""
KI-Modelle für Preis-Vorhersage und Trading-Signale.

Enthält:
- LSTM: Long Short-Term Memory für Zeitreihen
- Transformer: Multi-Head-Attention Modell
- ModelManager: Einheitliche Schnittstelle für alle Modelle
"""

from .lstm_model import LSTMPredictor, LSTMTrainer, create_sequences, create_trend_labels
from .transformer_model import TransformerPredictor, TransformerTrainer
from .model_manager import ModelManager

__all__ = [
    'LSTMPredictor', 'LSTMTrainer', 'create_sequences', 'create_trend_labels',
    'TransformerPredictor', 'TransformerTrainer',
    'ModelManager'
]
