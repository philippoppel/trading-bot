"""
Unit Tests für KI-Modelle.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.lstm_model import LSTMPredictor, LSTMTrainer, create_sequences, create_trend_labels
from src.models.transformer_model import TransformerPredictor, TransformerTrainer
from src.models.model_manager import ModelManager


class TestLSTMModel:
    """Tests für LSTM-Modell."""

    @pytest.fixture
    def model(self):
        return LSTMPredictor(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            num_classes=3
        )

    @pytest.fixture
    def sample_data(self):
        # (batch, seq_len, features)
        return torch.randn(8, 60, 10)

    def test_model_creation(self, model):
        """Test Modell-Erstellung."""
        assert model is not None
        assert model.input_size == 10
        assert model.hidden_size == 32

    def test_forward_pass(self, model, sample_data):
        """Test Forward Pass."""
        logits, attention = model(sample_data)

        assert logits.shape == (8, 3)
        assert attention.shape == (8, 60)

    def test_predict(self, model, sample_data):
        """Test Vorhersage."""
        probs = model.predict(sample_data)

        assert probs.shape == (8, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_class(self, model, sample_data):
        """Test Klassenvorhersage."""
        classes = model.predict_class(sample_data)

        assert classes.shape == (8,)
        assert all(c in [0, 1, 2] for c in classes)


class TestTransformerModel:
    """Tests für Transformer-Modell."""

    @pytest.fixture
    def model(self):
        return TransformerPredictor(
            input_size=10,
            d_model=32,
            nhead=4,
            num_encoder_layers=2,
            num_classes=3
        )

    @pytest.fixture
    def sample_data(self):
        return torch.randn(8, 100, 10)

    def test_model_creation(self, model):
        """Test Modell-Erstellung."""
        assert model is not None
        assert model.input_size == 10
        assert model.d_model == 32

    def test_forward_pass(self, model, sample_data):
        """Test Forward Pass."""
        logits, encoded = model(sample_data)

        assert logits.shape == (8, 3)
        assert encoded.shape == (8, 100, 32)

    def test_predict(self, model, sample_data):
        """Test Vorhersage."""
        probs = model.predict(sample_data)

        assert probs.shape == (8, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


class TestModelManager:
    """Tests für Model-Manager."""

    @pytest.fixture
    def manager(self):
        return ModelManager(device='cpu')

    def test_create_lstm(self, manager):
        """Test LSTM-Erstellung."""
        model = manager.create_lstm('test_lstm', input_size=10)

        assert 'test_lstm' in manager.list_models()
        assert isinstance(model, LSTMPredictor)

    def test_create_transformer(self, manager):
        """Test Transformer-Erstellung."""
        model = manager.create_transformer('test_transformer', input_size=10)

        assert 'test_transformer' in manager.list_models()
        assert isinstance(model, TransformerPredictor)

    def test_predict(self, manager):
        """Test Vorhersage über Manager."""
        manager.create_lstm('lstm', input_size=10)

        X = np.random.randn(4, 60, 10).astype(np.float32)
        probs = manager.predict('lstm', X)

        assert probs.shape == (4, 3)

    def test_ensemble_predict(self, manager):
        """Test Ensemble-Vorhersage."""
        manager.create_lstm('lstm1', input_size=10)
        manager.create_lstm('lstm2', input_size=10)

        X = np.random.randn(4, 60, 10).astype(np.float32)
        probs = manager.predict_ensemble(['lstm1', 'lstm2'], X)

        assert probs.shape == (4, 3)

    def test_model_info(self, manager):
        """Test Modell-Info."""
        manager.create_lstm('lstm', input_size=10, hidden_size=64)
        info = manager.get_model_info('lstm')

        assert info['name'] == 'lstm'
        assert info['input_size'] == 10
        assert info['hidden_size'] == 64


class TestDataUtils:
    """Tests für Daten-Utilities."""

    def test_create_sequences(self):
        """Test Sequenzerstellung."""
        data = np.random.randn(100, 10)
        targets = np.random.randint(0, 3, 100)

        X, y = create_sequences(data, targets, seq_length=20)

        assert X.shape == (80, 20, 10)
        assert y.shape == (80,)

    def test_create_trend_labels(self):
        """Test Trend-Label-Erstellung."""
        prices = np.array([100, 102, 101, 99, 103])
        labels = create_trend_labels(prices, threshold=0.01)

        assert len(labels) == 4
        assert all(l in [0, 1, 2] for l in labels)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
