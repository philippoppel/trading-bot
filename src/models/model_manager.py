"""
Model Manager für einheitliche Modell-Verwaltung.

Bietet eine zentrale Schnittstelle für:
- LSTM, Transformer und RL-Modelle
- Laden/Speichern von Modellen
- TorchScript-Export für Produktion
- Ensemble-Predictions
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal
from loguru import logger

from .lstm_model import LSTMPredictor, LSTMTrainer
from .transformer_model import TransformerPredictor, TransformerTrainer


ModelType = Literal['lstm', 'transformer', 'ensemble']


class ModelManager:
    """
    Zentrale Verwaltung für alle Vorhersage-Modelle.

    Features:
    - Einheitliche API für verschiedene Modelltypen
    - Automatisches Device-Management
    - TorchScript-Export
    - Ensemble-Predictions
    """

    def __init__(self, device: str = 'auto'):
        """
        Args:
            device: 'cuda', 'cpu' oder 'auto'
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.models: Dict[str, torch.nn.Module] = {}
        self.trainers: Dict[str, Union[LSTMTrainer, TransformerTrainer]] = {}

        logger.info(f"ModelManager initialized on {self.device}")

    def create_lstm(
        self,
        name: str,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        learning_rate: float = 0.001
    ) -> LSTMPredictor:
        """
        Erstellt ein neues LSTM-Modell.

        Args:
            name: Eindeutiger Name für das Modell
            input_size: Anzahl der Input-Features
            hidden_size: LSTM Hidden Size
            num_layers: Anzahl der LSTM-Schichten
            num_classes: Anzahl der Output-Klassen
            dropout: Dropout-Rate
            learning_rate: Lernrate für Training

        Returns:
            LSTMPredictor Instanz
        """
        model = LSTMPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )

        trainer = LSTMTrainer(model, learning_rate=learning_rate, device=str(self.device))

        self.models[name] = model
        self.trainers[name] = trainer

        logger.info(f"Created LSTM model '{name}' with {sum(p.numel() for p in model.parameters()):,} parameters")

        return model

    def create_transformer(
        self,
        name: str,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_classes: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 0.0001
    ) -> TransformerPredictor:
        """
        Erstellt ein neues Transformer-Modell.

        Args:
            name: Eindeutiger Name für das Modell
            input_size: Anzahl der Input-Features
            d_model: Embedding-Dimension
            nhead: Anzahl der Attention-Heads
            num_encoder_layers: Anzahl der Encoder-Schichten
            num_classes: Anzahl der Output-Klassen
            dropout: Dropout-Rate
            learning_rate: Lernrate für Training

        Returns:
            TransformerPredictor Instanz
        """
        model = TransformerPredictor(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_classes=num_classes,
            dropout=dropout
        )

        trainer = TransformerTrainer(model, learning_rate=learning_rate, device=str(self.device))

        self.models[name] = model
        self.trainers[name] = trainer

        logger.info(f"Created Transformer model '{name}' with {sum(p.numel() for p in model.parameters()):,} parameters")

        return model

    def get_model(self, name: str) -> torch.nn.Module:
        """Gibt ein Modell anhand des Namens zurück."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found. Available: {list(self.models.keys())}")
        return self.models[name]

    def get_trainer(self, name: str) -> Union[LSTMTrainer, TransformerTrainer]:
        """Gibt einen Trainer anhand des Namens zurück."""
        if name not in self.trainers:
            raise ValueError(f"Trainer '{name}' not found. Available: {list(self.trainers.keys())}")
        return self.trainers[name]

    def predict(
        self,
        name: str,
        X: np.ndarray,
        return_probs: bool = True
    ) -> np.ndarray:
        """
        Vorhersage mit einem spezifischen Modell.

        Args:
            name: Modellname
            X: Input-Daten (samples, seq_len, features)
            return_probs: Wahrscheinlichkeiten oder Klassen zurückgeben

        Returns:
            Vorhersagen als NumPy-Array
        """
        model = self.get_model(name)
        model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            logits, _ = model(X_tensor)
            probs = torch.softmax(logits, dim=1)

        if return_probs:
            return probs.cpu().numpy()
        else:
            return torch.argmax(probs, dim=1).cpu().numpy()

    def predict_ensemble(
        self,
        model_names: List[str],
        X: np.ndarray,
        weights: Optional[List[float]] = None,
        return_probs: bool = True
    ) -> np.ndarray:
        """
        Ensemble-Vorhersage mit mehreren Modellen.

        Args:
            model_names: Liste der Modellnamen
            X: Input-Daten
            weights: Gewichte für jedes Modell (default: gleich)
            return_probs: Wahrscheinlichkeiten oder Klassen zurückgeben

        Returns:
            Gemittelte Vorhersagen
        """
        if weights is None:
            weights = [1.0 / len(model_names)] * len(model_names)

        if len(weights) != len(model_names):
            raise ValueError("Number of weights must match number of models")

        # Normalisiere Gewichte
        weights = np.array(weights) / np.sum(weights)

        # Sammle Vorhersagen
        all_probs = []
        for name in model_names:
            probs = self.predict(name, X, return_probs=True)
            all_probs.append(probs)

        # Gewichteter Durchschnitt
        ensemble_probs = np.zeros_like(all_probs[0])
        for w, probs in zip(weights, all_probs):
            ensemble_probs += w * probs

        if return_probs:
            return ensemble_probs
        else:
            return np.argmax(ensemble_probs, axis=1)

    def save_model(self, name: str, path: str):
        """
        Speichert ein Modell.

        Args:
            name: Modellname
            path: Speicherpfad
        """
        trainer = self.get_trainer(name)
        trainer.save_model(path)
        logger.info(f"Model '{name}' saved to {path}")

    def load_model(self, name: str, path: str):
        """
        Lädt ein Modell aus Datei.

        Args:
            name: Modellname
            path: Ladepfad
        """
        if name not in self.trainers:
            raise ValueError(f"Trainer '{name}' not found. Create model first.")

        trainer = self.trainers[name]
        trainer.load_model(path)
        logger.info(f"Model '{name}' loaded from {path}")

    def export_torchscript(
        self,
        name: str,
        path: str,
        example_input_shape: tuple
    ):
        """
        Exportiert ein Modell als TorchScript.

        Args:
            name: Modellname
            path: Exportpfad
            example_input_shape: Shape des Beispiel-Inputs (batch, seq_len, features)
        """
        model = self.get_model(name)
        model.eval()

        # Erstelle Beispiel-Input
        example_input = torch.randn(example_input_shape).to(self.device)

        # Trace das Modell
        try:
            traced = torch.jit.trace(model, example_input)
            traced.save(path)
            logger.info(f"TorchScript model '{name}' exported to {path}")
        except Exception as e:
            logger.warning(f"Tracing failed, trying scripting: {e}")
            scripted = torch.jit.script(model)
            scripted.save(path)
            logger.info(f"TorchScript model '{name}' exported to {path} (scripted)")

    def load_torchscript(self, name: str, path: str):
        """
        Lädt ein TorchScript-Modell.

        Args:
            name: Name für das geladene Modell
            path: Pfad zum TorchScript-Modell
        """
        model = torch.jit.load(path, map_location=self.device)
        self.models[name] = model
        logger.info(f"TorchScript model loaded as '{name}' from {path}")

    def list_models(self) -> List[str]:
        """Gibt Liste aller Modellnamen zurück."""
        return list(self.models.keys())

    def get_model_info(self, name: str) -> Dict:
        """
        Gibt Informationen über ein Modell zurück.

        Args:
            name: Modellname

        Returns:
            Dictionary mit Modell-Informationen
        """
        model = self.get_model(name)

        # Zähle Parameter
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info = {
            'name': name,
            'type': type(model).__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device)
        }

        # Modell-spezifische Informationen
        if hasattr(model, 'input_size'):
            info['input_size'] = model.input_size
        if hasattr(model, 'num_classes'):
            info['num_classes'] = model.num_classes
        if hasattr(model, 'hidden_size'):
            info['hidden_size'] = model.hidden_size
        if hasattr(model, 'd_model'):
            info['d_model'] = model.d_model

        return info

    def train_model(
        self,
        name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Trainiert ein Modell.

        Args:
            name: Modellname
            X_train: Training Features
            y_train: Training Labels
            X_val: Validation Features
            y_val: Validation Labels
            epochs: Maximale Epochen
            batch_size: Batch-Größe
            early_stopping_patience: Geduld für Early Stopping
            save_path: Pfad zum Speichern des besten Modells

        Returns:
            Training History
        """
        trainer = self.get_trainer(name)

        # Erstelle DataLoader
        train_loader = trainer.create_dataloader(X_train, y_train, batch_size, shuffle=True)
        val_loader = trainer.create_dataloader(X_val, y_val, batch_size, shuffle=False)

        # Training
        history = trainer.train(
            train_loader,
            val_loader,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            save_path=save_path
        )

        return history

    def evaluate_model(
        self,
        name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32
    ) -> Dict:
        """
        Evaluiert ein Modell.

        Args:
            name: Modellname
            X_test: Test Features
            y_test: Test Labels
            batch_size: Batch-Größe

        Returns:
            Dictionary mit Metriken
        """
        trainer = self.get_trainer(name)
        test_loader = trainer.create_dataloader(X_test, y_test, batch_size, shuffle=False)

        loss, accuracy = trainer.evaluate(test_loader)

        # Zusätzliche Metriken
        predictions = self.predict(name, X_test, return_probs=False)

        # Per-Class Accuracy
        from collections import Counter
        class_correct = Counter()
        class_total = Counter()

        for pred, true in zip(predictions, y_test):
            class_total[true] += 1
            if pred == true:
                class_correct[true] += 1

        class_accuracy = {
            cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
            for cls in sorted(class_total.keys())
        }

        return {
            'loss': loss,
            'accuracy': accuracy,
            'class_accuracy': class_accuracy,
            'predictions': predictions
        }

    def remove_model(self, name: str):
        """Entfernt ein Modell aus dem Manager."""
        if name in self.models:
            del self.models[name]
        if name in self.trainers:
            del self.trainers[name]
        logger.info(f"Model '{name}' removed")
