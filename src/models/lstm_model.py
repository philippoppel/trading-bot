"""
LSTM-basiertes Modell für Preis-Trendvorhersage.

Verwendet Multi-Layer LSTM mit Attention für die Klassifikation
von Markttrends (up/down/sideways).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path
from loguru import logger


class AttentionLayer(nn.Module):
    """Attention-Mechanismus für LSTM-Output."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: Shape (batch, seq_len, hidden_size)

        Returns:
            context: Gewichteter Kontext-Vektor (batch, hidden_size)
            weights: Attention-Gewichte (batch, seq_len)
        """
        # Berechne Attention-Scores
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len)

        # Gewichtete Summe
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)

        return context, weights


class LSTMPredictor(nn.Module):
    """
    LSTM-Modell für Trendklassifikation.

    Features:
    - Multi-Layer Bidirectional LSTM
    - Attention-Mechanismus
    - Dropout für Regularisierung
    - Batch Normalization
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,  # up, down, sideways
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Args:
            input_size: Anzahl der Input-Features
            hidden_size: LSTM Hidden Size
            num_layers: Anzahl der LSTM-Schichten
            num_classes: Anzahl der Output-Klassen
            dropout: Dropout-Rate
            bidirectional: Bidirektionales LSTM
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input Batch Normalization
        self.batch_norm = nn.BatchNorm1d(input_size)

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention Layer
        self.attention = AttentionLayer(hidden_size * self.num_directions)

        # Fully Connected Layers
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # Initialisierung
        self._init_weights()

    def _init_weights(self):
        """Xavier-Initialisierung für bessere Konvergenz."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward Pass.

        Args:
            x: Input-Tensor (batch, seq_len, features)

        Returns:
            logits: Klassifikations-Logits (batch, num_classes)
            attention_weights: Attention-Gewichte (batch, seq_len)
        """
        batch_size, seq_len, features = x.shape

        # Batch Normalization (über Features)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden * directions)

        # Attention
        context, attention_weights = self.attention(lstm_out)

        # Classification
        logits = self.fc(context)

        return logits, attention_weights

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Vorhersage mit Softmax-Wahrscheinlichkeiten.

        Args:
            x: Input-Tensor

        Returns:
            Klassenwahrscheinlichkeiten als NumPy-Array
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict_class(self, x: torch.Tensor) -> np.ndarray:
        """
        Vorhersage der Klasse (0=down, 1=sideways, 2=up).

        Args:
            x: Input-Tensor

        Returns:
            Vorhergesagte Klassen als NumPy-Array
        """
        probs = self.predict(x)
        return np.argmax(probs, axis=1)


class LSTMTrainer:
    """Trainer-Klasse für LSTM-Modell."""

    def __init__(
        self,
        model: LSTMPredictor,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = 'auto'
    ):
        """
        Args:
            model: LSTM-Modell
            learning_rate: Lernrate
            weight_decay: L2-Regularisierung
            device: 'cuda', 'cpu' oder 'auto'
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        self.criterion = nn.CrossEntropyLoss()
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Erstellt DataLoader aus NumPy-Arrays.

        Args:
            X: Features (samples, seq_len, features)
            y: Labels (samples,)
            batch_size: Batch-Größe
            shuffle: Daten mischen

        Returns:
            PyTorch DataLoader
        """
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Trainiert eine Epoche.

        Returns:
            (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            logits, _ = self.model(X_batch)
            loss = self.criterion(logits, y_batch)

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluiert das Modell.

        Returns:
            (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits, _ = self.model(X_batch)
                loss = self.criterion(logits, y_batch)

                total_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(logits, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Vollständiges Training mit Early Stopping.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Maximale Epochen
            early_stopping_patience: Geduld für Early Stopping
            save_path: Pfad zum Speichern des besten Modells

        Returns:
            Training History
        """
        best_val_loss = float('inf')
        patience_counter = 0

        logger.info(f"Training LSTM on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation
            val_loss, val_acc = self.evaluate(val_loader)

            # Learning Rate Scheduling
            self.scheduler.step(val_loss)

            # History speichern
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            # Early Stopping & Best Model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                if save_path:
                    self.save_model(save_path)
                    logger.info(f"Best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return self.history

    def save_model(self, path: str):
        """Speichert Modell und Konfiguration."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'num_classes': self.model.num_classes,
                'bidirectional': self.model.bidirectional
            },
            'history': self.history
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, path)

    def load_model(self, path: str):
        """Lädt Modell aus Datei."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)

        logger.info(f"Model loaded from {path}")

    def export_torchscript(self, path: str, example_input: torch.Tensor):
        """
        Exportiert Modell als TorchScript für Produktion.

        Args:
            path: Speicherpfad
            example_input: Beispiel-Input für Tracing
        """
        self.model.eval()

        # Tracing
        traced_model = torch.jit.trace(
            self.model,
            example_input.to(self.device)
        )

        traced_model.save(path)
        logger.info(f"TorchScript model exported to {path}")


def create_sequences(
    data: np.ndarray,
    targets: np.ndarray,
    seq_length: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Erstellt Sequenzen für LSTM-Training.

    Args:
        data: Feature-Daten (samples, features)
        targets: Target-Labels (samples,)
        seq_length: Sequenzlänge

    Returns:
        X: Sequenzen (num_sequences, seq_length, features)
        y: Labels (num_sequences,)
    """
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(targets[i + seq_length])

    return np.array(X), np.array(y)


def create_trend_labels(
    prices: np.ndarray,
    threshold: float = 0.01
) -> np.ndarray:
    """
    Erstellt Trend-Labels aus Preisdaten.

    Args:
        prices: Preise (z.B. Close)
        threshold: Schwelle für Trend-Erkennung (1% default)

    Returns:
        Labels: 0=down, 1=sideways, 2=up
    """
    returns = np.diff(prices) / prices[:-1]

    labels = np.ones(len(returns), dtype=np.int64)  # sideways default
    labels[returns > threshold] = 2  # up
    labels[returns < -threshold] = 0  # down

    return labels
