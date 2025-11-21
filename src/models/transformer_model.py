"""
Transformer-basiertes Modell für Zeitreihen-Vorhersage.

Verwendet Multi-Head-Attention für die Klassifikation
von Markttrends mit langen Sequenzen.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from typing import Tuple, Optional, Dict, List
from pathlib import Path
from loguru import logger


class PositionalEncoding(nn.Module):
    """Sinusoidales Positional Encoding für Zeitreihen."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Positional Encoding Matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, d_model)

        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Transformer-Modell für Zeitreihen-Klassifikation.

    Features:
    - Multi-Head Self-Attention
    - Positional Encoding
    - Feedforward Networks
    - Layer Normalization
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        """
        Args:
            input_size: Anzahl der Input-Features
            d_model: Embedding-Dimension
            nhead: Anzahl der Attention-Heads
            num_encoder_layers: Anzahl der Encoder-Schichten
            dim_feedforward: Feedforward-Dimension
            num_classes: Anzahl der Output-Klassen
            dropout: Dropout-Rate
            max_seq_len: Maximale Sequenzlänge
        """
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.num_classes = num_classes

        # Input Embedding
        self.input_embedding = nn.Linear(input_size, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Output Layers
        self.output_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        # Initialisierung
        self._init_weights()

    def _init_weights(self):
        """Xavier-Initialisierung."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward Pass.

        Args:
            x: Input (batch, seq_len, input_size)
            src_mask: Optional Attention-Mask

        Returns:
            logits: Klassifikations-Logits (batch, num_classes)
            attention_weights: Letzter Attention-Output (batch, seq_len, d_model)
        """
        # Input Embedding
        x = self.input_embedding(x)  # (batch, seq_len, d_model)

        # Positional Encoding
        x = self.pos_encoder(x)

        # Transformer Encoder
        encoded = self.transformer_encoder(x, src_mask)  # (batch, seq_len, d_model)

        # Global Average Pooling über Sequenz
        pooled = encoded.mean(dim=1)  # (batch, d_model)

        # Normalization
        pooled = self.output_norm(pooled)

        # Classification
        logits = self.classifier(pooled)

        return logits, encoded

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


class TemporalConvEmbedding(nn.Module):
    """
    Optionales CNN-Embedding vor Transformer.
    Kann lokale Muster erfassen bevor Attention angewendet wird.
    """

    def __init__(self, input_size: int, d_model: int, kernel_size: int = 3):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, input_size)

        Returns:
            Embedded output (batch, seq_len, d_model)
        """
        # Conv1d erwartet (batch, channels, length)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = x.permute(0, 2, 1)

        return x


class TransformerWithConvEmbedding(TransformerPredictor):
    """Transformer mit optionalem CNN-Embedding."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        use_conv_embedding: bool = True
    ):
        super().__init__(
            input_size, d_model, nhead, num_encoder_layers,
            dim_feedforward, num_classes, dropout, max_seq_len
        )

        if use_conv_embedding:
            self.input_embedding = TemporalConvEmbedding(input_size, d_model)


class TransformerTrainer:
    """Trainer-Klasse für Transformer-Modell."""

    def __init__(
        self,
        model: TransformerPredictor,
        learning_rate: float = 0.0001,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        device: str = 'auto'
    ):
        """
        Args:
            model: Transformer-Modell
            learning_rate: Maximale Lernrate
            weight_decay: L2-Regularisierung
            warmup_steps: Warmup-Schritte für Scheduler
            device: 'cuda', 'cpu' oder 'auto'
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.warmup_steps = warmup_steps

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98)
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }

        self._step = 0

    def _get_lr(self, step: int, d_model: int) -> float:
        """Transformer Learning Rate Schedule mit Warmup."""
        if step < self.warmup_steps:
            return step / self.warmup_steps * self.optimizer.defaults['lr']

        # Inverse square root decay
        return self.optimizer.defaults['lr'] * min(
            1.0,
            step ** (-0.5) * self.warmup_steps ** 0.5
        )

    def _update_lr(self):
        """Aktualisiert Lernrate basierend auf Schritt."""
        lr = self._get_lr(self._step, self.model.d_model)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> DataLoader:
        """Erstellt DataLoader aus NumPy-Arrays."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Trainiert eine Epoche."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Learning Rate Update
            lr = self._update_lr()
            self._step += 1

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
        """Evaluiert das Modell."""
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
        early_stopping_patience: int = 15,
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

        logger.info(f"Training Transformer on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation
            val_loss, val_acc = self.evaluate(val_loader)

            # Aktuelle Learning Rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # History speichern
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
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
                'd_model': self.model.d_model,
                'num_classes': self.model.num_classes
            },
            'history': self.history,
            'step': self._step
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, path)

    def load_model(self, path: str):
        """Lädt Modell aus Datei."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self._step = checkpoint.get('step', 0)

        logger.info(f"Model loaded from {path}")

    def export_torchscript(self, path: str, example_input: torch.Tensor):
        """Exportiert Modell als TorchScript."""
        self.model.eval()

        # Verwende scripting statt tracing für bessere Kompatibilität
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(path)

        logger.info(f"TorchScript model exported to {path}")
