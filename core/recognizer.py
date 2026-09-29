"""
ClarifySign - ISL Recognition Engine
Deep Bidirectional LSTM neural network for isolated Indian Sign Language recognition.
Consumes temporal sequences of 225-dimensional skeletal landmark features and
outputs a calibrated softmax probability distribution over gesture classes.
"""

import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from config import (
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    DENSE_HIDDEN_DIM,
    DROPOUT_RATE,
    MODEL_PATH,
    LABELS_PATH,
    SHOPKEEPER_CLASSES
)


class ISLBiLSTM(nn.Module):
    """
    Bidirectional LSTM network with Layer Normalization and Dropout regularization
    for temporal landmark feature sequence modeling.
    """

    def __init__(
        self,
        input_dim=FEATURE_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS,
        num_classes=len(SHOPKEEPER_CLASSES),
        dense_dim=DENSE_HIDDEN_DIM,
        dropout=DROPOUT_RATE
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Temporal feature normalization
        self.layer_norm = nn.LayerNorm(input_dim)

        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Non-linear projection head
        self.fc1 = nn.Linear(hidden_dim * 2, dense_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dense_dim, num_classes)

    def forward(self, x):
        """
        x: tensor of shape (batch_size, sequence_length, feature_dim)
        returns logits of shape (batch_size, num_classes)
        """
        # x is (B, T, D)
        x = self.layer_norm(x)
        lstm_out, _ = self.lstm(x)

        # Global average + max temporal pooling across time steps
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        pooled = (avg_pool + max_pool) * 0.5

        dense = self.act(self.fc1(pooled))
        dense = self.dropout(dense)
        logits = self.classifier(dense)
        return logits


class ISLRecognizer:
    """
    Inference interface for our trained ISL recognition model.
    Loads PyTorch weights and class label mappings, performs sequence normalization,
    and returns full probability distributions over classes.
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        labels_path=LABELS_PATH,
        temperature=1.0,
        device=None
    ):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.temperature = max(1e-3, float(temperature))

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.labels = []
        self.model = None

        self._load_labels()
        self._load_model()

    def _load_labels(self):
        if self.labels_path.exists():
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    self.labels = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load labels from {self.labels_path}: {e}")

    def _load_model(self):
        if self.model_path.exists() and self.labels:
            try:
                num_classes = len(self.labels)
                self.model = ISLBiLSTM(
                    input_dim=FEATURE_DIM,
                    num_classes=num_classes
                ).to(self.device)

                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                elif isinstance(checkpoint, dict):
                    self.model.load_state_dict(checkpoint)
                else:
                    self.model = checkpoint

                self.model.eval()
            except Exception as e:
                print(f"Warning: Failed to load PyTorch model from {self.model_path}: {e}")
                self.model = None

    @property
    def available(self):
        """Returns True if a trained model and class labels are available."""
        return self.model is not None and len(self.labels) > 0

    def predict(self, sequence, top_k=5):
        """
        Runs model inference on a sequence of landmark features.
        sequence: shape (sequence_length, feature_dim) or (batch, sequence_length, feature_dim)
        returns: list of tuples [(class_name, probability), ...] sorted descending.
        """
        if not self.available:
            raise RuntimeError(
                f"Model or labels not found at {self.model_path}. "
                "Please train the model first using: python RUN_ME.py or python training/train.py"
            )

        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim == 2:
            seq = seq[np.newaxis, ...]  # Add batch dim -> (1, T, D)

        # Pad or slice to expected sequence length
        b, t, d = seq.shape
        if t != SEQUENCE_LENGTH or d != FEATURE_DIM:
            # Resample sequence
            resampled = np.zeros((b, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
            from core.landmarks import resample_sequence
            for i in range(b):
                resampled[i] = resample_sequence(seq[i], SEQUENCE_LENGTH)
            seq = resampled

        x_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            # Temperature scaling for calibrated probability estimation
            scaled_logits = logits / self.temperature
            probs = torch.softmax(scaled_logits, dim=-1).cpu().numpy()[0]

        top_indices = np.argsort(probs)[::-1][:min(top_k, len(self.labels))]
        return [(self.labels[i], float(probs[i])) for i in top_indices]

    def predict_distribution(self, sequence):
        """
        Returns full probability distribution dictionary and candidate ranking.
        """
        if not self.available:
            raise RuntimeError("Model not loaded.")
        candidates = self.predict(sequence, top_k=len(self.labels))
        dist = {label: prob for label, prob in candidates}
        return dist, candidates
