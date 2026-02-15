"""
LSTM model for time-series forecasting.
Captures long-range dependencies; no stationarity assumption.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm(
    lookback: int,
    n_features: int = 1,
    units: int = 32,
    dropout: float = 0.2,
) -> keras.Model:
    model = keras.Sequential([
        layers.LSTM(units, return_sequences=True, input_shape=(lookback, n_features)),
        layers.Dropout(dropout),
        layers.LSTM(units // 2, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def create_sequences(y: np.ndarray, lookback: int) -> tuple:
    """Create (X, y) for supervised learning. X: (samples, lookback, 1)."""
    y = np.asarray(y, dtype=np.float32)
    X, targets = [], []
    for i in range(lookback, len(y)):
        X.append(y[i - lookback : i].reshape(-1, 1))
        targets.append(y[i])
    return np.array(X), np.array(targets)


class LSTMForecaster:
    def __init__(
        self,
        lookback: int = 20,
        units: int = 32,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 0,
    ):
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None
        self.last_sequence_ = None  # last lookback values for multi-step

    def fit(self, y: np.ndarray) -> "LSTMForecaster":
        y = np.asarray(y, dtype=np.float32)
        X, targets = create_sequences(y, self.lookback)
        self.model_ = build_lstm(self.lookback, n_features=1, units=self.units)
        self.model_.fit(
            X, targets,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=0.1,
        )
        self.last_sequence_ = y[-self.lookback :].copy()
        return self

    def predict(self, steps: int) -> np.ndarray:
        preds = []
        seq = self.last_sequence_.copy()
        for _ in range(steps):
            x = seq[-self.lookback :].reshape(1, self.lookback, 1)
            p = self.model_.predict(x, verbose=0)[0, 0]
            preds.append(p)
            seq = np.append(seq, p)
        return np.array(preds, dtype=np.float32)
