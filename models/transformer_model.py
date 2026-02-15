"""
Transformer-based time-series forecasting.
Attention over past values; no explicit stationarity assumption.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_transformer(
    lookback: int,
    d_model: int = 32,
    num_heads: int = 4,
    ff_dim: int = 32,
    dropout: float = 0.1,
) -> keras.Model:
    inputs = keras.Input(shape=(lookback, 1))
    x = layers.Dense(d_model)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def create_sequences(y: np.ndarray, lookback: int) -> tuple:
    y = np.asarray(y, dtype=np.float32)
    X, targets = [], []
    for i in range(lookback, len(y)):
        X.append(y[i - lookback : i].reshape(-1, 1))
        targets.append(y[i])
    return np.array(X), np.array(targets)


class TransformerForecaster:
    def __init__(
        self,
        lookback: int = 20,
        d_model: int = 32,
        num_heads: int = 4,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 0,
    ):
        self.lookback = lookback
        self.d_model = d_model
        self.num_heads = num_heads
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None
        self.last_sequence_ = None

    def fit(self, y: np.ndarray) -> "TransformerForecaster":
        y = np.asarray(y, dtype=np.float32)
        X, targets = create_sequences(y, self.lookback)
        self.model_ = build_transformer(
            self.lookback,
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
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
