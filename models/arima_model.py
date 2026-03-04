"""
ARIMA model for time series forecasting.
Classical approach: assumes (or enforces via differencing) stationarity.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def _suggest_difference(series: np.ndarray, max_d: int = 2) -> int:
    """Suggest integration order d via ADF test."""
    for d in range(max_d + 1):
        if d == 0:
            y = series
        else:
            y = np.diff(series, n=d)
        if len(y) < 10:
            return max(0, d - 1)
        try:
            adf_result = adfuller(y, autolag="AIC")
            if adf_result[1] < 0.05:
                return d
        except Exception:
            pass
    return 1


def fit_arima(
    y: np.ndarray,
    order: tuple = None,
    seasonal_order: tuple = None,
) -> object:
    """
    Fit ARIMA model. If order is None, use (1, 0, 1) or (1, 1, 1) based on stationarity.
    """
    if order is None:
        d = _suggest_difference(y)
        order = (1, d, 1)
    model = ARIMA(y, order=order, seasonal_order=seasonal_order or (0, 0, 0, 0))
    return model.fit()


def predict_arima(model, steps: int) -> np.ndarray:
    """Forecast next `steps` points."""
    fc = model.forecast(steps=steps)
    return np.asarray(fc)


class ARIMAForecaster:
    """Wrapper for benchmark: fit on train, predict on test horizon."""

    def __init__(self, order: tuple = None):
        self.order = order
        self.model_ = None

    def fit(self, y: np.ndarray) -> "ARIMAForecaster":
        self.model_ = fit_arima(y, order=self.order)
        return self

    def predict(self, steps: int) -> np.ndarray:
        return predict_arima(self.model_, steps)
