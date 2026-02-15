"""
Prophet model for time-series forecasting.
Handles seasonality and trend; robust to missing data.
"""

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
except ImportError:
    Prophet = None


class ProphetForecaster:
    def __init__(self, yearly_seasonality: bool = True, weekly_seasonality: bool = True):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.model_ = None
        self.last_date_ = None

    def fit(self, y: np.ndarray, dates: pd.DatetimeIndex = None) -> "ProphetForecaster":
        if Prophet is None:
            raise ImportError("Install prophet: pip install prophet")
        n = len(y)
        if dates is None:
            dates = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame({"ds": dates, "y": np.asarray(y)})
        self.model_ = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
        )
        self.model_.fit(df)
        self.last_date_ = df["ds"].iloc[-1]
        return self

    def predict(self, steps: int) -> np.ndarray:
        future = pd.date_range(
            start=self.last_date_ + pd.Timedelta(days=1),
            periods=steps,
            freq="B",
        )
        future_df = pd.DataFrame({"ds": future})
        fc = self.model_.predict(future_df)
        return np.asarray(fc["yhat"].values)
