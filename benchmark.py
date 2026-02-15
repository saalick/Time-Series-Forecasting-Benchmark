"""
Time-series forecasting benchmark: same train/test split for all models.
Compares ARIMA, LSTM, Prophet, Transformer, moving average, and last-value baseline.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Repo root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.utils import evaluate
from models.arima_model import ARIMAForecaster
from models.lstm_model import LSTMForecaster
from models.prophet_model import ProphetForecaster
from models.transformer_model import TransformerForecaster


# ---------- Data ----------
def load_data(csv_path: str = None, target_column: str = "close") -> tuple:
    csv_path = csv_path or str(ROOT / "data" / "stock_data.csv")
    if not os.path.isfile(csv_path):
        # Try to generate
        gen = ROOT / "scripts" / "generate_data.py"
        if gen.exists():
            import subprocess
            subprocess.run([sys.executable, str(gen)], check=True, cwd=str(ROOT))
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    y = df[target_column].values
    dates = df["date"]
    return y, dates, df


# ---------- Baselines ----------
def baseline_last_value(y_train: np.ndarray, steps: int) -> np.ndarray:
    return np.full(steps, float(y_train[-1]))


def baseline_moving_average(y_train: np.ndarray, steps: int, window: int = 20) -> np.ndarray:
    last_ma = np.mean(y_train[-window:])
    return np.full(steps, float(last_ma))


# ---------- Run benchmark ----------
def run_benchmark(
    train_ratio: float = 0.8,
    csv_path: str = None,
    lstm_epochs: int = 30,
    transformer_epochs: int = 30,
) -> tuple:
    y, dates, df = load_data(csv_path)
    n = len(y)
    split = int(n * train_ratio)
    y_train, y_test = y[:split], y[split:]
    dates_train = dates[:split]
    test_steps = len(y_test)

    results = {}
    predictions = {}

    # 1) Last value
    pred_lv = baseline_last_value(y_train, test_steps)
    predictions["Last value"] = pred_lv
    results["Last value"] = evaluate(y_test, pred_lv)

    # 2) Moving average
    pred_ma = baseline_moving_average(y_train, test_steps)
    predictions["Moving average"] = pred_ma
    results["Moving average"] = evaluate(y_test, pred_ma)

    # 3) ARIMA
    try:
        arima = ARIMAForecaster()
        arima.fit(y_train)
        pred_arima = arima.predict(test_steps)
        predictions["ARIMA"] = pred_arima
        results["ARIMA"] = evaluate(y_test, pred_arima)
    except Exception as e:
        results["ARIMA"] = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "Directional_Accuracy": np.nan}
        predictions["ARIMA"] = np.full(test_steps, np.nan)
        print(f"ARIMA failed: {e}")

    # 4) LSTM
    try:
        lstm = LSTMForecaster(lookback=20, epochs=lstm_epochs, verbose=0)
        lstm.fit(y_train)
        pred_lstm = lstm.predict(test_steps)
        predictions["LSTM"] = pred_lstm
        results["LSTM"] = evaluate(y_test, pred_lstm)
    except Exception as e:
        results["LSTM"] = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "Directional_Accuracy": np.nan}
        predictions["LSTM"] = np.full(test_steps, np.nan)
        print(f"LSTM failed: {e}")

    # 5) Prophet (needs dates for fit)
    try:
        prophet = ProphetForecaster()
        prophet.fit(y_train, dates=dates_train)
        pred_prophet = prophet.predict(test_steps)
        predictions["Prophet"] = pred_prophet
        results["Prophet"] = evaluate(y_test, pred_prophet)
    except Exception as e:
        results["Prophet"] = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "Directional_Accuracy": np.nan}
        predictions["Prophet"] = np.full(test_steps, np.nan)
        print(f"Prophet failed: {e}")

    # 6) Transformer
    try:
        trans = TransformerForecaster(lookback=20, epochs=transformer_epochs, verbose=0)
        trans.fit(y_train)
        pred_trans = trans.predict(test_steps)
        predictions["Transformer"] = pred_trans
        results["Transformer"] = evaluate(y_test, pred_trans)
    except Exception as e:
        results["Transformer"] = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "Directional_Accuracy": np.nan}
        predictions["Transformer"] = np.full(test_steps, np.nan)
        print(f"Transformer failed: {e}")

    return results, predictions, y_test, dates[split:].reset_index(drop=True), df


def save_metrics(results: dict, path: str) -> None:
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name, **metrics}
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_comparison(
    dates_test: pd.Series,
    y_test: np.ndarray,
    predictions: dict,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(y_test))
    ax.plot(x, y_test, "k-", label="Actual", linewidth=2)

    for name, pred in predictions.items():
        if np.any(np.isfinite(pred)):
            ax.plot(x, pred, "--", label=name, alpha=0.8)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Close price")
    ax.set_title("Time-Series Forecasting Benchmark: All Models vs Actual")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print("Running benchmark (same train/test split for all models)...")
    results, predictions, y_test, dates_test, _ = run_benchmark(
        lstm_epochs=30,
        transformer_epochs=30,
    )

    # Metrics table
    metrics_path = results_dir / "metrics_table.csv"
    save_metrics(results, str(metrics_path))
    print(f"Metrics saved to {metrics_path}")

    # Comparison plot
    plot_path = results_dir / "model_comparison.png"
    plot_comparison(dates_test, y_test, predictions, str(plot_path))
    print(f"Plot saved to {plot_path}")

    # Print summary
    print("\n--- Metrics summary ---")
    metrics_df = pd.DataFrame(results).T
    print(metrics_df.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
