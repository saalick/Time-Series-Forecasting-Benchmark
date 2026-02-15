# Time-Series Forecasting Benchmark

A research-oriented benchmark comparing **classical statistics** (ARIMA), **modern ML** (LSTM, Transformer), **structured forecasting** (Prophet), and **simple baselines** on financial time-series data.

## Methods Compared

| Method | Type | Strengths | When to Use |
|--------|------|-----------|-------------|
| **ARIMA** | Classical | Theoretically grounded, interpretable, assumes stationarity | Short-term, stationary series; need confidence intervals |
| **LSTM** | Deep learning | Captures long-range dependencies, flexible | Complex patterns, large data, non-stationary |
| **Prophet** | Decomposition | Handles seasonality & holidays, robust to missing data | Business/calendar effects, interpretable trends |
| **Transformer** | Deep learning | Attention over time, state-of-the-art potential | Long sequences, when you have enough data |
| **Moving average** | Baseline | Simple, no fit | Sanity check, very short horizon |
| **Last value** | Baseline | Minimal assumption | Naive benchmark |

## Research Angle

> *"Empirically, LSTMs often outperform ARIMA on financial data, but ARIMA is theoretically grounded in stationarity assumptions. Research question: Can we combine statistical rigor with ML flexibility?"*

This repo supports experiments toward hybrid or ensemble approaches.

## Repository Structure

```
time-series-forecasting-benchmark/
├── README.md
├── requirements.txt
├── data/
│   └── stock_data.csv
├── models/
│   ├── arima_model.py
│   ├── lstm_model.py
│   ├── prophet_model.py
│   └── transformer_model.py
├── notebooks/
│   └── comparative_analysis.ipynb
├── results/
│   ├── model_comparison.png
│   └── metrics_table.csv
└── benchmark.py
```

## Setup

```bash
cd time-series-forecasting-benchmark
pip install -r requirements.txt
```

## Usage

**Run full benchmark** (trains all models, evaluates, saves metrics and plots):

```bash
python benchmark.py
```

**Explore in Jupyter:**

```bash
jupyter notebook notebooks/comparative_analysis.ipynb
```

## Metrics

- **RMSE** – Root mean squared error (scale-dependent)
- **MAE** – Mean absolute error (robust to outliers)
- **MAPE** – Mean absolute percentage error (%)
- **Directional accuracy** – % of steps where predicted direction (up/down) matches actual

## Data

Default: `data/stock_data.csv` — date and closing price. Use your own CSV with columns `date` and `close` (or set `target_column` in code).

## Conclusion (Typical Findings)

- **ARIMA**: Good when series is near-stationary (e.g. returns); weak on strong trends.
- **LSTM**: Often best raw accuracy on price levels with enough history; needs tuning.
- **Prophet**: Good for daily data with weekly/yearly seasonality; interpretable.
- **Transformer**: Can match or beat LSTM with sufficient data and sequence length; more compute.
- **Baselines**: Last value and moving average set a floor; beating them is necessary but not sufficient.

## Author

**Salik Riyaz**

B.Tech in Artificial Intelligence & Data Science
Indian Institute of Technology, Jodhpur

This repository explores empirical performance trade-offs between statistical rigor and deep learning flexibility in financial time-series forecasting.

## License

MIT.
