"""Generate sample stock_data.csv if not present. Run from repo root: python scripts/generate_data.py"""

import os
import sys

import numpy as np
import pandas as pd

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(base, "data", "stock_data.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.random.seed(42)
    n = 504  # ~2 years of trading days
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    returns = np.random.randn(n) * 0.01 + 0.0002
    close = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({"date": dates, "close": np.round(close, 2)})
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(df)} rows.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
