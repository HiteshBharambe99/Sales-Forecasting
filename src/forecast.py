"""
Monthly car sales forecasting.

Takes ~9 years of monthly car sales, does a bit of EDA, then compares a few
forecasting approaches on a held-out final year and forecasts the next 12
months. Writes charts and metrics to outputs/.

    python src/forecast.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "monthly-car-sales.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TEST_MONTHS = 12   # hold out the final year
N_LAGS = 12        # use the previous 12 months as features
HORIZON = 12       # how far ahead to forecast


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["Month"])
    df = df.sort_values("Month").reset_index(drop=True)
    df = df.rename(columns={"Sales": "sales"})
    print(f"{len(df)} months, {df['Month'].min():%Y-%m} to {df['Month'].max():%Y-%m}")
    return df


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def eda(df: pd.DataFrame) -> None:
    # full series
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Month"], df["sales"], color="#4C72B0")
    ax.set_title("Monthly car sales")
    ax.set_ylabel("Units sold")
    fig.tight_layout(); fig.savefig(FIG / "01_series.png", dpi=120); plt.close(fig)

    # seasonality: average by calendar month
    by_month = df.assign(m=df["Month"].dt.month).groupby("m")["sales"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(by_month.index, by_month.values, color="#4C72B0")
    ax.set_title("Average sales by calendar month (seasonality)")
    ax.set_xlabel("Month"); ax.set_ylabel("Avg units")
    ax.set_xticks(range(1, 13))
    fig.tight_layout(); fig.savefig(FIG / "02_seasonality.png", dpi=120); plt.close(fig)

    # year-on-year trend
    by_year = df.assign(y=df["Month"].dt.year).groupby("y")["sales"].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(by_year.index, by_year.values, marker="o", color="#DD8452")
    ax.set_title("Total sales per year (trend)")
    ax.set_xlabel("Year"); ax.set_ylabel("Units sold")
    fig.tight_layout(); fig.savefig(FIG / "03_yearly_trend.png", dpi=120); plt.close(fig)
    print("Saved EDA figures")


def make_supervised(series: np.ndarray, n_lags: int):
    """Turn a 1-D series into a lag-feature table for regression."""
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i - n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def seasonal_naive(train: np.ndarray, steps: int, season: int = 12) -> np.ndarray:
    """Baseline: repeat the value from one season (12 months) ago."""
    preds = []
    history = list(train)
    for _ in range(steps):
        preds.append(history[-season])
        history.append(history[-season])
    return np.array(preds)


def recursive_forecast(model, history: list[float], steps: int, n_lags: int) -> np.ndarray:
    """Forecast `steps` ahead, feeding each prediction back in as a new lag."""
    preds = []
    hist = list(history)
    for _ in range(steps):
        x = np.array(hist[-n_lags:]).reshape(1, -1)
        yhat = float(model.predict(x)[0])
        preds.append(yhat)
        hist.append(yhat)
    return np.array(preds)


def evaluate(df: pd.DataFrame) -> dict:
    sales = df["sales"].to_numpy(dtype=float)
    train, test = sales[:-TEST_MONTHS], sales[-TEST_MONTHS:]

    results = {}

    # 1. seasonal-naive baseline
    sn = seasonal_naive(train, TEST_MONTHS)
    results["seasonal_naive"] = sn

    # 2. linear regression on lag features
    Xtr, ytr = make_supervised(train, N_LAGS)
    lin = LinearRegression().fit(Xtr, ytr)
    results["linear_regression"] = recursive_forecast(lin, list(train), TEST_MONTHS, N_LAGS)

    # 3. random forest on lag features
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(Xtr, ytr)
    results["random_forest"] = recursive_forecast(rf, list(train), TEST_MONTHS, N_LAGS)

    metrics = {}
    for name, pred in results.items():
        metrics[name] = {
            "MAE": round(mean_absolute_error(test, pred), 1),
            "RMSE": round(float(np.sqrt(mean_squared_error(test, pred))), 1),
            "MAPE_pct": round(mape(test, pred), 2),
        }
    print("\nBacktest on final 12 months:")
    for name, m in metrics.items():
        print(f"  {name:20s} MAE={m['MAE']:>7}  RMSE={m['RMSE']:>7}  MAPE={m['MAPE_pct']}%")

    best = min(metrics, key=lambda k: metrics[k]["MAPE_pct"])
    print(f"\nBest model: {best} (lowest MAPE)")

    # plot actual vs each model over the test window
    test_months = df["Month"].to_numpy()[-TEST_MONTHS:]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Month"], df["sales"], color="#B0B0B0", label="history")
    ax.plot(test_months, test, color="black", marker="o", label="actual (test)")
    colors = {"seasonal_naive": "#8C8C8C", "linear_regression": "#4C72B0", "random_forest": "#DD8452"}
    for name, pred in results.items():
        ax.plot(test_months, pred, "--", marker=".", color=colors[name], label=name)
    ax.set_xlim(df["Month"].iloc[-TEST_MONTHS * 3], df["Month"].iloc[-1])
    ax.set_title("Backtest: forecasts vs actual (final 12 months)")
    ax.set_ylabel("Units sold"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "04_backtest.png", dpi=120); plt.close(fig)

    return metrics, best


def final_forecast(df: pd.DataFrame, best: str) -> pd.DataFrame:
    sales = df["sales"].to_numpy(dtype=float)

    if best == "seasonal_naive":
        fc = seasonal_naive(sales, HORIZON)
    else:
        X, y = make_supervised(sales, N_LAGS)
        model = (LinearRegression() if best == "linear_regression"
                 else RandomForestRegressor(n_estimators=300, random_state=42))
        model.fit(X, y)
        fc = recursive_forecast(model, list(sales), HORIZON, N_LAGS)

    future_months = pd.date_range(
        df["Month"].iloc[-1] + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS"
    )
    out = pd.DataFrame({"Month": future_months, "forecast": np.round(fc).astype(int)})
    out.to_csv(OUT / "forecast_next_12_months.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Month"], df["sales"], color="#4C72B0", label="history")
    ax.plot(future_months, fc, color="#DD8452", marker="o", label=f"forecast ({best})")
    ax.axvline(df["Month"].iloc[-1], color="grey", linestyle=":")
    ax.set_title("12-month forecast")
    ax.set_ylabel("Units sold"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "05_forecast.png", dpi=120); plt.close(fig)

    print(f"\nNext 12 months (using {best}):")
    print(out.to_string(index=False))
    return out


def main() -> None:
    df = load()
    eda(df)
    metrics, best = evaluate(df)
    final_forecast(df, best)
    with open(OUT / "metrics.json", "w") as f:
        json.dump({"backtest": metrics, "best_model": best}, f, indent=2)
    print(f"\nSaved metrics to {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
