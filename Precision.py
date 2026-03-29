"""
precision.py — Model Precision Evaluator
Calculates multiple metrics to assess the quality of the trained linear regression.
Requires thetas.json (run Train.py first) and data.csv.
"""

import csv
import json
import os


def load_thetas(filepath: str = "thetas.json") -> tuple[float, float]:
    if not os.path.exists(filepath):
        print("No trained model found. Run Train.py first.")
        exit(1)
    with open(filepath) as f:
        data = json.load(f)
    return data.get("theta0", 0.0), data.get("theta1", 0.0)


def read_dataset(filepath: str) -> tuple[list[float], list[float]]:
    mileages, prices = [], []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mileages.append(float(row["km"]))
            prices.append(float(row["price"]))
    return mileages, prices


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    return theta0 + theta1 * mileage


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def compute_metrics(mileages, prices, theta0, theta1) -> dict:
    m = len(mileages)
    predictions = [estimate_price(km, theta0, theta1) for km in mileages]
    residuals   = [p - pred for p, pred in zip(prices, predictions)]
    p_mean      = mean(prices)

    # Mean Absolute Error
    mae = sum(abs(r) for r in residuals) / m

    # Mean Squared Error
    mse = sum(r ** 2 for r in residuals) / m

    # Root Mean Squared Error
    rmse = mse ** 0.5

    # R² (coefficient of determination) — 1.0 = perfect fit
    ss_res = sum(r ** 2 for r in residuals)
    ss_tot = sum((p - p_mean) ** 2 for p in prices)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # Mean Absolute Percentage Error
    mape = mean([abs(r / p) * 100 for r, p in zip(residuals, prices) if p != 0])

    return {
        "MAE":  mae,
        "MSE":  mse,
        "RMSE": rmse,
        "R2":   r2,
        "MAPE": mape,
        "predictions": predictions,
        "residuals":   residuals,
    }


def bar(value: float, max_val: float, width: int = 30) -> str:
    filled = int(round(value / max_val * width)) if max_val else 0
    return "█" * filled + "░" * (width - filled)


def main():
    theta0, theta1 = load_thetas()
    mileages, prices = read_dataset("data.csv")
    m = len(mileages)

    metrics = compute_metrics(mileages, prices, theta0, theta1)

    print("=" * 52)
    print("  MODEL PRECISION REPORT")
    print("=" * 52)
    print(f"  Samples          : {m}")
    print(f"  theta0           : {theta0:>12.4f}")
    print(f"  theta1           : {theta1:>12.6f}")
    print()
    print(f"  MAE  (mean abs error)      : ${metrics['MAE']:>8.2f}")
    print(f"  RMSE (root mean sq error)  : ${metrics['RMSE']:>8.2f}")
    print(f"  MSE  (mean squared error)  : ${metrics['MSE']:>8.2f}")
    print(f"  MAPE (mean abs % error)    :  {metrics['MAPE']:>7.2f}%")
    print()
    print(f"  R² score  : {metrics['R2']:.4f}  {bar(max(0, metrics['R2']), 1.0)}")
    print(f"  Accuracy  : {max(0, 100 - metrics['MAPE']):.2f}%")
    print()

    # Per-sample breakdown
    print("-" * 52)
    print(f"  {'km':>8}  {'actual':>8}  {'predicted':>9}  {'error':>8}")
    print("-" * 52)
    for km, price, pred in zip(mileages, prices, metrics["predictions"]):
        err = price - pred
        print(f"  {km:>8.0f}  ${price:>7.0f}  ${pred:>8.0f}  {'+' if err >= 0 else ''}{err:>7.0f}")
    print("=" * 52)

    r2 = metrics["R2"]
    if r2 >= 0.95:
        verdict = "Excellent fit"
    elif r2 >= 0.85:
        verdict = "Good fit"
    elif r2 >= 0.70:
        verdict = "Acceptable fit"
    else:
        verdict = "Poor fit — consider more features"
    print(f"  Verdict: {verdict} (R²={r2:.4f})")
    print("=" * 52)


if __name__ == "__main__":
    main()