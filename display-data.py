"""
bonus.py — Data Plot + Regression Line + Precision Metrics
Covers all three bonus requirements:
  1. Plot the data distribution
  2. Plot the regression line on the same graph
  3. Calculate and print the precision of the algorithm
Requires: thetas.json (run train.py first), data.csv, matplotlib
Run: python3 bonus.py
"""

import csv
import json
import os

import matplotlib.pyplot as plt


# ── helpers ───────────────────────────────────────────────────────────────────

def load_thetas(filepath="thetas.json"):
    if not os.path.exists(filepath):
        print("No trained model found. Run train.py first.")
        exit(1)
    with open(filepath) as f:
        data = json.load(f)
    return data.get("theta0", 0.0), data.get("theta1", 0.0)


def read_dataset(filepath="data.csv"):
    mileages, prices = [], []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mileages.append(float(row["km"]))
            prices.append(float(row["price"]))
    return mileages, prices


def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


def mean(values):
    return sum(values) / len(values)


# ── precision metrics ─────────────────────────────────────────────────────────

def compute_metrics(mileages, prices, theta0, theta1):
    m           = len(mileages)
    predictions = [estimate_price(km, theta0, theta1) for km in mileages]
    residuals   = [p - pred for p, pred in zip(prices, predictions)]
    p_mean      = mean(prices)

    mae    = sum(abs(r) for r in residuals) / m
    mse    = sum(r ** 2 for r in residuals) / m
    rmse   = mse ** 0.5
    ss_res = sum(r ** 2 for r in residuals)
    ss_tot = sum((p - p_mean) ** 2 for p in prices)
    r2     = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    mape   = mean([abs(r / p) * 100 for r, p in zip(residuals, prices) if p != 0])

    return {
        "MAE":  mae,
        "MSE":  mse,
        "RMSE": rmse,
        "R2":   r2,
        "MAPE": mape,
    }


def print_precision(theta0, theta1, metrics):
    print("=" * 52)
    print("  PRECISION REPORT")
    print("=" * 52)
    print(f"  theta0           : {theta0:>12.4f}")
    print(f"  theta1           : {theta1:>12.6f}")
    print()
    print(f"  MAE  (mean abs error)      : ${metrics['MAE']:>8.2f}")
    print(f"  RMSE (root mean sq error)  : ${metrics['RMSE']:>8.2f}")
    print(f"  MSE  (mean squared error)  : ${metrics['MSE']:>8.2f}")
    print(f"  MAPE (mean abs % error)    :  {metrics['MAPE']:>7.2f}%")
    print(f"  R²   (coeff. of det.)      :   {metrics['R2']:>7.4f}")
    print(f"  Accuracy (1 - MAPE)        :  {max(0, 100 - metrics['MAPE']):>7.2f}%")
    print()
    r2 = metrics["R2"]
    verdict = (
        "Excellent fit" if r2 >= 0.95 else
        "Good fit"      if r2 >= 0.85 else
        "Acceptable"    if r2 >= 0.70 else
        "Poor fit — consider more features"
    )
    print(f"  Verdict: {verdict} (R²={r2:.4f})")
    print("=" * 52)


# ── plot ──────────────────────────────────────────────────────────────────────

def plot(theta0, theta1, mileages, prices, metrics):
    x_min, x_max = min(mileages), max(mileages)
    pad    = (x_max - x_min) * 0.05
    line_x = [x_min - pad, x_max + pad]
    line_y = [estimate_price(x, theta0, theta1) for x in line_x]

    r2   = metrics["R2"]
    rmse = metrics["RMSE"]
    mae  = metrics["MAE"]
    mape = metrics["MAPE"]
    acc  = max(0, 100 - mape)

    plt.figure(figsize=(9, 6))

    # 1. data distribution
    plt.scatter(mileages, prices, color="#378ADD", s=60, label="Data", zorder=2)

    # 2. regression line
    plt.plot(line_x, line_y, color="red", linewidth=2,
             label=f"Linear Regression  (θ₀={theta0:.1f}, θ₁={theta1:.5f})",
             zorder=3)

    # 3. precision annotation
    box_text = (
        f"R²   = {r2:.4f}\n"
        f"RMSE = ${rmse:,.0f}\n"
        f"MAE  = ${mae:,.0f}\n"
        f"Acc  = {acc:.1f}%"
    )
    plt.gca().text(
        0.02, 0.05, box_text,
        transform=plt.gca().transAxes,
        fontsize=9, verticalalignment="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9)
    )

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price ($)")
    plt.title("Car Price vs Mileage")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bonus_plot.png", dpi=150)
    print("Plot saved -> bonus_plot.png")
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    theta0, theta1 = load_thetas()
    mileages, prices = read_dataset()
    metrics = compute_metrics(mileages, prices, theta0, theta1)
    print_precision(theta0, theta1, metrics)
    plot(theta0, theta1, mileages, prices, metrics)


if __name__ == "__main__":
    main()