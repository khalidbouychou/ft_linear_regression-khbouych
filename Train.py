"""
Train.py — Linear Regression Training with Gradient Descent
Reads data.csv, learns theta0 and theta1, saves them to thetas.json.
"""

import csv
import json
import os


def read_dataset(filepath: str) -> tuple[list[float], list[float]]:
    mileages, prices = [], []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mileages.append(float(row["km"]))
            prices.append(float(row["price"]))
    return mileages, prices


def normalize(values: list[float]) -> tuple[list[float], float, float]:
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    if std == 0:
        raise ValueError("Standard deviation is zero — cannot normalize.")
    return [(v - mean) / std for v in values], mean, std


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    return theta0 + theta1 * mileage


def compute_mse(mileages, prices, theta0, theta1) -> float:
    m = len(mileages)
    return sum((estimate_price(km, theta0, theta1) - p) ** 2
               for km, p in zip(mileages, prices)) / m


def gradient_descent(
    mileages: list[float],
    prices: list[float],
    learning_rate: float = 0.1,
    iterations: int = 1000,
) -> tuple[float, float]:
    theta0, theta1 = 0.0, 0.0
    m = len(mileages)

    for i in range(iterations):
        # Compute gradients simultaneously
        tmp_theta0 = learning_rate * (1 / m) * sum(
            estimate_price(km, theta0, theta1) - p
            for km, p in zip(mileages, prices)
        )
        tmp_theta1 = learning_rate * (1 / m) * sum(
            (estimate_price(km, theta0, theta1) - p) * km
            for km, p in zip(mileages, prices)
        )
        # Simultaneous update
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

        if (i + 1) % 100 == 0:
            mse = compute_mse(mileages, prices, theta0, theta1)
            print(f"  iter {i+1:>5}: MSE = {mse:.6f}")

    return theta0, theta1


def main():
    dataset_path = "data.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: '{dataset_path}' not found.")
        return

    print("Reading dataset...")
    mileages, prices = read_dataset(dataset_path)
    m = len(mileages)
    print(f"  {m} samples loaded.")

    # Normalize features and targets
    norm_km,  km_mean,  km_std  = normalize(mileages)
    norm_p,   p_mean,   p_std   = normalize(prices)

    print("\nRunning gradient descent on normalized data...")
    theta0_n, theta1_n = gradient_descent(norm_km, norm_p,
                                          learning_rate=0.1,
                                          iterations=1000)

    # Denormalize thetas back to original scale
    # price = theta0_n * p_std + p_mean  +  theta1_n * (km - km_mean)/km_std * p_std
    # => theta1_real = theta1_n * p_std / km_std
    # => theta0_real = theta0_n * p_std + p_mean - theta1_real * km_mean
    theta1_real = theta1_n * p_std / km_std
    theta0_real = theta0_n * p_std + p_mean - theta1_real * km_mean

    print(f"\nTraining complete!")
    print(f"  theta0 = {theta0_real:.4f}")
    print(f"  theta1 = {theta1_real:.6f}")

    # Save thetas
    with open("thetas.json", "w") as f:
        json.dump({"theta0": theta0_real, "theta1": theta1_real}, f, indent=2)
    print("\nSaved to thetas.json")


if __name__ == "__main__":
    main()