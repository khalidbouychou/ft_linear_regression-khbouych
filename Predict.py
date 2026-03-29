"""
Predict.py — Car Price Predictor
Loads theta0 and theta1 from thetas.json (set to 0 if not found),
prompts for mileage, returns estimated price.
"""

import json
import os


def load_thetas(filepath: str = "thetas.json") -> tuple[float, float]:
    if not os.path.exists(filepath):
        print("No trained model found (thetas.json missing). Using theta0=0, theta1=0.")
        return 0.0, 0.0
    with open(filepath) as f:
        data = json.load(f)
    return data.get("theta0", 0.0), data.get("theta1", 0.0)


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    return theta0 + theta1 * mileage


def main():
    theta0, theta1 = load_thetas()
    print(f"Model loaded: theta0 = {theta0:.4f}, theta1 = {theta1:.6f}\n")

    while True:
        raw = input("Enter mileage (km) to predict price (or 'q' to quit): ").strip()
        if raw.lower() == "q":
            break
        try:
            mileage = float(raw)
            if mileage < 0:
                print("  Mileage cannot be negative.\n")
                continue
            price = estimate_price(mileage, theta0, theta1)
            print(f"  Estimated price for {mileage:.0f} km: ${price:,.2f}\n")
        except ValueError:
            print("  Invalid input. Please enter a number.\n")


if __name__ == "__main__":
    main()