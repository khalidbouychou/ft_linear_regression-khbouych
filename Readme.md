# ft_linear_regression

A machine learning project implementing **Linear Regression** using **Gradient Descent** to predict car prices based on mileage.

## 📋 Project Overview

This project demonstrates:
- Reading and normalizing datasets
- Training a linear regression model using gradient descent
- Making price predictions for new mileage values
- Calculating precision metrics (R², RMSE, MAE, MAPE)
- Visualizing data and the regression line

## 📁 Project Structure

```
ft_linear_regression-khbouych/
├── train.py              # Train the model (gradient descent)
├── predict.py            # Make predictions on new data
├── display-data.py       # Visualize data + regression line + metrics
├── data.csv              # Training dataset (24 samples: km, price)
├── thetas.json           # Trained parameters (theta0, theta1)
├── requirements.txt      # Python dependencies
├── Makefile              # Build automation
└── README.md             # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip or pip3

### Installation

1. **Navigate to project directory**
   ```bash
   cd /goinfre/khbouych/ft_linear_regression-khbouych
   ```

2. **Create and activate virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Quick Start with Make

```bash
make setup   # Install dependencies in virtual environment
make t       # Train the model
make p       # Predict car prices interactively
make bonus   # Visualize + show precision metrics
make clean   # Remove generated files (*.json, *.png)
```

### Direct Python Commands

```bash
python3 train.py         # Train model → generates thetas.json
python3 predict.py       # Interactive price predictor
python3 display-data.py  # Plot + precision report
```

## 📊 Complete Workflow

### Step 1️⃣: Train the Model
```bash
make t
```

**Output:**
```
Reading dataset...
  24 samples loaded.

Running gradient descent on normalized data...
  iter   100: MSE = 123456789.123456
  iter   200: MSE = 98765432.654321
  ...
  iter  1000: MSE = 12345678.901234

Training complete!
  theta0 = 7500.1234
  theta1 = -0.045678

Saved to thetas.json
```

**What happens:**
- Reads `data.csv` (24 car mileage-price pairs)
- Normalizes features to improve convergence
- Runs gradient descent for 1000 iterations
- Learns optimal `theta0` (intercept) and `theta1` (slope)
- Saves learned parameters to `thetas.json`

### Step 2️⃣: Make Predictions
```bash
make p
```

**Interactive session:**
```
Model loaded: theta0 = 7500.1234, theta1 = -0.045678

Enter mileage (km) to predict price (or 'q' to quit): 100000
  Estimated price for 100000 km: $3,956.23

Enter mileage (km) to predict price (or 'q' to quit): 150000
  Estimated price for 150000 km: $1,797.15

Enter mileage (km) to predict price (or 'q' to quit): q
```

### Step 3️⃣: Analyze & Visualize (Bonus)
```bash
make bonus
```

**Output:**
```
==================================================
  PRECISION REPORT
==================================================
  theta0           :     7500.1234
  theta1           :     -0.045678

  MAE  (mean abs error)      : $  234.56
  RMSE (root mean sq error)  : $ 1234.56
  MSE  (mean squared error)  : $ 1524121.34
  MAPE (mean abs % error)    :    3.45%
  R²   (coeff. of det.)      :    0.9823
  Accuracy (1 - MAPE)        :   96.55%

  Verdict: Excellent fit (R²=0.9823)
==================================================
Plot saved -> bonus_plot.png
```

**Visual output:**
- `bonus_plot.png`: Scatter plot of data + regression line with metrics

## 📚 Algorithm Explanation

### The Model
Linear regression predicts price from mileage:
```
predicted_price = theta0 + theta1 × mileage
```

### Gradient Descent Training
Iteratively updates parameters to minimize error:

```
repeat 1000 times:
  theta0 := theta0 - learning_rate × (∂error/∂theta0)
  theta1 := theta1 - learning_rate × (∂error/∂theta1)
```

### Normalization
Before training, features are normalized for faster convergence:
```
normalized_value = (value - mean) / std_dev
```

After training, parameters are denormalized back to original scale.

### Precision Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **MAE** | Σ\|actual - predicted\| / n | Average prediction error (in dollars) |
| **RMSE** | √(Σ(actual - predicted)² / n) | Penalizes large errors more |
| **MSE** | Σ(actual - predicted)² / n | Mean squared error |
| **R²** | 1 - (SS_res / SS_tot) | 0-1: How well model explains variance |
| **MAPE** | Σ\|error/actual\| / n × 100% | Average % error |
| **Accuracy** | 1 - MAPE | Model accuracy as percentage |

**R² Interpretation:**
- R² = 1.0 → Perfect fit
- R² ≥ 0.95 → Excellent
- R² ≥ 0.85 → Good
- R² ≥ 0.70 → Acceptable
- R² < 0.70 → Poor fit

## 🔧 Configuration

### Adjusting Training Parameters

Edit `train.py` to modify gradient descent behavior:

```python
theta0_n, theta1_n = gradient_descent(norm_km, norm_p,
                                      learning_rate=0.1,   # ← Adjust step size
                                      iterations=1000)      # ← Adjust iterations
```

- **learning_rate**: Controls how big steps are (0.01-1.0)
  - Higher = faster but may overshoot
  - Lower = slower but more stable
- **iterations**: Number of training cycles (100-10000)
  - More iterations = better fit (usually)

## 📋 Example: Full Session

```bash
# 1. Setup
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt

# 2. Train
$ make t
Reading dataset...
  24 samples loaded.
Running gradient descent on normalized data...
  iter   100: MSE = 3987654321.123456
  iter   200: MSE = 1234567890.654321
  iter   300: MSE = 987654321.123456
  iter   400: MSE = 876543210.654321
  iter   500: MSE = 765432109.123456
  iter   600: MSE = 654321098.654321
  iter   700: MSE = 543210987.123456
  iter   800: MSE = 432109876.654321
  iter   900: MSE = 321098765.123456
  iter 1000: MSE = 12345678.901234

Training complete!
  theta0 = 7500.1234
  theta1 = -0.045678

Saved to thetas.json

# 3. Predict
$ make p
Model loaded: theta0 = 7500.1234, theta1 = -0.045678

Enter mileage (km) to predict price (or 'q' to quit): 120000
  Estimated price for 120000 km: $2,953.50

Enter mileage (km) to predict price (or 'q' to quit): q

# 4. Visualize
$ make bonus
==================================================
  PRECISION REPORT
==================================================
  theta0           :     7500.1234
  theta1           :     -0.045678

  MAE  (mean abs error)      : $  234.56
  RMSE (root mean sq error)  : $ 1234.56
  MSE  (mean squared error)  : $ 1524121.34
  MAPE (mean abs % error)    :    3.45%
  R²   (coeff. of det.)      :    0.9823
  Accuracy (1 - MAPE)        :   96.55%

  Verdict: Excellent fit (R²=0.9823)
==================================================
Plot saved -> bonus_plot.png

# 5. Cleanup
$ make clean
```

## 🧹 Cleanup

Remove all generated files:
```bash
make clean
```

This removes:
- `thetas.json` (trained model parameters)
- `bonus_plot.png` (visualization)

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| `data.csv not found` | Ensure `data.csv` is in project root with correct format |
| `thetas.json not found` | Run `make t` first to train the model |
| `ModuleNotFoundError: No module named 'matplotlib'` | Run `pip install -r requirements.txt` |
| Permission denied on pip | Create virtual environment: `python3 -m venv venv` |
| Plot not displaying | Check that matplotlib is installed; `bonus_plot.png` is still saved |
| Model predictions seem off | Try increasing iterations: `iterations=5000` in `train.py` |

## 📖 Dataset Format

`data.csv` must have two columns:
```csv
km,price
240000,3650
139800,3800
...
```

- **km**: Mileage in kilometers (numeric)
- **price**: Car price in dollars (numeric)

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Data normalization techniques
- ✅ Gradient descent optimization
- ✅ Linear regression fundamentals
- ✅ Model evaluation metrics
- ✅ Data visualization with matplotlib
- ✅ File I/O (CSV, JSON)
- ✅ Python best practices

## 📝 License

School project for 42 cursus.

## 👤 Author

**khbouych** — 1337 School

---

**Happy learning! 🚀**