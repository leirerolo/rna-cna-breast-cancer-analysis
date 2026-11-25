"""
LINEAR REGRESSION VISUALIZATIONS:
- Predicted vs actual values
- Residual plots
- RMSE comparison between scenarios and baseline
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================================================
# PATHS
# ==================================================================================
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

PREDICTIONS_FILE = RESULTS / "linear_regression_predictions.csv"
RESULTS_FILE = RESULTS / "linear_regression_results.csv"
BASELINE_FILE = RESULTS / "regression_baseline.csv"


# ==================================================================================
# LOAD RESULTS
# ==================================================================================
predictions = pd.read_csv(PREDICTIONS_FILE)
results = pd.read_csv(RESULTS_FILE)

scenarios = ["RNA", "CNA", "RNA+CNA"]


# ==================================================================================
# 1. PREDICTED VS ACTUAL VALUES
# ==================================================================================
for scenario in scenarios:

    data = predictions[predictions["scenario"] == scenario]

    y_true = data["y_true"]
    y_pred = data["y_pred"]

    plt.figure(figsize=(7, 6))

    plt.scatter(y_true, y_pred, alpha=0.6)

    # Ideal predictions: points should be close to this line
    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual log1p(TMB)")
    plt.ylabel("Predicted log1p(TMB)")
    plt.title(f"Predicted vs actual values - {scenario}")

    plt.tight_layout()
    plt.savefig(
        FIGURES / f"linear_regression_predicted_vs_actual_{scenario}.png",
        dpi=300
    )
    plt.close()


# ==================================================================================
# 2. RESIDUAL PLOTS
# ==================================================================================
for scenario in scenarios:

    data = predictions[predictions["scenario"] == scenario].copy()

    data["residual"] = data["y_true"] - data["y_pred"]

    plt.figure(figsize=(7, 6))

    plt.scatter(
        data["y_pred"],
        data["residual"],
        alpha=0.6
    )

    plt.axhline(y=0, linestyle="--")

    plt.xlabel("Predicted log1p(TMB)")
    plt.ylabel("Residuals")
    plt.title(f"Residual plot - {scenario}")

    plt.tight_layout()
    plt.savefig(
        FIGURES / f"linear_regression_residuals_{scenario}.png",
        dpi=300
    )
    plt.close()


# ==================================================================================
# 3. RMSE COMPARISON
# ==================================================================================
plt.figure(figsize=(8, 6))

plt.bar(
    results["scenario"],
    results["test_rmse"]
)

# Add the best constant baseline as a reference line
if BASELINE_FILE.exists():

    baseline = pd.read_csv(BASELINE_FILE)
    best_baseline = baseline.loc[baseline["rmse"].idxmin()]

    plt.axhline(
        y=best_baseline["rmse"],
        linestyle="--",
        label=f"Best baseline ({best_baseline['baseline']})"
    )

    plt.legend()

plt.ylabel("Test RMSE")
plt.xlabel("Scenario")
plt.title("Linear Regression - RMSE comparison")

plt.tight_layout()
plt.savefig(
    FIGURES / "linear_regression_rmse_comparison.png",
    dpi=300
)
plt.close()

print(f"Visualizations saved in {FIGURES}")
