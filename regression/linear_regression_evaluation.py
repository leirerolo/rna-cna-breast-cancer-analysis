"""
LINEAR REGRESSION EVALUATION:
Paired bootstrap comparison between scenarios:
- RNA+CNA vs RNA
- RNA+CNA vs CNA
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==================================================================================
# SETTINGS
# ==================================================================================
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

PREDICTIONS_FILE = RESULTS / "linear_regression_predictions.csv"

SEED = 42
N_BOOTSTRAP = 20000
CONFIDENCE = 0.95

rng = np.random.default_rng(SEED)


# ==================================================================================
# LOAD PREDICTIONS
# ==================================================================================
predictions = pd.read_csv(PREDICTIONS_FILE)

#each scenario must contain predictions for the same test samples
rna = predictions[predictions["scenario"] == "RNA"].copy()
cna = predictions[predictions["scenario"] == "CNA"].copy()
combined = predictions[predictions["scenario"] == "RNA+CNA"].copy()


# ==================================================================================
# ALIGN TEST SAMPLES
# ==================================================================================
def align_predictions(data1, data2):

    merged = data1[["sample_id", "y_true", "y_pred"]].merge(
        data2[["sample_id", "y_true", "y_pred"]],
        on="sample_id",
        suffixes=("_1", "_2")
    )

    #check that both models have the same real target values
    if not np.allclose(
        merged["y_true_1"],
        merged["y_true_2"]
    ):
        raise ValueError("The models do not have matching true values.")

    return merged


# ==================================================================================
# PAIRED BOOTSTRAP
# ==================================================================================
def bootstrap_comparison(data1, data2, name1, name2):

    merged = align_predictions(data1, data2)

    y_true = merged["y_true_1"].to_numpy()
    pred1 = merged["y_pred_1"].to_numpy()
    pred2 = merged["y_pred_2"].to_numpy()

    n = len(y_true)

    rmse_differences = []
    mae_differences = []

    for _ in range(N_BOOTSTRAP):

        #resample the same test samples for both models
        indexes = rng.integers(0, n, size=n)

        true_sample = y_true[indexes]
        pred1_sample = pred1[indexes]
        pred2_sample = pred2[indexes]

        rmse1 = np.sqrt(mean_squared_error(
            true_sample, pred1_sample
        ))

        rmse2 = np.sqrt(mean_squared_error(
            true_sample, pred2_sample
        ))

        mae1 = mean_absolute_error(
            true_sample, pred1_sample
        )

        mae2 = mean_absolute_error(
            true_sample, pred2_sample
        )

        # Negative difference means model 1 is better
        rmse_differences.append(rmse1 - rmse2)
        mae_differences.append(mae1 - mae2)

    rmse_differences = np.array(rmse_differences)
    mae_differences = np.array(mae_differences)

    alpha = 1 - CONFIDENCE

    rmse_ci = np.percentile(
        rmse_differences,
        [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )

    mae_ci = np.percentile(
        mae_differences,
        [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )

    return {
        "comparison": f"{name1} vs {name2}",
        "n_samples": n,
        "rmse_difference_mean": rmse_differences.mean(),
        "rmse_ci_lower": rmse_ci[0],
        "rmse_ci_upper": rmse_ci[1],
        "mae_difference_mean": mae_differences.mean(),
        "mae_ci_lower": mae_ci[0],
        "mae_ci_upper": mae_ci[1],
    }


# ==================================================================================
# RUN COMPARISONS
# ==================================================================================
results = []

results.append(
    bootstrap_comparison(
        combined, rna,
        "RNA+CNA", "RNA"
    )
)

results.append(
    bootstrap_comparison(
        combined, cna,
        "RNA+CNA", "CNA"
    )
)

results_df = pd.DataFrame(results)

# ==================================================================================
# SAVE RESULTS
# ==================================================================================
output_file = RESULTS / "linear_regression_bootstrap.csv"

results_df.to_csv(output_file, index=False)

print("\n=== Paired bootstrap comparison ===")
print(results_df.round(4).to_string(index=False))

print(f"\nResults saved in {output_file}")
