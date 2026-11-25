"""
REGRESSION: 
- inspection of target feature
- constant baseline for log1p(TMB_NONSYNONYMOUS)
- metadata lines of the clinical file with the TMB field definition, coverage and distribution

Every model must beat this computed baseline
- always predict a constant learned on train (mean or median of log1p(TMB)), evaluated on test.

Output: results/regression_baseline.csv
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Preprocessing"))
from regression_data import TARGET, load_regression_data, regression_metrics

RAW = ROOT / "data" / "raw" / "brca_tcga_pan_can_atlas_2018"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


# ==================================================================================
# 1) DEFINITION OF THE FIELD (the 4 lines with '#' above the header)
# ==================================================================================
print("=== Definition of", TARGET, "===")
try:
    raw = (RAW / "data_clinical_sample.txt").read_text(encoding="utf-8").splitlines()
    header = raw[4].split("\t") #first line without '#'
    j = header.index(TARGET)

    for k, label in enumerate(["name", "description", "datatype", "priority"]):
        print(f"{label}: {raw[k].lstrip('#').split(chr(9))[j]}")

except Exception as error: 
    print("Could not read the definition:", error)


# ==================================================================================
# 2) COVERAGE AND DISTRIBUTION
# ==================================================================================
data = load_regression_data()
y, tmb, is_train, subtype = data["y"], data["tmb_raw"], data["is_train"], data["subtype"]

print("\n=== Coverage ===")
print(f"Samples with TMB: {len(y)} of 945 | train: {is_train.sum()} | test: {(~is_train).sum()}")
print("Per subtype (train/test):")
print(pd.crosstab(subtype, np.where(is_train, "train", "test")))

print("\n=== Distribution ===")
print("TMB (raw):")
print(tmb.describe().round(2).to_string())
print(f"skewness raw: {tmb.skew():.2f} | zeros: {(tmb == 0).sum()} | above 20: {(tmb > 20).sum()}")
print("\nlog1p(TMB):")
print(y.describe().round(2).to_string())
print(f"skewness log1p: {y.skew():.2f}")
print("\nMedian TMB by subtype:")
print(tmb.groupby(subtype).median().round(2).to_string())


# ==================================================================================
# 3) CONSTANT BASELINE (learned on train, evaluated on test)
# ==================================================================================
y_train, y_test = y[is_train], y[~is_train]
constants = {
    "constant_mean_train": y_train.mean(),
    "constant_median_train": y_train.median(),
}
rows = []

print("\n=== Constant baseline on test (log1p scale) ===")
for name, c in constants.items():
    metrics = regression_metrics(y_test, np.full(len(y_test), c))
    rows.append({"baseline": name, "constant": float(c), **metrics})
    print(f"{name}: constant = {c:.3f} | RMSE {metrics['rmse']:.3f} | "
          f"MAE {metrics['mae']:.3f} | R2 {metrics['r2']:.3f}")

#store the baseline
pd.DataFrame(rows).to_csv(RESULTS / "regression_baseline.csv", index=False)
print(f"\nStored in {RESULTS / 'regression_baseline.csv'}")