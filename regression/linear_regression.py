"""
REGRESSION:
Linear regression for log1p(TMB_NONSYNONYMOUS)
- combinations: RNA, CNA and RNA + CNA
- same patient split for all of them (patient_split.csv, with regression_data.py)
- hyperparameters optimised with Optuna
- minimising RMSE with 5-fold cross-validation on the train set only

genes >> patients, so least squares could overfit training set and generalize badly.
- so we will reduce the attributes before training the model, inside the pipeline

steps:
1) filter
2) standardization
3) select best k or PCA
4) linear regression model

"""

from pathlib import Path
import sys
import time
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# filters.py and regression_data.py live in the Preprocessing folder
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Preprocessing"))
from filters import ModalFrequencyFilter, ZeroFractionFilter
from regression_data import load_regression_data, regression_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==================================================================================
# PATHS AND SETTINGS
# ==================================================================================
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

SEED = 42
N_FOLDS = 5          # 5-fold cross-validation on the train set
N_TRIALS = 30        # Optuna trials per scenario
K_MAX = 300          # maximum number of selected genes / components (must stay below the
                     # number of training patients of each fold, ~480)
CNA_KIND = "disc"    # 'disc' (as in classification) or 'log2'. GROUP DECISION: keep it
                     # the same in every regression model

# ==================================================================================
# DATA (shared with the other regression models)
# ==================================================================================
data = load_regression_data(cna_kind=CNA_KIND)
datasets, y, is_train = data["datasets"], data["y"], data["is_train"]
y_train, y_test = y[is_train], y[~is_train]
print(f"Samples with TMB -> train: {is_train.sum()} | test: {(~is_train).sum()}")


# ==================================================================================
# SEARCH SPACE: how many attributes to keep and how to reduce them
# ==================================================================================
def suggest_linear(trial):
    reducer = trial.suggest_categorical("reducer", ["kbest", "pca"])
    k = trial.suggest_int("k", 5, K_MAX, log=True)
    if reducer == "kbest":
        return {"reduce": SelectKBest(f_regression, k=k)}
    return {"reduce": PCA(n_components=k, random_state=SEED)}


# ==================================================================================
# COMBINATIONS: data + Pipeline
# ==================================================================================
def cna_filter():
    # the near-constant filter exists for the discrete CNA only
    return ModalFrequencyFilter(threshold=0.95) if CNA_KIND == "disc" else "passthrough"


def tail():
    """Steps shared by every scenario: scaling, reduction (replaced by Optuna) and model."""
    return [
        ("scale", StandardScaler()),
        ("reduce", SelectKBest(f_regression)),
        ("lr", LinearRegression()),
    ]


rna_cna = datasets["RNA+CNA"]
combinations = {
    "RNA": (datasets["RNA"], Pipeline(
        [("filter", ZeroFractionFilter(threshold=0.95))] + tail())),

    "CNA": (datasets["CNA"], Pipeline(
        [("filter", cna_filter())] + tail())),

    "RNA+CNA": (rna_cna, Pipeline(
        [("select", ColumnTransformer([
            ("rna", ZeroFractionFilter(threshold=0.95),
             [c for c in rna_cna if c.startswith("RNA_")]),
            ("cna", cna_filter(),
             [c for c in rna_cna if c.startswith("CNA_")]),
        ]))] + tail())),
}

# ==================================================================================
# OPTUNA TUNING (cross-validation on train samples only)
# ==================================================================================
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def tune(pipe, X_train, y_train):
    """Optuna minimising the mean RMSE over the folds (train only)."""

    def objective(trial):
        params = suggest_linear(trial)
        model = clone(pipe).set_params(**params)
        res = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
            error_score="raise",
        )
        rmse = -res["test_rmse"]
        trial.set_user_attr("fold_rmse", rmse.tolist())
        trial.set_user_attr("fold_r2", res["test_r2"].tolist())
        return rmse.mean()

    t0 = time.time()

    def log_trial(study, trial):
        print(f"  trial {trial.number + 1}/{N_TRIALS} | RMSE={trial.value:.3f} "
              f"| {trial.params} | {time.time() - t0:.0f}s", flush=True)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[log_trial])

    # rebuild the best configuration and refit on the whole train set
    best_params = suggest_linear(optuna.trial.FixedTrial(study.best_params))
    best_model = clone(pipe).set_params(**best_params).fit(X_train, y_train)
    attrs = study.best_trial.user_attrs
    return best_model, study.best_params, attrs["fold_rmse"], attrs["fold_r2"], study


# ==================================================================================
# TRAINING AND FINAL EVALUATION (test, once)
# ==================================================================================
rows, fold_rows, pred_rows = [], [], []

for comb, (X, pipe) in combinations.items():
    print(f"\n=== {comb} ===")
    X_train, X_test = X[is_train], X[~is_train]

    model, best_params, fold_rmse, fold_r2, study = tune(pipe, X_train, y_train)
    study.trials_dataframe().to_csv(RESULTS / f"linear_regression_trials_{comb}.csv", index=False)
    joblib.dump(model, MODELS / f"linear_regression_{comb.replace('+', '_')}.joblib")

    for i, (r, r2) in enumerate(zip(fold_rmse, fold_r2)):
        fold_rows.append({"scenario": comb, "model": "LinearRegression",
                          "fold": i, "rmse": r, "r2": r2})

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    train_m = regression_metrics(y_train, pred_train)
    test_m = regression_metrics(y_test, pred_test)

    pred_rows.append(pd.DataFrame({
        "scenario": comb, "sample_id": X_test.index,
        "y_true": y_test.to_numpy(), "y_pred": pred_test}))

    row = {
        "scenario": comb,
        "model": "LinearRegression",
        "n_trials": N_TRIALS,
        "cna_kind": CNA_KIND,
        "best_params": str(best_params),
        "cv_rmse_mean": float(np.mean(fold_rmse)),
        "cv_rmse_std": float(np.std(fold_rmse)),
        "cv_r2_mean": float(np.mean(fold_r2)),
        "train_rmse": train_m["rmse"],
        "train_r2": train_m["r2"],
        "test_rmse": test_m["rmse"],
        "test_mae": test_m["mae"],
        "test_r2": test_m["r2"],
        "test_spearman": test_m["spearman"],
    }
    rows.append(row)

    print("Best params:", best_params)
    print(f"CV RMSE: {row['cv_rmse_mean']:.3f} +/- {row['cv_rmse_std']:.3f} "
          f"| CV R2: {row['cv_r2_mean']:.3f}")
    print(f"TRAIN RMSE: {row['train_rmse']:.3f} | R2: {row['train_r2']:.3f}")
    print(f"TEST  RMSE: {row['test_rmse']:.3f} | MAE: {row['test_mae']:.3f} "
          f"| R2: {row['test_r2']:.3f} | Spearman: {row['test_spearman']:.3f}")

    # saved after every scenario, so an interruption does not lose finished ones
    pd.DataFrame(rows).to_csv(RESULTS / "linear_regression_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RESULTS / "linear_regression_cv_folds.csv", index=False)
    pd.concat(pred_rows).to_csv(RESULTS / "linear_regression_predictions.csv", index=False)

# reference: constant baseline (regression_baseline.py), if it has been run
baseline = RESULTS / "regression_baseline.csv"
if baseline.exists():
    print("\nConstant baseline on test:")
    print(pd.read_csv(baseline)[["baseline", "rmse", "mae", "r2"]].round(3).to_string(index=False))

print(f"\nStored in {RESULTS} and {MODELS}")