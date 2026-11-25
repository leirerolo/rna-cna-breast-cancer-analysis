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
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#import filters.py and regression_data.py from the Preprocessing folder
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
N_FOLDS = 5          
N_TRIALS = 30     
#K_MAX = maximum number of selected genes
#we set it to 300, because it must be < the number of training patients per fold (~480)
#otherwise, training set would be memorised   
K_MAX = 300          
CNA_KIND = "disc" #our decision to enhance interpretability, in the same line of classification

# ==================================================================================
# DATA
# ==================================================================================
data = load_regression_data(cna_kind=CNA_KIND)

datasets = data["datasets"]
y = data["y"]
is_train = data["is_train"]
patient_ids = data["patient_ids"]

y_train, y_test = y[is_train], y[~is_train]

#patient IDs for train and test
groups_train = patient_ids[is_train].to_numpy()
groups_test = patient_ids[~is_train].to_numpy()

print(f"Samples with TMB -> train: {is_train.sum()} | test: {(~is_train).sum()}")
print(f"Patients -> train: {len(np.unique(groups_train))} | test: {len(np.unique(groups_test))}")

#check that no patient appears in both train and test
assert set(groups_train).isdisjoint(set(groups_test)), \
    "A patient appears in both train and test!"


# ==================================================================================
# SEARCH SPACE: how many attributes to keep and how to reduce them
# ==================================================================================
def suggest_linear(trial, k_max):
    reducer = trial.suggest_categorical("reducer", ["kbest", "pca"])
    k = trial.suggest_int("k", 5, k_max, log=True)

    if reducer == "kbest":
        return {"reduce": SelectKBest(f_regression, k=k)}
    return {"reduce": PCA(n_components=k, random_state=SEED)}


# ==================================================================================
# COMBINATIONS
# ==================================================================================
def cna_filter():
    #we use the following function from filters.py to filter near-constant genes
    return ModalFrequencyFilter(threshold=0.95) if CNA_KIND == "disc" else "passthrough"

def tail():
    #shared steps for every scenario: scaling, reduction and the model itself
    return [
        ("scale", StandardScaler()),
        #in reality, we are now defining reduction step, which starts with the best form
        #selected by the previous function. However, in training stage, the one selected by optuna
        #will be the one we will use. It's set now to predefine the steps
        ("reduce", SelectKBest(f_regression)),
        ("lr", LinearRegression()),
    ]

rna_cna = datasets["RNA+CNA"]

combinations = {
    "RNA": (datasets["RNA"], Pipeline(
        #use the filter from filters.py for RNA genes
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
cv = GroupKFold(n_splits=N_FOLDS)

def get_k_max(pipe, X_train, y_train, groups_train):
    min_features = float("inf")
    min_train_samples = float("inf")

    for train_idx, val_idx in cv.split(X_train, y_train, groups_train):
        #check that no patient is shared between train and validation
        fold_train_groups = set(groups_train[train_idx])
        fold_val_groups = set(groups_train[val_idx])

        assert fold_train_groups.isdisjoint(fold_val_groups), \
            "A patient appears in both training and validation!"

        #fit the first preprocessing step only on the fold training data
        filter_step = clone(pipe.steps[0][1])

        X_filtered = filter_step.fit_transform(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx]
        )

        n_features = X_filtered.shape[1] #number of features remaining after filtering
        min_features = min(min_features, n_features)
        min_train_samples = min(min_train_samples, len(train_idx))

    #safe maximum for both SelectKBest and PCA
    k_max = min(
        K_MAX,
        int(min_features),
        int(min_train_samples)
    )
    if k_max < 5:
        raise ValueError(
            f"Not enough features or training samples. "
            f"Maximum allowed k: {k_max}"
        )
    return k_max


def tune(pipe, X_train, y_train, groups_train, k_max):
    #optuna will minimize the mean RMSE over the training folds

    def objective(trial):
        params = suggest_linear(trial, k_max)
        model = clone(pipe).set_params(**params)
        
        res = cross_validate(
            model, X_train, y_train, cv=cv, groups=groups_train,
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

    #rebuild the best configuration and refit on the whole train set
    best_params = suggest_linear(optuna.trial.FixedTrial(study.best_params), k_max)
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

    #calculate the maximum valid number of features/components
    k_max = get_k_max(pipe, X_train, y_train, groups_train)
    print(f"Maximum allowed k: {k_max}")

    #hyperparameter optimization using patient-grouped CV
    model, best_params, fold_rmse, fold_r2, study = tune(
        pipe,
        X_train,
        y_train,
        groups_train,
        k_max
    )
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
        "patient_id": groups_test,
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

    #saved after every scenario, so an interruption does not lose finished ones
    pd.DataFrame(rows).to_csv(RESULTS / "linear_regression_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RESULTS / "linear_regression_cv_folds.csv", index=False)
    pd.concat(pred_rows).to_csv(RESULTS / "linear_regression_predictions.csv", index=False)


# ==================================================================================
# COMPARISON WITH CONSTANT BASELINE
# ==================================================================================
baseline = RESULTS / "regression_baseline.csv"

if baseline.exists():
    baseline_df = pd.read_csv(baseline)

    print("\n=== Comparison with constant baseline ===")

    # Best baseline according to RMSE
    best_baseline = baseline_df.loc[baseline_df["rmse"].idxmin()]

    print(f"\nBest baseline: {best_baseline['baseline']}")
    print(f"Baseline RMSE: {best_baseline['rmse']:.3f}")
    print(f"Baseline MAE:  {best_baseline['mae']:.3f}")
    print(f"Baseline R2:   {best_baseline['r2']:.3f}")

    results_df = pd.DataFrame(rows)

    # Compare every scenario with the best baseline
    results_df["beats_baseline_rmse"] = (
        results_df["test_rmse"] < best_baseline["rmse"]
    )

    results_df["rmse_improvement"] = (
        best_baseline["rmse"] - results_df["test_rmse"]
    )

    results_df["rmse_improvement_pct"] = (
        results_df["rmse_improvement"] / best_baseline["rmse"] * 100
    )

    print("\n=== Linear Regression vs baseline ===")

    print(results_df[[
        "scenario",
        "test_rmse",
        "test_mae",
        "test_r2",
        "beats_baseline_rmse",
        "rmse_improvement_pct"
    ]].round(3).to_string(index=False))

    # Save comparison
    results_df.to_csv(
        RESULTS / "linear_regression_results.csv",
        index=False
    )

else:
    print("\nBaseline file not found. Run regression_baseline.py first.")

print(f"\nStored in {RESULTS} and {MODELS}")