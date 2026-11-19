#Random Forest

"""
CLASSIFICATION:
Random Forest classifier for PAM50 subtypes (LumA, LumB, Basal, Her2)
- combinations: RNA, CNA (discrete), RNA+CNA
- same patient train/test split for all models

Hyperparameters: optimized with Optuna
- 30 trials per combination
- 5-fold stratified cross-validation on train only
- maximize macro-F1
- evaluate on test only after choosing the best parameters

The fitted models, hyperparameters and results are saved in results/.
"""

from pathlib import Path
import sys
import json
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Import the same filters used in Naive Bayes and CART
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Preprocessing"))
from filters import ModalFrequencyFilter, ZeroFractionFilter

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==================================================================================
# PATHS AND SETTINGS
# ==================================================================================
P = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
N_FOLDS = 5
N_TRIALS = 30

# ==================================================================================
# DATA LOADING --> rows are samples, splitting is by patient
# ==================================================================================
clinical = pd.read_csv(P / "clinical_945.csv",
                       usecols=["SAMPLE_ID", "PATIENT_ID", "SUBTYPE"],
                       index_col="SAMPLE_ID")
split = pd.read_csv(P / "patient_split.csv").set_index("PATIENT_ID")["split"]

rna = pd.read_parquet(P / "rna_log2.parquet")
cna_disc = pd.read_parquet(P / "cna_disc_raw.parquet")

for name, df in [("RNA", rna), ("CNA discrete", cna_disc)]:
    assert df.index.equals(clinical.index), f"{name}: samples differ from clinical table"
    assert np.isfinite(df.to_numpy()).all(), f"{name}: missing or infinite values"

assert set(np.unique(cna_disc.to_numpy())) <= {-2, -1, 0, 1, 2}, "Unexpected CNA values"
assert split.index.is_unique, "Repeated PATIENT_ID in patient_split.csv"
assert clinical["PATIENT_ID"].isin(split.index).all(), "Patients missing from split"
assert set(split) == {"train", "test"}, "Unexpected split values"
assert clinical["PATIENT_ID"].is_unique, "Multiple samples per patient: grouped CV is needed"

is_train = clinical["PATIENT_ID"].map(split).eq("train").to_numpy()
y = clinical["SUBTYPE"]
y_train, y_test = y[is_train], y[~is_train]
print(f"Train: {is_train.sum()} | Test: {(~is_train).sum()}")

# Prefix gene names so RNA and CNA columns are different in the combined dataset
rna_cna = pd.concat([rna.add_prefix("RNA_"), cna_disc.add_prefix("CNA_")], axis=1)

# ==================================================================================
# SEARCH SPACE --> hyperparameters for Random Forest
# ==================================================================================
def suggest_forest(trial):
    # number of trees in the forest
    n_estimators = trial.suggest_int("n_estimators", 100, 300, step=50)
    # number of features considered at each split
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    # maximum depth of each tree; None means no fixed depth limit
    max_depth = trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20])
    # minimum samples needed to split a node
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    # minimum samples in each leaf
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    return {
        "rf__n_estimators": n_estimators,
        "rf__max_features": max_features,
        "rf__max_depth": max_depth,
        "rf__min_samples_split": min_samples_split,
        "rf__min_samples_leaf": min_samples_leaf
    }

# ==================================================================================
# COMBINATIONS --> same filters as CART, no scaling for Random Forest
# ==================================================================================
combinations = {
    "RNA": (rna, Pipeline([
        ("filter", ZeroFractionFilter(threshold=0.95)),
        ("rf", RandomForestClassifier(criterion="gini", bootstrap=True,
                                      random_state=SEED, n_jobs=-1))
    ])),
    "CNA": (cna_disc, Pipeline([
        ("filter", ModalFrequencyFilter(threshold=0.95)),
        ("rf", RandomForestClassifier(criterion="gini", bootstrap=True,
                                      random_state=SEED, n_jobs=-1))
    ])),
    "RNA+CNA": (rna_cna, Pipeline([
        ("select", ColumnTransformer([
            ("rna", ZeroFractionFilter(threshold=0.95),
             [c for c in rna_cna if c.startswith("RNA_")]),
            ("cna", ModalFrequencyFilter(threshold=0.95),
             [c for c in rna_cna if c.startswith("CNA_")])
        ])),
        ("rf", RandomForestClassifier(criterion="gini", bootstrap=True,
                                      random_state=SEED, n_jobs=-1))
    ]))
}

# ==================================================================================
# OPTUNA TUNING --> 5-fold CV, only on training samples
# ==================================================================================
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def tune(pipe, X_train, y_train):
    def objective(trial):
        params = suggest_forest(trial)
        model = clone(pipe).set_params(**params)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring="f1_macro", error_score="raise")
        trial.set_user_attr("fold_scores", scores.tolist())
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)

    best_params = suggest_forest(optuna.trial.FixedTrial(study.best_params))
    best_model = clone(pipe).set_params(**best_params).fit(X_train, y_train)
    return best_model, study.best_params, study.best_trial.user_attrs["fold_scores"], study

# ==================================================================================
# TRAINING, SAVING AND FINAL EVALUATION
# ==================================================================================
rows, fold_rows = [], []

for comb, (X, pipe) in combinations.items():
    print(f"\n=== {comb} ===")
    X_train, X_test = X[is_train], X[~is_train]

    model, best_params, fold_scores, study = tune(pipe, X_train, y_train)

    # Save the complete fitted pipeline (including the gene filters)
    joblib.dump(model, RESULTS / f"random_forest_model_{comb}.joblib")

    # Save the best parameters separately, plus every Optuna trial
    with open(RESULTS / f"random_forest_best_params_{comb}.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)
    study.trials_dataframe().to_csv(RESULTS / f"random_forest_trials_{comb}.csv", index=False)

    for i, score in enumerate(fold_scores, start=1):
        fold_rows.append({"scenario": comb, "model": "RandomForest", "fold": i,
                          "macro_f1": float(score)})

    # Check how much importance each data type has in the combined forest
    if comb == "RNA+CNA":
        forest = model.named_steps["rf"]
        selector = model.named_steps["select"]
        rna_cols = selector.output_indices_["rna"]
        cna_cols = selector.output_indices_["cna"]
        importance = forest.feature_importances_
        print(f"Feature importance: RNA={importance[rna_cols].sum():.3f}, "
              f"CNA={importance[cna_cols].sum():.3f}")

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)
    classes = model.classes_
    row = {
        "scenario": comb,
        "model": "RandomForest",
        "n_trials": N_TRIALS,
        "best_params": str(best_params),
        "cv_macro_f1_mean": float(np.mean(fold_scores)),
        "cv_macro_f1_std": float(np.std(fold_scores)),
        "test_macro_f1": f1_score(y_test, pred, average="macro"),
        "test_mcc": matthews_corrcoef(y_test, pred),
        "test_accuracy": accuracy_score(y_test, pred),
        "test_auc_ovr": roc_auc_score(y_test, prob, multi_class="ovr",
                                       average="macro", labels=classes)
    }
    rows.append(row)

    matrix = confusion_matrix(y_test, pred, labels=classes)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        RESULTS / f"random_forest_confusion_matrix_{comb}.csv")

    print("Best params:", best_params)
    print(f"CV macro-F1: {row['cv_macro_f1_mean']:.3f} +/- {row['cv_macro_f1_std']:.3f}")
    print(f"TEST macro-F1: {row['test_macro_f1']:.3f} | MCC: {row['test_mcc']:.3f} "
          f"| acc: {row['test_accuracy']:.3f} | AUC ovr: {row['test_auc_ovr']:.3f}")
    print("Confusion matrix (rows = real, cols = predicted):", list(classes))
    print(matrix)

    # Save after each scenario, so completed results are kept if a later one fails
    pd.DataFrame(rows).to_csv(RESULTS / "random_forest_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RESULTS / "random_forest_cv_folds.csv", index=False)

print(f"\nModels, best parameters and results saved in {RESULTS}")