"""
CLASSIFICATION:
CART decision tree classifier for PAM50 subtype (LumA, LumB, Basal, Her2)
- combinations: RNA, CNA (discrete) and RNA + CNA
- we use the same patient split for all of them (patient_split.csv)

Hyperparameters: optimized with Optuna
- maximising macro-F1 with 5-fold
- stratified cross-validation on the train set only
- test set used once at the end

Pipeline:
- gene filters inside the Pipeline, so they're refitted on the training of each fold
- therefore, no data leakage from feature filtering

Outputs in results/ folder
"""

from pathlib import Path
import sys
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

#we must import filters.py from the project
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
N_FOLDS = 5 #we use 5-fold stratified cross-validation
N_TRIALS = 30 #number of trials optuna will perform to look for the best hyperparameters


# ==================================================================================
# DATA LOADING --> rows = samples, the split is by patient
# ==================================================================================
clinical = pd.read_csv(P / "clinical_945.csv",
                       usecols=["SAMPLE_ID", "PATIENT_ID", "SUBTYPE"],
                       index_col="SAMPLE_ID")
split = pd.read_csv(P / "patient_split.csv").set_index("PATIENT_ID")["split"]

rna = pd.read_parquet(P / "rna_log2.parquet")
cna_disc = pd.read_parquet(P / "cna_disc_raw.parquet")

#check if the samples match
for name, df in [("RNA", rna), ("CNA discrete", cna_disc)]:
    assert df.index.equals(clinical.index), f"{name}: samples differ from clinical table"
    assert np.isfinite(df.to_numpy()).all(), f"{name}: missing or infinite values"

#check if cna discrete contains values -2, -1, 0, 1 and 2
assert set(np.unique(cna_disc.to_numpy())) <= {-2, -1, 0, 1, 2}, "Unexpected CNA discrete values"

#every patient must have a split assignment
assert split.index.is_unique, "Repeated PATIENT_ID in patient_split.csv"
assert clinical["PATIENT_ID"].isin(split.index).all(), "Patients missing from the split"
assert set(split) == {"train", "test"}, "Unexpected split values"

#we use stratified k-fold on samples, so each patient must have only one sample
assert clinical["PATIENT_ID"].is_unique, "Multiple samples per patient: grouped CV is needed"

#we identify the training samples with their patient id
is_train = clinical["PATIENT_ID"].map(split).eq("train").to_numpy()

#separate labels
y = clinical["SUBTYPE"] #y = subtype of each sample
y_train, y_test = y[is_train], y[~is_train]
print(f"Train: {is_train.sum()} | Test: {(~is_train).sum()}")

#RNA + CNA: we give the columns a prefix so they don't collide
#and combine both dfs horizontally
rna_cna = pd.concat([
    rna.add_prefix("RNA_"),
    cna_disc.add_prefix("CNA_")],
    axis=1)


# ==================================================================================
# SEARCH SPACE:
# defining the hyperparameters that optuna will try for CART
# ==================================================================================

def suggest_tree(trial):
    #max_depth controls how deep the tree can grow
    max_depth = trial.suggest_int("max_depth", 2, 15)

    #minimum number of samples needed to split a node
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

    #minimum number of samples in each leaf
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    #cost-complexity pruning: larger values may produce smaller trees
    ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.02)

    #the prefix tree__ means the parameter belongs to step tree in the pipeline
    return {
        "tree__max_depth": max_depth,
        "tree__min_samples_split": min_samples_split,
        "tree__min_samples_leaf": min_samples_leaf,
        "tree__ccp_alpha": ccp_alpha
    }


# =============================================================================
# COMBINATIONS
# =============================================================================
#for each combination, we define the data, preprocessing and CART classifier
#there is no StandardScaler because decision trees do not need feature scaling

combinations = {
    "RNA": (rna, Pipeline([
        ("filter", ZeroFractionFilter(threshold=0.95)),
        ("tree", DecisionTreeClassifier(criterion="gini", random_state=SEED)),
    ]), suggest_tree),

    "CNA": (cna_disc, Pipeline([
        ("filter", ModalFrequencyFilter(threshold=0.95)),
        ("tree", DecisionTreeClassifier(criterion="gini", random_state=SEED)),
    ]), suggest_tree),

    "RNA+CNA": (rna_cna, Pipeline([
        ("select", ColumnTransformer([
            ("rna", ZeroFractionFilter(threshold=0.95),
             [c for c in rna_cna if c.startswith("RNA_")]),
            ("cna", ModalFrequencyFilter(threshold=0.95),
             [c for c in rna_cna if c.startswith("CNA_")]),
        ])),
        ("tree", DecisionTreeClassifier(criterion="gini", random_state=SEED)),
    ]), suggest_tree)
}


# ====================================================================================
# OPTUNA TUNING (cross-validation just in training samples)
# ====================================================================================
cv = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True, #mix data before dividing it
    random_state=SEED
)


def tune(pipe, suggest, X_train, y_train):
    #optuna maximizing mean macro-F1 over the folds

    def objective(trial):
        params = suggest(trial) #suggest a hyperparameter combination
        model = clone(pipe).set_params(**params) #apply it

        #evaluate the model
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring="f1_macro",
            error_score="raise"
        )

        #store the individual scores of the 5 folds and return their mean
        trial.set_user_attr("fold_scores", scores.tolist())
        return scores.mean() #score that optuna will try to maximize

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=N_TRIALS) #30 trials with 5 folds each

    #get the best model and train it on the complete training set
    best_params = suggest(optuna.trial.FixedTrial(study.best_params))
    best_model = clone(pipe).set_params(**best_params).fit(X_train, y_train)

    return (
        best_model,
        study.best_params,
        study.best_trial.user_attrs["fold_scores"],
        study
    )


# ============================================================================================
# TRAINING & FINAL EVALUATION
# ============================================================================================
#final results of each combination and individual results of each fold
rows, fold_rows = [], []

for comb, (X, pipe, suggest) in combinations.items():
    print(f"\n=== {comb} ===")

    X_train, X_test = X[is_train], X[~is_train]

    #call tune to optimize hyperparameters + train the best model
    model, best_params, fold_scores, study = tune(
        pipe, suggest, X_train, y_train
    )

import joblib

joblib.dump(model, RESULTS / f"cart_model_{comb}.joblib")

if comb == "RNA+CNA":
    tree = model.named_steps["tree"]
    features = model.named_steps["select"].get_feature_names_out()

    used_features = features[tree.feature_importances_ > 0]

    print("RNA features used:",
          sum(f.startswith("rna__") for f in used_features))

    print("CNA features used:",
          sum(f.startswith("cna__") for f in used_features))
    
    #store the trials tested by optuna
    study.trials_dataframe().to_csv(
        RESULTS / f"cart_trials_{comb}.csv",
        index=False
    )

    #store scores of each fold
    for i, s in enumerate(fold_scores):
        fold_rows.append({
            "scenario": comb,
            "model": "CART",
            "fold": i,
            "macro_f1": s
        })

    #predict with test set
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)
    classes = model.classes_

    #calculate the metrics
    row = {
        "scenario": comb,
        "model": "CART",
        "n_trials": N_TRIALS,
        "best_params": str(best_params),
        "cv_macro_f1_mean": float(np.mean(fold_scores)),
        "cv_macro_f1_std": float(np.std(fold_scores)),
        "test_macro_f1": f1_score(y_test, pred, average="macro"),
        "test_mcc": matthews_corrcoef(y_test, pred),
        "test_accuracy": accuracy_score(y_test, pred),
        "test_auc_ovr": roc_auc_score(
            y_test, prob,
            multi_class="ovr",
            average="macro",
            labels=classes
        ),
    }
    rows.append(row)

    print("Best params:", best_params)
    print(
        f"CV macro-F1: {row['cv_macro_f1_mean']:.3f} "
        f"+/- {row['cv_macro_f1_std']:.3f}"
    )
    print(
        f"TEST macro-F1: {row['test_macro_f1']:.3f} "
        f"| MCC: {row['test_mcc']:.3f} "
        f"| acc: {row['test_accuracy']:.3f} "
        f"| AUC ovr: {row['test_auc_ovr']:.3f}"
    )

    print("Confusion matrix (rows = real, cols = predicted):", list(classes))
    matrix = confusion_matrix(y_test, pred, labels=classes)
    print(matrix)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        RESULTS / f"cart_confusion_matrix_{comb}.csv"
    )

pd.DataFrame(rows).to_csv(
    RESULTS / "cart_results.csv",
    index=False
)
pd.DataFrame(fold_rows).to_csv(
    RESULTS / "cart_cv_folds.csv",
    index=False
)

print(f"\nStored in {RESULTS}")
