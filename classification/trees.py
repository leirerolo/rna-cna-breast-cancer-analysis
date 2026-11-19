"""
CLASSIFICATION: CART for the four BRCA molecular subtypes.
Compare RNA, discrete CNA and RNA+CNA with the same patient split.
Use the same evaluation procedure as the Naive Bayes experiment.
"""

from pathlib import Path
import sys

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# We use the same gene filters as in the Naive Bayes script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Preprocessing"))
from filters import ModalFrequencyFilter, ZeroFractionFilter

optuna.logging.set_verbosity(optuna.logging.WARNING)


# PATHS AND SETTINGS
P = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
N_TRIALS = 30

# DATA LOADING: rows = samples, split = by patient
clinical = pd.read_csv(
    P / "clinical_945.csv",
    usecols=["SAMPLE_ID", "PATIENT_ID", "SUBTYPE"],
    index_col="SAMPLE_ID"
)
split = pd.read_csv(P / "patient_split.csv").set_index("PATIENT_ID")["split"]

rna = pd.read_parquet(P / "rna_log2.parquet")
cna_disc = pd.read_parquet(P / "cna_disc_raw.parquet")

for name, df in [("RNA", rna), ("CNA discrete", cna_disc)]:
    if not df.index.equals(clinical.index):
        raise ValueError(f"{name}: samples differ from clinical data")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f"{name}: missing or infinite values")

if not set(np.unique(cna_disc.to_numpy())) <= {-2, -1, 0, 1, 2}:
    raise ValueError("Unexpected CNA discrete values")

if not split.index.is_unique:
    raise ValueError("Repeated patient IDs in the split")

sample_split = clinical["PATIENT_ID"].map(split)
if sample_split.isna().any() or set(sample_split) != {"train", "test"}:
    raise ValueError("Missing or invalid patient split assignments")

is_train = sample_split.eq("train").to_numpy()
y = clinical["SUBTYPE"]
y_train, y_test = y.iloc[np.flatnonzero(is_train)], y.iloc[np.flatnonzero(~is_train)]
print(f"Train samples: {is_train.sum()} | Test samples: {(~is_train).sum()}")

rna_cna = pd.concat([
    rna.add_prefix("RNA_"),
    cna_disc.add_prefix("CNA_")
], axis=1)

# PIPELINES: fit gene filters only on the training data of each CV fold
combinations = {
    "RNA": (rna, Pipeline([
        ("filter", ZeroFractionFilter(threshold=0.95)),
        ("tree", DecisionTreeClassifier(criterion="gini", random_state=SEED)),
    ])),
    "CNA": (cna_disc, Pipeline([
        ("filter", ModalFrequencyFilter(threshold=0.95)),
        ("tree", DecisionTreeClassifier(criterion="gini", random_state=SEED)),
    ])),
    "RNA+CNA": (rna_cna, Pipeline([
        ("select", ColumnTransformer([
            ("rna", ZeroFractionFilter(threshold=0.95),
             [c for c in rna_cna.columns if c.startswith("RNA_")]),
            ("cna", ModalFrequencyFilter(threshold=0.95),
             [c for c in rna_cna.columns if c.startswith("CNA_")]),
        ])),
        ("tree", DecisionTreeClassifier(criterion="gini", random_state=SEED)),
    ])),
}

# HYPERPARAMETER SEARCH: CART, as in the classroom notebook
def suggest_tree(trial):
    return {
        "tree__max_depth": trial.suggest_int("max_depth", 2, 15),
        "tree__min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "tree__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "tree__ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.02),
    }

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def tune(pipe, X_train, y_train):
    def objective(trial):
        params = suggest_tree(trial)
        model = clone(pipe).set_params(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="f1_macro", error_score="raise"
        )
        trial.set_user_attr("fold_scores", scores.tolist())
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=N_TRIALS)
    best_params = suggest_tree(optuna.trial.FixedTrial(study.best_params))
    best_model = clone(pipe).set_params(**best_params).fit(X_train, y_train)
    return best_model, study.best_params, study.best_trial.user_attrs["fold_scores"], study


# TRAINING AND FINAL TEST EVALUATION
rows, fold_rows = [], []

for comb, (X, pipe) in combinations.items():
    print(f"\n=== CART: {comb} ===")
    X_train, X_test = X.iloc[np.flatnonzero(is_train)], X.iloc[np.flatnonzero(~is_train)]
    model, best_params, fold_scores, study = tune(pipe, X_train, y_train)

    study.trials_dataframe().to_csv(RESULTS / f"cart_trials_{comb}.csv", index=False)
    for i, score in enumerate(fold_scores, start=1):
        fold_rows.append({
            "scenario": comb, "model": "CART", "fold": i, "macro_f1": score
        })

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)
    classes = model.classes_

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
            y_test, prob, multi_class="ovr", average="macro", labels=classes
        ),
    }
    rows.append(row)

    print("Best parameters:", best_params)
    print(f"CV macro-F1: {row['cv_macro_f1_mean']:.3f} +/- {row['cv_macro_f1_std']:.3f}")
    print(f"TEST macro-F1: {row['test_macro_f1']:.3f} | MCC: {row['test_mcc']:.3f} "
          f"| Accuracy: {row['test_accuracy']:.3f} | AUC OVR: {row['test_auc_ovr']:.3f}")
    print("Confusion matrix (rows = real, columns = predicted):", list(classes))

    matrix = confusion_matrix(y_test, pred, labels=classes)
    print(matrix)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        RESULTS / f"cart_confusion_matrix_{comb}.csv"
    )

    # Show the first three levels; a full tree with thousands of genes is hard to read.
    plt.figure(figsize=(16, 8))
    plot_tree(model.named_steps["tree"], max_depth=3, class_names=list(classes),
              filled=True, fontsize=7)
    plt.title(f"CART - {comb} (first 3 levels)")
    plt.tight_layout()
    plt.savefig(RESULTS / f"cart_tree_{comb}.png", dpi=150)
    plt.close()

pd.DataFrame(rows).to_csv(RESULTS / "cart_results.csv", index=False)
pd.DataFrame(fold_rows).to_csv(RESULTS / "cart_cv_folds.csv", index=False)
print(f"\nResults stored in {RESULTS}")
