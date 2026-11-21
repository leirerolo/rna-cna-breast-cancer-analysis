"""
CLASSIFICATION:
Rebuilds the Naive Bayes models from the best hyperparameters stored in
results/naive_bayes_results.csv (no new Optuna search) and saves them
with joblib in models/.

Use it when naive_bayes.py was run without saving the models.
At the end it checks that the test macro-F1 matches the stored one.
"""

from pathlib import Path
import ast
import sys
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# filters.py (and shift_cna) must be importable, also when the model is loaded later
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Preprocessing"))
from filters import ModalFrequencyFilter, ZeroFractionFilter, shift_cna

# ==================================================================================
# PATHS AND SETTINGS
# ==================================================================================
P = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

UNIFORM_PRIORS = [0.25, 0.25, 0.25, 0.25]

# ==================================================================================
# DATA LOADING (same as the training script)
# ==================================================================================
clinical = pd.read_csv(P / "clinical_945.csv",
                       usecols=["SAMPLE_ID", "PATIENT_ID", "SUBTYPE"],
                       index_col="SAMPLE_ID")
split = pd.read_csv(P / "patient_split.csv").set_index("PATIENT_ID")["split"]

rna = pd.read_parquet(P / "rna_log2.parquet")
cna_disc = pd.read_parquet(P / "cna_disc_raw.parquet")

for name, df in [("RNA", rna), ("CNA discrete", cna_disc)]:
    assert df.index.equals(clinical.index), f"{name}: samples differ from clinical table"
assert clinical["PATIENT_ID"].isin(split.index).all(), "Patients missing from the split"

is_train = clinical["PATIENT_ID"].map(split).eq("train").to_numpy()
y = clinical["SUBTYPE"]
y_train, y_test = y[is_train], y[~is_train]

rna_cna = pd.concat([rna.add_prefix("RNA_"), cna_disc.add_prefix("CNA_")], axis=1)


# ==================================================================================
# SEARCH SPACES (needed to turn the stored params back into Pipeline params)
# ==================================================================================
def suggest_gaussian(trial):
    var_smoothing = trial.suggest_float("var_smoothing", 1e-9, 1e-1, log=True)
    priors = trial.suggest_categorical("priors", ["empirical", "uniform"])
    return {
        "nb__var_smoothing": var_smoothing,
        "nb__priors": None if priors == "empirical" else UNIFORM_PRIORS,
    }


def suggest_categorical(trial):
    alpha = trial.suggest_float("alpha", 0.01, 5.0, log=True)
    fit_prior = trial.suggest_categorical("fit_prior", [True, False])
    return {"nb__alpha": alpha, "nb__fit_prior": fit_prior}


# ==================================================================================
# COMBINATIONS (same pipelines as the training script)
# ==================================================================================
combinations = {
    "RNA": (rna, Pipeline([
        ("filter", ZeroFractionFilter(threshold=0.95)),
        ("scale", StandardScaler()),
        ("nb", GaussianNB()),
    ]), suggest_gaussian),

    "CNA": (cna_disc, Pipeline([
        ("filter", ModalFrequencyFilter(threshold=0.95)),
        ("shift", FunctionTransformer(shift_cna)),
        ("nb", CategoricalNB(min_categories=5)),
    ]), suggest_categorical),

    "RNA+CNA": (rna_cna, Pipeline([
        ("select", ColumnTransformer([
            ("rna", ZeroFractionFilter(threshold=0.95),
             [c for c in rna_cna if c.startswith("RNA_")]),
            ("cna", ModalFrequencyFilter(threshold=0.95),
             [c for c in rna_cna if c.startswith("CNA_")]),
        ])),
        ("scale", StandardScaler()),
        ("nb", GaussianNB()),
    ]), suggest_gaussian),
}

# ==================================================================================
# REBUILD, SAVE AND CHECK
# ==================================================================================
saved = pd.read_csv(RESULTS / "naive_bayes_results.csv").set_index("scenario")

for comb, (X, pipe, suggest) in combinations.items():
    print(f"\n=== {comb} ===")
    X_train, X_test = X[is_train], X[~is_train]

    best = ast.literal_eval(saved.loc[comb, "best_params"])  # string -> dict
    params = suggest(optuna.trial.FixedTrial(best))
    model = clone(pipe).set_params(**params).fit(X_train, y_train)

    joblib.dump(model, MODELS / f"naive_bayes_{comb.replace('+', '_')}.joblib")

    f1_now = f1_score(y_test, model.predict(X_test), average="macro")
    f1_csv = saved.loc[comb, "test_macro_f1"]
    print(f"TEST macro-F1 now: {f1_now:.3f} | stored: {f1_csv:.3f}")
    if not np.isclose(f1_now, f1_csv, atol=1e-6):
        print("  WARNING: it does not match. The stored results probably come from a "
              "different version of the pipeline (e.g. before filtering the CNA in "
              "RNA+CNA). Rerun naive_bayes.py for this scenario before using the results.")

print(f"\nModels saved in {MODELS}")