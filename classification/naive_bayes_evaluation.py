"""
CLASSIFICATION:
Bootstrap comparison of the Naive Bayes scenarios on the test set.

For each comparison (RNA+CNA minus RNA, RNA+CNA minus CNA) it estimates the
difference in macro-F1 and a 95% confidence interval, resampling the test
patients with replacement inside each true subtype (stratified bootstrap).

The models are NOT retrained: it loads the saved models (models/ folder), so
it only captures the uncertainty due to the test sample, not the uncertainty
of retraining.

Output: results/naive_bayes_bootstrap_comparison.csv
"""

from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# ==================================================================================
# PATHS AND SETTINGS
# ==================================================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Preprocessing"))
import filters  # noqa: F401  (needed to load the saved pipelines: filters + shift_cna)

P = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODEL_NAME = "naive_bayes"
SEED = 42
N_BOOTSTRAP = 20_000

# ==================================================================================
# DATA (same as the training script)
# ==================================================================================
clinical = pd.read_csv(P / "clinical_945.csv",
                       usecols=["SAMPLE_ID", "PATIENT_ID", "SUBTYPE"],
                       index_col="SAMPLE_ID")
split = pd.read_csv(P / "patient_split.csv").set_index("PATIENT_ID")["split"]
rna = pd.read_parquet(P / "rna_log2.parquet")
cna_disc = pd.read_parquet(P / "cna_disc_raw.parquet")
rna_cna = pd.concat([rna.add_prefix("RNA_"), cna_disc.add_prefix("CNA_")], axis=1)

is_train = clinical["PATIENT_ID"].map(split).eq("train").to_numpy()
y_test = clinical["SUBTYPE"][~is_train]
test_ids = y_test.index

if not clinical.loc[test_ids, "PATIENT_ID"].is_unique:
    raise ValueError("Only one test sample per patient was expected.")

datasets = {"RNA": rna, "CNA": cna_disc, "RNA+CNA": rna_cna}

# ==================================================================================
# LOAD MODELS AND PREDICT THE SAME TEST PATIENTS
# ==================================================================================
labels, y_true = np.unique(y_test.to_numpy(), return_inverse=True)
predictions = {}

for scenario, X in datasets.items():
    path = MODELS / f"{MODEL_NAME}_{scenario.replace('+', '_')}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run {MODEL_NAME}_model_creation.py first.")
    model = joblib.load(path)

    predicted = model.predict(X.loc[test_ids])      # .loc keeps the order of y_test
    if not np.isin(predicted, labels).all():
        raise ValueError(f"{scenario}: predictions with unexpected labels.")
    predictions[scenario] = np.searchsorted(labels, predicted)

# ==================================================================================
# STRATIFIED BOOTSTRAP
# ==================================================================================
# indexes of the test patients grouped by their true subtype
groups = [np.flatnonzero(y_true == s) for s in np.unique(y_true)]


def macro_f1_difference(indexes, pred_a, pred_b):
    """macro-F1(model a) - macro-F1(model b) on the given test patients."""
    true = y_true[indexes]
    return (f1_score(true, pred_a[indexes], average="macro")
            - f1_score(true, pred_b[indexes], average="macro"))


comparisons = [
    ("RNA+CNA minus RNA", "RNA+CNA", "RNA"),
    ("RNA+CNA minus CNA", "RNA+CNA", "CNA"),
]

rng = np.random.default_rng(SEED)
all_idx = np.arange(len(y_true))
rows = []

for name, a, b in comparisons:
    pred_a, pred_b = predictions[a], predictions[b]
    observed = macro_f1_difference(all_idx, pred_a, pred_b)

    differences = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        differences[i] = macro_f1_difference(idx, pred_a, pred_b)

    low, high = np.quantile(differences, [0.025, 0.975])
    print(f"{name}: macro-F1 difference {observed:+.3f} | 95% CI [{low:+.3f}, {high:+.3f}]")

    rows.append({
        "model": MODEL_NAME,
        "comparison": name,
        "n_test_patients": len(y_true),
        "n_bootstrap": N_BOOTSTRAP,
        "delta_macro_f1": observed,
        "ci_95_lower": low,
        "ci_95_upper": high,
    })

output = RESULTS / f"{MODEL_NAME}_bootstrap_comparison.csv"
pd.DataFrame(rows).to_csv(output, index=False)
print(f"Results saved to {output}")