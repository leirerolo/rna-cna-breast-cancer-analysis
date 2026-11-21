from pathlib import Path 
import sys 
import joblib 
import  numpy as np
import pandas as pd 
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score)

#=================================================================================================================================
# Path definition 
#=================================================================================================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT /"Preprocessing"))

P = ROOT / "data" / "processed"


#=================================================================================================================================
# Data (same from the training script)
#=================================================================================================================================

clinical = pd.read_csv(P / "clinical_945.csv", usecols=["SAMPLE_ID", "PATIENT_ID", "SUBTYPE"], index_col="SAMPLE_ID")
split = pd.read_csv(P / "patient_split.csv").set_index("PATIENT_ID")["split"]
rna = pd.read_parquet(P / "rna_log2.parquet")
cna_disc = pd.read_parquet(P / "cna_disc_raw.parquet")
rna_cna = pd.concat([rna.add_prefix("RNA_"), cna_disc.add_prefix("CNA_")], axis=1)

is_train = clinical["PATIENT_ID"].map(split).eq("train").to_numpy()
y_test = clinical["SUBTYPE"][~is_train]

datasets = {"RNA": rna, "CNA": cna_disc, "RNA+CNA": rna_cna}

#=================================================================================================================================
# model loagind and predicting for the same patients
#=================================================================================================================================

sys.path.insert(0, str(ROOT / "Preprocessing"))

test_ids = y_test.index
test_patients = clinical.loc[test_ids, "PATIENT_ID"]

if not test_ids.is_unique or not test_patients.is_unique:
    raise ValueError("ONly one test sample per patient was expected.")

labels, y_true = np.unique(y_test.to_numpy(), return_inverse=True)
n_classes = len(labels)
predictions = {}

for scenario, X in datasets.items():
    model_name = f"svm_{scenario.replace('+', '_')}.joblib"
    model = joblib.load(ROOT / "models" / model_name)

    # .loc preserves the same sample order as y_test

    predicted = model.predict(X.loc[test_ids])
    if not np.isin(predicted, labels).all():
        raise ValueError(f"{scenario}: There are predictions with unextepcted labels.")

    predictions[scenario] = np.searchsorted(labels, predicted)


pred_rna = predictions["RNA"]
pred_rna_cna = predictions["RNA+CNA"]

#=================================================================================================================================
# indexes of the Test's patients, grouped by their real subtype. 
#=================================================================================================================================

groups = [
    np.flatnonzero(y_true == subtype)
    for subtype in np.unique(y_true)
]

def macro_f1_difference(indexes):
    labels = y_true[indexes]
    f1_rna = f1_score(labels, pred_rna[indexes], average="macro")
    f1_rna_cna = f1_score(labels, pred_rna_cna[indexes], average="macro")
    return f1_rna_cna - f1_rna

#=================================================================================================================================
#difference in the original test, usign each patients once. 
#=================================================================================================================================
observed_difference = macro_f1_difference(np.arange(len(y_true)))

rng = np.random.default_rng(42)
n_bootstrap = 20_000
differences = np.empty(n_bootstrap)

for i in range(n_bootstrap):
    indexes = np.concatenate([
        rng.choice(group, size=len(group), replace=True)
        for group in groups
    ])
    differences[i] = macro_f1_difference(indexes)

lower_threshold, upper_threshold = np.quantile(differences, [0.025, 0.975])

print(f"macro-F1 difference (RNA+CNA - RNA): {observed_difference:+.3f}")
print(f"95% confidence interval: [{lower_threshold:+.3f}, {upper_threshold:+.3f}]")


output = ROOT / "results" / "svm_bootstrap_comparison.csv"
output.parent.mkdir(parents=True, exist_ok=True)

summary = pd.DataFrame([{
    "comparison": "RNA+CNA minus RNA",
    "n_test_patients": len(y_true),
    "n_bootstrap": n_bootstrap,
    "delta_macro_f1": observed_difference,
    "ci_95_lower": lower_threshold,
    "ci_95_upper": upper_threshold,
}])

summary.to_csv(output, index=False)
print(f"Results saved to {output}")

#=================================================================================================================================
# Results storage
#=================================================================================================================================

output = ROOT / "results" / "svm_bootstrap_comparison.csv"
output.parent.mkdir(parents=True, exist_ok=True)

summary = pd.DataFrame([{
    "comparison": "RNA+CNA minus RNA",
    "n_test_patients": len(y_true),
    "n_bootstrap": n_bootstrap,
    "delta_macro_f1": observed_difference,
    "ci_95_lower": lower_threshold,
    "ci_95_upper": upper_threshold,
}])

summary.to_csv(output, index=False)
print(f"Results saved to {output}")