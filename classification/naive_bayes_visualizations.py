"""
CLASSIFICATION:
Figures for the Naive Bayes models (loaded from the models/ folder):
- precision-recall curves: one panel per scenario, one curve per subtype
- confusion matrices normalised by row (recall per subtype)

Outputs: results/figures/naive_bayes_pr_curves.png
         results/figures/naive_bayes_confusion_matrices.png
"""

from pathlib import Path
import sys
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no window: it only writes image files
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             precision_recall_curve)
from sklearn.preprocessing import label_binarize

# ==================================================================================
# PATHS AND SETTINGS
# ==================================================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Preprocessing"))
import filters  # noqa: F401  (needed to load the saved pipelines: filters + shift_cna)

P = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "naive_bayes"
MODEL_LABEL = "Naive Bayes"

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

datasets = {"RNA": rna, "CNA": cna_disc, "RNA+CNA": rna_cna}


def load_model(scenario):
    path = MODELS / f"{MODEL_NAME}_{scenario.replace('+', '_')}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run {MODEL_NAME}_model_creation.py first.")
    return joblib.load(path)


# ==================================================================================
# PRECISION-RECALL CURVES
# ==================================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, (scenario, X) in zip(axes, datasets.items()):
    model = load_model(scenario)
    prob = model.predict_proba(X[~is_train])
    classes = model.classes_
    Y = label_binarize(y_test, classes=classes)

    for i, c in enumerate(classes):
        p, r, _ = precision_recall_curve(Y[:, i], prob[:, i])
        ap = average_precision_score(Y[:, i], prob[:, i])
        line, = ax.plot(r, p, label=f"{c} (AP={ap:.2f})")
        # dotted line = prevalence of the subtype = performance of a random classifier
        ax.axhline(Y[:, i].mean(), ls=":", lw=0.8, color=line.get_color())

    ax.set_title(f"{MODEL_LABEL} - {scenario}")
    ax.set_xlabel("Recall")
    ax.legend(loc="lower left")
axes[0].set_ylabel("Precision")
fig.tight_layout()
fig.savefig(FIGS / f"{MODEL_NAME}_pr_curves.png", dpi=200)
plt.close(fig)

# ==================================================================================
# CONFUSION MATRICES (normalised by row)
# ==================================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (scenario, X) in zip(axes, datasets.items()):
    model = load_model(scenario)
    ConfusionMatrixDisplay.from_estimator(
        model, X[~is_train], y_test, normalize="true",
        values_format=".2f", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"{MODEL_LABEL} - {scenario}")
fig.tight_layout()
fig.savefig(FIGS / f"{MODEL_NAME}_confusion_matrices.png", dpi=200)
plt.close(fig)

print(f"Figures saved in {FIGS}")