from pathlib import Path 
import sys 
import joblib 
import  numpy as npp 
import pandas as pd 
import matplotlib
matplotlib.use("Agg")   # backend sin ventana: solo genera imágenes
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import label_binarize

#=================================================================================================================================
# Path definition 
#=================================================================================================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT /"Preprocessing"))
from filters import ModalFrequencyFilter, ZeroFractionFilter

P = ROOT / "data" / "processed"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

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
# PR curves: a pannel for each scenario and a curve per subtype
#=================================================================================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, (comb, X) in zip(axes, datasets.items()):
    model = joblib.load(ROOT / "models" / f"svm_{comb.replace('+', '_')}.joblib")
    X_test = X[~is_train]
    prob = model.predict_proba(X_test)
    classes = model.classes_
    Y = label_binarize(y_test, classes=classes)

    for i, c in enumerate(classes):
        p, r, _ = precision_recall_curve(Y[:, i], prob[:, i])
        ap = average_precision_score(Y[:, i], prob[:, i])
        line, = ax.plot(r, p, label=f"{c} (AP={ap:.2f})")
        ax.axhline(Y[:, i].mean(), ls=":", lw=0.8, color=line.get_color())  # prevalencia = azar

    ax.set_title(f"SVM – {comb}")
    ax.set_xlabel("Recall")
    ax.legend(loc="lower left")
axes[0].set_ylabel("Precision")
fig.tight_layout()
fig.savefig(FIGS / "svm_pr_curves.png", dpi=200)

#=================================================================================================================================
# normalized per row confussion matrix 
#=================================================================================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (comb, X) in zip(axes, datasets.items()):
    model = joblib.load(ROOT / "models" / f"svm_{comb.replace('+', '_')}.joblib")
    ConfusionMatrixDisplay.from_estimator(
        model, X[~is_train], y_test, normalize="true",
        values_format=".2f", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"SVM – {comb}")
fig.tight_layout()
fig.savefig(FIGS / "svm_confusion_matrices.png", dpi=200)

plt.show()