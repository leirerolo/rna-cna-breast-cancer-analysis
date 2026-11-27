#Bootstrap evaluation
"""
STATISTICAL EVALUATION WITH BOOTSTRAP:
Uses the fitted models saved in results/ (CART and Random Forest)
- same patient test set as in the training scripts
- stratified bootstrap of the test set: each class is resampled separately,
  so every bootstrap sample has all PAM50 subtypes (needed for the AUC)
- the SAME bootstrap samples are used for every model -> paired comparisons

What we get:
1. 95% confidence interval of each metric (macro-F1, MCC, accuracy, AUC)
2. Paired comparisons (e.g. RF vs CART, RNA+CNA vs RNA) with a 95% CI of the
   difference and a bootstrap p-value (Holm-corrected for multiple tests)
3. Plots saved in results/figures/

Note: the bootstrap only measures the uncertainty due to the test sample.
The models are NOT retrained.
"""

from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)

# The saved pipelines contain our filters, so they must be importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Preprocessing"))
from filters import ModalFrequencyFilter, ZeroFractionFilter  # noqa: F401

# ==================================================================================
# PATHS AND SETTINGS
# ==================================================================================
P = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOT = 2000      # number of bootstrap samples
ALPHA = 0.05       # 95% confidence intervals

# model name -> prefix used when the model was saved
# (add Naive Bayes here, e.g. "NaiveBayes": "naive_bayes", if it is saved the same way)
MODELS = {"CART": "cart", "RandomForest": "random_forest"}
SCENARIOS = ["RNA", "CNA", "RNA+CNA"]
METRICS = ["macro_f1", "mcc", "accuracy", "auc_ovr"]
METRIC_NAMES = {"macro_f1": "Macro-F1", "mcc": "MCC",
                "accuracy": "Accuracy", "auc_ovr": "AUC (OvR)"}

# ==================================================================================
# DATA LOADING --> exactly the same test set as in the training scripts
# ==================================================================================
clinical = pd.read_csv(P / "clinical_945.csv",
                       usecols=["SAMPLE_ID", "PATIENT_ID", "SUBTYPE"],
                       index_col="SAMPLE_ID")
split = pd.read_csv(P / "patient_split.csv").set_index("PATIENT_ID")["split"]

rna = pd.read_parquet(P / "rna_log2.parquet")
cna_disc = pd.read_parquet(P / "cna_disc_raw.parquet")
rna_cna = pd.concat([rna.add_prefix("RNA_"), cna_disc.add_prefix("CNA_")], axis=1)

is_test = clinical["PATIENT_ID"].map(split).eq("test").to_numpy()
y_test = clinical["SUBTYPE"].to_numpy()[is_test]
X_test = {"RNA": rna[is_test], "CNA": cna_disc[is_test], "RNA+CNA": rna_cna[is_test]}
print(f"Test samples: {len(y_test)}")
print(pd.Series(y_test).value_counts().to_string())

# ==================================================================================
# PREDICTIONS --> computed only once per model, then reused in every bootstrap
# ==================================================================================
preds = {}
for model_name, prefix in MODELS.items():
    for scen in SCENARIOS:
        model = joblib.load(RESULTS / f"{prefix}_model_{scen}.joblib")
        preds[(model_name, scen)] = {
            "pred": model.predict(X_test[scen]),
            "prob": model.predict_proba(X_test[scen]),
            "classes": model.classes_,
        }
CLASSES = preds[(list(MODELS)[0], SCENARIOS[0])]["classes"]


def compute_metrics(y, pred, prob, classes):
    return {
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "accuracy": accuracy_score(y, pred),
        "auc_ovr": roc_auc_score(y, prob, multi_class="ovr",
                                 average="macro", labels=classes),
    }

# ==================================================================================
# BOOTSTRAP --> stratified resampling of the test set
# ==================================================================================
def stratified_resample(y, rng):
    """Resample with replacement inside each class (keeps class sizes)."""
    idx = []
    for c in np.unique(y):
        class_idx = np.flatnonzero(y == c)
        idx.append(rng.choice(class_idx, size=len(class_idx), replace=True))
    return np.concatenate(idx)


rng = np.random.default_rng(SEED)
boot_idx = [stratified_resample(y_test, rng) for _ in range(N_BOOT)]

boot_rows = []
for (model_name, scen), p in preds.items():
    for b, idx in enumerate(boot_idx):
        m = compute_metrics(y_test[idx], p["pred"][idx], p["prob"][idx], p["classes"])
        boot_rows.append({"model": model_name, "scenario": scen, "boot": b, **m})
boot = pd.DataFrame(boot_rows)
boot.to_csv(RESULTS / "trees_bootstrap_samples.csv", index=False)


def get_boot(model_name, scen, metric):
    """Bootstrap values of one model, ordered by bootstrap sample (for pairing)."""
    sub = boot[(boot["model"] == model_name) & (boot["scenario"] == scen)]
    return sub.sort_values("boot")[metric].to_numpy()

# ==================================================================================
# 1) CONFIDENCE INTERVALS (percentile method)
# ==================================================================================
q_low, q_high = 100 * ALPHA / 2, 100 * (1 - ALPHA / 2)
ci_rows = []
for (model_name, scen), p in preds.items():
    observed = compute_metrics(y_test, p["pred"], p["prob"], p["classes"])
    for metric in METRICS:
        values = get_boot(model_name, scen, metric)
        low, high = np.percentile(values, [q_low, q_high])
        ci_rows.append({"model": model_name, "scenario": scen, "metric": metric,
                        "estimate": observed[metric], "ci_low": low, "ci_high": high,
                        "boot_std": values.std()})
ci = pd.DataFrame(ci_rows)
ci.to_csv(RESULTS / "trees_bootstrap_ci.csv", index=False)

print("\n=== 95% bootstrap confidence intervals ===")
for _, r in ci.iterrows():
    print(f"{r['model']:>12} | {r['scenario']:<7} | {r['metric']:<8} "
          f"{r['estimate']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")

# ==================================================================================
# 2) PAIRED COMPARISONS --> difference A - B on the same bootstrap samples
# ==================================================================================
comparisons = []
for scen in SCENARIOS:                      # models compared within each data type
    comparisons.append((("RandomForest", scen), ("CART", scen)))
for model_name in MODELS:                   # data types compared within each model
    comparisons.append(((model_name, "RNA+CNA"), (model_name, "RNA")))
    comparisons.append(((model_name, "RNA+CNA"), (model_name, "CNA")))
    comparisons.append(((model_name, "RNA"), (model_name, "CNA")))


def holm(pvalues):
    """Holm correction for multiple comparisons."""
    p = np.asarray(pvalues)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0
    for rank, i in enumerate(np.argsort(p)):
        running_max = max(running_max, (m - rank) * p[i])
        adjusted[i] = min(running_max, 1.0)
    return adjusted


comp_rows = []
for metric in METRICS:
    rows_metric = []
    for (a_model, a_scen), (b_model, b_scen) in comparisons:
        diff = get_boot(a_model, a_scen, metric) - get_boot(b_model, b_scen, metric)
        observed = (ci.query("model == @a_model and scenario == @a_scen and metric == @metric")["estimate"].iloc[0]
                    - ci.query("model == @b_model and scenario == @b_scen and metric == @metric")["estimate"].iloc[0])
        low, high = np.percentile(diff, [q_low, q_high])
        # two-sided bootstrap p-value: how often the difference crosses 0
        p_value = min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
        rows_metric.append({"metric": metric,
                            "comparison": f"{a_model} {a_scen} - {b_model} {b_scen}",
                            "difference": observed, "ci_low": low, "ci_high": high,
                            "p_value": p_value})
    p_holm = holm([r["p_value"] for r in rows_metric])
    for r, ph in zip(rows_metric, p_holm):
        r["p_holm"] = ph
        r["significant"] = ph < ALPHA
    comp_rows += rows_metric
comp = pd.DataFrame(comp_rows)
comp.to_csv(RESULTS / "trees_bootstrap_comparisons.csv", index=False)

print("\n=== Paired comparisons (macro-F1) ===")
for _, r in comp[comp["metric"] == "macro_f1"].iterrows():
    star = "*" if r["significant"] else ""
    print(f"{r['comparison']:<40} {r['difference']:+.3f} "
          f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  p_holm={r['p_holm']:.3f} {star}")

# ==================================================================================
# 3) PLOTS
# ==================================================================================
colors = {name: plt.cm.tab10(i) for i, name in enumerate(MODELS)}
x = np.arange(len(SCENARIOS))
offsets = np.linspace(-0.15, 0.15, len(MODELS))

# --- Plot 1: point estimate + 95% CI for every metric ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, metric in zip(axes.ravel(), METRICS):
    for off, model_name in zip(offsets, MODELS):
        sub = ci[(ci["model"] == model_name) & (ci["metric"] == metric)].set_index("scenario").loc[SCENARIOS]
        err = [sub["estimate"] - sub["ci_low"], sub["ci_high"] - sub["estimate"]]
        ax.errorbar(x + off, sub["estimate"], yerr=err, fmt="o", capsize=5,
                    color=colors[model_name], label=model_name)
    ax.set_xticks(x, SCENARIOS)
    ax.set_title(METRIC_NAMES[metric])
    ax.grid(axis="y", alpha=0.3)
axes[0, 0].legend()
fig.suptitle(f"Test performance with 95% bootstrap CI ({N_BOOT} resamples)")
fig.tight_layout()
fig.savefig(FIGURES / "bootstrap_ci_all_metrics.png", dpi=300)
plt.close(fig)

# --- Plot 2: full bootstrap distribution of macro-F1 ----------------------------
fig, ax = plt.subplots(figsize=(9, 5))
data, positions, labels = [], [], []
for off, model_name in zip(offsets, MODELS):
    for i, scen in enumerate(SCENARIOS):
        data.append(get_boot(model_name, scen, "macro_f1"))
        positions.append(i + off * 2)
parts = ax.violinplot(data, positions=positions, widths=0.25, showmedians=True)
for body, (model_name, _) in zip(parts["bodies"],
                                 [(m, s) for m in MODELS for s in SCENARIOS]):
    body.set_facecolor(colors[model_name])
    body.set_alpha(0.6)
ax.set_xticks(x, SCENARIOS)
ax.set_ylabel("Macro-F1")
ax.set_title("Bootstrap distribution of test macro-F1")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=colors[m], alpha=0.6) for m in MODELS],
          list(MODELS))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "bootstrap_macro_f1_violin.png", dpi=300)
plt.close(fig)

# --- Plot 3: paired differences (macro-F1 and MCC) ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, metric in zip(axes, ["macro_f1", "mcc"]):
    sub = comp[comp["metric"] == metric].reset_index(drop=True)
    ypos = np.arange(len(sub))[::-1]
    for yp, (_, r) in zip(ypos, sub.iterrows()):
        color = "tab:red" if r["significant"] else "gray"
        ax.errorbar(r["difference"], yp,
                    xerr=[[r["difference"] - r["ci_low"]], [r["ci_high"] - r["difference"]]],
                    fmt="o", capsize=4, color=color)
        ax.text(r["ci_high"] + 0.005, yp, f"p={r['p_holm']:.3f}", va="center", fontsize=8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    left, right = ax.get_xlim()
    ax.set_xlim(left, right + 0.15 * (right - left))   # room for the p-value labels
    ax.set_yticks(ypos, sub["comparison"])
    ax.set_xlabel(f"Difference in {METRIC_NAMES[metric]} (A - B)")
    ax.set_title(METRIC_NAMES[metric])
    ax.grid(axis="x", alpha=0.3)
fig.suptitle("Paired bootstrap comparisons (95% CI, Holm-corrected p; red = significant)")
fig.tight_layout()
fig.savefig(FIGURES / "bootstrap_paired_differences.png", dpi=300)
plt.close(fig)

# --- Plot 4: normalized confusion matrices (where does each model fail?) --------
fig, axes = plt.subplots(len(MODELS), len(SCENARIOS),
                         figsize=(4 * len(SCENARIOS), 3.8 * len(MODELS)), squeeze=False)
for i, model_name in enumerate(MODELS):
    for j, scen in enumerate(SCENARIOS):
        ax = axes[i, j]
        matrix = confusion_matrix(y_test, preds[(model_name, scen)]["pred"],
                                  labels=CLASSES, normalize="true")
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        for r in range(len(CLASSES)):
            for c in range(len(CLASSES)):
                ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center",
                        color="white" if matrix[r, c] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(len(CLASSES)), CLASSES)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_title(f"{model_name} - {scen}")
        if j == 0:
            ax.set_ylabel("Real")
        if i == len(MODELS) - 1:
            ax.set_xlabel("Predicted")
fig.suptitle("Confusion matrices on test (row-normalized = recall per subtype)")
fig.tight_layout()
fig.savefig(FIGURES / "confusion_matrices_normalized.png", dpi=300)
plt.close(fig)

print(f"\nTables saved in {RESULTS} and figures in {FIGURES}")
