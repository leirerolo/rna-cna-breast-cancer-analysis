"""
CLASSIFICATION:
Support vector machine classifier for PAM50 subtype (LumA, LumB, Basal, Her2)
- combinations: RNA, CNA (discrete) and RNA + CNA
- we use the same patient split for all of them (patient_split.csv)

Hyperparameters: optimized with Optuna
- maximising macro-F1 with 5-fold
- stratified cross-validation on the train set only
- test set used once at the end

Pipeline: 
- gene filters and scaling inside the Pipeline, so they're refitted on the training of each fold
- therefore, no data leakage

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
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  StandardScaler
import joblib

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
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

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

rna = pd.read_parquet(P/ "rna_log2.parquet")
cna_disc = pd.read_parquet(P / "cna_disc_raw.parquet")

#check if the samples match
for name, df in [("RNA", rna), ("CNA discrete", cna_disc)]:
    assert df.index.equals(clinical.index), f"{name}: samples differ from clinical table"

#check if there are NaNs in cna, because in that case ModalFrequencyFilter would fail
assert not cna_disc.isna().any().any(), "CNA discrete has NaN"
#check if cna discrete contains values -2, -1, 0, 1 and 2
assert set(np.unique(cna_disc.to_numpy())) <= {-2, -1, 0, 1, 2}, "Unexpected CNA discrete values"

#every patient must have a split assignment
assert clinical["PATIENT_ID"].isin(split.index).all(), "Patients missing from the split"

#we identify the training samples with their patient id
is_train = clinical["PATIENT_ID"].map(split).eq("train").to_numpy()

#separate labels
y = clinical["SUBTYPE"] #y = subtype of each sample
y_train, y_test = y[is_train], y[~is_train] #y_train = labels of train samples, y_test = same for test samples
print(f"Train: {is_train.sum()} | Test: {(~is_train).sum()}")

#RNA + CNA: we give the columns a prefix so they don't collide
#and combine both dfs horizontally, so that we add cna columns to those of rna
#this creates a third dataset that contains both
rna_cna = pd.concat([
    rna.add_prefix("RNA_"), 
    cna_disc.add_prefix("CNA_")], 
    axis=1)


# =================================================================================
# SEARCH SPACE:
# hyperparameters that optuna will explore for the SVM
# ==================================================================================

#C, kernel and class_weight are always tuned; gamma only applies to the RBF kernel
def suggest_svm(trial):
    C = trial.suggest_float("C", 1e-3, 1e1, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    params = {
        "svm__C": C,
        "svm__kernel": kernel,
        "svm__class_weight": class_weight,
    }
    # only meakes sense with RBF (conditional parameter)
    if kernel == "rbf":
        params["svm__gamma"] = trial.suggest_float("gamma", 1e-6, 1e-3, log=True)
    return params

# =============================================================================
# COMBINATIONS
# =============================================================================
#for each combination (RNA, CNA or RNA+CNA), we define:
#- the data to be used
#- the preprocessing and classification pipeline
#- the function optuna will use to find hyperparameters

combinations = {
    "RNA": (rna, Pipeline([
        ("filter", ZeroFractionFilter(threshold=0.95)),
        ("scale", StandardScaler()), # standarization so the scale doesn't bias the model
        ("svm", SVC(probability=True, random_state=SEED)),
    ]), suggest_svm),

    "CNA": (cna_disc, Pipeline([
        ("filter", ModalFrequencyFilter(threshold=0.95)),
        ("scale", StandardScaler()),
        ("svm", SVC(probability=True, random_state=SEED)),
    ]), suggest_svm),

    "RNA+CNA": (rna_cna, Pipeline([
        ("select", ColumnTransformer([
            ("rna", ZeroFractionFilter(threshold=0.95),
             [c for c in rna_cna if c.startswith("RNA_")]),
            ("cna", ModalFrequencyFilter(threshold=0.95),
             [c for c in rna_cna if c.startswith("CNA_")]),
        ])),
        ("scale", StandardScaler()),
        ("svm", SVC(probability=True, random_state=SEED)),
    ]), suggest_svm),
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
        params = suggest(trial) #suggest a hyperparam combination
        model = clone(pipe).set_params(**params, svm__probability=False)
        #evaluate the model
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="f1_macro",
            error_score="raise", n_jobs=5   # un proceso por fold
        )

        #store the individual scores of the 5 folds and return their mean
        trial.set_user_attr("fold_scores", scores.tolist())
        return scores.mean() # = score that optuna will try to maximize

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    # allows us to see the progress of the optimization
    def log_trial(study, trial):
        print(f"  trial {trial.number + 1}/{N_TRIALS} | F1={trial.value:.3f} | {trial.params}", flush=True)

    study.optimize(objective, n_trials=N_TRIALS,callbacks=[log_trial]) #30 trials, and in each, call objective() and store the score

    #get the best model and train it
    best_params = suggest(optuna.trial.FixedTrial(study.best_params))
    best_model = clone(pipe).set_params(**best_params).fit(X_train, y_train)

    return(
        best_model,
        study.best_params,
        study.best_trial.user_attrs["fold_scores"],
        study
    )


# ============================================================================================
# TRAINING & FINAL EVALUATION 
# ============================================================================================
#final results of each comb & individual results of each fold
rows, fold_rows = [], []

for comb, (X, pipe, suggest) in combinations.items():
    print(f"\n=== {comb} ===")

    X_train, X_test = X[is_train], X[~is_train]

    #call tune to optimize hyperparams + train the best model
    model, best_params, fold_scores, study = tune(
        pipe, suggest, X_train, y_train
    )

    #save the trained model 
    joblib.dump(model, MODELS / f"svm_{comb.replace('+', '_')}.joblib")

    
    # checking nº of genes that survived the filter
    d = model[:-1].transform(X_train).shape[1]
    print(f"genes after the filter: {d} | 1/d = {1/d:.2e}")
   
    #best model trained with its params, scores...
    study.trials_dataframe().to_csv(
        RESULTS / f"svm_trials_{comb}.csv",
        index=False
    )

    #store scores of each fold
    for i, s in enumerate(fold_scores):
        fold_rows.append({
            "scenario": comb,
            "model": "SVM",
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
        "model": "SVM",
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
    print(confusion_matrix(y_test, pred, labels=classes))

pd.DataFrame(rows).to_csv(
    RESULTS / "svm_results.csv",
    index=False
)
pd.DataFrame(fold_rows).to_csv(
    RESULTS / "svm_cv_folds.csv",
    index=False
)
print(f"\nStored in {RESULTS}")