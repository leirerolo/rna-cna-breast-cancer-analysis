"""
REGRESSION: 
This is a module of shared data for all models in this task
- predictors: RNA / CNA / RNA+CNA 
- target: scaled log1p(TMB_NONSYNONYMOUS) -> because tmb is very right-skewed

Here we will ensure that the following models are aligned with the previous patient split
(patient_split.csv)

- load RNA, CNA, RNA+CNA
- calculate log1p (TMB)
- keep the patients that have TMB
- respect the already stored split
- give metrics: RMSE, MAE, R^2, Spearman to evaluate all regression models the same way

*every regression model script imports this file, so all of us use exactly the
same patients, target and metrics.
    sys.path.append(str(ROOT / "Preprocessing"))
    from regression_data import load_regression_data, regression_metrics

"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
TARGET = "TMB_NONSYNONYMOUS"


#load regression data with cna as discrete GISTIC, like did in classification
#return: dictionary with the following:
#   - datasets of the 3 scenarios, with their samples with TMB as rows
#   - y = log1p(TMB)
#   - tmb_raw, without transformation
#   - boolean array to know if it's train or test
#   - subtype of each sample

def load_regression_data(cna_kind="disc"):
    clinical = pd.read_csv(
        P / "clinical_945.csv",
        usecols=["SAMPLE_ID", "PATIENT_ID", "SUBTYPE", TARGET],
        index_col="SAMPLE_ID",
    )
    split = pd.read_csv(P / "patient_split.csv").set_index("PATIENT_ID")["split"]

    cna_files = {"disc": "cna_disc_raw.parquet", "log2": "cna_log2_raw.parquet"}
    if cna_kind not in cna_files:
        raise ValueError("cna_kind must be 'disc' or 'log2'")

    rna = pd.read_parquet(P / "rna_log2.parquet")
    cna = pd.read_parquet(P / cna_files[cna_kind])

    for name, df in [("RNA", rna), ("CNA", cna)]:
        assert df.index.equals(clinical.index), f"{name}: samples differ from clinical table"
    assert clinical["PATIENT_ID"].isin(split.index).all(), "Patients missing from the split"

    #we keep only samples with a TMB value while not changing the split
    has_tmb = clinical[TARGET].notna()
    clinical = clinical[has_tmb]
    assert (clinical[TARGET] >= 0).all(), "Negative TMB values"
    rna, cna = rna.loc[clinical.index], cna.loc[clinical.index]

    #prefixes so RNA and CNA columns don't collide
    rna_cna = pd.concat([rna.add_prefix("RNA_"), cna.add_prefix("CNA_")], axis=1)

    is_train = clinical["PATIENT_ID"].map(split).eq("train").to_numpy()

    #patient ID of every sample, to keep the datasets' order
    patient_ids = clinical["PATIENT_ID"]

    return {
        "datasets": {"RNA": rna, "CNA": cna, "RNA+CNA": rna_cna},
        "y": np.log1p(clinical[TARGET]),
        "tmb_raw": clinical[TARGET],
        "is_train": is_train,
        "subtype": clinical["SUBTYPE"],
        "patient_ids": patient_ids,
    }


def tmb_from_log(y_log):
    #back to the original TMB scale (mutations per megabase), in case it's necessary to check anything
    return np.expm1(y_log)

#RMSE, MAE, R2 and Spearman, which is NaN if the predictions are constant
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    #spearman computations: to see if our predictions are keeping the same order/ranking of the real values
    if np.ptp(y_pred) == 0:
        rho = np.nan
    else:
        rho = spearmanr(y_true, y_pred)[0] #rho measures correlation between the two variables' ranks (-1,+1)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(rho),
    }