"""
SCRIPT 03: preprocessing

Only sample-wise transformations are applied here.
Gene filters (see filters.py) and any scaling are fitted inside the
model pipeline using the training portion of each CV fold.

- RNA: log2(x+1) -> rna_log2.parquet
- CNA (discrete and log2): use the Parquet files from script 01
"""

from pathlib import Path
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1] / "data" / "processed"

for f in ["clinical_945.csv", "patient_split.csv", "rna_raw.parquet",
          "cna_log2_raw.parquet", "cna_disc_raw.parquet"]:
    if not (P / f).exists():
        raise FileNotFoundError(f"Missing {f}. Run scripts 01 and 02 first.")

split = pd.read_csv(P / "patient_split.csv").set_index("PATIENT_ID")["split"]

clinical = pd.read_csv(
    P / "clinical_945.csv", usecols=["SAMPLE_ID", "PATIENT_ID"]
).set_index("SAMPLE_ID")

rna = pd.read_parquet(P / "rna_raw.parquet")
cna_log2 = pd.read_parquet(P / "cna_log2_raw.parquet")
cna_disc = pd.read_parquet(P / "cna_disc_raw.parquet")

#======================================================================================================
# Sanity checks: same samples in each matrix, one split assignment per patient
#======================================================================================================

if clinical.index.hasnans or not clinical.index.is_unique:
    raise ValueError("Missing or repeated SAMPLE_ID in clinical_945.csv")

if split.index.hasnans or not split.index.is_unique:
    raise ValueError("Missing or repeated PATIENT_ID in patient_split.csv")

if set(split) != {"train", "test"}:
    raise ValueError("Expected train and test assignments in patient_split.csv")

if set(clinical["PATIENT_ID"]) != set(split.index):
    raise ValueError("Clinical data and split contain different patients")

for name, df in [("RNA", rna), ("CNA log2", cna_log2), ("CNA discrete", cna_disc)]:
    if not df.index.equals(clinical.index):
        raise ValueError(
            f"{name}: SAMPLE_IDs differ from the clinical table or are out of order"
        )

#======================================================================================================
# RNA: log2(x+1)
#======================================================================================================

rna_values = rna.to_numpy()
if not np.isfinite(rna_values).all():
    raise ValueError("RNA contains NaN or infinite values")
if (rna_values < 0).any():
    raise ValueError("RNA contains negative expression values")

print("RNA max before log2:", float(rna_values.max()))

rna_log2 = np.log2(rna + 1)
rna_log2.to_parquet(P / "rna_log2.parquet")

#======================================================================================================
# CNA: diagnostics
#======================================================================================================

print("CNA discrete values:", sorted(pd.unique(cna_disc.to_numpy().ravel())))
print("NaN CNA discrete:", int(cna_disc.isna().sum().sum()),
      "| NaN CNA log2:", int(cna_log2.isna().sum().sum()))

print("\nRNA log2:", rna_log2.shape)
print("CNA discrete:", cna_disc.shape)
print("CNA log2:", cna_log2.shape)
print(f"Stored in {P / 'rna_log2.parquet'}")