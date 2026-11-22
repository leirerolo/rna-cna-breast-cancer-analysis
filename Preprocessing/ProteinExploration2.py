"""
EXPLORATION: protein and phosphoprotein files of the study.

For each file it prints: the description in its meta file, the shape, the
identifier columns, how many of our 945 patients have data, and how many
proteins have values in at least 80% of those patients.
Nothing is saved: it is only to decide which file can be used as target.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "brca_tcga_pan_can_atlas_2018"
P = ROOT / "data" / "processed"

clinical = pd.read_csv(P / "clinical_945.csv", usecols=["SAMPLE_ID", "PATIENT_ID"])
samples_945 = set(clinical["SAMPLE_ID"])
patients_945 = set(clinical["PATIENT_ID"])

files = [
    "data_rppa.txt",
    "data_rppa_zscores.txt",
    "data_protein_quantification.txt",
    "data_protein_quantification_zscores.txt",
    "data_phosphoprotein_quantification.txt",
]

for name in files:
    print("\n" + "=" * 90)
    print(name)

    meta = RAW / name.replace("data_", "meta_", 1)
    if meta.exists():
        print("--- meta file (first 15 lines) ---")
        print("\n".join(meta.read_text(encoding="utf-8", errors="replace").splitlines()[:15]))

    df = pd.read_csv(RAW / name, sep="\t", low_memory=False)
    sample_cols = [c for c in df.columns if str(c).startswith("TCGA-")]
    id_cols = [c for c in df.columns if c not in sample_cols]

    print("--- data ---")
    print("Shape (rows x columns):", df.shape)
    print("Identifier columns:", id_cols)
    print("First identifiers:", df[id_cols[0]].astype(str).head(8).tolist())
    print("Sample columns:", len(sample_cols), "| example:", sample_cols[:2])

    common_samples = [c for c in sample_cols if c in samples_945]
    common_patients = {c[:12] for c in sample_cols} & patients_945
    print("Samples in our 945:", len(common_samples),
          "| patients in our 945 (matching by the first 12 characters):", len(common_patients))

    if common_samples:
        values = df[common_samples].apply(pd.to_numeric, errors="coerce")
        missing_by_row = values.isna().mean(axis=1)
        print(f"Overall missing values (in our patients): {values.isna().mean().mean():.1%}")
        print("Rows with values in >= 80% of our patients:",
              int((missing_by_row <= 0.2).sum()), "of", len(df))