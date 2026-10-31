"""
SCRIPT 01: we will load RNA, CNA and clinical data,
as well as intersecting the patients that have both

As the data we are extracting are too big, we are storing
them in .parquet files rather than CSV
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] #folder of the repository, with "scripts" folder inside
#we build the paths to the original and the processed data
RAW = ROOT / "data" / "raw" / "brca_tcga_pan_can_atlas_2018"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True) #create an output folder if it doesn't exist


def read_gene_matrix(name):
    #the file has genes in rows and samples in columns
    #the function returns patients in rows and genes in cols

    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(f"{path} path not found. Review folder data/raw.")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    initial_n = len(df)

    df = df.dropna(subset=["Hugo_Symbol"])
    n_without_symbol = initial_n - len(df)
    n_before_dup = len(df)
    df = df.drop_duplicates(subset="Hugo_Symbol", keep="first")

    n_dup = n_before_dup - len(df)
    print(f"{name}: {initial_n} rows | without symbol: {n_without_symbol}"
          f" | eliminated duplicates: {n_dup} | final genes: {len(df)}")

    df = df.set_index("Hugo_Symbol")
    #we keep only sample columns (format: TCGA-XX-XXXX-01)
    sample_cols = [c for c in df.columns if str(c).startswith("TCGA-")]
    df = df[sample_cols]

    #ID of the sample --> ID of the patient (first 12 characters)
    df.columns = [c[:12] for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()] #in case a patient has 2 samples
    
    return df.T.astype(float) #this returns patients x genes


def read_clinical(name):
    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(f"path {path} not found")
    
    #as there are 4 lines of metadata with # after the normal header:
    return pd.read_csv(path, sep="\t", comment="#")

print("\nReading RNA...")
rna = read_gene_matrix("data_mrna_seq_v2_rsem.txt")
print("\nReading CNA log2...")
cna_log2 = read_gene_matrix("data_log2_cna.txt")
print("\nReading CNA discrete...")
cna_disc = read_gene_matrix("data_cna.txt")

print("\n--------------------------------------------------")

print("\nReading clinical...")
pat = read_clinical("data_clinical_patient.txt").set_index("PATIENT_ID")
sample = read_clinical("data_clinical_sample.txt")
sample["PATIENT_ID"] = sample["PATIENT_ID"].astype(str)
n_sample = len(sample)
sample = sample.drop_duplicates(subset="PATIENT_ID").set_index("PATIENT_ID")
print(f"Clinical samples: {n_sample} | after removing repeated patients: {len(sample)}")


#exploration of the subtypes registered in the source file
print("\nSUBTYPE values in the file:")
print(pat["SUBTYPE"].value_counts(dropna=False))

#we now list the observed subtypes 
SUBTYPES = ["BRCA_LumA", "BRCA_LumB", "BRCA_Basal", "BRCA_Her2"]

left = [s for s in SUBTYPES if s not in set(pat["SUBTYPE"].dropna())]
if left:
    raise SystemExit(f"\nERROR: these expected subtypes don't appear: {left}. "
                     "Adjust SUBTYPES with the names in the table above.")
pat = pat[pat["SUBTYPE"].isin(SUBTYPES)]


#INTERSECTION: RNA, CNA, both and the valid subtype
common = sorted(set(rna.index) & set(cna_log2.index)
                & set(cna_disc.index) & set(pat.index))
print("\n--------------------------------------------------------------------------")
print(f"\nPatients in the intersection: {len(common)} (expected: 945)")
#explanation on "expected: 945" -> the sum of the observed subtype counts (substracting the NaNs and BRCA_Normal)

rna = rna.loc[common]
cna_log2 = cna_log2.loc[common]
cna_disc = cna_disc.loc[common]
clinical = pat.loc[common].join(sample, how="left", rsuffix="_sample")
 
print("\nSubtypes in the intersection:")
print(clinical["SUBTYPE"].value_counts())
print("\nRNA form:", rna.shape, "| CNA log2:", cna_log2.shape,
      "| CNA disc:", cna_disc.shape)
print("NaN in RNA:", int(rna.isna().sum().sum()),
      "| CNA log2:", int(cna_log2.isna().sum().sum()),
      "| CNA disc:", int(cna_disc.isna().sum().sum()))
 
#saving (parquet needs pyarrow, so pip install pyarrow)
rna.to_parquet(OUT / "rna_raw.parquet")
cna_log2.to_parquet(OUT / "cna_log2_raw.parquet")
cna_disc.to_parquet(OUT / "cna_disc_raw.parquet")
clinical.to_csv(OUT / "clinical_945.csv")
print(f"\nStored in {OUT}")
