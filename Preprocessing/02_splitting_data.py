"""
SCRIPT 02: we will perform the train/validation/test set splits, stratifying
by subtype, and storing by patient ID
This split is shared by the 3 scenarios: RNA, CNA and RNA+CNA, and also by all tasks

We are generating an output in the shape of a csv file with the processed and splitted data of the patients
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"

path = OUT / "clinical_945.csv"
if not path.exists():
    raise FileNotFoundError("clinical_945.csv missing: execute first script 01")

TEST_SIZE = 0.2 #our decision -> we'll document it in Metodology section
SEED = 42 #fixed, so we all get the same split

#read the clinical data file and extract the patient ids to then split them
clinical = pd.read_csv(path, usecols=["PATIENT_ID", "SUBTYPE"])

if clinical[["PATIENT_ID", "SUBTYPE"]].isna().any().any():
    raise ValueError("Missing PATIENT_ID or SUBTYPE")

patients = clinical.drop_duplicates().set_index("PATIENT_ID").sort_index()

if not patients.index.is_unique:
    raise ValueError("A patient has conflicting subtypes")

ids = patients.index.to_numpy()
y = patients["SUBTYPE"]

#split1: train and test
train_ids, test_ids = train_test_split(
    ids, test_size=TEST_SIZE, stratify=y, random_state=SEED
)



#build a table with the split
split = pd.DataFrame({
    "PATIENT_ID": list(train_ids) + list(test_ids),
    "split": (["train"] * len(train_ids) 
    + ["test"] * len(test_ids)),
})

#store the split
split.to_csv(OUT / "patient_split.csv", index=False)

#printing an overview of the split
print("Train:", len(train_ids), "| Test:", len(test_ids))
for name, group in [("train", train_ids),("test", test_ids)]:
    print(f"\nSubtypes in {name}:\n", y.loc[group].value_counts())
print(f"\nStored in {OUT / 'patient_split.csv'}")

