"""
SCRIPT 02: we will perform the train/test set splits, stratifying
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
clinical = pd.read_csv(path, index_col=0)
ids = clinical.index.to_numpy()
#we store each patient's subtype column so that the split respects subtype proportions
y = clinical["SUBTYPE"] 

#split
train_ids, test_ids = train_test_split(
    ids, test_size=TEST_SIZE, stratify=y, random_state=SEED
)

#build a table with the split
split = pd.DataFrame({
    "PATIENT_ID": list(train_ids) + list(test_ids),
    "split": ["train"] * len(train_ids) + ["test"] * len(test_ids),
})

#store the split
split.to_csv(OUT / "split_patients.csv", index=False)

#printing an overview of the split
print("Train:", len(train_ids), "| Test:", len(test_ids))
print("\nSubtypes in train:\n", y.loc[train_ids].value_counts())
print("\nSubtypes in test:\n", y.loc[test_ids].value_counts())
print(f"\nStored in {OUT / 'split_patients.csv'}")