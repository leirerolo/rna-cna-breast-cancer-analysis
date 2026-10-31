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
VAL_SIZE = 0.2 #also decision
SEED = 42 #fixed, so we all get the same split

#read the clinical data file and extract the patient ids to then split them
clinical = pd.read_csv(path, index_col=0)
ids = clinical.index.to_numpy()
#we store each patient's subtype column so that the split respects subtype proportions
y = clinical["SUBTYPE"] 

#split1: train and test
trainval_ids, test_ids = train_test_split(
    ids, test_size=TEST_SIZE, stratify=y, random_state=SEED
)

#split2: inside train, split a validation set
train_ids, val_ids = train_test_split(
    trainval_ids,
    test_size=VAL_SIZE / (1-TEST_SIZE),
    stratify=y.loc[trainval_ids],
    random_state=SEED,
)


#build a table with the split
split = pd.DataFrame({
    "PATIENT_ID": list(train_ids) + list(val_ids) + list(test_ids),
    "split": (["train"] * len(train_ids) 
    + ["val"] * len(val_ids)
    + ["test"] * len(test_ids)),
})

#store the split
split.to_csv(OUT / "patient_split.csv", index=False)

#printing an overview of the split
print("Train:", len(train_ids), "| Validation:", len(val_ids), "| Test:", len(test_ids))
for name, group in [("train", train_ids), ("validation", val_ids), ("test", test_ids)]:
    print(f"\nSubtypes in {name}:\n", y.loc[group].value_counts())
print(f"\nStored in {OUT / 'patient_split.csv'}")