"""
SCRIPT 03: preprocessing
HAY QUE DECIDIR ESTO!!!!!!!
lo que me sugiere la IA es esto:
- RNA: log2(x+1) y filtro de genes casi siempre a cero (calculado SOLO con train)
- CNA: filtro de genes casi constantes (calculado SOLO con train)
y luego me dice que el escalado y la selección de genes van dentro del Pipeline de los modelos

We will store this in .parquet files: 
filtered_rna, filtered_cna_disc, filtered_cna_log2
"""

from pathlib import Path
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1] / "data" / "processed"

for f in ["patient_split.csv", "rna_raw.parquet", 
          "cna_log2_raw.parquet", "cna_disc_raw.parquet"]:
    if not (P / f).exists:
        raise FileNotFoundError(f"You don't have {f}: execute firstly 01 and 02 scripts.")

#we get the ids of the training set
split = pd.read_csv(P / "patient_split.csv").set_index("PATIENT_ID")["split"]
train_ids = split[split == "train"].index

rna = pd.read_parquet(P / "rna_raw.parquet")
cna_log2 = pd.read_parquet(P / "cna_log2_raw.parquet")
cna_disc = pd.read_parquet(P / "cna_disc_raw.parquet")


#PROVISIONAL THRESHOLDS DECIDIR!!!!!!!!!!!!
#habría que mirar cuántos genes quedan y por qué
THRESHOLD_ZEROS = 0.95 #RNA: we delete genes with >= 95% of zeros in train
THRESHOLD_CONST = 0.95 #CNA discrete: most frequent value in >=95% of train
THRESHOLD_VAR = 0.01 #CNA log2: we delete genes with variance <=0.1 in train

#a partir de aquí, he hecho con la sugerencia de claude,
#pero si decidimos otra cosa solo sería cambiar un par de cosas
#en esta estructura

#--------- RNA: log2(x+1)
assert not rna.isna().any().any(), "There is NaN in RNA"
assert (rna.values >= 0).all(), "RNA with negative values"
rna = np.log2(rna + 1)

prop_zeros = (rna.loc[train_ids] == 0).mean()
rna = rna[prop_zeros[prop_zeros < THRESHOLD_ZEROS].index]


#---------- CNA discrete: almost constant gen = a value that dominates in train
print("CNA: NaN discrete:", int(cna_disc.isna().sum().sum()),
      "| NaN log2:", int(cna_log2.isna().sum().sum()))

def freq_modal(col):
    return col.value_counts(normalize=True).iloc[0]

fm = cna_disc.loc[train_ids].apply(freq_modal)
cna_disc = cna_disc[fm[fm < THRESHOLD_CONST].index]


#----------- CNA log2 (continuous): it uses the variance
var = cna_log2.loc[train_ids].var()
cna_log2 = cna_log2[var[var > THRESHOLD_VAR].index]

print("\nRNA after filtering:", rna.shape)
print("CNA discrete after filtering:", cna_disc.shape)
print("CNA log2 after filtering:", cna_log2.shape)

rna.to_parquet(P / "filtered_rna.parquet")
cna_disc.to_parquet(P / "filtered_cna_disc.parquet")
cna_log2.to_parquet(P / "filtered_cna_log2.parquet")
print(f"Stored in {P}")
