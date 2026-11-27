import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Preprocessing"))

from filters import ZeroFractionFilter, ModalFrequencyFilter

modelo = joblib.load(ROOT / "results" / "cart_model_RNA+CNA.joblib")

arbol = modelo.named_steps["tree"]
selector = modelo.named_steps["select"]

rna = pd.read_parquet(ROOT / "data" / "processed" / "rna_log2.parquet")

filtro = selector.named_transformers_["rna"]
genes = rna.columns[filtro.mask_]

importancias = arbol.feature_importances_

rna_idx = selector.output_indices_["rna"]
cna_idx = selector.output_indices_["cna"]

print("Genes utilizados por CART:")

for i in np.argsort(importancias)[::-1]:
    if importancias[i] > 0 and i in range(rna_idx.start, rna_idx.stop):
        print(genes[i - rna_idx.start], round(importancias[i] * 100, 2), "%")

print("CNA utilizadas:", sum(
    importancias[i] > 0 for i in range(cna_idx.start, cna_idx.stop)
))