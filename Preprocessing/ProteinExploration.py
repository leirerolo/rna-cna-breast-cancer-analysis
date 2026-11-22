from pathlib import Path
import pandas as pd

RAW = Path("data/raw/brca_tcga_pan_can_atlas_2018")
print([f.name for f in RAW.iterdir()
       if any(k in f.name.lower() for k in ("rppa", "prot"))])