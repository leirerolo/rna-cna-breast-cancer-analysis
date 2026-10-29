"""
The purpose of this script is mainly exploratory. We will observe the following things: 
- missing values
- Genes and repeated ID
- repeated samples 
- correlation between sample and patient 
- percentage of zeros
- missing values per gene and per sample 
"""
#=======================================================================================
# libraries
#=======================================================================================
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


#1 data_mrna_seq_v2_rsem.txt exploration 
RNA_path = Path("rna-cna-breast-cancer-analysis/data/raw/brca_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt")
RNA_dataset = pd.read_csv(RNA_path)

#dataset structure
RNA_dataset.head()
print(f"Number of rows and columns: {RNA_dataset.shape}")
RNA_dataset.info()
#2 data_cna.txt exploration 


#3 data_log2.txt exploration 