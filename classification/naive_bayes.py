"""
CLASSIFICATION:
Naïve Bayes classifier for PAM50 subtype (LumA, LumB, Basal, Her2)
- combinations: RNA, CNA (discrete) and RNA + CNA
- we use the same patient split for all of them (patient_split.csv)

Hyperparameters: optimized with Optuna
- maximising macro-F1 with 5-fold
- stratified cross-validation on the train set only
- test set used once at the end

Pipeline: 
- gene filters and scaling inside the Pipeline, so they're refitted on the training of each fold
- therefore, no data leakage

Outputs in results/ folder
"""

from pathlib import Path
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from filters import ModalFrequencyFilter, ZeroFractionFilter  # scripts/filters.py

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==================================================================================
# PATHS AND SETTINGS
# ==================================================================================
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
N_FOLDS = 5
N_TRIALS = 30                              # Optuna trials per scenario
UNIFORM_PRIORS = [0.25, 0.25, 0.25, 0.25]  # 4 classes (alphabetical order)