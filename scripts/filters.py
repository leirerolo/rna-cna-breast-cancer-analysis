#======================================================================
#functions for removing statistically unsignificant genes
#======================================================================

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ZeroFractionFilter(BaseEstimator, TransformerMixin):
    """RNA: drops genes with a fraction of zeros >= threshold (fitted on train only)."""
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.mask_ = (X == 0).mean(axis=0) < self.threshold
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.mask_]


class ModalFrequencyFilter(BaseEstimator, TransformerMixin):
    """CNA discrete: drops genes whose most frequent value is in >= threshold of patients."""
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X)
        freqs = np.max([(X == v).mean(axis=0) for v in np.unique(X)], axis=0)
        self.mask_ = freqs < self.threshold
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.mask_]