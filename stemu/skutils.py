"""Utilities for working with scikit-learn."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CDFTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X):
        Xprime = np.interp(np.linspace(0, 1, len(X)), self.cdf, X)
        return Xprime

    def inverse_transform(self, Xprime):
        X = np.interp(Xprime, np.linspace(0, 1, len(self.cdf)), self.cdf)
        return X

    def fit(self, X, y):
        """
        X == t
        y == y
        """
        X = X.copy()
        diff = y.max(axis=0) - y.min(axis=0)
        self.cdf = diff.cumsum() / diff.sum()
        return self

class ResampleYTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, t, tpri):
        self.t = t
        self.tpri = tpri
        return self
    
    def transform(self, X):
        X = X.copy()
        X = np.interp(self.tpri, self.t, X)
        return X
    
    def inverse_transform(self, X):
        X = X.copy()
        X = np.interp(self.t, self.tpri, X)
        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """Logarithm transformer."""

    def __init__(self, index=None):
        """Initialize the transformer.
        Parameters
        ----------
        index : int or None, optional
            The index of the column to transform. If None, transform all columns.
        """
        self.index = index
        pass

    def fit(self, X):
        """Fit the transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        """
        return self

    def transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        """
        X = X.copy()
        
        if self.index is not None:
            X[:, self.index] = np.log10(X[:, self.index])
        else:
            X = np.log10(X)
        return X
    
    def inverse_transform(self, X):
        """Inverse transform the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        """
        X = X.copy()
        if self.index is not None:
            X[:, self.index] = 10**X[:, self.index]
        else:
            X = 10**X

        return X