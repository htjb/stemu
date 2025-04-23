"""Utility functions for stemu."""

import numpy as np


def stack(X, t, y=None):
    """Stack the data for training.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.
    t : array-like of shape (n_target)
        The independent variable for the target
    y : array-like of shape (n_samples, n_target)
        The dependent variable for the target

    Returns
    -------
    X : array-like of shape (n_samples*n_target,n_features)
        The input data.
    y : array-like of shape (n_samples*n_target,)
        The dependent variable for the target
    """
    tiledX = np.tile(X, (len(t), 1))
    tiledt = np.tile(t, (len(X), 1))
    X = np.hstack([tiledX, tiledt])
    if y is not None:
        y = y.flatten()
    return X, y


def unstack(y, t):
    """Unstack the data for prediction.

    Parameters
    ----------
    X : array-like of shape (n_samples*n_target, n_features)
        The input data.
    y : array-like of shape (n_samples*n_target,)
        The dependent variable for the target

    Returns
    -------
    X : array-like of shape (n_samples, n_features)
        The input data.
    t : array-like of shape (n_target,)
        The independent variable for the target
    y : array-like of shape (n_samples, n_target)
        The dependent variable for the target
    """
    y = y.reshape(-1, len(t))
    return y
