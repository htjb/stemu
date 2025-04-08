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
    X = np.concatenate([tiledX, 
                        np.repeat(t, len(X), axis=0).reshape(-1, 1)], axis=1)
    if y is not None:
        y = y.flatten()
    return X, y


def unstack(X, y, t):
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
    X = X[:, :-1]
    t = X[:, -1]
    return X, t, y
