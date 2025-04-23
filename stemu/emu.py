"""Emulator base class."""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from stemu.skutils import CDFTransformer, ResampleYTransformer
from tensorflow import keras
from stemu.utils import stack, unstack


class Emu(object):
    """General Emulation base class.

    This fits an emulator for y=f(t|X) in the style of sklearn models.

    Anything with a default initialisation in the __init__ method is considered
    a hyperparameter and can be adjusted by the user after initialisation.

    Attributes
    ----------
    model : keras model, default is a simple dense network
    epochs : int, default=100
    loss : keras loss, default='mse'
    optimizer : keras optimizer, default='adam'
    callbacks : list of keras.callbacks
    X_pipeline : sklearn.pipeline to transform input data X
    t_pipeline : sklearn.pipeline to transform independent variable t
    y_pipeline : sklearn.pipeline to transform dependent variable y
    """

    def __init__(self, **kwargs):
        self.epochs = kwargs.get("epochs", 100)
        self.loss = kwargs.get("loss", "mse")
        self.callbacks = [keras.callbacks.EarlyStopping(monitor="loss", patience=100)]

        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)

        self.X_pipeline = kwargs.get('xpipe', 
                                     Pipeline([("scaler", StandardScaler())]))
        self.t_pipeline = [Pipeline([("cdf", CDFTransformer())]), 
                 Pipeline([("minmax", MinMaxScaler())])]
        self.y_pipeline = kwargs.get('ypipe',
                                     Pipeline(['scale', StandardScaler()]))

        self.network = [
                keras.layers.Dense(16, activation="tanh"),
                keras.layers.Dense(16, activation="tanh"),
                keras.layers.Dense(16, activation="tanh")
            ]

    def fit(self, X, t, y):
        """Fit the emulator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        t : array-like of shape (n_target,)
            The independent variable for the target
        y : array-like of shape (n_samples, n_target)
            The dependent variable for the target

        Returns
        -------
        self : object
            Returns self.
        """
        self.t = t.copy()

        Xprime = self.X_pipeline.fit_transform(X)
        tprime = self.t_pipeline[0].fit_transform(t, y)
        y = np.array([np.interp(tprime, t, yi) for yi in y])
        yprime = self.y_pipeline.fit_transform(y)

        tprime = self.t_pipeline[1].fit_transform(t.reshape(-1, 1))

        X, y = stack(Xprime, tprime, yprime)

        self.model = keras.models.Sequential(
            [keras.layers.Input(X.shape[-1:])]
            + self.network
            + [keras.layers.Dense(1, activation="linear")]
        )

        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        self.history = self.model.fit(
            X, y, epochs=self.epochs, batch_size=len(t), callbacks=self.callbacks,
            verbose=1, shuffle=True
        )

    def predict(self, X, t=None):
        """Predict the target.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        t : array-like of shape (n_target,)
            The independent variable for the target
            Defaults to the original training t

        Returns
        -------
        y : array-like of shape (n_samples, n_target)
            The predicted target
        """
        if t is None:
            t = self.t
        tprime = self.t_pipeline[1].transform(t.reshape(-1, 1))
        X = self.X_pipeline.transform(np.atleast_2d(X))
        X, _ = stack(X, np.atleast_1d(tprime))
        ypred = self.model.predict(X)
        yunstacked = unstack(ypred, t)
        y = self.y_pipeline.inverse_transform(yunstacked)
        return y
