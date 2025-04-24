"""Emulator base class."""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from stemu.skutils import CDFTransformer, ResampleYTransformer
from tensorflow import keras
from stemu.utils import stack, unstack


class Emu(object):

    def __init__(self, **kwargs):
        self.epochs = kwargs.get("epochs", 100)
        self.loss = kwargs.get("loss", "mse")
        self.callbacks = [keras.callbacks.EarlyStopping(monitor="loss", patience=100)]

        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)

        self.params_pipeline = kwargs.get('parampipe', 
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

    def fit(self, params, t, y):
        self.t = t.copy()

        paramsprime = self.params_pipeline.fit_transform(params)
        tprime = self.t_pipeline[0].fit_transform(t, y) # cdf
        y = np.array([np.interp(tprime, t, yi) for yi in y]) # resampling
        yprime = self.y_pipeline.fit_transform(y) # division by std

        tprime = self.t_pipeline[1].fit_transform(t.reshape(-1, 1)) # minmax

        params, y = stack(paramsprime, tprime, yprime)

        self.model = keras.models.Sequential(
            [keras.layers.Input(params.shape[-1:])]
            + self.network
            + [keras.layers.Dense(1, activation="linear")]
        )

        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        self.history = self.model.fit(
            params, y, epochs=self.epochs, 
            batch_size=len(t), callbacks=self.callbacks,
            verbose=1, shuffle=True
        )

    def predict(self, params, t=None):
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
        print(t.shape)
        tprime = self.t_pipeline[1].transform(t.reshape(-1, 1))
        print(type(tprime))
        #print(self.t_pipeline[0].__dict__)
        tprime = np.interp(t, self.t, self.t_pipeline[0].named_steps['cdf'].cdf)
        print(type(tprime))
        exit()
        #print(tprime.shape)
        params = self.params_pipeline.transform(params)
        print(params.shape)
        params, _ = stack(params, tprime)
        ypred = self.model.predict(params)

        """plt.plot(self.t_pipeline[1].inverse_transform(params[:450, -1].reshape(-1, 1)), ypred[:450], "o", label="pred")
        plt.show()
        exit()"""
        yunstacked = unstack(ypred, t)
        y = self.y_pipeline.inverse_transform(yunstacked)
        return y
