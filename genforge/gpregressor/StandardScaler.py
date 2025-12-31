# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
import pandas as pd

class StandardScaler:
    """
    Minimal clone of sklearn.preprocessing.StandardScaler (with_mean=True, with_std=True).

    - Fits per-column mean and population variance (ddof=0), like sklearn.
    - During transform, divides by std; zero-variance columns use scale=1.0
      to avoid division by zero (matching sklearn behavior).
    - Accepts numpy arrays or pandas DataFrames. Always returns a numpy array.
    """

    def __init__(self):
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X):
        Xv = self._to_numpy_2d(X)
        self.n_features_in_ = Xv.shape[1]
        self.mean_ = Xv.mean(axis=0)
        # population variance (ddof=0), same as sklearn
        self.var_ = Xv.var(axis=0)
        scale = np.sqrt(self.var_)
        # sklearn avoids division by zero by replacing 0 std with 1.0 in scale_
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(list(X.columns), dtype=object)
        else:
            self.feature_names_in_ = None
        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fit before transform.")
        Xv = self._to_numpy_2d(X)
        if Xv.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {Xv.shape[1]}."
            )
        return (Xv - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        Xs = self._to_numpy_2d(X_scaled)
        if Xs.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {Xs.shape[1]}."
            )
        return Xs * self.scale_ + self.mean_

    @staticmethod
    def _to_numpy_2d(X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            Xv = np.asarray(X)
        else:
            Xv = X
        Xv = np.asarray(Xv)
        if Xv.ndim == 1:
            Xv = Xv.reshape(-1, 1)
        if Xv.ndim != 2:
            raise ValueError("Input must be 2D after conversion.")
        return Xv