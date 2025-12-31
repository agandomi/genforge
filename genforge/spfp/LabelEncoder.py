# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
import pandas as pd

class LabelEncoder:
    """
    Minimal clone of sklearn.preprocessing.LabelEncoder.

    - Fits sorted unique classes_. (Sklearn sorts unique labels.)
    - transform() maps labels -> integers [0..n_classes-1].
    - inverse_transform() maps integers back to original labels.
    - Accepts 1D numpy arrays / pandas Series; returns 1D numpy arrays.
    """

    def __init__(self):
        self.classes_ = None
        self._class_to_index = None

    def fit(self, y):
        yv = self._to_numpy_1d(y)
        # sklearn sorts unique classes
        self.classes_ = np.unique(yv)
        self._class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self

    def transform(self, y):
        if self._class_to_index is None:
            raise RuntimeError("LabelEncoder must be fit before transform.")
        yv = self._to_numpy_1d(y)
        try:
            return np.array([self._class_to_index[val] for val in yv], dtype=int)
        except KeyError as e:
            raise ValueError(f"y contains previously unseen label: {e.args[0]!r}")

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y_idx):
        if self.classes_ is None:
            raise RuntimeError("LabelEncoder must be fit before inverse_transform.")
        yi = np.asarray(y_idx, dtype=int).ravel()
        if yi.min(initial=0) < 0 or yi.max(initial=-1) >= len(self.classes_):
            raise ValueError("y contains class indices out of range.")
        return self.classes_[yi]

    @staticmethod
    def _to_numpy_1d(y):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            yv = np.asarray(y).ravel()
        else:
            yv = np.asarray(y).ravel()
        return yv