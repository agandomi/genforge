# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np

class LinearRegressionModel:
    def __init__(self, **params):
        self.params = {
            'xtrain': None,
            'ytrain': None,
            'xval': None,
            'yval': None,
            **params
        }
        self.weight = None
        self.bias = None
        
    @classmethod
    def compiler(cls, **params):
        return cls(**params)
    
    def predict(self, X):
        return X @ self.weight + self.bias
    
    def loss(self, y_pred, y_true):
        return np.mean((y_pred.flatten() - y_true.flatten()) ** 2)
    
    def fit(self):
        X_train = self.params['xtrain']
        y_train = self.params['ytrain']
        
        X_augmented = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        w = np.linalg.pinv(X_augmented.T @ X_augmented) @ X_augmented.T @ y_train
        
        self.weight = np.array(w[:-1]).flatten()
        self.bias = np.array([w[-1]])
        