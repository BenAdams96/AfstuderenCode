from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

# import csv_to_dataframes

import matplotlib.pyplot as plt
import itertools
from typing import List
import numpy as np
from pathlib import Path

import pandas as pd
import math
import re
import os

class BaseModel:
    def __init__(self, model, random_state=42, **kwargs):
        self.model = model
        # Set the random_state directly in the model's parameters
        self.model.set_params(random_state=random_state, **kwargs)
        self.best_model = None

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """Evaluate the model's performance on the provided data."""
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        return rmse, mse, r2
    
    def hyperparameter_tuning(self, X, y, param_grid, cv, scoring='r2') -> GridSearchCV:
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X, y)
        self.best_model = grid_search.best_estimator_
        return grid_search

# Model classes
class RandomForestModel(BaseModel):
    def __init__(self, random_state=42, **kwargs):
        rf_model = RandomForestRegressor()
        super().__init__(model=rf_model, random_state=random_state, **kwargs)

class XGBoostModel(BaseModel):
    def __init__(self, random_state=42, **kwargs):
        xgb_model = XGBRegressor()
        super().__init__(model=xgb_model, random_state=random_state, **kwargs)

class SVRModel(BaseModel):
    def __init__(self, random_state=42, **kwargs):
        svr_model = SVR()
        super().__init__(model=svr_model, random_state=random_state, **kwargs)



# class RandomForestModel2(BaseModel):
#     def __init__(self, random_state=42, n_trees=100, max_depth=10, min_samples_split=5, max_features='sqrt'):
#         rf_model = RandomForestRegressor(
#             n_estimators=n_trees,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             max_features=max_features,
#             random_state=random_state
#         )
#         super().__init__(model=rf_model, random_state=random_state)

# class XGBoostModel2(BaseModel):
#     def __init__(self, random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1):
#         xgb_model = XGBRegressor(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             learning_rate=learning_rate,
#             random_state=random_state
#         )
#         super().__init__(model=xgb_model, random_state=random_state)

# class SVRModel2(BaseModel):
#     def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
#         svr_model = SVR(kernel=kernel, C=C, epsilon=epsilon)
#         super().__init__(model=svr_model)

# # Usage:
# rf_model = RandomForestModel()
# grid_search_rf = rf_model.hyperparameter_tuning(X, y, param_grid, cv=custom_inner_splits, scoring='r2')

# xgb_model = XGBoostModel()
# grid_search_xgb = xgb_model.hyperparameter_tuning(X, y, param_grid, cv=custom_inner_splits, scoring='r2')

# svr_model = SVRModel()
# grid_search_svr = svr_model.hyperparameter_tuning(X, y, param_grid, cv=custom_inner_splits, scoring='r2')

