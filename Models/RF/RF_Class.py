from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

class RandomForestModel:
    def __init__(self, n_trees, max_depth=10, min_samples_split=5, max_features='sqrt'):
        """Initialize the RandomForestModel with the given parameters."""
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_samples_split
        self.max_features = max_features
        self.model = RandomForestRegressor(n_estimators=n_trees,
                                           max_depth = max_depth,
                                           min_samples_split=min_samples_split,
                                           max_features=max_features)
        self.top_features = None

    def __repr__(self):
        top_features_length = str(len(self.top_features))
        trees = str(self.n_trees)
        maxdepth = str(self.max_depth)
        return f'n={trees} md={maxdepth} f={top_features_length}'
        # attrs = ', '.join(f"{key}={value}" for key, value in self.__dict__.items() if key not in ['model', 'top_features'])
        
        # #NOTE: to see the length of the top_features
        # def __repr__(self):
        #     attrs = ', '.join(f"{key}={value}" for key, value in self.__dict__.items() if key != 'model')
            
        #     if isinstance(self.top_features, pd.Series):
        #         top_features_length = len(self.top_features)
        #         attrs += f", top_features_length={top_features_length}"
        #     else:
        #         attrs += f", top_features={self.top_features}"

        # #NOTE: to see the top 10
        # # if isinstance(self.top_features, pd.Series):
        # #     top_features_list = self.top_features.head(10).tolist()
        # #     attrs += f", top_features={top_features_list}"
        # # else:
        # #     attrs += f", top_features={self.top_features}"

        # return f"{self.__class__.__name__}({attrs})"
    
    def __str__(self):
        return self.__repr__()
    
    def fit(self, X_train, y_train):
        """Fit the Random Forest model to the training data."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Predict using the trained Random Forest model."""
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model's performance on the test data."""
        predictions = self.model.predict(X_test)   ### use self.model.predict or self.predict?
        rmse = root_mean_squared_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return rmse, mse, r2
    
    def hyperparameter_tuning(self, X_data, y_data, param_grid, cv, scoring_):
        """Perform hyperparameter tuning using GridSearchCV."""
        grid_search = GridSearchCV(estimator=self.model,
                                   param_grid=param_grid,
                                   cv=cv,
                                   scoring=scoring_,
                                   verbose=1,
                                   n_jobs=-1)
        grid_search.fit(X_data, y_data)
        # print('done search')

        # # Store validation predictions and true values
        # val_predictions = []
        # val_true_values = []

        # for train_idx, val_idx in cv:
        #     X_val_fold = X_data.iloc[val_idx]
        #     y_val_fold = y_data.iloc[val_idx]

        #     best_model = grid_search.best_estimator_
        #     # Predict for validation set
        #     y_val_pred = best_model.predict(X_val_fold)
        #     val_predictions.extend(y_val_pred)
        #     val_true_values.extend(y_val_fold)
        #     self.plot_predicted_vs_real_pKi(y_val_fold, y_val_pred,8)
        
        # for i, (train_indices, test_indices) in enumerate(grid_search.cv):
        #     print(f"Split {i}: Train indices: {train_indices}, Test indices: {test_indices}")

        # Update the model with the best parameters found
        self.model = grid_search.best_estimator_
        self.n_trees = len(self.model.estimators_)
        self.max_features = self.model.get_params()['max_features']
        self.max_depth = self.model.get_params()['max_depth']
        self.min_size = self.model.get_params()['min_samples_split']
        #self.feature_importances = grid_search.best_estimator_.feature_importances_
        self.top_features = self.feature_importances(X_data)
        return grid_search
    
    def feature_importances(self, X_train):
        """Get feature importances from the trained model."""
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("The model does not have feature_importances_ attribute.")
        feature_importances = pd.Series(self.model.feature_importances_, index=X_train.columns)
        return feature_importances.sort_values(ascending=False) #NOTE: this needs to be a pandas series!
    
    def plot_predicted_vs_real_pKi(self, y_true, y_pred, number):
        """
        Plots predicted pKi values against real pKi values for model evaluation.

        Parameters:
            y_true (array-like): The true pKi values.
            y_pred (array-like): The predicted pKi values.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolors='k', s=80)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')  # Ideal line
        plt.xlabel('Real pKi Values')
        plt.ylabel('Predicted pKi Values')
        plt.title('Predicted vs Real pKi Values')
        plt.grid(True)
        plt.savefig(f'true_vs_predict_{number}.png')
