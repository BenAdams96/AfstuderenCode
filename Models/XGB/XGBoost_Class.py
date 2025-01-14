from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


import pandas as pd
import numpy as np

class XGBoostModel:
    def __init__(self, n_trees=100, max_depth=10, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, gamma=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma  # Added gamma parameter
        self.model: XGBRegressor = XGBRegressor(n_estimators=n_trees,
                                  max_depth=max_depth,
                                  learning_rate=learning_rate,
                                  subsample=subsample,
                                  colsample_bytree=colsample_bytree,
                                  gamma=gamma)  # Include gamma in the model
        self.top_features = None

    def __repr__(self):
        top_features_length = str(len(self.top_features)) if self.top_features is not None else 'None'
        return f'n={self.n_trees} md={self.max_depth} lr={self.learning_rate} gamma={self.gamma} f={top_features_length}'

    def __str__(self):
        return self.__repr__()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)  # For Root Mean Squared Error (RMSE)
        r2 = r2_score(y_test, predictions)
        return rmse, mse, r2

    def hyperparameter_tuning(self, X_data, y_data, param_grid, cv, scoring_):
        grid_search = GridSearchCV(estimator=self.model,
                                   param_grid=param_grid,
                                   cv=cv,
                                   scoring=scoring_,
                                   verbose=1,
                                   n_jobs=-1)
        grid_search.fit(X_data, y_data)
        self.model = grid_search.best_estimator_
        self.n_trees = self.model.get_params()['n_estimators']
        self.max_depth = self.model.get_params()['max_depth']
        self.learning_rate = self.model.get_params()['learning_rate']
        self.subsample = self.model.get_params()['subsample']
        self.colsample_bytree = self.model.get_params()['colsample_bytree']
        self.gamma = self.model.get_params()['gamma']  # Capture the gamma parameter
        self.top_features = self.feature_importances(X_data)
        return grid_search

    def feature_importances(self, X_train):
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("The model does not have feature_importances_ attribute.")
        feature_importances = pd.Series(self.model.feature_importances_, index=X_train.columns)
        return feature_importances.sort_values(ascending=False)  # NOTE: this needs to be a panda series

