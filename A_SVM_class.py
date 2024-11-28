from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

class SupportVectorMachineRegressor:
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.top_features = None

    def __repr__(self):
        return f'kernel={self.kernel} C={self.C} epsilon={self.epsilon} gamma={self.gamma}'

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
                                   verbose=2,
                                   n_jobs=-1)
        grid_search.fit(X_data, y_data)
        self.model = grid_search.best_estimator_
        self.kernel = self.model.get_params()['kernel']
        self.C = self.model.get_params()['C']
        self.epsilon = self.model.get_params()['epsilon']
        self.gamma = self.model.get_params()['gamma']
        self.top_features = self.get_feature_weights(X_data)  # Calculate pseudo-feature importance
        return grid_search

    def get_feature_weights(self, X_train):
        # Although SVR doesn't provide feature importances, we can estimate them based on model coefficients if linear
        if self.kernel == 'linear':
            weights = np.abs(self.model.coef_).flatten()
            feature_importances = pd.Series(weights, index=X_train.columns)
            return feature_importances.sort_values(ascending=False)
        else:
            return "Feature importance not available for non-linear kernels."
