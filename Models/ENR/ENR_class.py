from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

class ElasticNetRegressor:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=2000, tol=1e-4):
        """
        ElasticNet regression model.
        
        Parameters:
        - alpha: Regularization strength
        - l1_ratio: The mixing parameter between L1 and L2 regularization
        - max_iter: Maximum number of iterations
        - tol: Tolerance for stopping criteria
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol)
        self.feature_importances = None

    def __repr__(self):
        return f'alpha={self.alpha} l1_ratio={self.l1_ratio} max_iter={self.max_iter} tol={self.tol}'

    def __str__(self):
        return self.__repr__()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.feature_importances = self.get_feature_weights(X_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)  # Root Mean Squared Error
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
        self.alpha = self.model.alpha
        self.l1_ratio = self.model.l1_ratio
        self.max_iter = self.model.max_iter
        self.tol = self.model.tol
        self.feature_importances = self.get_feature_weights(X_data)
        return grid_search

    def get_feature_weights(self, X_train):
        """
        Return feature importances based on coefficients.
        """
        coefficients = np.abs(self.model.coef_)
        feature_importances = pd.Series(coefficients, index=X_train.columns)
        return feature_importances.sort_values(ascending=False)
    
    def print_feature_stats(self, X_train):
        """
        Print statistics about the features, such as:
        - Top 10 most important features
        - Number of zero coefficients (discarded features)
        - Max, Min, and Mean importance
        """
        feature_importances = self.get_feature_weights(X_train)

        print("\nFeature Importance Statistics:")
        
        # Print top 10 important features
        print("\nTop 10 Features (by Importance):")
        print(feature_importances.head(10))
        
        # Count the number of zero coefficients (features excluded)
        zero_coeffs = np.sum(feature_importances == 0)
        print(f"\nNumber of Features with Zero Coefficients (Excluded from Model): {zero_coeffs}")
        
        # Max, Min, Mean importance
        print(f"\nMax Feature Importance: {feature_importances.max()}")
        print(f"Min Feature Importance: {feature_importances.min()}")
        print(f"Mean Feature Importance: {feature_importances.mean()}")