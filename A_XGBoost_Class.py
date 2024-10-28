from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

class XGBoostModel:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, min_child_weight=1, subsample=1, colsample_bytree=1):
        """Initialize the XGBoostModel with the given parameters."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective='reg:squarederror',  # For regression tasks
            n_jobs=-1
        )
        self.top_features = None

    def __repr__(self):
        top_features_length = str(len(self.top_features) if self.top_features is not None else 0)
        return f'XGBoostModel(n_estimators={self.n_estimators}, max_depth={self.max_depth}, top_features={top_features_length})'

    def __str__(self):
        return self.__repr__()
    
    def fit(self, X_train, y_train):
        """Fit the XGBoost model to the training data."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Predict using the trained XGBoost model."""
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model's performance on the test data."""
        predictions = self.predict(X_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return rmse, mse, r2
    
    def hyperparameter_tuning(self, X_data, y_data, param_grid, cv, scoring_):
        """Perform hyperparameter tuning using GridSearchCV."""
        grid_search = GridSearchCV(estimator=self.model,
                                   param_grid=param_grid,
                                   cv=cv,
                                   scoring=scoring_,
                                   verbose=0,
                                   n_jobs=-1)
        grid_search.fit(X_data, y_data)
        # Update the model with the best parameters found
        self.model = grid_search.best_estimator_
        self.n_estimators = self.model.get_params()['n_estimators']
        self.max_depth = self.model.get_params()['max_depth']
        self.learning_rate = self.model.get_params()['learning_rate']
        self.min_child_weight = self.model.get_params()['min_child_weight']
        self.subsample = self.model.get_params()['subsample']
        self.colsample_bytree = self.model.get_params()['colsample_bytree']
        self.top_features = self.feature_importances(X_data)
        return grid_search
    
    def feature_importances(self, X_train):
        """Get feature importances from the trained model."""
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("The model does not have feature_importances_ attribute.")
        feature_importances = pd.Series(self.model.feature_importances_, index=X_train.columns)
        return feature_importances.sort_values(ascending=False)  # NOTE: this needs to be a pandas series!