"""
Baseline Models for NHL Player Performance Prediction
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class BaselineModels:
    """Collection of baseline models for NHL player performance prediction."""

    def __init__(self, random_state: int = 42):
        """Initialize baseline models.

        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.fitted_models = {}
        self.model_scores = {}

    def get_dummy_regressor(self) -> DummyRegressor:
        """Get a dummy regressor (simplest baseline).

        Returns:
            Configured DummyRegressor
        """
        return DummyRegressor(strategy='mean')

    def get_linear_regression(self) -> LinearRegression:
        """Get a linear regression model.

        Returns:
            Configured LinearRegression
        """
        return LinearRegression()

    def get_ridge_regression(self, alpha: float = 1.0) -> Ridge:
        """Get a Ridge regression model.

        Args:
            alpha: Regularization strength

        Returns:
            Configured Ridge regression
        """
        return Ridge(alpha=alpha, random_state=self.random_state)

    def get_lasso_regression(self, alpha: float = 0.1) -> Lasso:
        """Get a Lasso regression model.

        Args:
            alpha: Regularization strength

        Returns:
            Configured Lasso regression
        """
        return Lasso(alpha=alpha, random_state=self.random_state, max_iter=1000)

    def get_elastic_net(self, alpha: float = 0.1, l1_ratio: float = 0.5) -> ElasticNet:
        """Get an Elastic Net regression model.

        Args:
            alpha: Regularization strength
            l1_ratio: L1 ratio in the penalty

        Returns:
            Configured ElasticNet regression
        """
        return ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=self.random_state,
            max_iter=1000
        )

    def get_decision_tree(self, max_depth: int = 10) -> DecisionTreeRegressor:
        """Get a decision tree regressor.

        Args:
            max_depth: Maximum depth of the tree

        Returns:
            Configured DecisionTreeRegressor
        """
        return DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=self.random_state
        )

    def get_knn_regressor(self, n_neighbors: int = 5) -> KNeighborsRegressor:
        """Get a k-nearest neighbors regressor.

        Args:
            n_neighbors: Number of neighbors

        Returns:
            Configured KNeighborsRegressor
        """
        return KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')

    def get_all_baseline_models(self) -> Dict[str, Any]:
        """Get all baseline models with default parameters.

        Returns:
            Dictionary of model name to model instance
        """
        models = {
            'dummy': self.get_dummy_regressor(),
            'linear': self.get_linear_regression(),
            'ridge': self.get_ridge_regression(),
            'lasso': self.get_lasso_regression(),
            'elastic_net': self.get_elastic_net(),
            'decision_tree': self.get_decision_tree(),
            'knn': self.get_knn_regressor()
        }

        self.models = models
        return models

    def tune_ridge_regression(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Ridge:
        """Tune Ridge regression hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable
            cv: Cross-validation folds

        Returns:
            Best Ridge regression model
        """
        logger.info("Tuning Ridge regression...")

        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        }

        ridge = Ridge(random_state=self.random_state)
        grid_search = GridSearchCV(
            ridge, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
        )

        grid_search.fit(X, y)
        logger.info(f"Best Ridge alpha: {grid_search.best_params_['alpha']}")

        return grid_search.best_estimator_

    def tune_lasso_regression(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Lasso:
        """Tune Lasso regression hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable
            cv: Cross-validation folds

        Returns:
            Best Lasso regression model
        """
        logger.info("Tuning Lasso regression...")

        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        }

        lasso = Lasso(random_state=self.random_state, max_iter=1000)
        grid_search = GridSearchCV(
            lasso, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
        )

        grid_search.fit(X, y)
        logger.info(f"Best Lasso alpha: {grid_search.best_params_['alpha']}")

        return grid_search.best_estimator_

    def tune_decision_tree(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> DecisionTreeRegressor:
        """Tune Decision Tree hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable
            cv: Cross-validation folds

        Returns:
            Best DecisionTreeRegressor model
        """
        logger.info("Tuning Decision Tree...")

        param_grid = {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [10, 20, 50],
            'min_samples_leaf': [5, 10, 20]
        }

        dt = DecisionTreeRegressor(random_state=self.random_state)
        grid_search = GridSearchCV(
            dt, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
        )

        grid_search.fit(X, y)
        logger.info(f"Best Decision Tree params: {grid_search.best_params_}")

        return grid_search.best_estimator_

    def tune_knn_regressor(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> KNeighborsRegressor:
        """Tune k-NN hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable
            cv: Cross-validation folds

        Returns:
            Best KNeighborsRegressor model
        """
        logger.info("Tuning k-NN regressor...")

        param_grid = {
            'n_neighbors': [3, 5, 7, 10, 15, 20],
            'weights': ['uniform', 'distance']
        }

        knn = KNeighborsRegressor()
        grid_search = GridSearchCV(
            knn, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
        )

        grid_search.fit(X, y)
        logger.info(f"Best k-NN params: {grid_search.best_params_}")

        return grid_search.best_estimator_

    def fit_all_models(self, X: np.ndarray, y: np.ndarray, tune: bool = False, cv: int = 5) -> Dict[str, Any]:
        """Fit all baseline models.

        Args:
            X: Feature matrix
            y: Target variable
            tune: Whether to tune hyperparameters
            cv: Cross-validation folds for tuning

        Returns:
            Dictionary of fitted models
        """
        logger.info(f"Fitting {'tuned' if tune else 'default'} baseline models...")

        if not self.models:
            self.get_all_baseline_models()

        fitted_models = {}

        for name, model in self.models.items():
            try:
                logger.info(f"Fitting {name}...")

                if tune and name in ['ridge', 'lasso', 'decision_tree', 'knn']:
                    if name == 'ridge':
                        fitted_model = self.tune_ridge_regression(X, y, cv)
                    elif name == 'lasso':
                        fitted_model = self.tune_lasso_regression(X, y, cv)
                    elif name == 'decision_tree':
                        fitted_model = self.tune_decision_tree(X, y, cv)
                    elif name == 'knn':
                        fitted_model = self.tune_knn_regressor(X, y, cv)
                    else:
                        fitted_model = model.fit(X, y)
                else:
                    fitted_model = model.fit(X, y)

                fitted_models[name] = fitted_model
                logger.info(f"✅ {name} fitted successfully")

            except Exception as e:
                logger.error(f"❌ Error fitting {name}: {e}")
                continue

        self.fitted_models = fitted_models
        logger.info(f"Fitted {len(fitted_models)} baseline models")

        return fitted_models

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate all fitted models.

        Args:
            X_test: Test feature matrix
            y_test: Test target variable

        Returns:
            Dictionary of model scores
        """
        if not self.fitted_models:
            raise ValueError("No models have been fitted. Call fit_all_models first.")

        logger.info("Evaluating baseline models...")

        scores = {}

        for name, model in self.fitted_models.items():
            try:
                y_pred = model.predict(X_test)

                model_scores = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                }

                scores[name] = model_scores
                logger.info(f"{name} - RMSE: {model_scores['rmse']:.4f}, R²: {model_scores['r2']:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                continue

        self.model_scores = scores
        return scores

    def get_feature_importance(self, feature_names: Optional[list] = None) -> Dict[str, np.ndarray]:
        """Get feature importance from fitted models that support it.

        Args:
            feature_names: Optional list of feature names

        Returns:
            Dictionary of feature importances
        """
        if not self.fitted_models:
            raise ValueError("No models have been fitted. Call fit_all_models first.")

        importance_dict = {}

        for name, model in self.fitted_models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    importance_dict[name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # Linear models
                    importance_dict[name] = np.abs(model.coef_)
                else:
                    logger.info(f"Model {name} doesn't support feature importance")
                    continue

                logger.info(f"Feature importance extracted for {name}")

            except Exception as e:
                logger.error(f"Error getting feature importance for {name}: {e}")
                continue

        return importance_dict

    def get_best_model(self, metric: str = 'rmse') -> Tuple[str, Any, float]:
        """Get the best performing model based on a metric.

        Args:
            metric: Metric to use for selection ('rmse', 'mae', 'r2')

        Returns:
            Tuple of (model_name, model_instance, best_score)
        """
        if not self.model_scores:
            raise ValueError("No model scores available. Call evaluate_models first.")

        if metric == 'r2':
            # For R², higher is better
            best_score = max(self.model_scores.items(), key=lambda x: x[1][metric])
        else:
            # For RMSE, MAE, lower is better
            best_score = min(self.model_scores.items(), key=lambda x: x[1][metric])

        best_name = best_score[0]
        best_model = self.fitted_models[best_name]
        score_value = best_score[1][metric]

        logger.info(f"Best model: {best_name} with {metric}={score_value:.4f}")

        return best_name, best_model, score_value

    def get_model_summary(self) -> pd.DataFrame:
        """Get a summary DataFrame of all model performances.

        Returns:
            DataFrame with model performance metrics
        """
        if not self.model_scores:
            raise ValueError("No model scores available. Call evaluate_models first.")

        summary_data = []
        for name, scores in self.model_scores.items():
            row = {'model': name}
            row.update(scores)
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('rmse')  # Sort by RMSE (lower is better)

        return summary_df