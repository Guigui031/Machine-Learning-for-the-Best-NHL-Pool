"""
Advanced Models for NHL Player Performance Prediction
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("LightGBM not available")


class AdvancedModels:
    """Collection of advanced models for NHL player performance prediction."""

    def __init__(self, random_state: int = 42):
        """Initialize advanced models.

        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.fitted_models = {}
        self.model_scores = {}

    def get_random_forest(self, n_estimators: int = 100, max_depth: int = 15) -> RandomForestRegressor:
        """Get a Random Forest regressor.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees

        Returns:
            Configured RandomForestRegressor
        """
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1
        )

    def get_gradient_boosting(self, n_estimators: int = 100, max_depth: int = 6) -> GradientBoostingRegressor:
        """Get a Gradient Boosting regressor.

        Args:
            n_estimators: Number of boosting stages
            max_depth: Maximum depth of trees

        Returns:
            Configured GradientBoostingRegressor
        """
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=self.random_state
        )

    def get_xgboost(self, n_estimators: int = 100, max_depth: int = 6) -> Any:
        """Get an XGBoost regressor.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees

        Returns:
            Configured XGBRegressor or None if not available
        """
        if not HAS_XGB:
            logger.warning("XGBoost not available")
            return None

        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='rmse'
        )

    def get_lightgbm(self, n_estimators: int = 100, max_depth: int = 6) -> Any:
        """Get a LightGBM regressor.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees

        Returns:
            Configured LGBMRegressor or None if not available
        """
        if not HAS_LGB:
            logger.warning("LightGBM not available")
            return None

        return lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )

    def get_svr(self, kernel: str = 'rbf', C: float = 1.0) -> SVR:
        """Get a Support Vector Regressor.

        Args:
            kernel: Kernel type
            C: Regularization parameter

        Returns:
            Configured SVR
        """
        return SVR(kernel=kernel, C=C, gamma='auto')

    def get_neural_network(self, hidden_layer_sizes: Tuple[int, ...] = (100, 50)) -> MLPRegressor:
        """Get a Multi-layer Perceptron regressor.

        Args:
            hidden_layer_sizes: Size of hidden layers

        Returns:
            Configured MLPRegressor
        """
        return MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )

    def get_all_advanced_models(self) -> Dict[str, Any]:
        """Get all advanced models with default parameters.

        Returns:
            Dictionary of model name to model instance
        """
        models = {
            'random_forest': self.get_random_forest(),
            'gradient_boosting': self.get_gradient_boosting(),
            'svr': self.get_svr(),
            'neural_network': self.get_neural_network()
        }

        # Add XGBoost if available
        if HAS_XGB:
            models['xgboost'] = self.get_xgboost()

        # Add LightGBM if available
        if HAS_LGB:
            models['lightgbm'] = self.get_lightgbm()

        self.models = models
        return models

    def tune_random_forest(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> RandomForestRegressor:
        """Tune Random Forest hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable
            cv: Cross-validation folds

        Returns:
            Best RandomForestRegressor model
        """
        logger.info("Tuning Random Forest...")

        param_distributions = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10]
        }

        rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        random_search = RandomizedSearchCV(
            rf, param_distributions, n_iter=20, cv=cv,
            scoring='neg_mean_squared_error', n_jobs=-1, random_state=self.random_state
        )

        random_search.fit(X, y)
        logger.info(f"Best Random Forest params: {random_search.best_params_}")

        return random_search.best_estimator_

    def tune_xgboost(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Any:
        """Tune XGBoost hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable
            cv: Cross-validation folds

        Returns:
            Best XGBRegressor model or None if not available
        """
        if not HAS_XGB:
            logger.warning("XGBoost not available for tuning")
            return None

        logger.info("Tuning XGBoost...")

        param_distributions = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        xgb_model = xgb.XGBRegressor(
            random_state=self.random_state, n_jobs=-1, eval_metric='rmse'
        )
        random_search = RandomizedSearchCV(
            xgb_model, param_distributions, n_iter=20, cv=cv,
            scoring='neg_mean_squared_error', n_jobs=-1, random_state=self.random_state
        )

        random_search.fit(X, y)
        logger.info(f"Best XGBoost params: {random_search.best_params_}")

        return random_search.best_estimator_

    def tune_svr(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> SVR:
        """Tune SVR hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable
            cv: Cross-validation folds

        Returns:
            Best SVR model
        """
        logger.info("Tuning SVR...")

        param_distributions = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['auto', 'scale', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly']
        }

        svr = SVR()
        random_search = RandomizedSearchCV(
            svr, param_distributions, n_iter=20, cv=cv,
            scoring='neg_mean_squared_error', n_jobs=-1, random_state=self.random_state
        )

        random_search.fit(X, y)
        logger.info(f"Best SVR params: {random_search.best_params_}")

        return random_search.best_estimator_

    def tune_neural_network(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> MLPRegressor:
        """Tune Neural Network hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable
            cv: Cross-validation folds

        Returns:
            Best MLPRegressor model
        """
        logger.info("Tuning Neural Network...")

        param_distributions = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100), (150, 100, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }

        mlp = MLPRegressor(
            max_iter=500, random_state=self.random_state,
            early_stopping=True, validation_fraction=0.1
        )
        random_search = RandomizedSearchCV(
            mlp, param_distributions, n_iter=10, cv=cv,
            scoring='neg_mean_squared_error', n_jobs=-1, random_state=self.random_state
        )

        random_search.fit(X, y)
        logger.info(f"Best Neural Network params: {random_search.best_params_}")

        return random_search.best_estimator_

    def fit_all_models(self, X: np.ndarray, y: np.ndarray, tune: bool = False, cv: int = 5) -> Dict[str, Any]:
        """Fit all advanced models.

        Args:
            X: Feature matrix
            y: Target variable
            tune: Whether to tune hyperparameters
            cv: Cross-validation folds for tuning

        Returns:
            Dictionary of fitted models
        """
        logger.info(f"Fitting {'tuned' if tune else 'default'} advanced models...")

        if not self.models:
            self.get_all_advanced_models()

        fitted_models = {}

        for name, model in self.models.items():
            if model is None:  # Skip unavailable models
                continue

            try:
                logger.info(f"Fitting {name}...")

                if tune:
                    if name == 'random_forest':
                        fitted_model = self.tune_random_forest(X, y, cv)
                    elif name == 'xgboost' and HAS_XGB:
                        fitted_model = self.tune_xgboost(X, y, cv)
                    elif name == 'svr':
                        fitted_model = self.tune_svr(X, y, cv)
                    elif name == 'neural_network':
                        fitted_model = self.tune_neural_network(X, y, cv)
                    else:
                        fitted_model = model.fit(X, y)
                else:
                    fitted_model = model.fit(X, y)

                fitted_models[name] = fitted_model
                logger.info(f"{name} fitted successfully")

            except Exception as e:
                logger.error(f"Error fitting {name}: {e}")
                continue

        self.fitted_models = fitted_models
        logger.info(f"Fitted {len(fitted_models)} advanced models")

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

        logger.info("Evaluating advanced models...")

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

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
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
                    # Tree-based models (RF, XGBoost, LightGBM, etc.)
                    importance_dict[name] = model.feature_importances_
                elif hasattr(model, 'coef_') and hasattr(model.coef_, '__len__'):
                    # Linear models with coefficients
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

        logger.info(f"Best advanced model: {best_name} with {metric}={score_value:.4f}")

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