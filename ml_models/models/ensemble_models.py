"""
Ensemble Models for NHL Player Performance Prediction
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import VotingRegressor, StackingRegressor, BaggingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Import from our modules
from .baseline_models import BaselineModels
from .advanced_models import AdvancedModels

logger = logging.getLogger(__name__)

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class EnsembleModels:
    """Collection of ensemble models for NHL player performance prediction."""

    def __init__(self, random_state: int = 42):
        """Initialize ensemble models.

        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.baseline_models = BaselineModels(random_state)
        self.advanced_models = AdvancedModels(random_state)
        self.ensemble_models = {}
        self.fitted_ensembles = {}
        self.ensemble_scores = {}

    def get_voting_regressor(self, voting: str = 'soft') -> VotingRegressor:
        """Get a Voting Regressor ensemble.

        Args:
            voting: Voting strategy ('hard' or 'soft')

        Returns:
            Configured VotingRegressor
        """
        # Base estimators for voting
        estimators = [
            ('ridge', Ridge(alpha=1.0, random_state=self.random_state)),
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=self.random_state, n_jobs=-1)),
            ('svr', SVR(kernel='rbf', C=1.0))
        ]

        # Add XGBoost if available
        if HAS_XGB:
            estimators.append(('xgb', xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, n_jobs=-1, eval_metric='rmse'
            )))

        return VotingRegressor(estimators=estimators)

    def get_stacking_regressor(self, final_estimator=None) -> StackingRegressor:
        """Get a Stacking Regressor ensemble.

        Args:
            final_estimator: Meta-learner (default: Ridge)

        Returns:
            Configured StackingRegressor
        """
        if final_estimator is None:
            final_estimator = Ridge(alpha=1.0, random_state=self.random_state)

        # Base estimators for stacking
        estimators = [
            ('ridge', Ridge(alpha=1.0, random_state=self.random_state)),
            ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=self.random_state, n_jobs=-1)),
            ('svr', SVR(kernel='rbf', C=1.0))
        ]

        # Add XGBoost if available
        if HAS_XGB:
            estimators.append(('xgb', xgb.XGBRegressor(
                n_estimators=50, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, n_jobs=-1, eval_metric='rmse'
            )))

        return StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5
        )

    def get_bagging_regressor(self, base_estimator=None, n_estimators: int = 10) -> BaggingRegressor:
        """Get a Bagging Regressor ensemble.

        Args:
            base_estimator: Base estimator to bag (default: RandomForest)
            n_estimators: Number of base estimators

        Returns:
            Configured BaggingRegressor
        """
        if base_estimator is None:
            base_estimator = RandomForestRegressor(
                n_estimators=50, max_depth=15, random_state=self.random_state
            )

        return BaggingRegressor(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )

    def get_custom_weighted_ensemble(self, models: Dict[str, Any], weights: Optional[List[float]] = None):
        """Create a custom weighted ensemble.

        Args:
            models: Dictionary of model name to fitted model
            weights: Optional weights for each model

        Returns:
            Custom ensemble class
        """
        class WeightedEnsemble:
            def __init__(self, models, weights=None):
                self.models = models
                self.weights = weights or [1.0 / len(models)] * len(models)
                self.model_names = list(models.keys())

            def predict(self, X):
                predictions = []
                for name in self.model_names:
                    pred = self.models[name].predict(X)
                    predictions.append(pred)

                predictions = np.array(predictions)
                weighted_pred = np.average(predictions, axis=0, weights=self.weights)
                return weighted_pred

            def fit(self, X, y):
                # Models are already fitted
                return self

        return WeightedEnsemble(models, weights)

    def get_all_ensemble_models(self) -> Dict[str, Any]:
        """Get all ensemble models.

        Returns:
            Dictionary of ensemble model name to model instance
        """
        ensembles = {
            'voting': self.get_voting_regressor(),
            'stacking': self.get_stacking_regressor(),
            'stacking_linear': self.get_stacking_regressor(LinearRegression()),
            'bagging_rf': self.get_bagging_regressor(),
        }

        # Add different voting strategies
        if HAS_XGB:
            ensembles['voting_xgb_heavy'] = VotingRegressor([
                ('xgb1', xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=self.random_state, eval_metric='rmse')),
                ('xgb2', xgb.XGBRegressor(n_estimators=150, max_depth=8, learning_rate=0.05, random_state=self.random_state+1, eval_metric='rmse')),
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=self.random_state))
            ])

        self.ensemble_models = ensembles
        return ensembles

    def fit_ensemble_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit all ensemble models.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Dictionary of fitted ensemble models
        """
        logger.info("Fitting ensemble models...")

        if not self.ensemble_models:
            self.get_all_ensemble_models()

        fitted_ensembles = {}

        for name, model in self.ensemble_models.items():
            try:
                logger.info(f"Fitting ensemble {name}...")
                fitted_model = model.fit(X, y)
                fitted_ensembles[name] = fitted_model
                logger.info(f"✅ Ensemble {name} fitted successfully")

            except Exception as e:
                logger.error(f"❌ Error fitting ensemble {name}: {e}")
                continue

        self.fitted_ensembles = fitted_ensembles
        logger.info(f"Fitted {len(fitted_ensembles)} ensemble models")

        return fitted_ensembles

    def create_adaptive_ensemble(self, X: np.ndarray, y: np.ndarray,
                                validation_size: float = 0.2) -> Any:
        """Create an adaptive ensemble that weights models based on validation performance.

        Args:
            X: Feature matrix
            y: Target variable
            validation_size: Fraction of data to use for validation

        Returns:
            Adaptive ensemble model
        """
        logger.info("Creating adaptive ensemble...")

        from sklearn.model_selection import train_test_split

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_size, random_state=self.random_state
        )

        # Fit base models
        base_models = {
            'ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'rf': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=self.random_state, n_jobs=-1),
            'svr': SVR(kernel='rbf', C=1.0)
        }

        if HAS_XGB:
            base_models['xgb'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, n_jobs=-1, eval_metric='rmse'
            )

        fitted_models = {}
        validation_scores = {}

        for name, model in base_models.items():
            try:
                # Fit on training data
                fitted_model = model.fit(X_train, y_train)
                fitted_models[name] = fitted_model

                # Evaluate on validation data
                y_pred = fitted_model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                validation_scores[name] = mse

                logger.info(f"{name} validation MSE: {mse:.4f}")

            except Exception as e:
                logger.error(f"Error in adaptive ensemble for {name}: {e}")
                continue

        # Calculate weights inversely proportional to validation error
        if validation_scores:
            min_mse = min(validation_scores.values())
            weights = []
            model_names = []

            for name, mse in validation_scores.items():
                # Inverse weight: better models (lower MSE) get higher weights
                weight = min_mse / mse
                weights.append(weight)
                model_names.append(name)

            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            logger.info(f"Adaptive ensemble weights: {dict(zip(model_names, weights))}")

            # Refit on full training data
            for name in model_names:
                fitted_models[name] = base_models[name].fit(X, y)

            return self.get_custom_weighted_ensemble(fitted_models, weights)

        else:
            logger.error("No models succeeded in adaptive ensemble")
            return None

    def evaluate_ensembles(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate all fitted ensemble models.

        Args:
            X_test: Test feature matrix
            y_test: Test target variable

        Returns:
            Dictionary of ensemble scores
        """
        if not self.fitted_ensembles:
            raise ValueError("No ensemble models have been fitted. Call fit_ensemble_models first.")

        logger.info("Evaluating ensemble models...")

        scores = {}

        for name, model in self.fitted_ensembles.items():
            try:
                y_pred = model.predict(X_test)

                model_scores = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                }

                scores[name] = model_scores
                logger.info(f"Ensemble {name} - RMSE: {model_scores['rmse']:.4f}, R²: {model_scores['r2']:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating ensemble {name}: {e}")
                continue

        self.ensemble_scores = scores
        return scores

    def get_cross_validated_scores(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Dict[str, float]]:
        """Get cross-validated scores for ensemble models.

        Args:
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds

        Returns:
            Dictionary of cross-validated scores
        """
        if not self.ensemble_models:
            self.get_all_ensemble_models()

        logger.info(f"Getting {cv}-fold cross-validated scores...")

        cv_scores = {}

        for name, model in self.ensemble_models.items():
            try:
                logger.info(f"Cross-validating ensemble {name}...")

                # Get cross-validated scores
                scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
                rmse_scores = np.sqrt(-scores)

                cv_scores[name] = {
                    'mean_rmse': rmse_scores.mean(),
                    'std_rmse': rmse_scores.std(),
                    'min_rmse': rmse_scores.min(),
                    'max_rmse': rmse_scores.max()
                }

                logger.info(f"Ensemble {name} CV RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")

            except Exception as e:
                logger.error(f"Error in cross-validation for ensemble {name}: {e}")
                continue

        return cv_scores

    def get_best_ensemble(self, metric: str = 'rmse') -> Tuple[str, Any, float]:
        """Get the best performing ensemble model.

        Args:
            metric: Metric to use for selection ('rmse', 'mae', 'r2')

        Returns:
            Tuple of (model_name, model_instance, best_score)
        """
        if not self.ensemble_scores:
            raise ValueError("No ensemble scores available. Call evaluate_ensembles first.")

        if metric == 'r2':
            # For R², higher is better
            best_score = max(self.ensemble_scores.items(), key=lambda x: x[1][metric])
        else:
            # For RMSE, MAE, lower is better
            best_score = min(self.ensemble_scores.items(), key=lambda x: x[1][metric])

        best_name = best_score[0]
        best_model = self.fitted_ensembles[best_name]
        score_value = best_score[1][metric]

        logger.info(f"Best ensemble: {best_name} with {metric}={score_value:.4f}")

        return best_name, best_model, score_value

    def get_ensemble_summary(self) -> pd.DataFrame:
        """Get a summary DataFrame of all ensemble performances.

        Returns:
            DataFrame with ensemble performance metrics
        """
        if not self.ensemble_scores:
            raise ValueError("No ensemble scores available. Call evaluate_ensembles first.")

        summary_data = []
        for name, scores in self.ensemble_scores.items():
            row = {'ensemble': name}
            row.update(scores)
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('rmse')  # Sort by RMSE (lower is better)

        return summary_df