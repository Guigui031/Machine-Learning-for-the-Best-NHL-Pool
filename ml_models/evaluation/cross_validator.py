"""
Cross-Validation Utilities for NHL Models
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import (
    cross_val_score, TimeSeriesSplit, KFold, StratifiedKFold,
    cross_validate, validation_curve, learning_curve
)
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class CrossValidator:
    """Cross-validation utilities for NHL prediction models."""

    def __init__(self, random_state: int = 42):
        """Initialize cross-validator.

        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.cv_results = {}

    def time_series_cv(self, model: Any, X: np.ndarray, y: np.ndarray,
                      n_splits: int = 5) -> Dict[str, np.ndarray]:
        """Perform time series cross-validation.

        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target variable
            n_splits: Number of splits

        Returns:
            Dictionary of cross-validation scores
        """
        logger.info(f"Performing time series CV with {n_splits} splits...")

        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Define scoring metrics
        scoring = {
            'neg_mse': 'neg_mean_squared_error',
            'r2': 'r2',
            'neg_mae': 'neg_mean_absolute_error'
        }

        cv_results = cross_validate(
            model, X, y, cv=tscv, scoring=scoring,
            return_train_score=True, n_jobs=-1
        )

        # Convert to positive values and calculate RMSE
        results = {
            'test_rmse': np.sqrt(-cv_results['test_neg_mse']),
            'train_rmse': np.sqrt(-cv_results['train_neg_mse']),
            'test_r2': cv_results['test_r2'],
            'train_r2': cv_results['train_r2'],
            'test_mae': -cv_results['test_neg_mae'],
            'train_mae': -cv_results['train_neg_mae']
        }

        logger.info(f"Time series CV completed - Test RMSE: {results['test_rmse'].mean():.4f} ± {results['test_rmse'].std():.4f}")

        return results

    def standard_cv(self, model: Any, X: np.ndarray, y: np.ndarray,
                   cv: int = 5) -> Dict[str, np.ndarray]:
        """Perform standard k-fold cross-validation.

        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target variable
            cv: Number of folds

        Returns:
            Dictionary of cross-validation scores
        """
        logger.info(f"Performing standard CV with {cv} folds...")

        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        scoring = {
            'neg_mse': 'neg_mean_squared_error',
            'r2': 'r2',
            'neg_mae': 'neg_mean_absolute_error'
        }

        cv_results = cross_validate(
            model, X, y, cv=kfold, scoring=scoring,
            return_train_score=True, n_jobs=-1
        )

        results = {
            'test_rmse': np.sqrt(-cv_results['test_neg_mse']),
            'train_rmse': np.sqrt(-cv_results['train_neg_mse']),
            'test_r2': cv_results['test_r2'],
            'train_r2': cv_results['train_r2'],
            'test_mae': -cv_results['test_neg_mae'],
            'train_mae': -cv_results['train_neg_mae']
        }

        logger.info(f"Standard CV completed - Test RMSE: {results['test_rmse'].mean():.4f} ± {results['test_rmse'].std():.4f}")

        return results

    def compare_cv_strategies(self, model: Any, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Compare different cross-validation strategies.

        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target variable

        Returns:
            DataFrame comparing CV strategies
        """
        logger.info("Comparing CV strategies...")

        strategies = {
            'Standard 5-Fold': self.standard_cv(model, X, y, cv=5),
            'Time Series 5-Split': self.time_series_cv(model, X, y, n_splits=5)
        }

        comparison_data = []
        for strategy_name, results in strategies.items():
            row = {
                'strategy': strategy_name,
                'test_rmse_mean': results['test_rmse'].mean(),
                'test_rmse_std': results['test_rmse'].std(),
                'test_r2_mean': results['test_r2'].mean(),
                'test_r2_std': results['test_r2'].std(),
                'train_test_gap': results['train_rmse'].mean() - results['test_rmse'].mean()
            }
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df

    def validation_curve_analysis(self, model_class: Any, X: np.ndarray, y: np.ndarray,
                                 param_name: str, param_range: List, cv: int = 5) -> Dict[str, np.ndarray]:
        """Generate validation curve for hyperparameter analysis.

        Args:
            model_class: Model class (not instance)
            X: Feature matrix
            y: Target variable
            param_name: Parameter name to vary
            param_range: Range of parameter values
            cv: Number of CV folds

        Returns:
            Dictionary with validation curve results
        """
        logger.info(f"Generating validation curve for {param_name}...")

        train_scores, validation_scores = validation_curve(
            model_class(), X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
        )

        # Convert to RMSE
        train_rmse = np.sqrt(-train_scores)
        validation_rmse = np.sqrt(-validation_scores)

        results = {
            'param_range': param_range,
            'train_scores_mean': train_rmse.mean(axis=1),
            'train_scores_std': train_rmse.std(axis=1),
            'validation_scores_mean': validation_rmse.mean(axis=1),
            'validation_scores_std': validation_rmse.std(axis=1)
        }

        logger.info("Validation curve analysis completed")
        return results

    def learning_curve_analysis(self, model: Any, X: np.ndarray, y: np.ndarray,
                               train_sizes: Optional[np.ndarray] = None, cv: int = 5) -> Dict[str, np.ndarray]:
        """Generate learning curve to analyze model performance vs training size.

        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target variable
            train_sizes: Training sizes to use
            cv: Number of CV folds

        Returns:
            Dictionary with learning curve results
        """
        logger.info("Generating learning curve...")

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        train_sizes_abs, train_scores, validation_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=cv,
            scoring='neg_mean_squared_error', n_jobs=-1, random_state=self.random_state
        )

        # Convert to RMSE
        train_rmse = np.sqrt(-train_scores)
        validation_rmse = np.sqrt(-validation_scores)

        results = {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': train_rmse.mean(axis=1),
            'train_scores_std': train_rmse.std(axis=1),
            'validation_scores_mean': validation_rmse.mean(axis=1),
            'validation_scores_std': validation_rmse.std(axis=1)
        }

        logger.info("Learning curve analysis completed")
        return results

    def plot_validation_curve(self, validation_results: Dict[str, np.ndarray],
                             param_name: str, title: str = "Validation Curve") -> None:
        """Plot validation curve results.

        Args:
            validation_results: Results from validation_curve_analysis
            param_name: Parameter name for x-axis label
            title: Plot title
        """
        plt.figure(figsize=(10, 6))

        param_range = validation_results['param_range']
        train_mean = validation_results['train_scores_mean']
        train_std = validation_results['train_scores_std']
        val_mean = validation_results['validation_scores_mean']
        val_std = validation_results['validation_scores_std']

        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

        plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

        plt.xlabel(param_name)
        plt.ylabel('RMSE')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_learning_curve(self, learning_results: Dict[str, np.ndarray],
                           title: str = "Learning Curve") -> None:
        """Plot learning curve results.

        Args:
            learning_results: Results from learning_curve_analysis
            title: Plot title
        """
        plt.figure(figsize=(10, 6))

        train_sizes = learning_results['train_sizes']
        train_mean = learning_results['train_scores_mean']
        train_std = learning_results['train_scores_std']
        val_mean = learning_results['validation_scores_mean']
        val_std = learning_results['validation_scores_std']

        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

        plt.xlabel('Training Set Size')
        plt.ylabel('RMSE')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.show()

    def cross_validate_multiple_models(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                                     cv_strategy: str = 'standard', cv: int = 5) -> pd.DataFrame:
        """Cross-validate multiple models and compare results.

        Args:
            models: Dictionary of model_name -> model_instance
            X: Feature matrix
            y: Target variable
            cv_strategy: 'standard' or 'time_series'
            cv: Number of folds/splits

        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Cross-validating {len(models)} models with {cv_strategy} strategy...")

        results_data = []

        for model_name, model in models.items():
            try:
                if cv_strategy == 'time_series':
                    cv_results = self.time_series_cv(model, X, y, n_splits=cv)
                else:
                    cv_results = self.standard_cv(model, X, y, cv=cv)

                row = {
                    'model': model_name,
                    'test_rmse_mean': cv_results['test_rmse'].mean(),
                    'test_rmse_std': cv_results['test_rmse'].std(),
                    'test_r2_mean': cv_results['test_r2'].mean(),
                    'test_r2_std': cv_results['test_r2'].std(),
                    'overfitting_score': cv_results['train_rmse'].mean() - cv_results['test_rmse'].mean()
                }
                results_data.append(row)

            except Exception as e:
                logger.error(f"Error cross-validating {model_name}: {e}")
                continue

        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('test_rmse_mean')

        logger.info("Multi-model cross-validation completed")
        return results_df