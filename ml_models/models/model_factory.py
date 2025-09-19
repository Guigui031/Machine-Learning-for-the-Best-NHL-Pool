"""
Model Factory for Creating NHL Prediction Models
"""

import logging
from typing import Dict, Any, Optional, List
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR

logger = logging.getLogger(__name__)

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class ModelFactory:
    """Factory class for creating different types of NHL prediction models."""

    @staticmethod
    def create_baseline_model(model_type: str, **kwargs) -> Any:
        """Create a baseline model.

        Args:
            model_type: Type of model to create
            **kwargs: Additional parameters for the model

        Returns:
            Configured model instance
        """
        if model_type == 'linear':
            return LinearRegression(**kwargs)
        elif model_type == 'ridge':
            return Ridge(alpha=kwargs.get('alpha', 1.0),
                        random_state=kwargs.get('random_state', 42))
        elif model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 15),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown baseline model type: {model_type}")

    @staticmethod
    def create_advanced_model(model_type: str, **kwargs) -> Any:
        """Create an advanced model.

        Args:
            model_type: Type of model to create
            **kwargs: Additional parameters for the model

        Returns:
            Configured model instance
        """
        if model_type == 'xgboost' and HAS_XGB:
            return xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1,
                eval_metric='rmse'
            )
        elif model_type == 'svr':
            return SVR(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                gamma=kwargs.get('gamma', 'auto')
            )
        else:
            raise ValueError(f"Unknown advanced model type: {model_type}")

    @staticmethod
    def create_ensemble_model(ensemble_type: str, base_models: Optional[List] = None, **kwargs) -> Any:
        """Create an ensemble model.

        Args:
            ensemble_type: Type of ensemble to create
            base_models: List of base models for ensemble
            **kwargs: Additional parameters

        Returns:
            Configured ensemble model
        """
        if ensemble_type == 'voting':
            if base_models is None:
                estimators = [
                    ('ridge', Ridge(alpha=1.0, random_state=42)),
                    ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
                ]
                if HAS_XGB:
                    estimators.append(('xgb', xgb.XGBRegressor(
                        n_estimators=100, random_state=42, n_jobs=-1, eval_metric='rmse'
                    )))
            else:
                estimators = base_models

            return VotingRegressor(estimators=estimators)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    @staticmethod
    def get_model_recommendations(data_size: int, feature_count: int) -> Dict[str, List[str]]:
        """Get model recommendations based on data characteristics.

        Args:
            data_size: Number of training samples
            feature_count: Number of features

        Returns:
            Dictionary of recommended models by category
        """
        recommendations = {
            'baseline': ['linear', 'ridge'],
            'advanced': [],
            'ensemble': []
        }

        if data_size < 100:
            # Small dataset
            recommendations['baseline'].extend(['ridge'])
            recommendations['advanced'] = ['random_forest']
        elif data_size < 1000:
            # Medium dataset
            recommendations['baseline'].extend(['random_forest'])
            recommendations['advanced'] = ['random_forest', 'svr']
            if HAS_XGB:
                recommendations['advanced'].append('xgboost')
        else:
            # Large dataset
            recommendations['advanced'] = ['random_forest', 'svr']
            recommendations['ensemble'] = ['voting']
            if HAS_XGB:
                recommendations['advanced'].append('xgboost')

        return recommendations

    @staticmethod
    def create_hockey_specific_model(**kwargs) -> Any:
        """Create a model specifically tuned for hockey data.

        Returns:
            Hockey-optimized model
        """
        # Hockey-specific model configuration
        # Based on hockey data characteristics, we prefer tree-based models
        return RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=kwargs.get('random_state', 42),
            n_jobs=-1
        )