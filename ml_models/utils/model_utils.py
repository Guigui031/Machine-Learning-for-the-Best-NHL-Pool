"""
Model Utility Functions
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ModelUtils:
    """Utility functions for NHL prediction models."""

    @staticmethod
    def save_model(model: Any, filepath: str, metadata: Optional[Dict] = None) -> None:
        """Save a model with optional metadata.

        Args:
            model: Model to save
            filepath: Path to save the model
            metadata: Optional metadata dictionary
        """
        try:
            # Save the model
            joblib.dump(model, filepath)
            logger.info(f"Model saved to: {filepath}")

            # Save metadata if provided
            if metadata:
                metadata_path = Path(filepath).with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Metadata saved to: {metadata_path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")

    @staticmethod
    def load_model(filepath: str) -> Tuple[Any, Optional[Dict]]:
        """Load a model with optional metadata.

        Args:
            filepath: Path to the saved model

        Returns:
            Tuple of (model, metadata)
        """
        try:
            # Load the model
            model = joblib.load(filepath)
            logger.info(f"Model loaded from: {filepath}")

            # Try to load metadata
            metadata = None
            metadata_path = Path(filepath).with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Metadata loaded from: {metadata_path}")

            return model, metadata

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None

    @staticmethod
    def model_size_analysis(model: Any) -> Dict[str, Any]:
        """Analyze model complexity and size.

        Args:
            model: Model to analyze

        Returns:
            Dictionary with model analysis
        """
        analysis = {}

        try:
            # Model type
            analysis['model_type'] = type(model).__name__

            # Try to get model parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
                analysis['num_parameters'] = len(params)
                analysis['parameters'] = params

            # Tree-based model analysis
            if hasattr(model, 'n_estimators'):
                analysis['n_estimators'] = model.n_estimators
            if hasattr(model, 'max_depth'):
                analysis['max_depth'] = model.max_depth

            # Feature count
            if hasattr(model, 'n_features_in_'):
                analysis['n_features'] = model.n_features_in_

            # Model complexity score (rough estimate)
            complexity_score = 1
            if hasattr(model, 'n_estimators'):
                complexity_score *= model.n_estimators
            if hasattr(model, 'max_depth') and model.max_depth:
                complexity_score *= model.max_depth

            analysis['complexity_score'] = complexity_score

        except Exception as e:
            logger.warning(f"Error in model analysis: {e}")
            analysis['error'] = str(e)

        return analysis

    @staticmethod
    def compare_model_performance(model_scores: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Compare multiple models' performance.

        Args:
            model_scores: Dictionary of model_name -> metrics_dict

        Returns:
            DataFrame with model comparison
        """
        comparison_data = []

        for model_name, scores in model_scores.items():
            row = {'model': model_name}
            row.update(scores)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Add ranking columns
        if 'rmse' in df.columns:
            df['rmse_rank'] = df['rmse'].rank()
        if 'r2' in df.columns:
            df['r2_rank'] = df['r2'].rank(ascending=False)
        if 'mae' in df.columns:
            df['mae_rank'] = df['mae'].rank()

        # Overall rank (average of individual ranks)
        rank_columns = [col for col in df.columns if col.endswith('_rank')]
        if rank_columns:
            df['overall_rank'] = df[rank_columns].mean(axis=1)
            df = df.sort_values('overall_rank')

        return df

    @staticmethod
    def create_model_ensemble_weights(individual_scores: Dict[str, float],
                                    method: str = 'inverse_error') -> Dict[str, float]:
        """Create ensemble weights based on individual model performance.

        Args:
            individual_scores: Dictionary of model_name -> error_score
            method: Method for weight calculation

        Returns:
            Dictionary of model_name -> weight
        """
        if method == 'inverse_error':
            # Weights inversely proportional to error
            min_error = min(individual_scores.values())
            weights = {}
            total_weight = 0

            for name, error in individual_scores.items():
                weight = min_error / error
                weights[name] = weight
                total_weight += weight

            # Normalize weights
            for name in weights:
                weights[name] /= total_weight

        elif method == 'uniform':
            # Equal weights
            n_models = len(individual_scores)
            weights = {name: 1.0 / n_models for name in individual_scores}

        else:
            raise ValueError(f"Unknown weighting method: {method}")

        return weights

    @staticmethod
    def validate_model_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[bool, List[str]]:
        """Validate model inputs for common issues.

        Args:
            X: Feature matrix
            y: Target variable (optional)

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check for NaN values
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            issues.append(f"Features contain {nan_count} NaN values")

        # Check for infinite values
        if np.isinf(X).any():
            inf_count = np.isinf(X).sum()
            issues.append(f"Features contain {inf_count} infinite values")

        # Check for constant features
        feature_variances = np.var(X, axis=0)
        constant_features = np.sum(feature_variances == 0)
        if constant_features > 0:
            issues.append(f"Found {constant_features} constant features")

        # Check target variable if provided
        if y is not None:
            if np.isnan(y).any():
                nan_targets = np.isnan(y).sum()
                issues.append(f"Target contains {nan_targets} NaN values")

            if np.isinf(y).any():
                inf_targets = np.isinf(y).sum()
                issues.append(f"Target contains {inf_targets} infinite values")

        # Check matrix shape
        if X.shape[0] == 0:
            issues.append("Empty feature matrix")

        if y is not None and X.shape[0] != len(y):
            issues.append(f"Mismatch between features ({X.shape[0]}) and targets ({len(y)})")

        return len(issues) == 0, issues

    @staticmethod
    def feature_importance_summary(models: Dict[str, Any], feature_names: List[str],
                                 top_k: int = 10) -> pd.DataFrame:
        """Create feature importance summary across multiple models.

        Args:
            models: Dictionary of model_name -> fitted_model
            feature_names: List of feature names
            top_k: Number of top features to include

        Returns:
            DataFrame with feature importance summary
        """
        importance_data = []

        for model_name, model in models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_)
                else:
                    continue

                # Create importance records
                for i, importance in enumerate(importances):
                    if i < len(feature_names):
                        importance_data.append({
                            'model': model_name,
                            'feature': feature_names[i],
                            'importance': importance,
                            'rank': None
                        })

            except Exception as e:
                logger.warning(f"Could not extract importance for {model_name}: {e}")
                continue

        if not importance_data:
            return pd.DataFrame()

        df = pd.DataFrame(importance_data)

        # Add ranks within each model
        df['rank'] = df.groupby('model')['importance'].rank(ascending=False)

        # Create summary with top features
        summary_data = []
        for feature in feature_names:
            feature_data = df[df['feature'] == feature]
            if not feature_data.empty:
                summary_data.append({
                    'feature': feature,
                    'avg_importance': feature_data['importance'].mean(),
                    'std_importance': feature_data['importance'].std(),
                    'avg_rank': feature_data['rank'].mean(),
                    'models_count': len(feature_data)
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('avg_importance', ascending=False)

        return summary_df.head(top_k)

    @staticmethod
    def prediction_confidence_intervals(predictions: np.ndarray, residuals: np.ndarray,
                                      confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate prediction confidence intervals.

        Args:
            predictions: Model predictions
            residuals: Model residuals (true - predicted)
            confidence_level: Confidence level for intervals

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        # Calculate residual standard deviation
        residual_std = np.std(residuals)

        # Calculate confidence interval width
        from scipy import stats
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        interval_width = z_score * residual_std

        # Calculate bounds
        lower_bounds = predictions - interval_width
        upper_bounds = predictions + interval_width

        return lower_bounds, upper_bounds