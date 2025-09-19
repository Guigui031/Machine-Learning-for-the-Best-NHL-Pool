"""
NHL-Specific Metrics and Evaluation Functions
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

logger = logging.getLogger(__name__)


class NHLMetrics:
    """NHL-specific metrics for evaluating player performance predictions."""

    @staticmethod
    def points_per_game_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                                tolerance: float = 0.1) -> float:
        """Calculate accuracy within tolerance for points per game predictions.

        Args:
            y_true: True values
            y_pred: Predicted values
            tolerance: Tolerance for considering prediction accurate

        Returns:
            Accuracy score
        """
        absolute_errors = np.abs(y_true - y_pred)
        accurate_predictions = absolute_errors <= tolerance
        return np.mean(accurate_predictions)

    @staticmethod
    def performance_tier_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate accuracy for different performance tiers.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of tier-specific accuracies
        """
        # Define performance tiers based on points per game
        def get_tier(ppg):
            if ppg < 0.3:
                return 'low'
            elif ppg < 0.7:
                return 'average'
            elif ppg < 1.0:
                return 'good'
            else:
                return 'elite'

        true_tiers = [get_tier(ppg) for ppg in y_true]
        pred_tiers = [get_tier(ppg) for ppg in y_pred]

        # Calculate accuracy for each tier
        tier_accuracies = {}
        unique_tiers = set(true_tiers)

        for tier in unique_tiers:
            tier_mask = np.array(true_tiers) == tier
            if tier_mask.any():
                tier_true = np.array(true_tiers)[tier_mask]
                tier_pred = np.array(pred_tiers)[tier_mask]
                tier_accuracies[f'{tier}_accuracy'] = np.mean(tier_true == tier_pred)

        # Overall tier accuracy
        tier_accuracies['overall_tier_accuracy'] = np.mean(np.array(true_tiers) == np.array(pred_tiers))

        return tier_accuracies

    @staticmethod
    def top_performer_identification(y_true: np.ndarray, y_pred: np.ndarray,
                                   top_percentile: float = 90) -> Dict[str, float]:
        """Evaluate model's ability to identify top performers.

        Args:
            y_true: True values
            y_pred: Predicted values
            top_percentile: Percentile to define top performers

        Returns:
            Dictionary with identification metrics
        """
        threshold = np.percentile(y_true, top_percentile)

        true_top = y_true >= threshold
        pred_top = y_pred >= threshold

        # Calculate metrics
        true_positives = np.sum(true_top & pred_top)
        false_positives = np.sum(~true_top & pred_top)
        false_negatives = np.sum(true_top & ~pred_top)
        true_negatives = np.sum(~true_top & ~pred_top)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            f'top_{top_percentile}th_precision': precision,
            f'top_{top_percentile}th_recall': recall,
            f'top_{top_percentile}th_f1': f1_score
        }

    @staticmethod
    def position_specific_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                positions: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each position.

        Args:
            y_true: True values
            y_pred: Predicted values
            positions: Position labels

        Returns:
            Dictionary of position-specific metrics
        """
        position_metrics = {}
        unique_positions = np.unique(positions)

        for position in unique_positions:
            pos_mask = positions == position
            if pos_mask.any():
                pos_true = y_true[pos_mask]
                pos_pred = y_pred[pos_mask]

                position_metrics[position] = {
                    'rmse': np.sqrt(mean_squared_error(pos_true, pos_pred)),
                    'mae': mean_absolute_error(pos_true, pos_pred),
                    'r2': r2_score(pos_true, pos_pred),
                    'mape': mean_absolute_percentage_error(pos_true, pos_pred) * 100,
                    'count': len(pos_true)
                }

        return position_metrics

    @staticmethod
    def consistency_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate consistency-related metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of consistency metrics
        """
        residuals = y_true - y_pred
        absolute_residuals = np.abs(residuals)

        return {
            'prediction_std': np.std(residuals),
            'absolute_residual_std': np.std(absolute_residuals),
            'max_error': np.max(absolute_residuals),
            'min_error': np.min(absolute_residuals),
            'median_absolute_error': np.median(absolute_residuals),
            'q75_absolute_error': np.percentile(absolute_residuals, 75),
            'q95_absolute_error': np.percentile(absolute_residuals, 95)
        }

    @staticmethod
    def bias_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Analyze prediction bias.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of bias metrics
        """
        residuals = y_pred - y_true

        return {
            'mean_bias': np.mean(residuals),
            'median_bias': np.median(residuals),
            'positive_bias_pct': np.mean(residuals > 0) * 100,
            'negative_bias_pct': np.mean(residuals < 0) * 100,
            'bias_magnitude': np.mean(np.abs(residuals))
        }

    @staticmethod
    def hockey_specific_evaluation(y_true: np.ndarray, y_pred: np.ndarray,
                                 positions: Optional[np.ndarray] = None,
                                 ages: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Comprehensive hockey-specific evaluation.

        Args:
            y_true: True values
            y_pred: Predicted values
            positions: Position labels (optional)
            ages: Age values (optional)

        Returns:
            Comprehensive evaluation dictionary
        """
        evaluation = {}

        # Basic metrics
        evaluation['basic_metrics'] = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }

        # Hockey-specific metrics
        evaluation['ppg_accuracy'] = {
            'tolerance_0.05': NHLMetrics.points_per_game_accuracy(y_true, y_pred, 0.05),
            'tolerance_0.10': NHLMetrics.points_per_game_accuracy(y_true, y_pred, 0.10),
            'tolerance_0.15': NHLMetrics.points_per_game_accuracy(y_true, y_pred, 0.15)
        }

        # Performance tier analysis
        evaluation['tier_metrics'] = NHLMetrics.performance_tier_accuracy(y_true, y_pred)

        # Top performer identification
        evaluation['top_performer_metrics'] = {}
        for percentile in [80, 90, 95]:
            top_metrics = NHLMetrics.top_performer_identification(y_true, y_pred, percentile)
            evaluation['top_performer_metrics'].update(top_metrics)

        # Consistency analysis
        evaluation['consistency_metrics'] = NHLMetrics.consistency_metrics(y_true, y_pred)

        # Bias analysis
        evaluation['bias_metrics'] = NHLMetrics.bias_analysis(y_true, y_pred)

        # Position-specific metrics if available
        if positions is not None:
            evaluation['position_metrics'] = NHLMetrics.position_specific_metrics(y_true, y_pred, positions)

        # Age-specific metrics if available
        if ages is not None:
            evaluation['age_metrics'] = NHLMetrics.age_specific_metrics(y_true, y_pred, ages)

        return evaluation

    @staticmethod
    def age_specific_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           ages: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for different age groups.

        Args:
            y_true: True values
            y_pred: Predicted values
            ages: Age values

        Returns:
            Dictionary of age group-specific metrics
        """
        # Define age groups
        age_groups = pd.cut(ages, bins=[0, 23, 27, 31, 50],
                           labels=['Young (≤23)', 'Prime (24-27)', 'Veteran (28-31)', 'Old (32+)'])

        age_metrics = {}

        for age_group in age_groups.categories:
            group_mask = age_groups == age_group
            if group_mask.any():
                group_true = y_true[group_mask]
                group_pred = y_pred[group_mask]

                age_metrics[str(age_group)] = {
                    'rmse': np.sqrt(mean_squared_error(group_true, group_pred)),
                    'mae': mean_absolute_error(group_true, group_pred),
                    'r2': r2_score(group_true, group_pred),
                    'count': len(group_true),
                    'mean_true': np.mean(group_true),
                    'mean_pred': np.mean(group_pred)
                }

        return age_metrics

    @staticmethod
    def fantasy_hockey_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             salaries: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate fantasy hockey-specific metrics.

        Args:
            y_true: True values (points per game)
            y_pred: Predicted values
            salaries: Player salaries (optional)

        Returns:
            Dictionary of fantasy-specific metrics
        """
        fantasy_metrics = {}

        # Value-based metrics if salaries available
        if salaries is not None:
            # Points per dollar (value)
            true_value = y_true / (salaries / 1000000)  # Points per million dollars
            pred_value = y_pred / (salaries / 1000000)

            fantasy_metrics['value_prediction_accuracy'] = r2_score(true_value, pred_value)
            fantasy_metrics['value_rmse'] = np.sqrt(mean_squared_error(true_value, pred_value))

            # Identify undervalued players (predicted value > actual value)
            value_diff = pred_value - true_value
            fantasy_metrics['undervalued_identification_rate'] = np.mean(value_diff > 0.5)

        # Sleeper pick identification (low actual, high predicted)
        sleeper_mask = (y_true < 0.5) & (y_pred > 0.7)
        fantasy_metrics['sleeper_identification_rate'] = np.mean(sleeper_mask)

        # Bust avoidance (high actual, low predicted - we want this to be low)
        bust_mask = (y_true > 0.8) & (y_pred < 0.5)
        fantasy_metrics['bust_miss_rate'] = np.mean(bust_mask)

        return fantasy_metrics