"""
Comprehensive Model Evaluation for NHL Prediction Models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive evaluation of NHL prediction models."""

    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}
        self.comparison_data = {}

    def evaluate_single_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                            model_name: str = "model") -> Dict[str, float]:
        """Evaluate a single model comprehensively.

        Args:
            model: Fitted model to evaluate
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model: {model_name}")

        try:
            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
            }

            # Additional hockey-specific metrics
            metrics.update(self._calculate_hockey_metrics(y_test, y_pred))

            # Store results
            self.evaluation_results[model_name] = {
                'metrics': metrics,
                'predictions': y_pred,
                'true_values': y_test
            }

            logger.info(f"Model {model_name} - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            return {}

    def _calculate_hockey_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate hockey-specific evaluation metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of hockey-specific metrics
        """
        hockey_metrics = {}

        try:
            # Prediction accuracy for different performance tiers
            # Low performers (< 0.5 PPG)
            low_mask = y_true < 0.5
            if low_mask.any():
                hockey_metrics['rmse_low_performers'] = np.sqrt(mean_squared_error(
                    y_true[low_mask], y_pred[low_mask]
                ))

            # Average performers (0.5 - 1.0 PPG)
            avg_mask = (y_true >= 0.5) & (y_true < 1.0)
            if avg_mask.any():
                hockey_metrics['rmse_avg_performers'] = np.sqrt(mean_squared_error(
                    y_true[avg_mask], y_pred[avg_mask]
                ))

            # High performers (>= 1.0 PPG)
            high_mask = y_true >= 1.0
            if high_mask.any():
                hockey_metrics['rmse_high_performers'] = np.sqrt(mean_squared_error(
                    y_true[high_mask], y_pred[high_mask]
                ))

            # Top player identification accuracy (top 10%)
            top_10_threshold = np.percentile(y_true, 90)
            top_10_true = y_true >= top_10_threshold
            top_10_pred = y_pred >= top_10_threshold

            if top_10_true.any():
                # Precision and recall for top 10% players
                true_positives = np.sum(top_10_true & top_10_pred)
                false_positives = np.sum(~top_10_true & top_10_pred)
                false_negatives = np.sum(top_10_true & ~top_10_pred)

                if (true_positives + false_positives) > 0:
                    hockey_metrics['top_10_precision'] = true_positives / (true_positives + false_positives)
                if (true_positives + false_negatives) > 0:
                    hockey_metrics['top_10_recall'] = true_positives / (true_positives + false_negatives)

            # Prediction consistency (lower is better)
            residuals = np.abs(y_true - y_pred)
            hockey_metrics['prediction_consistency'] = np.std(residuals)

            # Bias metrics
            hockey_metrics['mean_bias'] = np.mean(y_pred - y_true)
            hockey_metrics['median_bias'] = np.median(y_pred - y_true)

        except Exception as e:
            logger.warning(f"Error calculating hockey metrics: {e}")

        return hockey_metrics

    def compare_models(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Compare multiple models side by side.

        Args:
            models: Dictionary of model_name -> fitted_model
            X_test: Test features
            y_test: Test targets

        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(models)} models...")

        comparison_data = []

        for name, model in models.items():
            metrics = self.evaluate_single_model(model, X_test, y_test, name)
            if metrics:
                row = {'model': name}
                row.update(metrics)
                comparison_data.append(row)

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('rmse')  # Sort by RMSE
            self.comparison_data = comparison_df
            return comparison_df
        else:
            logger.error("No models could be evaluated")
            return pd.DataFrame()

    def plot_predictions_vs_actual(self, model_names: Optional[List[str]] = None,
                                  figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot predictions vs actual values for models.

        Args:
            model_names: List of model names to plot (default: all)
            figsize: Figure size
        """
        if not self.evaluation_results:
            logger.error("No evaluation results available")
            return

        if model_names is None:
            model_names = list(self.evaluation_results.keys())

        n_models = len(model_names)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()

        for i, model_name in enumerate(model_names):
            if model_name not in self.evaluation_results:
                continue

            ax = axes[i] if i < len(axes) else None
            if ax is None:
                continue

            results = self.evaluation_results[model_name]
            y_true = results['true_values']
            y_pred = results['predictions']
            r2 = results['metrics']['r2']
            rmse = results['metrics']['rmse']

            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.6, s=20)

            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

            # Formatting
            ax.set_xlabel('Actual Points per Game')
            ax.set_ylabel('Predicted Points per Game')
            ax.set_title(f'{model_name}\nR² = {r2:.3f}, RMSE = {rmse:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_residuals(self, model_names: Optional[List[str]] = None,
                      figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot residual analysis for models.

        Args:
            model_names: List of model names to plot (default: all)
            figsize: Figure size
        """
        if not self.evaluation_results:
            logger.error("No evaluation results available")
            return

        if model_names is None:
            model_names = list(self.evaluation_results.keys())

        n_models = len(model_names)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()

        for i, model_name in enumerate(model_names):
            if model_name not in self.evaluation_results:
                continue

            ax = axes[i] if i < len(axes) else None
            if ax is None:
                continue

            results = self.evaluation_results[model_name]
            y_true = results['true_values']
            y_pred = results['predictions']
            residuals = y_true - y_pred

            # Residuals vs predicted
            ax.scatter(y_pred, residuals, alpha=0.6, s=20)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)

            # Formatting
            ax.set_xlabel('Predicted Points per Game')
            ax.set_ylabel('Residuals')
            ax.set_title(f'{model_name} - Residual Analysis')
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, models: Dict[str, Any], feature_names: List[str],
                              top_k: int = 20, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot feature importance for models that support it.

        Args:
            models: Dictionary of model_name -> fitted_model
            feature_names: List of feature names
            top_k: Number of top features to show
            figsize: Figure size
        """
        importance_data = []

        for name, model in models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_)
                else:
                    logger.info(f"Model {name} doesn't support feature importance")
                    continue

                # Create importance dataframe
                for i, importance in enumerate(importances):
                    if i < len(feature_names):
                        importance_data.append({
                            'model': name,
                            'feature': feature_names[i],
                            'importance': importance
                        })

            except Exception as e:
                logger.error(f"Error getting feature importance for {name}: {e}")
                continue

        if not importance_data:
            logger.warning("No feature importance data available")
            return

        importance_df = pd.DataFrame(importance_data)

        # Plot top features for each model
        unique_models = importance_df['model'].unique()
        n_models = len(unique_models)

        if n_models == 0:
            return

        cols = min(2, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()

        for i, model_name in enumerate(unique_models):
            if i >= len(axes):
                break

            ax = axes[i]
            model_data = importance_df[importance_df['model'] == model_name]
            top_features = model_data.nlargest(top_k, 'importance')

            # Horizontal bar plot
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_features['importance'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_name} - Top {top_k} Features')
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_model_comparison(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot model comparison metrics.

        Args:
            figsize: Figure size
        """
        if not hasattr(self, 'comparison_data') or self.comparison_data.empty:
            logger.error("No comparison data available. Run compare_models first.")
            return

        # Select key metrics for comparison
        metrics_to_plot = ['rmse', 'mae', 'r2']
        available_metrics = [m for m in metrics_to_plot if m in self.comparison_data.columns]

        if not available_metrics:
            logger.error("No comparison metrics available")
            return

        fig, axes = plt.subplots(1, len(available_metrics), figsize=figsize)
        if len(available_metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(available_metrics):
            ax = axes[i]

            # Sort data for better visualization
            sorted_data = self.comparison_data.sort_values(metric, ascending=(metric != 'r2'))

            # Bar plot
            bars = ax.bar(range(len(sorted_data)), sorted_data[metric])
            ax.set_xticks(range(len(sorted_data)))
            ax.set_xticklabels(sorted_data['model'], rotation=45, ha='right')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'Model Comparison - {metric.upper()}')
            ax.grid(True, alpha=0.3)

            # Color best performing model
            if metric == 'r2':
                best_idx = sorted_data[metric].idxmax()
            else:
                best_idx = sorted_data[metric].idxmin()

            best_pos = list(sorted_data.index).index(best_idx)
            bars[best_pos].set_color('gold')

        plt.tight_layout()
        plt.show()

    def generate_evaluation_report(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray,
                                 feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report.

        Args:
            models: Dictionary of model_name -> fitted_model
            X_test: Test features
            y_test: Test targets
            feature_names: Optional list of feature names

        Returns:
            Comprehensive evaluation report
        """
        logger.info("Generating comprehensive evaluation report...")

        report = {
            'summary': {},
            'detailed_metrics': {},
            'best_models': {},
            'recommendations': []
        }

        # Compare all models
        comparison_df = self.compare_models(models, X_test, y_test)

        if not comparison_df.empty:
            # Summary statistics
            report['summary'] = {
                'num_models_evaluated': len(comparison_df),
                'best_rmse': comparison_df['rmse'].min(),
                'best_r2': comparison_df['r2'].max(),
                'rmse_range': (comparison_df['rmse'].min(), comparison_df['rmse'].max()),
                'r2_range': (comparison_df['r2'].min(), comparison_df['r2'].max())
            }

            # Detailed metrics
            report['detailed_metrics'] = comparison_df.to_dict('records')

            # Best models by different metrics
            report['best_models'] = {
                'best_overall_rmse': comparison_df.loc[comparison_df['rmse'].idxmin(), 'model'],
                'best_r2': comparison_df.loc[comparison_df['r2'].idxmax(), 'model'],
                'best_mae': comparison_df.loc[comparison_df['mae'].idxmin(), 'model']
            }

            # Recommendations
            best_rmse_model = report['best_models']['best_overall_rmse']
            best_r2_model = report['best_models']['best_r2']

            recommendations = [
                f"Best overall model by RMSE: {best_rmse_model}",
                f"Best explanatory model (R²): {best_r2_model}"
            ]

            # Check for overfitting indicators
            high_r2_threshold = 0.8
            if comparison_df['r2'].max() > high_r2_threshold:
                recommendations.append("High R² detected - check for potential overfitting")

            # Performance tier recommendations
            if comparison_df['rmse'].min() < 0.1:
                recommendations.append("Excellent model performance achieved")
            elif comparison_df['rmse'].min() < 0.2:
                recommendations.append("Good model performance - consider ensemble methods")
            else:
                recommendations.append("Model performance could be improved - try feature engineering")

            report['recommendations'] = recommendations

        logger.info("Evaluation report generated successfully")
        return report