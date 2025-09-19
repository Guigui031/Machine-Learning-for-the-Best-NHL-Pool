"""
Feature Engineering Utility Functions
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class FeatureUtils:
    """Utility functions for feature engineering and analysis."""

    @staticmethod
    def detect_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None,
                       method: str = 'iqr', threshold: float = 1.5) -> Dict[str, np.ndarray]:
        """Detect outliers in specified columns.

        Args:
            df: DataFrame to analyze
            columns: Columns to check (default: all numeric columns)
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            Dictionary mapping column names to outlier indices
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outliers = {}

        for col in columns:
            if col not in df.columns:
                continue

            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)

            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > threshold

            else:
                raise ValueError(f"Unknown method: {method}")

            outliers[col] = df.index[outlier_mask].values

        return outliers

    @staticmethod
    def correlation_analysis(df: pd.DataFrame, target_col: str,
                           threshold: float = 0.8) -> Dict[str, Any]:
        """Analyze correlations in the dataset.

        Args:
            df: DataFrame to analyze
            target_col: Target column name
            threshold: Threshold for high correlation

        Returns:
            Dictionary with correlation analysis results
        """
        # Calculate correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        # Target correlations
        if target_col in corr_matrix.columns:
            target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
            target_corr = target_corr.drop(target_col)  # Remove self-correlation
        else:
            target_corr = pd.Series()

        # High feature-feature correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = abs(corr_matrix.iloc[i, j])

                if corr_value > threshold:
                    high_corr_pairs.append({
                        'feature1': col1,
                        'feature2': col2,
                        'correlation': corr_value
                    })

        return {
            'target_correlations': target_corr,
            'high_correlation_pairs': high_corr_pairs,
            'correlation_matrix': corr_matrix
        }

    @staticmethod
    def feature_importance_analysis(X: pd.DataFrame, y: pd.Series,
                                  methods: List[str] = ['f_regression', 'mutual_info']) -> pd.DataFrame:
        """Analyze feature importance using multiple methods.

        Args:
            X: Feature matrix
            y: Target variable
            methods: List of methods to use

        Returns:
            DataFrame with feature importance scores
        """
        results = []

        for feature in X.columns:
            feature_data = {'feature': feature}

            # F-regression score
            if 'f_regression' in methods:
                try:
                    f_scores, _ = f_regression(X[[feature]], y)
                    feature_data['f_score'] = f_scores[0]
                except:
                    feature_data['f_score'] = 0

            # Mutual information score
            if 'mutual_info' in methods:
                try:
                    mi_scores = mutual_info_regression(X[[feature]], y, random_state=42)
                    feature_data['mutual_info'] = mi_scores[0]
                except:
                    feature_data['mutual_info'] = 0

            # Simple correlation
            if 'correlation' in methods:
                try:
                    corr = abs(X[feature].corr(y))
                    feature_data['correlation'] = corr if not np.isnan(corr) else 0
                except:
                    feature_data['correlation'] = 0

            results.append(feature_data)

        importance_df = pd.DataFrame(results)

        # Add ranks for each method
        for method in methods:
            if method in importance_df.columns:
                importance_df[f'{method}_rank'] = importance_df[method].rank(ascending=False)

        # Calculate average rank
        rank_columns = [col for col in importance_df.columns if col.endswith('_rank')]
        if rank_columns:
            importance_df['avg_rank'] = importance_df[rank_columns].mean(axis=1)
            importance_df = importance_df.sort_values('avg_rank')

        return importance_df

    @staticmethod
    def missing_value_analysis(df: pd.DataFrame) -> pd.DataFrame:
        """Analyze missing values in the dataset.

        Args:
            df: DataFrame to analyze

        Returns:
            DataFrame with missing value statistics
        """
        missing_stats = []

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100

            missing_stats.append({
                'column': col,
                'missing_count': missing_count,
                'missing_percentage': missing_pct,
                'data_type': str(df[col].dtype)
            })

        missing_df = pd.DataFrame(missing_stats)
        missing_df = missing_df.sort_values('missing_percentage', ascending=False)

        return missing_df

    @staticmethod
    def feature_distribution_analysis(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Analyze feature distributions.

        Args:
            df: DataFrame to analyze
            columns: Columns to analyze (default: all numeric)

        Returns:
            Dictionary with distribution statistics
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        distributions = {}

        for col in columns:
            if col not in df.columns:
                continue

            series = df[col].dropna()

            distributions[col] = {
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis(),
                'min': series.min(),
                'max': series.max(),
                'q25': series.quantile(0.25),
                'q75': series.quantile(0.75),
                'unique_values': series.nunique(),
                'zero_count': (series == 0).sum(),
                'negative_count': (series < 0).sum()
            }

        return distributions

    @staticmethod
    def plot_feature_distributions(df: pd.DataFrame, columns: List[str],
                                 figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot distributions of specified features.

        Args:
            df: DataFrame with features
            columns: Columns to plot
            figsize: Figure size
        """
        n_cols = min(4, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        elif n_cols == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()

        for i, col in enumerate(columns):
            if i >= len(axes):
                break

            ax = axes[i]

            if col in df.columns:
                # Plot histogram
                df[col].dropna().hist(bins=30, ax=ax, alpha=0.7)
                ax.set_title(f'{col}\nMean: {df[col].mean():.3f}, Std: {df[col].std():.3f}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10),
                               target_col: Optional[str] = None) -> None:
        """Plot correlation heatmap.

        Args:
            df: DataFrame with features
            figsize: Figure size
            target_col: Optional target column to highlight
        """
        # Calculate correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        # Create heatmap
        plt.figure(figsize=figsize)

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Generate heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})

        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()

        # If target column specified, show target correlations separately
        if target_col and target_col in corr_matrix.columns:
            target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
            target_corr = target_corr.drop(target_col)  # Remove self-correlation

            plt.figure(figsize=(10, 8))
            target_corr.head(20).plot(kind='barh')
            plt.title(f'Top 20 Features Correlated with {target_col}')
            plt.xlabel('Absolute Correlation')
            plt.tight_layout()
            plt.show()

    @staticmethod
    def suggest_feature_transformations(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Suggest feature transformations based on distribution analysis.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with transformation suggestions
        """
        suggestions = {}
        distributions = FeatureUtils.feature_distribution_analysis(df)

        for col, stats in distributions.items():
            col_suggestions = []

            # Skewness-based suggestions
            if abs(stats['skewness']) > 2:
                if stats['skewness'] > 2:
                    col_suggestions.append('log_transform (right skewed)')
                else:
                    col_suggestions.append('square_transform (left skewed)')

            # Scale-based suggestions
            if stats['std'] > stats['mean'] * 2:
                col_suggestions.append('robust_scaling (high variance)')

            # Range-based suggestions
            if stats['max'] - stats['min'] > 1000:
                col_suggestions.append('min_max_scaling (large range)')

            # Zero/constant value suggestions
            if stats['zero_count'] > len(df) * 0.5:
                col_suggestions.append('consider_removing (many zeros)')

            if stats['unique_values'] == 1:
                col_suggestions.append('remove_constant_feature')

            suggestions[col] = col_suggestions

        return suggestions

    @staticmethod
    def create_interaction_features(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]],
                                  operations: List[str] = ['multiply', 'add', 'divide']) -> pd.DataFrame:
        """Create interaction features between specified feature pairs.

        Args:
            df: Input DataFrame
            feature_pairs: List of feature name tuples
            operations: List of operations to apply

        Returns:
            DataFrame with added interaction features
        """
        df_interactions = df.copy()

        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                base_name = f"{feat1}_{feat2}"

                if 'multiply' in operations:
                    df_interactions[f"{base_name}_multiply"] = df[feat1] * df[feat2]

                if 'add' in operations:
                    df_interactions[f"{base_name}_add"] = df[feat1] + df[feat2]

                if 'divide' in operations and not (df[feat2] == 0).any():
                    df_interactions[f"{base_name}_divide"] = df[feat1] / df[feat2]

                if 'subtract' in operations:
                    df_interactions[f"{base_name}_subtract"] = df[feat1] - df[feat2]

        return df_interactions