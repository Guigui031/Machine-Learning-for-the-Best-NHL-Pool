"""
Main Feature Engineering Class for NHL Data
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Main feature engineering class for NHL player data."""

    def __init__(self, scaler_type: str = 'standard'):
        """Initialize the feature engineer.

        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = self._get_scaler(scaler_type)
        self.feature_names = []
        self.is_fitted = False

    def _get_scaler(self, scaler_type: str):
        """Get the appropriate scaler."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        return scalers.get(scaler_type, StandardScaler())

    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic engineered features from raw data.

        Args:
            df: Input DataFrame with player data

        Returns:
            DataFrame with basic features added
        """
        logger.info("Creating basic features...")
        df_features = df.copy()

        # Performance ratios
        for season in ['1', '2']:
            games_col = f'games_{season}'
            if games_col in df_features.columns and df_features[games_col].sum() > 0:
                # Points per game
                df_features[f'ppg_{season}'] = (
                    df_features[f'goals_{season}'] + df_features[f'assists_{season}']
                ) / df_features[games_col].replace(0, np.nan)

                # Shooting percentage
                if f'shots_{season}' in df_features.columns:
                    df_features[f'shooting_pct_{season}'] = (
                        df_features[f'goals_{season}'] / df_features[f'shots_{season}'].replace(0, np.nan)
                    ) * 100

                # Time on ice per game (if in minutes)
                if f'time_{season}' in df_features.columns:
                    df_features[f'toi_per_game_{season}'] = df_features[f'time_{season}']

        # Performance trends (season 2 vs season 1)
        if all(col in df_features.columns for col in ['ppg_1', 'ppg_2']):
            df_features['ppg_trend'] = df_features['ppg_2'] - df_features['ppg_1']
            df_features['ppg_trend_pct'] = (
                (df_features['ppg_2'] - df_features['ppg_1']) / df_features['ppg_1'].replace(0, np.nan)
            ) * 100

        # Age-based features
        if 'age' in df_features.columns:
            df_features['age_squared'] = df_features['age'] ** 2
            df_features['age_cubed'] = df_features['age'] ** 3

            # Age groups
            df_features['age_group'] = pd.cut(
                df_features['age'],
                bins=[0, 23, 27, 31, 50],
                labels=['young', 'prime', 'veteran', 'old']
            )

        # Physical features
        if all(col in df_features.columns for col in ['height', 'weight']):
            df_features['bmi'] = df_features['weight'] / ((df_features['height'] / 100) ** 2)

        logger.info(f"Created {len(df_features.columns) - len(df.columns)} basic features")
        return df_features

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced hockey-specific features.

        Args:
            df: Input DataFrame with basic features

        Returns:
            DataFrame with advanced features added
        """
        logger.info("Creating advanced features...")
        df_features = df.copy()

        # Consistency metrics
        for stat in ['goals', 'assists', 'points']:
            if f'{stat}_1' in df_features.columns and f'{stat}_2' in df_features.columns:
                # Coefficient of variation (consistency)
                mean_stat = (df_features[f'{stat}_1'] + df_features[f'{stat}_2']) / 2
                std_stat = np.sqrt(((df_features[f'{stat}_1'] - mean_stat) ** 2 +
                                   (df_features[f'{stat}_2'] - mean_stat) ** 2) / 2)
                df_features[f'{stat}_consistency'] = std_stat / mean_stat.replace(0, np.nan)

        # Role-specific features
        if 'role' in df_features.columns:
            # One-hot encode positions
            role_dummies = pd.get_dummies(df_features['role'], prefix='role')
            df_features = pd.concat([df_features, role_dummies], axis=1)

            # Position-specific performance expectations
            df_features['is_attacker'] = (df_features['role'] == 'A').astype(int)
            df_features['is_defenseman'] = (df_features['role'] == 'D').astype(int)
            df_features['is_goalie'] = (df_features['role'] == 'G').astype(int)

        # Experience proxies
        if 'age' in df_features.columns:
            # Estimate career stage
            df_features['career_stage'] = pd.cut(
                df_features['age'],
                bins=[0, 21, 25, 29, 33, 50],
                labels=['rookie', 'developing', 'prime', 'veteran', 'declining']
            )

        # Interaction features
        if all(col in df_features.columns for col in ['age', 'ppg_1']):
            df_features['age_performance_interaction'] = df_features['age'] * df_features['ppg_1']

        if all(col in df_features.columns for col in ['games_1', 'games_2']):
            df_features['durability'] = (df_features['games_1'] + df_features['games_2']) / 2
            df_features['games_consistency'] = abs(df_features['games_1'] - df_features['games_2'])

        logger.info(f"Created {len(df_features.columns) - len(df.columns)} advanced features")
        return df_features

    def create_position_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create position-specific features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with position-specific features
        """
        logger.info("Creating position-specific features...")
        df_features = df.copy()

        if 'role' not in df_features.columns:
            logger.warning("No 'role' column found, skipping position-specific features")
            return df_features

        # Attacker-specific features
        attacker_mask = df_features['role'] == 'A'
        if attacker_mask.any():
            # Goals to assists ratio
            for season in ['1', '2']:
                assists_col = f'assists_{season}'
                goals_col = f'goals_{season}'
                if all(col in df_features.columns for col in [assists_col, goals_col]):
                    df_features.loc[attacker_mask, f'goals_assists_ratio_{season}'] = (
                        df_features.loc[attacker_mask, goals_col] /
                        df_features.loc[attacker_mask, assists_col].replace(0, np.nan)
                    )

        # Defenseman-specific features
        defenseman_mask = df_features['role'] == 'D'
        if defenseman_mask.any():
            # Plus/minus per game (defensive contribution)
            for season in ['1', '2']:
                pm_col = f'plus_minus_{season}'
                games_col = f'games_{season}'
                if all(col in df_features.columns for col in [pm_col, games_col]):
                    df_features.loc[defenseman_mask, f'pm_per_game_{season}'] = (
                        df_features.loc[defenseman_mask, pm_col] /
                        df_features.loc[defenseman_mask, games_col].replace(0, np.nan)
                    )

        # Goalie-specific features (if any goalies in data)
        goalie_mask = df_features['role'] == 'G'
        if goalie_mask.any():
            # Save percentage and GAA would go here if we had goalie stats
            logger.info("Goalie-specific features would be added here with goalie stats")

        logger.info("Position-specific features created")
        return df_features

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       k: int = 20, method: str = 'f_regression') -> Tuple[pd.DataFrame, List[str]]:
        """Select top k features using statistical tests.

        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            method: Selection method ('f_regression' or 'mutual_info')

        Returns:
            Tuple of (selected features DataFrame, selected feature names)
        """
        logger.info(f"Selecting top {k} features using {method}...")

        # Remove non-numeric columns for feature selection
        numeric_X = X.select_dtypes(include=[np.number])

        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=min(k, numeric_X.shape[1]))
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, numeric_X.shape[1]))
        else:
            raise ValueError(f"Unknown method: {method}")

        # Fit selector
        X_selected = selector.fit_transform(numeric_X, y)
        selected_features = numeric_X.columns[selector.get_support()].tolist()

        logger.info(f"Selected {len(selected_features)} features: {selected_features[:5]}...")

        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit the feature engineer and transform the data.

        Args:
            X: Input features
            y: Target variable (optional, used for feature selection)

        Returns:
            Transformed features
        """
        logger.info("Fitting and transforming features...")

        # Create features
        X_features = self.create_basic_features(X)
        X_features = self.create_advanced_features(X_features)
        X_features = self.create_position_specific_features(X_features)

        # Get numeric columns for scaling
        numeric_columns = X_features.select_dtypes(include=[np.number]).columns
        categorical_columns = X_features.select_dtypes(exclude=[np.number]).columns

        # Scale numeric features
        if len(numeric_columns) > 0:
            X_features[numeric_columns] = self.scaler.fit_transform(X_features[numeric_columns])

        # Store feature names
        self.feature_names = X_features.columns.tolist()
        self.is_fitted = True

        logger.info(f"Feature engineering complete. Total features: {len(self.feature_names)}")
        return X_features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted feature engineer.

        Args:
            X: Input features

        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        logger.info("Transforming features...")

        # Create features
        X_features = self.create_basic_features(X)
        X_features = self.create_advanced_features(X_features)
        X_features = self.create_position_specific_features(X_features)

        # Ensure we have the same columns as training
        missing_cols = set(self.feature_names) - set(X_features.columns)
        if missing_cols:
            logger.warning(f"Missing columns in transform: {missing_cols}")
            for col in missing_cols:
                X_features[col] = 0

        # Select only the columns we had during training
        X_features = X_features[self.feature_names]

        # Scale numeric features
        numeric_columns = X_features.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            X_features[numeric_columns] = self.scaler.transform(X_features[numeric_columns])

        return X_features

    def get_feature_importance_data(self) -> Dict[str, Any]:
        """Get data for feature importance analysis.

        Returns:
            Dictionary with feature information
        """
        return {
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names),
            'scaler_type': self.scaler_type,
            'is_fitted': self.is_fitted
        }