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
        self.label_encoders = {}
        self.imputer = None

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

            # Age groups (convert to string to avoid categorical issues)
            df_features['age_group'] = pd.cut(
                df_features['age'],
                bins=[0, 23, 27, 31, 50],
                labels=['young', 'prime', 'veteran', 'old']
            ).astype(str)

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
            # Estimate career stage (convert to string to avoid categorical issues)
            df_features['career_stage'] = pd.cut(
                df_features['age'],
                bins=[0, 21, 25, 29, 33, 50],
                labels=['rookie', 'developing', 'prime', 'veteran', 'declining']
            ).astype(str)

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

        # Handle categorical variables with label encoding
        if len(categorical_columns) > 0:
            logger.info(f"Encoding {len(categorical_columns)} categorical columns: {categorical_columns.tolist()}")
            from sklearn.preprocessing import LabelEncoder

            # Initialize label encoders for each categorical column
            self.label_encoders = {}

            for col in categorical_columns:
                # Handle pandas Categorical columns
                if hasattr(X_features[col], 'cat'):
                    # Convert categorical to string first
                    X_features[col] = X_features[col].astype(str)

                # Fill NaN values with 'Unknown' before encoding
                X_features[col] = X_features[col].fillna('Unknown')

                # Ensure 'Unknown' is in the data so it gets encoded
                col_values = X_features[col].astype(str)
                unique_values = col_values.unique().tolist()
                if 'Unknown' not in unique_values:
                    # Add 'Unknown' to ensure it's in the encoder classes
                    col_values_with_unknown = np.append(col_values.values, 'Unknown')
                    le = LabelEncoder()
                    le.fit(col_values_with_unknown)
                    X_features[col] = le.transform(col_values)
                else:
                    # Use label encoding for categorical variables
                    le = LabelEncoder()
                    X_features[col] = le.fit_transform(col_values)

                self.label_encoders[col] = le

            logger.info(f"Encoded categorical variables successfully")

        # Handle any remaining NaN values
        nan_count = X_features.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values, applying imputation...")
            from sklearn.impute import SimpleImputer

            # Initialize imputer if needed
            if self.imputer is None:
                self.imputer = SimpleImputer(strategy='median')

            # Apply imputation to numeric columns
            if len(numeric_columns) > 0:
                X_features[numeric_columns] = self.imputer.fit_transform(X_features[numeric_columns])

            # Fill any remaining NaN in other columns with 0
            X_features = X_features.fillna(0)

            logger.info(f"Imputation complete. Remaining NaN values: {X_features.isnull().sum().sum()}")

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

        # Handle categorical variables FIRST (before alignment)
        if hasattr(self, 'label_encoders') and self.label_encoders:
            for col, le in self.label_encoders.items():
                if col in X_features.columns:
                    # Handle pandas Categorical columns
                    if hasattr(X_features[col], 'cat'):
                        X_features[col] = X_features[col].astype(str)

                    # Fill NaN values with 'Unknown'
                    X_features[col] = X_features[col].fillna('Unknown')

                    # Handle unseen categories
                    col_values = X_features[col].astype(str)
                    unknown_mask = ~col_values.isin(le.classes_)

                    if unknown_mask.any():
                        logger.warning(f"Found {unknown_mask.sum()} unseen categories in {col}")
                        # Find fallback category
                        fallback_category = None
                        for potential_fallback in ['Unknown', 'unknown', 'other', 'Other']:
                            if potential_fallback in le.classes_:
                                fallback_category = potential_fallback
                                break
                        if fallback_category is None:
                            fallback_category = le.classes_[0]
                        X_features.loc[unknown_mask, col] = fallback_category

                    # Transform using the fitted label encoder
                    X_features[col] = le.transform(X_features[col].astype(str))

        # NOW align features to match training
        logger.info(f"Aligning features: current {len(X_features.columns)} -> target {len(self.feature_names)}")

        # Create aligned DataFrame with expected features
        aligned_features = pd.DataFrame(
            0,  # Fill missing features with 0
            index=X_features.index,
            columns=self.feature_names
        )

        # Copy over matching features
        matching_cols = set(X_features.columns) & set(self.feature_names)
        for col in matching_cols:
            aligned_features[col] = X_features[col]

        logger.info(f"Aligned {len(matching_cols)} matching features, filled {len(self.feature_names) - len(matching_cols)} missing with 0")

        X_features = aligned_features

        # Handle any remaining NaN values in aligned data
        nan_count = X_features.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values after alignment, applying imputation...")

            # Apply imputation if we have a fitted imputer
            if self.imputer is not None:
                numeric_columns = X_features.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0 and len(numeric_columns) == self.imputer.n_features_in_:
                    X_features[numeric_columns] = self.imputer.transform(X_features[numeric_columns])

            # Fill any remaining NaN with 0
            X_features = X_features.fillna(0)

        # Scale only numeric features (same as during training)
        numeric_columns = X_features.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            if hasattr(self.scaler, 'n_features_in_') and len(numeric_columns) == self.scaler.n_features_in_:
                logger.info(f"Scaling {len(numeric_columns)} numeric features (matches scaler training)")
                # Use .values to avoid feature name validation issues
                scaled_values = self.scaler.transform(X_features[numeric_columns].values)
                X_features[numeric_columns] = scaled_values
            else:
                logger.warning(f"Numeric feature count mismatch: have {len(numeric_columns)}, scaler expects {getattr(self.scaler, 'n_features_in_', 'unknown')}")
                # Create a subset of features that match what the scaler expects
                scaler_expected = getattr(self.scaler, 'n_features_in_', 0)
                if scaler_expected > 0 and len(numeric_columns) >= scaler_expected:
                    # Use the first N numeric columns that match the scaler expectation
                    logger.info(f"Using first {scaler_expected} numeric columns for scaling")
                    cols_to_scale = numeric_columns[:scaler_expected]
                    # Use .values to avoid feature name validation issues
                    scaled_values = self.scaler.transform(X_features[cols_to_scale].values)
                    X_features[cols_to_scale] = scaled_values
                else:
                    logger.warning("Cannot scale features - dimension mismatch too large")

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

    def save_components(self, filepath: str):
        """Save feature engineer components to JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        import json
        import numpy as np
        from pathlib import Path

        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before saving")

        logger.info(f"Saving feature engineer components to {filepath}")

        # Prepare components for JSON serialization
        components = {
            'scaler_type': self.scaler_type,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }

        # Save scaler parameters
        if hasattr(self.scaler, 'get_params'):
            components['scaler_params'] = self.scaler.get_params()

        # Save scaler fitted attributes
        if hasattr(self.scaler, 'mean_'):
            components['scaler_mean'] = self.scaler.mean_.tolist() if hasattr(self.scaler.mean_, 'tolist') else str(self.scaler.mean_)
        if hasattr(self.scaler, 'scale_'):
            components['scaler_scale'] = self.scaler.scale_.tolist() if hasattr(self.scaler.scale_, 'tolist') else str(self.scaler.scale_)
        if hasattr(self.scaler, 'var_'):
            components['scaler_var'] = self.scaler.var_.tolist() if hasattr(self.scaler.var_, 'tolist') else str(self.scaler.var_)

        # Save imputer information
        if self.imputer is not None:
            components['imputer_fitted'] = True
            components['imputer_strategy'] = self.imputer.strategy
            if hasattr(self.imputer, 'statistics_'):
                components['imputer_statistics'] = self.imputer.statistics_.tolist() if hasattr(self.imputer.statistics_, 'tolist') else str(self.imputer.statistics_)
            if hasattr(self.imputer, 'n_features_in_'):
                components['imputer_n_features_in'] = int(self.imputer.n_features_in_)
            if hasattr(self.imputer, 'feature_names_in_'):
                components['imputer_feature_names_in'] = self.imputer.feature_names_in_.tolist() if hasattr(self.imputer.feature_names_in_, 'tolist') else str(self.imputer.feature_names_in_)
        else:
            components['imputer_fitted'] = False

        # Save label encoders
        components['label_encoders'] = {}
        for col, le in self.label_encoders.items():
            if hasattr(le, 'classes_'):
                components['label_encoders'][col] = {
                    'classes': le.classes_.tolist()
                }

        # Write to file
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(components, f, indent=2)

        logger.info(f"Feature engineer components saved successfully")

    @classmethod
    def load_components(cls, filepath: str):
        """Load feature engineer from saved components.

        Args:
            filepath: Path to the JSON file

        Returns:
            Loaded FeatureEngineer instance
        """
        import json
        import numpy as np
        from pathlib import Path
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
        from sklearn.impute import SimpleImputer

        logger.info(f"Loading feature engineer components from {filepath}")

        with open(filepath, 'r') as f:
            components = json.load(f)

        # Create feature engineer instance
        feature_engineer = cls(scaler_type=components['scaler_type'])

        # Reconstruct scaler
        if components['scaler_type'] == 'standard':
            feature_engineer.scaler = StandardScaler()
            if 'scaler_mean' in components:
                # Handle both list and string formats
                if isinstance(components['scaler_mean'], str):
                    feature_engineer.scaler.mean_ = np.fromstring(components['scaler_mean'].strip('[]'), sep=' ')
                else:
                    feature_engineer.scaler.mean_ = np.array(components['scaler_mean'])

                if isinstance(components['scaler_scale'], str):
                    feature_engineer.scaler.scale_ = np.fromstring(components['scaler_scale'].strip('[]'), sep=' ')
                else:
                    feature_engineer.scaler.scale_ = np.array(components['scaler_scale'])

                if isinstance(components['scaler_var'], str):
                    feature_engineer.scaler.var_ = np.fromstring(components['scaler_var'].strip('[]'), sep=' ')
                else:
                    feature_engineer.scaler.var_ = np.array(components['scaler_var'])

                feature_engineer.scaler.n_features_in_ = len(feature_engineer.scaler.mean_)

        elif components['scaler_type'] == 'minmax':
            feature_engineer.scaler = MinMaxScaler()
            # Add MinMaxScaler reconstruction if needed

        # Reconstruct imputer
        if components.get('imputer_fitted', False):
            feature_engineer.imputer = SimpleImputer(strategy=components.get('imputer_strategy', 'median'))

            if 'imputer_statistics' in components:
                if isinstance(components['imputer_statistics'], str):
                    feature_engineer.imputer.statistics_ = np.fromstring(components['imputer_statistics'].strip('[]'), sep=' ')
                else:
                    feature_engineer.imputer.statistics_ = np.array(components['imputer_statistics'])

            if 'imputer_n_features_in' in components:
                feature_engineer.imputer.n_features_in_ = components['imputer_n_features_in']

            if 'imputer_feature_names_in' in components:
                if isinstance(components['imputer_feature_names_in'], str):
                    # Parse string representation of array
                    feature_names_str = components['imputer_feature_names_in'].strip("[]'").replace("'", "").replace('"', '')
                    feature_engineer.imputer.feature_names_in_ = np.array(feature_names_str.split())
                else:
                    feature_engineer.imputer.feature_names_in_ = np.array(components['imputer_feature_names_in'])
        else:
            feature_engineer.imputer = None

        # Reconstruct label encoders
        feature_engineer.label_encoders = {}
        for col, encoder_info in components['label_encoders'].items():
            le = LabelEncoder()
            le.classes_ = np.array(encoder_info['classes'])
            feature_engineer.label_encoders[col] = le

        # Set other attributes
        feature_engineer.feature_names = components['feature_names']
        feature_engineer.is_fitted = components['is_fitted']

        logger.info(f"Feature engineer components loaded successfully")

        return feature_engineer