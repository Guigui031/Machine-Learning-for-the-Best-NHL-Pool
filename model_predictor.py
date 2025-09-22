"""
NHL Model Predictor - Production-ready prediction module
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

from ml_models.features import FeatureEngineer, HockeyFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class NHLModelPredictor:
    """Production-ready NHL model predictor that handles all the complexities."""

    def __init__(self, models_dir: str = "models_saved"):
        """Initialize the predictor.

        Args:
            models_dir: Directory containing saved model artifacts
        """
        self.models_dir = Path(models_dir)
        self.model = None
        self.feature_engineer = None
        self.training_info = None
        self.metadata = None

    def load_model_artifacts(self) -> bool:
        """Load all model artifacts.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load metadata first
            metadata_path = self.models_dir / "model_metadata.json"
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

            # Load training info
            training_info_path = self.models_dir / "training_info.json"
            with open(training_info_path, 'r') as f:
                self.training_info = json.load(f)

            # Load the model
            model_name = self.metadata['best_model_name'].replace('_', '-')
            model_path = self.models_dir / f"best_nhl_model_{model_name}.joblib"
            self.model = joblib.load(model_path)

            # Reconstruct feature engineer
            self._reconstruct_feature_engineer()

            # Override feature names with model metadata (ground truth)
            self.feature_engineer.feature_names = self.metadata['feature_names']

            print(f"Loaded model: {self.metadata['best_model_name']}")
            print(f"Performance: RMSE={self.metadata['performance_metrics']['rmse']:.4f}")
            print(f"Features: {len(self.feature_engineer.feature_names)}")

            return True

        except Exception as e:
            print(f"Error loading model artifacts: {e}")
            return False

    def _reconstruct_feature_engineer(self):
        """Reconstruct the feature engineer from saved components."""
        components_path = self.models_dir / "feature_engineer_components.json"

        with open(components_path, 'r') as f:
            components = json.load(f)

        # Create feature engineer
        self.feature_engineer = FeatureEngineer(scaler_type=components['scaler_type'])

        # Reconstruct scaler
        if components['scaler_type'] == 'standard':
            self.feature_engineer.scaler = StandardScaler()
            if components['scaler_mean'] is not None:
                self.feature_engineer.scaler.mean_ = np.array(components['scaler_mean'])
                self.feature_engineer.scaler.scale_ = np.array(components['scaler_scale'])
                self.feature_engineer.scaler.var_ = np.array(components['scaler_var'])
                self.feature_engineer.scaler.n_features_in_ = len(components['scaler_mean'])
        elif components['scaler_type'] == 'minmax':
            self.feature_engineer.scaler = MinMaxScaler()
            # Add MinMaxScaler reconstruction if needed

        # Reconstruct imputer
        if components.get('imputer_fitted', False) and components.get('imputer_strategy'):
            self.feature_engineer.imputer = SimpleImputer(strategy=components['imputer_strategy'])

            # Set fitted attributes if available
            if components['imputer_statistics']:
                self.feature_engineer.imputer.statistics_ = np.array(components['imputer_statistics'])

            if components.get('imputer_n_features_in'):
                self.feature_engineer.imputer.n_features_in_ = components['imputer_n_features_in']

            if components.get('imputer_feature_names_in'):
                self.feature_engineer.imputer.feature_names_in_ = np.array(components['imputer_feature_names_in'])
        else:
            # If no imputer was fitted during training, set to None
            self.feature_engineer.imputer = None

        # Reconstruct label encoders
        self.feature_engineer.label_encoders = {}
        for col, encoder_info in components['label_encoders'].items():
            le = LabelEncoder()
            le.classes_ = np.array(encoder_info['classes'])
            self.feature_engineer.label_encoders[col] = le

        self.feature_engineer.feature_names = components['feature_names']
        self.feature_engineer.is_fitted = components['is_fitted']

    def predict(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on player data.

        Args:
            player_data: DataFrame with player statistics

        Returns:
            DataFrame with predictions
        """
        if self.model is None or self.feature_engineer is None:
            raise ValueError("Model not loaded. Call load_model_artifacts() first.")

        print(f"Input data shape: {player_data.shape}")

        # Validate required columns
        required_base_cols = ['goals_1', 'assists_1', 'games_1', 'goals_2', 'assists_2', 'games_2']
        missing_cols = [col for col in required_base_cols if col not in player_data.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Prepare features (same as training)
        exclude_cols = ['player_id', 'name', 'target_points']
        available_exclude_cols = [col for col in exclude_cols if col in player_data.columns]
        X_raw = player_data.drop(columns=available_exclude_cols)

        # Apply hockey feature engineering
        X_hockey = HockeyFeatures.create_all_hockey_features(X_raw)
        print(f"After hockey features: {X_hockey.shape}")

        # Apply feature engineering
        X_processed = self.feature_engineer.transform(X_hockey)
        print(f"After feature engineering: {X_processed.shape}")

        # Align features with training
        X_aligned = self._align_features(X_processed)
        print(f"After alignment: {X_aligned.shape}")

        # Make predictions
        predictions = self.model.predict(X_aligned.values)

        # Create results DataFrame
        results = pd.DataFrame({
            'player_name': player_data['name'].values if 'name' in player_data.columns else range(len(player_data)),
            'position': player_data['role'].values if 'role' in player_data.columns else 'Unknown',
            'predicted_ppg': predictions
        })

        return results.sort_values('predicted_ppg', ascending=False)

    def _align_features(self, X_processed: pd.DataFrame) -> pd.DataFrame:
        """Align features with training expectations."""
        expected_features = set(self.feature_engineer.feature_names)
        actual_features = set(X_processed.columns)

        missing_features = expected_features - actual_features
        extra_features = actual_features - expected_features

        if missing_features:
            print(f"Missing {len(missing_features)} features (will fill with 0):")
            for feat in sorted(list(missing_features))[:5]:  # Show first 5
                print(f"   - {feat}")
            if len(missing_features) > 5:
                print(f"   ... and {len(missing_features) - 5} more")

        if extra_features:
            print(f"{len(extra_features)} extra features (will ignore):")
            for feat in sorted(list(extra_features))[:5]:  # Show first 5
                print(f"   + {feat}")
            if len(extra_features) > 5:
                print(f"   ... and {len(extra_features) - 5} more")

        # Create aligned DataFrame with expected features in exact order
        aligned_data = pd.DataFrame(
            0,
            index=X_processed.index,
            columns=self.feature_engineer.feature_names
        )

        # Copy matching features
        matching_features = expected_features & actual_features
        print(f"Copying {len(matching_features)} matching features")

        for feature in matching_features:
            aligned_data[feature] = X_processed[feature]

        print(f"Final aligned shape: {aligned_data.shape}")
        return aligned_data

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.metadata is None:
            return {}

        return {
            'model_name': self.metadata['best_model_name'],
            'model_type': self.metadata['model_type'],
            'performance': self.metadata['performance_metrics'],
            'training_samples': self.metadata['training_samples'],
            'feature_count': len(self.metadata['feature_names']),
            'timestamp': self.metadata['timestamp']
        }


# Example usage function
def predict_nhl_performance(player_data: pd.DataFrame, models_dir: str = "models_saved") -> pd.DataFrame:
    """Convenience function for making NHL predictions.

    Args:
        player_data: DataFrame with player statistics
        models_dir: Directory containing saved models

    Returns:
        DataFrame with predictions sorted by performance
    """
    predictor = NHLModelPredictor(models_dir)

    if not predictor.load_model_artifacts():
        raise RuntimeError("Failed to load model artifacts")

    return predictor.predict(player_data)


if __name__ == "__main__":
    # Test the predictor
    print("Testing NHL Model Predictor...")

    # Create sample data
    sample_data = pd.DataFrame({
        'name': ['Player A', 'Player B', 'Player C'],
        'role': ['A', 'D', 'G'],
        'age': [25, 28, 22],
        'goals_1': [20, 8, 0],
        'assists_1': [30, 25, 0],
        'games_1': [70, 65, 50],
        'goals_2': [22, 10, 0],
        'assists_2': [32, 28, 0],
        'games_2': [72, 67, 52]
    })

    try:
        results = predict_nhl_performance(sample_data)
        print("Prediction successful!")
        print(results)
    except Exception as e:
        print(f"Test failed: {e}")