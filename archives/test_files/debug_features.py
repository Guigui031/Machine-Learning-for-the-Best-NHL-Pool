#!/usr/bin/env python3
"""
Debug script to compare expected vs actual features
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from ml_models.features import FeatureEngineer, HockeyFeatures

def debug_feature_mismatch():
    """Debug the feature mismatch issue"""

    print("Debugging feature mismatch...")

    # Load expected features from model metadata
    models_dir = Path("models_saved")
    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    expected_features = metadata['feature_names']
    print(f"Expected features from model: {len(expected_features)}")

    # Load saved components
    components_path = models_dir / "feature_engineer_components.json"
    with open(components_path, 'r') as f:
        components = json.load(f)

    saved_features = components['feature_names']
    print(f"Saved features from components: {len(saved_features)}")

    # Create test data similar to what would be used in prediction
    test_data = pd.DataFrame({
        'name': ['Test Player'],
        'role': ['A'],
        'position': ['C'],
        'age': [25],
        'height': [180],
        'weight': [80],
        'country': ['CAN'],
        'salary': [88000000],
        'goals_1': [20],
        'assists_1': [30],
        'pim_1': [20],
        'games_1': [70],
        'shots_1': [200],
        'time_1': [1400],
        'plus_minus_1': [10],
        'team_1': ['TOR'],
        'goals_2': [22],
        'assists_2': [32],
        'pim_2': [22],
        'games_2': [72],
        'shots_2': [210],
        'time_2': [1420],
        'plus_minus_2': [12],
        'team_2': ['TOR']
    })

    print(f"Test data shape: {test_data.shape}")
    print(f"Test data columns: {list(test_data.columns)}")

    # Apply hockey features
    X_hockey = HockeyFeatures.create_all_hockey_features(test_data)
    print(f"Hockey features shape: {X_hockey.shape}")

    # Apply feature engineering
    fe = FeatureEngineer(scaler_type='standard')
    X_engineered = fe.fit_transform(X_hockey, pd.Series([1.0]))  # dummy target

    actual_features = list(X_engineered.columns)
    print(f"Generated features: {len(actual_features)}")

    # Compare features
    expected_set = set(expected_features)
    saved_set = set(saved_features)
    actual_set = set(actual_features)

    print(f"\nFeature comparison:")
    print(f"Expected (model): {len(expected_set)}")
    print(f"Saved (components): {len(saved_set)}")
    print(f"Actual (generated): {len(actual_set)}")

    # Find differences
    missing_from_actual = expected_set - actual_set
    extra_in_actual = actual_set - expected_set

    if missing_from_actual:
        print(f"\nMissing from actual ({len(missing_from_actual)}):")
        for feat in sorted(missing_from_actual)[:10]:  # Show first 10
            print(f"  - {feat}")
        if len(missing_from_actual) > 10:
            print(f"  ... and {len(missing_from_actual) - 10} more")

    if extra_in_actual:
        print(f"\nExtra in actual ({len(extra_in_actual)}):")
        for feat in sorted(extra_in_actual)[:10]:  # Show first 10
            print(f"  + {feat}")
        if len(extra_in_actual) > 10:
            print(f"  ... and {len(extra_in_actual) - 10} more")

    # Check saved vs expected
    missing_from_saved = expected_set - saved_set
    extra_in_saved = saved_set - expected_set

    if missing_from_saved:
        print(f"\nMissing from saved components ({len(missing_from_saved)}):")
        for feat in sorted(missing_from_saved)[:10]:
            print(f"  - {feat}")

    if extra_in_saved:
        print(f"\nExtra in saved components ({len(extra_in_saved)}):")
        for feat in sorted(extra_in_saved)[:10]:
            print(f"  + {feat}")

    return {
        'expected': expected_features,
        'saved': saved_features,
        'actual': actual_features,
        'missing_from_actual': list(missing_from_actual),
        'extra_in_actual': list(extra_in_actual)
    }

if __name__ == "__main__":
    results = debug_feature_mismatch()