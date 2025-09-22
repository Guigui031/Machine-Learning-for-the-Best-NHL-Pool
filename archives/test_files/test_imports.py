#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

print("Testing imports...")

try:
    # Test ml_models imports
    from ml_models.features import FeatureEngineer, HockeyFeatures
    print("Features imports successful")

    from ml_models.models import BaselineModels, AdvancedModels, EnsembleModels
    print("Models imports successful")

    from ml_models.evaluation import ModelEvaluator, CrossValidator, NHLMetrics
    print("Evaluation imports successful")

    from ml_models.utils import ModelUtils, FeatureUtils
    print("Utils imports successful")

    # Test data pipeline imports
    from data_pipeline import NHLDataPipeline
    print("Data pipeline import successful")

    from config import config
    print("Config import successful")

    print("\nAll imports successful!")

except ImportError as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()

except Exception as e:
    print(f"Other error: {e}")
    import traceback
    traceback.print_exc()