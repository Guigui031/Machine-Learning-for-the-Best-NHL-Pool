"""
NHL Machine Learning Models Package

This package contains all ML models, feature engineering, and evaluation tools
for NHL player performance prediction and team optimization.
"""

__version__ = "1.0.0"
__author__ = "NHL Pool Optimizer"

from .features import FeatureEngineer, HockeyFeatures
from .models import BaselineModels, AdvancedModels, EnsembleModels
from .evaluation import ModelEvaluator, CrossValidator
from .utils import ModelUtils, FeatureUtils

__all__ = [
    'FeatureEngineer',
    'HockeyFeatures',
    'BaselineModels',
    'AdvancedModels',
    'EnsembleModels',
    'ModelEvaluator',
    'CrossValidator',
    'ModelUtils',
    'FeatureUtils'
]