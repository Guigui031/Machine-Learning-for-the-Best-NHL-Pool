"""
Feature Transformers for NHL Data Processing
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class FeatureTransformers:
    """Collection of custom feature transformers for NHL data."""

    @staticmethod
    def get_categorical_encoder():
        """Get a categorical encoder for NHL data."""
        from sklearn.preprocessing import LabelEncoder
        return LabelEncoder()

    @staticmethod
    def get_polynomial_features(degree: int = 2):
        """Get polynomial features transformer."""
        from sklearn.preprocessing import PolynomialFeatures
        return PolynomialFeatures(degree=degree, include_bias=False)

    @staticmethod
    def get_feature_selector(k: int = 20):
        """Get feature selector."""
        return SelectKBest(score_func=f_regression, k=k)

    @staticmethod
    def get_dimensionality_reducer(n_components: int = 10):
        """Get PCA dimensionality reducer."""
        return PCA(n_components=n_components)


class NHLDataTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for NHL player data."""

    def __init__(self, scale_features: bool = True, encode_categorical: bool = True):
        """Initialize the transformer.

        Args:
            scale_features: Whether to scale numerical features
            encode_categorical: Whether to encode categorical features
        """
        self.scale_features = scale_features
        self.encode_categorical = encode_categorical
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []

    def fit(self, X, y=None):
        """Fit the transformer to the data."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

            if self.scale_features:
                numeric_columns = X.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    self.scaler = StandardScaler()
                    self.scaler.fit(X[numeric_columns])

            if self.encode_categorical:
                categorical_columns = X.select_dtypes(exclude=[np.number]).columns
                for col in categorical_columns:
                    encoder = LabelEncoder()
                    encoder.fit(X[col].astype(str))
                    self.label_encoders[col] = encoder

        return self

    def transform(self, X):
        """Transform the data."""
        X_transformed = X.copy()

        if isinstance(X_transformed, pd.DataFrame):
            if self.scale_features and self.scaler is not None:
                numeric_columns = X_transformed.select_dtypes(include=[np.number]).columns
                X_transformed[numeric_columns] = self.scaler.transform(X_transformed[numeric_columns])

            if self.encode_categorical:
                for col, encoder in self.label_encoders.items():
                    if col in X_transformed.columns:
                        X_transformed[col] = encoder.transform(X_transformed[col].astype(str))

        return X_transformed