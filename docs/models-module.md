# Models Module Documentation

The models module (`ensemble_learning.py`) implements machine learning algorithms for predicting NHL player performance. It uses ensemble methods to combine multiple regression models for robust and accurate predictions of player Points Per Game (PPG).

## Module Overview

The ensemble learning approach combines multiple diverse algorithms to create a more robust prediction model than any single algorithm could provide. The module supports hyperparameter tuning, cross-validation, and GPU memory management.

## Core Components

### Supported Models

#### 1. XGBoost Regressor
- **Type**: Gradient Boosting Ensemble
- **Use Case**: Non-linear relationships, feature interactions
- **Hyperparameters**:
  - `learning_rate`: 0.01 to 0.21 (uniform distribution)
  - `n_estimators`: [200, 300, 400, 500]
  - `max_depth`: [3, 5, 7, 10]
  - `subsample`: 0.6 to 1.0 (uniform distribution)
- **Special Features**:
  - GPU acceleration support
  - Built-in regularization
  - Handles missing values automatically

#### 2. Support Vector Regression (SVR)
- **Type**: Kernel-based regression
- **Use Case**: Complex non-linear patterns, robust to outliers
- **Hyperparameters**:
  - `C`: Regularization (10^-3 to 10^3, log scale)
  - `epsilon`: Loss function tolerance (0.1 to 1.0)
  - `kernel`: ['linear', 'poly', 'rbf', 'sigmoid']
  - `degree`: Polynomial degree [2, 3, 4] (for poly kernel)
  - `gamma`: Kernel coefficient ['scale', 'auto']

#### 3. Stochastic Gradient Descent (SGD)
- **Type**: Linear regression with regularization
- **Use Case**: Large datasets, linear relationships
- **Configuration**: Modified Huber loss with Elastic Net penalty
- **Hyperparameters**:
  - `alpha`: Regularization strength (0.01 to 1.0, log scale)
  - `l1_ratio`: L1/L2 penalty mix (0.001 to 1.0)
- **Features**:
  - Fast training on large datasets
  - Robust to outliers (Modified Huber loss)
  - Elastic Net combines L1 and L2 regularization

#### 4. Logistic Regression (Repurposed for Regression)
- **Type**: Linear model with regularization
- **Use Case**: Baseline linear model
- **Hyperparameters**:
  - `C`: Inverse regularization (0.1 to 10.1, uniform)
  - `penalty`: 'l1' (Lasso regularization)
  - `solver`: 'liblinear' (for L1 penalty)

### Core Functions

#### `create_pipeline_and_params(model_name)`
**Purpose**: Creates scikit-learn pipeline and hyperparameter space for each model

**Parameters**:
- `model_name`: String identifier ('XGBoost', 'SVR', 'SGD', 'LogisticRegression')

**Returns**:
- `pipeline`: sklearn Pipeline object
- `param_grid`: Dictionary of hyperparameter distributions

**Architecture**: Each model is wrapped in a Pipeline for consistent preprocessing and parameter naming.

#### `tune_model(pipeline, param_grid, X_train, y_train)`
**Purpose**: Performs hyperparameter optimization using randomized search

**Process**:
1. **Randomized Search**: Uses `RandomizedSearchCV` with 10 iterations
2. **Cross-Validation**: 5-fold stratified cross-validation
3. **Scoring**: Negative mean squared error
4. **Prediction**: Generates cross-validated predictions

**Returns**: Best estimator from hyperparameter search

**Key Features**:
- Memory efficient (n_jobs=1 for GPU models)
- Deterministic results (random_state=0)
- Verbose output for monitoring progress

#### `train_ensemble(X_train, y_train, model_names)`
**Purpose**: Main entry point for training ensemble model

**Current Implementation Note**:
The function currently returns a simple SVR model for performance reasons. The full ensemble implementation is available but commented out for efficiency.

**Full Ensemble Process** (when enabled):
1. Train and tune each individual model
2. Combine models using `VotingRegressor`
3. Fit ensemble on training data
4. Return trained ensemble

**Parameters**:
- `X_train`: Feature matrix (numpy array or pandas DataFrame)
- `y_train`: Target values (PPG predictions)
- `model_names`: List of model names to include in ensemble

#### `estimate(X_test, model)`
**Purpose**: Generates predictions on test data

**Parameters**:
- `X_test`: Test feature matrix
- `model`: Trained model (ensemble or individual)

**Returns**: Array of predicted PPG values

### Cross-Validation Strategy

#### Stratified K-Fold (5 splits)
- **Purpose**: Ensures balanced representation across folds
- **Configuration**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Benefit**: More reliable performance estimates for regression tasks

#### Scoring Metric
- **Primary**: Mean Squared Error (MSE)
- **Implementation**: `make_scorer(mean_squared_error)`
- **Optimization**: Models minimize prediction error

### GPU Memory Management

#### `free_gpu_cache()`
**Purpose**: Manages GPU memory for XGBoost training

**Process**:
1. Display initial GPU utilization
2. Force garbage collection (`gc.collect()`)
3. Clear PyTorch CUDA cache
4. Display updated GPU utilization

**Dependencies**: GPUtil, torch

**Use Case**: Prevents GPU memory overflow during hyperparameter tuning

## Model Selection Strategy

### Ensemble Rationale
The combination of different model types provides several advantages:

1. **XGBoost**: Captures complex feature interactions and non-linear patterns
2. **SVR**: Robust to outliers, good generalization with kernel tricks
3. **SGD**: Fast linear baseline, good for large datasets
4. **Logistic Regression**: Simple linear model for interpretability

### Voting Strategy
- **Type**: `VotingRegressor` with equal weights
- **Aggregation**: Average of individual model predictions
- **Benefit**: Reduces overfitting, improves generalization

## Performance Optimization

### Hyperparameter Search Efficiency
- **Strategy**: Randomized search vs. grid search
- **Iterations**: Limited to 10 for practical runtime
- **Parallelization**: Controlled (n_jobs=1) to manage GPU memory

### Memory Management
- **GPU**: Automatic cache clearing between models
- **CPU**: Garbage collection after each model
- **Pipeline**: Efficient data flow through sklearn pipelines

## Training Workflow

```python
# 1. Prepare data
X_train, y_train = prepare_training_data()

# 2. Define models to use
model_names = ['XGBoost', 'SVR', 'SGD']

# 3. Train ensemble
ensemble_model = train_ensemble(X_train, y_train, model_names)

# 4. Make predictions
predictions = estimate(X_test, ensemble_model)
```

## Integration with NHL Pool System

### Input Features
The models expect normalized features from the data processing module:
- Historical performance metrics (goals, assists, games played)
- Player attributes (age, position, team performance)
- Time-series features (previous seasons' statistics)

### Output
- **Predicted PPG**: Points per game for the upcoming season
- **Use Case**: Input for optimization algorithms to select optimal fantasy teams

### Model Persistence
While not explicitly implemented in this module, trained models can be saved using joblib or pickle for reuse across seasons.

## Error Handling

### Model Creation
- **Invalid Model Names**: Raises `ValueError` with clear message
- **Parameter Validation**: sklearn handles parameter validation
- **Memory Issues**: GPU cache management prevents CUDA out-of-memory errors

### Training Robustness
- **Cross-Validation**: Handles data splits automatically
- **Convergence**: Models have appropriate max_iter settings
- **Numerical Stability**: Robust loss functions and regularization

## Configuration Parameters

### Global Settings
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(mean_squared_error)
```

### Model-Specific Defaults
- **XGBoost**: `eval_metric='logloss'` (though using for regression)
- **LogisticRegression**: `max_iter=1000` for convergence
- **SGD**: `max_iter=10000` with elastic net penalty
- **SVR**: RBF kernel as default

## Future Enhancements

### Potential Improvements
1. **Feature Engineering**: Automated feature selection and creation
2. **Model Stacking**: Meta-learners instead of simple voting
3. **Time Series**: Incorporate temporal dependencies explicitly
4. **Advanced Ensembles**: Boosting-based ensemble methods
5. **Online Learning**: Update models with new data incrementally

### Performance Monitoring
- Cross-validation scores per model
- Ensemble vs. individual model comparison
- Feature importance analysis (for tree-based models)

This models module provides a robust foundation for NHL player performance prediction, balancing accuracy with computational efficiency through ensemble methods and careful hyperparameter optimization.