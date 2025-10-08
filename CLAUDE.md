# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project for NHL pool optimization that predicts player performance and optimizes fantasy hockey team selection. The system uses historical NHL data to predict player points per game (PPG) and employs optimization algorithms to select the best possible team within salary and position constraints.

## Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Main Workflows

#### Complete ML Pipeline (Recommended)
```bash
jupyter notebook NHL_ML_Models_Complete.ipynb
```
- Full data collection, preprocessing, and feature engineering
- Model training with multiple algorithms (baseline, advanced, ensemble)
- Performance evaluation and model selection
- Saves trained models to `models_saved/`

#### Team Optimization
```bash
jupyter notebook Team_Optimization_Notebook.ipynb
```
- Loads pre-trained models for predictions
- Applies Linear Programming optimization with constraints
- Generates optimal team within $95.5M salary cap
- Exports results to CSV files

#### Original Analysis (Legacy)
```bash
jupyter notebook ML_for_NHL.ipynb
```
- Historical analysis and exploratory data analysis
- Algorithm comparison and development

### Testing Individual Components
```bash
# Data pipeline (downloads and validates data)
python data_pipeline.py

# Data download from NHL API
python data_download.py

# Data processing and normalization
python process_data.py

# Make predictions with trained models
python model_predictor.py

# Team optimization
python pool_classifier.py
```

## Architecture Overview

### Two-Tier Architecture
The project has both **legacy modules** (root level) and a **modern modular structure** (`ml_models/`, `src/`):

#### Core Data Models (Root Level)
- **`player.py`**: `Player` and `Season` classes
  - Player attributes: name, age, position, salary, height, weight, country
  - Season stats (comprehensive):
    - **Common**: games_played, goals, assists, points, pim
    - **Skaters** (A/D): shots, plus_minus, powerplay_goals/points, shorthanded_goals/points, game_winning_goals, overtime_goals, faceoff_percentage, shooting_percentage, time_on_ice_per_game, avg_shifts_per_game
    - **Goalies** (G): games_started, wins, losses, ties, overtime_losses, shutouts, goals_against, goals_against_avg, shots_against, saves, save_percentage, time_on_ice (total)
  - Role encoding: 'A' (Attacker), 'D' (Defenseman), 'G' (Goalie)
  - Position-specific points: Attackers (2×G + A), Defensemen (3×G + A), Goalies (3×W + 5×SO + 3×G + A)

- **`team.py`**: Team metadata (name, ID, season, points percentage)

- **`config.py`**: Centralized configuration using dataclasses
  - `APIConfig`: NHL API settings (base URL, retries, timeout)
  - `DataConfig`: Data paths and season settings
  - `PoolConfig`: Optimization constraints (salary cap, position limits)
  - `MLConfig`: ML hyperparameters (test size, CV folds, scoring)

#### Data Pipeline (Root Level)
- **`data_download.py`**: NHL API integration
  - Downloads player points, team rosters, season standings
  - API: `https://api-web.nhle.com/v1`
  - Implements caching to avoid redundant calls

- **`data_pipeline.py`**: Comprehensive pipeline class `NHLDataPipeline`
  - Orchestrates download, validation, and dataset creation
  - Methods: `download_all_data()`, `get_all_players_for_seasons()`, `create_training_dataset()`, `prepare_current_season_data()`
  - Integrates with data validation module

- **`process_data.py`**: Data processing utilities
  - `load_player()`: Loads player with multi-season data
  - `get_year_data_skaters()`: Extracts season-specific stats
  - `process_data_skaters()`: Normalizes and engineers features
  - Creates consecutive season pairs for training (seasons N-1, N → predict N+1)

- **`data_validation.py`**: Data quality checks via `DataValidator` class

#### Machine Learning (Modern Structure: `ml_models/`)

**Feature Engineering** (`ml_models/features/`):
- **`feature_engineer.py`**: Main `FeatureEngineer` class
  - `create_basic_features()`: PPG, shooting %, TOI per game, performance trends
  - `create_advanced_features()`: Consistency metrics, role-specific features, age interactions
  - `create_position_specific_features()`: Goals/assists ratios, plus/minus per game
  - Handles scaling (StandardScaler/MinMaxScaler), label encoding, imputation
  - `fit_transform()` for training, `transform()` for inference
  - `save_components()` / `load_components()` for model persistence

- **`hockey_features.py`**: Hockey-specific feature creation via `HockeyFeatures` class
  - Performance metrics, efficiency stats, consistency measures
  - `create_all_hockey_features()`: Applies full suite of hockey features

**Model Training** (`ml_models/models/`):
- **`model_factory.py`**: Factory pattern for creating models
  - `create_baseline_model()`: LinearRegression, Ridge, RandomForest
  - `create_advanced_model()`: XGBoost, SVR
  - `create_ensemble_model()`: VotingRegressor with multiple estimators

- **`baseline_models.py`**, **`advanced_models.py`**, **`ensemble_models.py`**: Model implementations

**Model Evaluation** (`ml_models/evaluation/`):
- **`model_evaluator.py`**: Performance evaluation (RMSE, R², MAE)
- **`cross_validator.py`**: Cross-validation with hyperparameter tuning
- **`metrics.py`**: Custom scoring functions

#### Prediction and Optimization (Root Level)
- **`model_predictor.py`**: Production prediction module
  - `NHLModelPredictor` class loads saved models from `models_saved/`
  - `load_model_artifacts()`: Loads model, feature engineer, metadata
  - `predict()`: Generates PPG predictions with feature alignment
  - Reconstructs `FeatureEngineer` from saved components

- **`pool_classifier.py`**: Team optimization
  - Linear Programming (PuLP) for optimal solution
  - Branch-and-Bound algorithm (educational alternative)
  - Constraints: salary cap ($95.5M default), position limits (12A/6D/2G)
  - Objective: maximize total predicted PPG

- **`ensemble_learning.py`**: Legacy ensemble training (still used in some notebooks)

### Data Structure

```
data/
├── {season}/                          # e.g., 20232024
│   ├── {season}_players_points.json   # Top scorers
│   ├── {season}_standings.json        # Team standings
│   └── teams/
│       └── {team_abbrev}.json         # Team rosters (e.g., TOR.json)

models_saved/                          # Trained model artifacts
├── best_nhl_model_{model_name}.joblib # Saved model
├── model_metadata.json                # Performance metrics
├── training_info.json                 # Training configuration
└── feature_engineer_components.json   # Feature engineering state
```

Season format: `YYYYYYY` (e.g., `20232024` for 2023-24 season)

## Key Workflows

### Training Pipeline
1. **Data Collection**: `data_pipeline.py` downloads multi-season data via NHL API
2. **Player Loading**: Create `Player` objects with `load_player()` from `process_data.py`
3. **Dataset Creation**: Generate training examples with consecutive season pairs (seasons N-1, N → predict N+1)
4. **Feature Engineering**: Apply `FeatureEngineer` to create hockey-specific features
5. **Model Training**: Use `model_factory.py` to create and train ensemble models
6. **Evaluation**: Assess performance with cross-validation (typical R² ~0.77)
7. **Model Saving**: Persist model and feature engineer to `models_saved/`

### Prediction and Optimization Pipeline
1. **Load Model**: `NHLModelPredictor.load_model_artifacts()` from `models_saved/`
2. **Prepare Data**: `prepare_current_season_data()` creates prediction dataset
3. **Generate Predictions**: `predict()` produces PPG estimates for all players
4. **Optimization**: `pool_classifier.py` or Team_Optimization_Notebook solves LP problem
5. **Results**: Export optimal roster to CSV with salary and performance breakdowns

## Configuration

Key settings in `config.py`:
- **Seasons**: Default training seasons 2020-2025, current season 2024-2025
- **Salary Cap**: $95.5M (configurable via `PoolConfig.salary_cap`)
- **Position Limits**: 12 Attackers, 6 Defensemen, 2 Goalies (configurable)
- **API Settings**: 1s request delay, 3 retries, 30s timeout
- **ML Settings**: 20% test split, 5-fold CV, random_state=42

Can load from `config.json` or environment variables (e.g., `NHL_SALARY_CAP`)

## Key Dependencies

- **beautifulsoup4**: Salary data scraping
- **numpy, pandas**: Data manipulation
- **scikit-learn**: ML models, preprocessing, evaluation
- **PuLP**: Linear programming optimization
- **requests**: NHL API fetching
- **xgboost**: Gradient boosting (optional but recommended)
- **joblib**: Model serialization

## Development Notes

- **Feature Alignment**: During prediction, `FeatureEngineer.transform()` aligns features to match training (fills missing with 0)
- **Label Encoders**: Categorical variables (age_group, career_stage, role) are label-encoded and persisted
- **Imputation**: Missing values filled with median strategy, saved with feature engineer
- **Model Persistence**: Models saved as `.joblib`, feature engineer as JSON components
- **Optimization**: LP solver guarantees optimal solution in <1 second; Branch-and-Bound provided for educational purposes
- **Performance**: Best models achieve R² ~0.768 for PPG prediction
- **Data Quality**: `DataValidator` checks for missing values, outliers, and feature quality