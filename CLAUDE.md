# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project for NHL pool optimization that predicts player performance and optimizes fantasy hockey team selection. The project uses historical NHL data to predict player points per game (PPG) and employs optimization algorithms to select the best possible team within salary constraints.

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Main Analysis
The primary workflow is contained in the Jupyter notebook:
```bash
# Launch Jupyter notebook to run the main analysis
jupyter notebook ML_for_NHL.ipynb
```

### Testing Individual Components
```bash
# Test data download functionality
python data_download.py

# Test player data processing
python process_data.py

# Test ensemble learning models
python ensemble_learning.py

# Test optimization algorithms
python pool_classifier.py
```

## Architecture Overview

The project follows a modular architecture with clear separation of concerns:

### Core Data Models
- **`player.py`**: Defines `Player` and `Season` classes that encapsulate player statistics and biographical information
  - Player attributes: name, age, position, salary, height, weight, country
  - Season-specific stats: goals, assists, games played, plus/minus, etc.
  - Points calculation varies by position (Attackers: 2×goals + assists, Defensemen: 3×goals + assists, Goalies: 3×wins + 5×shutouts + 3×goals + assists)

- **`team.py`**: Simple `Team` class for team metadata (name, ID, season, points percentage)

### Data Pipeline
- **`data_download.py`**: Handles NHL API data fetching
  - Downloads player statistics, team rosters, season standings
  - Uses NHL API endpoints (api-web.nhle.com)
  - Implements caching to avoid redundant API calls

- **`process_data.py`**: Data processing and normalization
  - Loads player data from JSON files
  - Normalizes statistics by games played
  - Handles data cleaning and feature engineering
  - Creates training datasets with consecutive season pairs to predict third season

### Machine Learning Components
- **`ensemble_learning.py`**: Implements ensemble regression models
  - Supports XGBoost, LogisticRegression, SVR, SGD models
  - Uses scikit-learn's VotingRegressor for ensemble predictions
  - Includes hyperparameter tuning with RandomizedSearchCV
  - Handles GPU memory management for XGBoost

### Optimization Engine
- **`pool_classifier.py`**: Fantasy team optimization
  - Implements both Linear Programming (PuLP) and Branch-and-Bound algorithms
  - Constraints: salary budget (88M), position limits (12 attackers, 6 defensemen, 2 goalies)
  - Objective: maximize total predicted points per game

### Main Workflow (`ML_for_NHL.ipynb`)
1. **Data Collection**: Extract player data for seasons 2020-2024
2. **Data Cleaning**: Filter players with minimum 3 seasons and 100 total games
3. **Feature Engineering**: Create training examples using consecutive season pairs
4. **Model Training**: Train ensemble models to predict PPG for third season
5. **Team Optimization**: Use trained models to predict current season and optimize team selection

## Data Structure

The project expects data in the following structure:
```
data/
├── {season}/
│   ├── {season}_players_points.json
│   ├── {season}_standings.json
│   └── teams/
│       └── {team_abbrev}.json
```

Where `{season}` follows format `YYYYYYY` (e.g., `20232024` for 2023-24 season).

## Key Dependencies

- **beautifulsoup4**: Web scraping for salary data
- **numpy, pandas**: Data manipulation and analysis  
- **scikit-learn**: Machine learning models and utilities
- **PuLP**: Linear programming optimization
- **requests**: API data fetching
- **xgboost**: Gradient boosting for ensemble models

## Development Notes

- The project uses a 4-season sliding window for training (predicting season N+1 from seasons N-1 and N)
- Player positions are encoded as roles: 'A' (Attacker), 'D' (Defenseman), 'G' (Goalie)
- Salary data is scraped separately and defaults to 88M (maximum budget) for missing values
- The optimization problem balances predicted performance against salary constraints
- Both LP solver and custom Branch-and-Bound implementations are provided for comparison